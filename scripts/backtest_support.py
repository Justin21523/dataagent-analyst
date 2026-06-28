from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx

TERMINAL_JOB_STATUSES = {"success", "completed_with_warnings", "failed"}


@dataclass
class StepResult:
    name: str
    status: str
    duration_seconds: float
    payload_path: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssertionResult:
    name: str
    status: str
    expected: Any = None
    actual: Any = None
    message: str | None = None


class ArtifactStore:
    def __init__(self, root: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        self.run_dir = root / self.run_id
        self.payload_dir = self.run_dir / "payloads"
        self.screenshot_dir = self.run_dir / "screenshots"
        self.steps: list[StepResult] = []
        self.assertions: list[AssertionResult] = []
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def payload_path(self, name: str) -> Path:
        return self.payload_dir / f"{safe_name(name)}.json"

    def screenshot_path(self, name: str) -> Path:
        return self.screenshot_dir / f"{safe_name(name)}.png"

    def write_payload(self, name: str, payload: Any) -> str:
        path = self.payload_path(name)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path.relative_to(self.run_dir))

    def add_step(self, result: StepResult) -> None:
        self.steps.append(result)

    def add_assertion(self, result: AssertionResult) -> None:
        self.assertions.append(result)

    def assert_that(
        self,
        name: str,
        condition: bool,
        *,
        expected: Any = None,
        actual: Any = None,
        message: str | None = None,
    ) -> None:
        status = "success" if condition else "failed"
        self.add_assertion(
            AssertionResult(
                name=name,
                status=status,
                expected=expected,
                actual=actual,
                message=message,
            )
        )
        if not condition:
            raise AssertionError(message or f"Backtest assertion failed: {name}")

    def status(self) -> str:
        if any(step.status == "failed" for step in self.steps):
            return "failed"
        if any(assertion.status == "failed" for assertion in self.assertions):
            return "failed"
        return "success"

    def write_run_json(self, metadata: dict[str, Any] | None = None) -> Path:
        payload = {
            "run_id": self.run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
            "status": self.status(),
            "steps": [
                {
                    "name": step.name,
                    "status": step.status,
                    "duration_seconds": step.duration_seconds,
                    "payload_path": step.payload_path,
                    "error": step.error,
                    "metadata": step.metadata,
                }
                for step in self.steps
            ],
            "assertions": [
                {
                    "name": assertion.name,
                    "status": assertion.status,
                    "expected": assertion.expected,
                    "actual": assertion.actual,
                    "message": assertion.message,
                }
                for assertion in self.assertions
            ],
        }
        path = self.run_dir / "run.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_summary(self, metadata: dict[str, Any] | None = None) -> Path:
        failed = [step for step in self.steps if step.status == "failed"]
        skipped = [step for step in self.steps if step.status == "skipped"]
        failed_assertions = [
            assertion for assertion in self.assertions if assertion.status == "failed"
        ]
        lines = [
            f"# Backtest Run {self.run_id}",
            "",
            f"- Status: {self.status()}",
            f"- Steps: {len(self.steps)}",
            f"- Failed: {len(failed)}",
            f"- Skipped: {len(skipped)}",
            f"- Assertions: {len(self.assertions)}",
            f"- Failed assertions: {len(failed_assertions)}",
        ]
        for key, value in (metadata or {}).items():
            lines.append(f"- {key}: `{value}`")
        lines.extend(["", "## Steps", ""])
        for step in self.steps:
            suffix = f" ({step.error})" if step.error else ""
            lines.append(f"- `{step.status}` {step.name} - {step.duration_seconds:.2f}s{suffix}")
        if self.assertions:
            lines.extend(["", "## Assertions", ""])
            for assertion in self.assertions:
                suffix = f" ({assertion.message})" if assertion.message else ""
                lines.append(f"- `{assertion.status}` {assertion.name}{suffix}")
        path = self.run_dir / "summary.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


class BacktestHttpClient:
    def __init__(self, base_url: str, store: ArtifactStore) -> None:
        self.base_url = base_url.rstrip("/")
        self.store = store
        self.client = httpx.Client(base_url=self.base_url, timeout=120)

    def close(self) -> None:
        self.client.close()

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        started_at = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.store.add_step(
                StepResult(
                    name=name,
                    status="failed",
                    duration_seconds=time.perf_counter() - started_at,
                    error=str(exc),
                )
            )
            raise

    def request_json(
        self,
        name: str,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        expected_status: set[int] | None = None,
    ) -> dict[str, Any]:
        expected_status = expected_status or {200, 201, 202}
        started_at = time.perf_counter()
        try:
            response = self.client.request(method, path, json=json_payload, files=files)
            try:
                payload = response.json()
            except ValueError as exc:
                body_preview = response.text[:250]
                raise RuntimeError(
                    f"{method} {path} returned non-JSON response "
                    f"{response.status_code}: {body_preview}"
                ) from exc
            if response.status_code not in expected_status:
                raise RuntimeError(f"{method} {path} returned {response.status_code}: {payload}")
            payload_path = self.store.write_payload(name, payload)
            self.store.add_step(
                StepResult(
                    name=name,
                    status="success",
                    duration_seconds=time.perf_counter() - started_at,
                    payload_path=payload_path,
                    metadata={"method": method, "path": path, "status_code": response.status_code},
                )
            )
            return payload
        except Exception as exc:
            self.store.add_step(
                StepResult(
                    name=name,
                    status="failed",
                    duration_seconds=time.perf_counter() - started_at,
                    error=str(exc),
                    metadata={"method": method, "path": path},
                )
            )
            raise

    def request_json_expected_error(
        self,
        name: str,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        expected_status: set[int] | None = None,
    ) -> dict[str, Any]:
        expected_status = expected_status or {400, 422}
        started_at = time.perf_counter()
        response = self.client.request(method, path, json=json_payload, files=files)
        try:
            payload = response.json()
        except ValueError as exc:
            body_preview = response.text[:250]
            self.store.add_step(
                StepResult(
                    name=name,
                    status="failed",
                    duration_seconds=time.perf_counter() - started_at,
                    error=(
                        f"{method} {path} returned non-JSON response "
                        f"{response.status_code}: {body_preview}"
                    ),
                    metadata={"method": method, "path": path, "status_code": response.status_code},
                )
            )
            raise RuntimeError("Expected JSON error response.") from exc

        if response.status_code not in expected_status:
            self.store.add_step(
                StepResult(
                    name=name,
                    status="failed",
                    duration_seconds=time.perf_counter() - started_at,
                    error=f"{method} {path} returned unexpected status {response.status_code}",
                    metadata={"method": method, "path": path, "status_code": response.status_code},
                )
            )
            raise RuntimeError(f"Expected error response, got {response.status_code}: {payload}")

        payload_path = self.store.write_payload(name, payload)
        self.store.add_step(
            StepResult(
                name=name,
                status="success",
                duration_seconds=time.perf_counter() - started_at,
                payload_path=payload_path,
                metadata={
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "expected_error": True,
                },
            )
        )
        return payload

    def request_json_optional(
        self,
        name: str,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        expected_status: set[int] | None = None,
    ) -> dict[str, Any] | None:
        try:
            return self.request_json(
                name,
                method,
                path,
                json_payload=json_payload,
                files=files,
                expected_status=expected_status,
            )
        except Exception as exc:
            if self.store.steps and self.store.steps[-1].name == name:
                self.store.steps.pop()
            self.skip(name, f"Skipped optional step: {exc}")
            return None

    def skip(self, name: str, reason: str) -> None:
        self.store.add_step(
            StepResult(name=name, status="skipped", duration_seconds=0.0, error=reason)
        )


class ManagedServices:
    def __init__(
        self,
        *,
        base_url: str,
        frontend_url: str,
        python: str,
        no_start: bool,
        keep_servers: bool,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.frontend_url = frontend_url.rstrip("/")
        self.python = python
        self.no_start = no_start
        self.keep_servers = keep_servers
        self.processes: list[subprocess.Popen[str]] = []

    def __enter__(self) -> ManagedServices:
        self.ensure_backend()
        self.ensure_frontend()
        return self

    def __exit__(self, *_: object) -> None:
        if self.keep_servers:
            return
        for process in reversed(self.processes):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def ensure_backend(self) -> None:
        if backend_health_ok(self.base_url):
            return
        if self.no_start:
            raise RuntimeError(f"Backend is not healthy: {self.base_url}/api/health")
        if port_open(self.base_url):
            raise RuntimeError(f"Backend port is occupied but health check failed: {self.base_url}")
        host, port = host_port(self.base_url)
        process = subprocess.Popen(
            [
                self.python,
                "-m",
                "uvicorn",
                "backend.app.main:app",
                "--host",
                host,
                "--port",
                str(port),
            ],
            text=True,
        )
        self.processes.append(process)
        wait_for_url(f"{self.base_url}/api/health", "backend")

    def ensure_frontend(self) -> None:
        if url_ok(self.frontend_url):
            return
        if self.no_start:
            raise RuntimeError(f"Frontend is not reachable: {self.frontend_url}")
        if port_open(self.frontend_url):
            raise RuntimeError(f"Frontend port is occupied but not reachable: {self.frontend_url}")
        _, port = host_port(self.frontend_url)
        process = subprocess.Popen(
            [
                self.python,
                "-m",
                "http.server",
                str(port),
                "--directory",
                "frontend",
            ],
            text=True,
        )
        self.processes.append(process)
        wait_for_url(self.frontend_url, "frontend")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument(
        "--frontend-url",
        default=os.getenv("FRONTEND_URL", "http://127.0.0.1:5173"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(os.getenv("BACKTEST_ARTIFACT_DIR", "data/backtests")),
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=Path(os.getenv("BACKTEST_SAMPLE_FILE", "data/samples/customer_churn_demo.csv")),
    )
    parser.add_argument("--python", default=os.getenv("PYTHON", ".venv/bin/python"))
    parser.add_argument("--run-id", default=os.getenv("BACKTEST_RUN_ID"))
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument(
        "--keep-servers",
        action="store_true",
        default=os.getenv("BACKTEST_KEEP_SERVERS") == "1",
    )


def ensure_demo_dataset(sample_file: Path, python: str) -> None:
    if sample_file.exists():
        return
    subprocess.run([python, "scripts/generate_demo_dataset.py"], check=True)


def wait_for_url(url: str, label: str, timeout_seconds: float = 45.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if url_ok(url):
            return
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {label}: {url}")


def url_ok(url: str) -> bool:
    try:
        response = httpx.get(url, timeout=2)
        return response.status_code < 500
    except httpx.HTTPError:
        return False


def backend_health_ok(base_url: str) -> bool:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/health", timeout=2)
        if response.status_code != 200:
            return False
        payload = response.json()
        return isinstance(payload, dict) and "status" in payload
    except (ValueError, httpx.HTTPError):
        return False


def port_open(url: str) -> bool:
    host, port = host_port(url)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.port:
        raise ValueError(f"URL must include host and port: {url}")
    return parsed.hostname, parsed.port


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def write_suite_summary(
    artifact_dir: Path,
    suite_name: str,
    runs: list[dict[str, Any]],
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": suite_name,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "failed" if any(run.get("status") == "failed" for run in runs) else "success",
        "runs": runs,
    }
    path = artifact_dir / f"{safe_name(suite_name)}_suite_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
