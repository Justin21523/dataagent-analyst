from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.app.schemas.backtest_schema import (
    BacktestAssertion,
    BacktestPayloadDetail,
    BacktestPayloadSummary,
    BacktestRunDetail,
    BacktestRunSummary,
    BacktestScreenshotSummary,
    BacktestStep,
    BacktestSuiteSummary,
)


class BacktestArtifactError(Exception):
    """Base exception for backtest artifact reads."""


class BacktestArtifactNotFoundError(BacktestArtifactError):
    """Raised when a requested artifact does not exist."""


class BacktestArtifactValidationError(BacktestArtifactError):
    """Raised when an artifact path or identifier is unsafe."""


class BacktestArtifactService:
    SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backtests_dir = settings.backtests_dir
        self.backtests_dir.mkdir(parents=True, exist_ok=True)

    def list_runs(self, limit: int = 50) -> list[BacktestRunSummary]:
        limit = max(1, min(limit, 200))
        run_dirs = [
            path
            for path in self.backtests_dir.iterdir()
            if path.is_dir() and (path / "run.json").exists()
        ]
        run_dirs.sort(key=lambda path: (path / "run.json").stat().st_mtime, reverse=True)

        return [self._read_run_summary(path) for path in run_dirs[:limit]]

    def get_run_detail(self, run_id: str) -> BacktestRunDetail:
        run_dir = self._safe_run_dir(run_id)
        run_path = run_dir / "run.json"

        if not run_path.exists():
            raise BacktestArtifactNotFoundError(f"Backtest run not found: {run_id}")

        summary = self._read_run_summary(run_dir)
        if summary.status == "invalid":
            return BacktestRunDetail(
                summary=summary,
                payloads=self._list_payloads(run_dir),
                screenshots=self._list_screenshots(run_dir),
                summary_markdown=self._read_optional_text(run_dir / "summary.md"),
            )

        payload = self._load_json_file(run_path)
        steps = [
            BacktestStep.model_validate(step)
            for step in payload.get("steps", [])
            if isinstance(step, dict)
        ]
        assertions = [
            BacktestAssertion.model_validate(assertion)
            for assertion in payload.get("assertions", [])
            if isinstance(assertion, dict)
        ]
        summary_markdown = self._read_optional_text(run_dir / "summary.md")

        return BacktestRunDetail(
            summary=summary,
            steps=steps,
            assertions=assertions,
            payloads=self._list_payloads(run_dir),
            screenshots=self._list_screenshots(run_dir),
            summary_markdown=summary_markdown,
        )

    def get_payload(self, run_id: str, payload_name: str) -> BacktestPayloadDetail:
        run_dir = self._safe_run_dir(run_id)
        payload_path = self._safe_child_file(run_dir / "payloads", payload_name, ".json")

        if not payload_path.exists():
            raise BacktestArtifactNotFoundError(f"Backtest payload not found: {payload_name}")

        return BacktestPayloadDetail(
            run_id=run_id,
            name=payload_path.name,
            payload=self._load_json_file(payload_path),
        )

    def get_screenshot_path(self, run_id: str, screenshot_name: str) -> Path:
        run_dir = self._safe_run_dir(run_id)
        screenshot_path = self._safe_child_file(run_dir / "screenshots", screenshot_name, ".png")

        if not screenshot_path.exists():
            raise BacktestArtifactNotFoundError(f"Backtest screenshot not found: {screenshot_name}")

        return screenshot_path

    def list_suites(self) -> list[BacktestSuiteSummary]:
        suite_paths = sorted(
            self.backtests_dir.glob("*_suite_summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return [self._read_suite_summary(path) for path in suite_paths]

    def get_suite(self, suite_id: str) -> BacktestSuiteSummary:
        path = self._safe_suite_path(suite_id)

        if not path.exists():
            raise BacktestArtifactNotFoundError(f"Backtest suite not found: {suite_id}")

        return self._read_suite_summary(path)

    def _read_run_summary(self, run_dir: Path) -> BacktestRunSummary:
        run_path = run_dir / "run.json"
        payloads = self._list_payloads(run_dir)
        screenshots = self._list_screenshots(run_dir)

        try:
            payload = self._load_json_file(run_path)
            steps = [step for step in payload.get("steps", []) if isinstance(step, dict)]
            assertions = [
                assertion
                for assertion in payload.get("assertions", [])
                if isinstance(assertion, dict)
            ]
            raw_metadata = payload.get("metadata")
            metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
            return BacktestRunSummary(
                run_id=str(payload.get("run_id") or run_dir.name),
                status=str(payload.get("status") or "unknown"),
                created_at=payload.get("created_at"),
                metadata=metadata,
                step_count=len(steps),
                failed_step_count=sum(1 for step in steps if step.get("status") == "failed"),
                assertion_count=len(assertions),
                failed_assertion_count=sum(
                    1 for assertion in assertions if assertion.get("status") == "failed"
                ),
                screenshot_count=len(screenshots),
                payload_count=len(payloads),
                has_summary=(run_dir / "summary.md").exists(),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return BacktestRunSummary(
                run_id=run_dir.name,
                status="invalid",
                screenshot_count=len(screenshots),
                payload_count=len(payloads),
                has_summary=(run_dir / "summary.md").exists(),
                error=str(exc),
            )

    def _read_suite_summary(self, path: Path) -> BacktestSuiteSummary:
        suite_id = path.name.removesuffix("_suite_summary.json")

        try:
            payload = self._load_json_file(path)
            raw_runs = payload.get("runs")
            runs = (
                [run for run in raw_runs if isinstance(run, dict)]
                if isinstance(raw_runs, list)
                else []
            )
            return BacktestSuiteSummary(
                suite_id=suite_id,
                suite=str(payload.get("suite") or suite_id),
                status=str(payload.get("status") or "unknown"),
                created_at=payload.get("created_at"),
                runs=runs,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return BacktestSuiteSummary(
                suite_id=suite_id,
                suite=suite_id,
                status="invalid",
                error=str(exc),
            )

    def _list_payloads(self, run_dir: Path) -> list[BacktestPayloadSummary]:
        payload_dir = run_dir / "payloads"
        if not payload_dir.exists():
            return []

        return [
            BacktestPayloadSummary(
                name=path.name,
                path=str(path.relative_to(run_dir)),
                size_bytes=path.stat().st_size,
            )
            for path in sorted(payload_dir.glob("*.json"))
            if path.is_file()
        ]

    def _list_screenshots(self, run_dir: Path) -> list[BacktestScreenshotSummary]:
        screenshot_dir = run_dir / "screenshots"
        if not screenshot_dir.exists():
            return []

        return [
            BacktestScreenshotSummary(
                name=path.name,
                path=str(path.relative_to(run_dir)),
                size_bytes=path.stat().st_size,
            )
            for path in sorted(screenshot_dir.glob("*.png"))
            if path.is_file()
        ]

    def _safe_run_dir(self, run_id: str) -> Path:
        if not self.SAFE_NAME_PATTERN.fullmatch(run_id) or "/" in run_id or "\\" in run_id:
            raise BacktestArtifactValidationError(f"Unsafe backtest run ID: {run_id}")

        run_dir = (self.backtests_dir / run_id).resolve()
        self._ensure_inside_backtests(run_dir)
        return run_dir

    def _safe_suite_path(self, suite_id: str) -> Path:
        if not self.SAFE_NAME_PATTERN.fullmatch(suite_id) or "/" in suite_id or "\\" in suite_id:
            raise BacktestArtifactValidationError(f"Unsafe backtest suite ID: {suite_id}")

        suite_path = (self.backtests_dir / f"{suite_id}_suite_summary.json").resolve()
        self._ensure_inside_backtests(suite_path)
        return suite_path

    def _safe_child_file(self, directory: Path, name: str, suffix: str) -> Path:
        if not self.SAFE_NAME_PATTERN.fullmatch(name) or "/" in name or "\\" in name:
            raise BacktestArtifactValidationError(f"Unsafe backtest artifact name: {name}")
        if Path(name).suffix != suffix:
            raise BacktestArtifactValidationError(f"Backtest artifact must end with {suffix}")

        file_path = (directory / name).resolve()
        self._ensure_inside_backtests(file_path)
        return file_path

    def _ensure_inside_backtests(self, path: Path) -> None:
        root = self.backtests_dir.resolve()
        if path != root and root not in path.parents:
            raise BacktestArtifactValidationError(
                "Backtest artifact path escapes runtime directory."
            )

    def _load_json_file(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path.name}")
        return payload

    def _read_optional_text(self, path: Path) -> str | None:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
