from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.schemas.backtest_schema import BacktestJobDetail, BacktestJobRequest
from backend.app.services.backtest_job_registry_service import BacktestJobRegistryService

TERMINAL_STATUSES = {"success", "failed"}


class BacktestJobRunnerError(Exception):
    """Raised when a backtest job cannot be prepared or executed."""


class BacktestJobRunnerService:
    SCRIPT_SEQUENCE = {
        "api": ("scripts/backtest_full_workflow.py",),
        "scenario": ("scripts/backtest_scenarios.py",),
        "ui": ("scripts/ui_backtest_workflow.py",),
        "full": (
            "scripts/backtest_full_workflow.py",
            "scripts/backtest_scenarios.py",
            "scripts/ui_backtest_workflow.py",
        ),
    }

    RUN_LINE_PATTERN = re.compile(r"^(?:UI backtest artifacts|Backtest artifacts): (.+)$")
    SUITE_LINE_PATTERN = re.compile(r"^Scenario suite summary: (.+)$")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = BacktestJobRegistryService(settings)

    def create_job(self, request: BacktestJobRequest) -> BacktestJobDetail:
        self._build_commands(request)
        return self.registry.create_job(request)

    def run_job(self, job_id: str, request: BacktestJobRequest) -> None:
        self.registry.mark_running(job_id)
        run_ids: list[str] = []
        suite_ids: list[str] = []
        commands = self._build_commands(request)

        for index, command in enumerate(commands, start=1):
            script_name = command[1]
            self.registry.append_event(
                job_id,
                event_type="command",
                status="running",
                message=f"Running {script_name}",
                payload={"command": command, "index": index, "total": len(commands)},
            )
            exit_code = self._run_command(job_id, command, run_ids, suite_ids)

            if exit_code != 0:
                result = {
                    "suite_type": request.suite_type,
                    "run_ids": run_ids,
                    "suite_ids": suite_ids,
                    "failed_command": command,
                    "exit_code": exit_code,
                }
                self.registry.mark_failed(
                    job_id,
                    f"Backtest command failed with exit code {exit_code}: {script_name}",
                    result,
                )
                return

        self.registry.mark_completed(
            job_id,
            {
                "suite_type": request.suite_type,
                "run_ids": run_ids,
                "suite_ids": suite_ids,
                "artifact_dir": str(self.settings.backtests_dir),
            },
        )

    def _run_command(
        self,
        job_id: str,
        command: list[str],
        run_ids: list[str],
        suite_ids: list[str],
    ) -> int:
        process = subprocess.Popen(
            command,
            cwd=self.settings.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        assert process.stdout is not None
        for line in process.stdout:
            message = line.rstrip()
            if not message:
                continue

            self._capture_artifact_reference(message, run_ids, suite_ids)
            self.registry.append_event(
                job_id,
                event_type="log",
                status="running",
                message=message,
                payload={},
            )

        return process.wait()

    def _build_commands(self, request: BacktestJobRequest) -> list[list[str]]:
        sample_file = self._safe_project_file(
            Path(request.sample_file or "data/samples/customer_churn_demo.csv")
        )
        scripts = self.SCRIPT_SEQUENCE[request.suite_type]
        base_url = f"http://127.0.0.1:{request.backend_port}"
        frontend_url = f"http://127.0.0.1:{request.frontend_port}"

        commands: list[list[str]] = []
        for script in scripts:
            script_path = self._safe_project_file(Path(script))
            command = [
                sys.executable,
                str(script_path.relative_to(self.settings.project_root)),
                "--base-url",
                base_url,
                "--frontend-url",
                frontend_url,
                "--sample-file",
                str(sample_file.relative_to(self.settings.project_root)),
                "--artifact-dir",
                str(self.settings.backtests_dir.relative_to(self.settings.project_root)),
                "--python",
                sys.executable,
            ]
            if request.keep_servers:
                command.append("--keep-servers")
            commands.append(command)

        return commands

    def _safe_project_file(self, path: Path) -> Path:
        resolved = (self.settings.project_root / path).resolve()
        root = self.settings.project_root.resolve()
        if resolved != root and root not in resolved.parents:
            raise BacktestJobRunnerError(f"Path escapes project root: {path}")
        return resolved

    def _capture_artifact_reference(
        self,
        message: str,
        run_ids: list[str],
        suite_ids: list[str],
    ) -> None:
        run_match = self.RUN_LINE_PATTERN.match(message)
        if run_match:
            run_id = Path(run_match.group(1)).name
            if run_id not in run_ids:
                run_ids.append(run_id)
            return

        suite_match = self.SUITE_LINE_PATTERN.match(message)
        if suite_match:
            suite_id = Path(suite_match.group(1)).name.removesuffix("_suite_summary.json")
            if suite_id not in suite_ids:
                suite_ids.append(suite_id)
