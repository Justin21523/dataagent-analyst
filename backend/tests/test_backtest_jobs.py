from pathlib import Path

import pytest
from backend.app.api.routes_backtests import (
    get_backtest_job_registry_service,
    get_backtest_job_runner_service,
)
from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.schemas.backtest_schema import BacktestJobRequest
from backend.app.services.backtest_job_registry_service import BacktestJobRegistryService
from backend.app.services.backtest_job_runner_service import (
    BacktestJobRunnerService,
)
from fastapi.testclient import TestClient


class FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def __iter__(self):
        return iter(self.lines)


class FakeProcess:
    def __init__(self, command: list[str], exit_code: int = 0) -> None:
        self.command = command
        self.stdout = FakeStdout(
            [
                "Backtest artifacts: data/backtests/20260629T000000Z-test\n",
                "Scenario suite summary: data/backtests/scenario_backtest_suite_summary.json\n",
            ]
        )
        self.exit_code = exit_code

    def wait(self) -> int:
        return self.exit_code


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        processed_data_dir=tmp_path / "processed",
        backtests_dir=tmp_path / "data" / "backtests",
    )


def test_backtest_job_runner_records_success_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_popen(command: list[str], **_: object) -> FakeProcess:
        commands.append(command)
        return FakeProcess(command)

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    settings = _settings(tmp_path)
    runner = BacktestJobRunnerService(settings)
    request = BacktestJobRequest(suite_type="api")

    job = runner.create_job(request)
    runner.run_job(job.job_id, request)

    completed = BacktestJobRegistryService(settings).get_job(job.job_id)

    assert completed.status == "success"
    assert completed.run_ids == ["20260629T000000Z-test"]
    assert completed.suite_ids == ["scenario_backtest"]
    assert any(event.event_type == "log" for event in completed.events)
    assert commands[0][1] == "scripts/backtest_full_workflow.py"


def test_backtest_job_runner_records_failed_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_popen(command: list[str], **_: object) -> FakeProcess:
        return FakeProcess(command, exit_code=2)

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    settings = _settings(tmp_path)
    runner = BacktestJobRunnerService(settings)
    request = BacktestJobRequest(suite_type="scenario")

    job = runner.create_job(request)
    runner.run_job(job.job_id, request)

    failed = BacktestJobRegistryService(settings).get_job(job.job_id)

    assert failed.status == "failed"
    assert failed.error_message
    assert failed.result and failed.result["exit_code"] == 2


def test_backtest_job_runner_rejects_sample_file_outside_project(tmp_path: Path) -> None:
    runner = BacktestJobRunnerService(_settings(tmp_path))

    with pytest.raises(Exception, match="escapes project root"):
        runner.create_job(
            BacktestJobRequest(
                suite_type="api",
                sample_file="../outside.csv",
            )
        )


def test_backtest_job_routes_start_and_list_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_popen(command: list[str], **_: object) -> FakeProcess:
        return FakeProcess(command)

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    settings = _settings(tmp_path)
    app = create_app()
    app.dependency_overrides[get_backtest_job_registry_service] = lambda: (
        BacktestJobRegistryService(settings)
    )
    app.dependency_overrides[get_backtest_job_runner_service] = lambda: BacktestJobRunnerService(
        settings
    )
    client = TestClient(app)

    start_response = client.post("/api/backtests/jobs", json={"suite_type": "api"})

    assert start_response.status_code == 202

    job_id = start_response.json()["job"]["job_id"]
    detail_response = client.get(f"/api/backtests/jobs/{job_id}")
    list_response = client.get("/api/backtests/jobs")
    events_response = client.get(f"/api/backtests/jobs/{job_id}/events")

    assert detail_response.status_code == 200
    assert detail_response.json()["status"] == "success"
    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1
    assert events_response.status_code == 200
    assert events_response.json()["total"] >= 3

    app.dependency_overrides.clear()
