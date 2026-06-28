import json
from pathlib import Path

import pytest
from backend.app.api.routes_backtests import get_backtest_service
from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.services.backtest_artifact_service import (
    BacktestArtifactService,
    BacktestArtifactValidationError,
)
from fastapi.testclient import TestClient


def _settings(tmp_path: Path) -> Settings:
    return Settings(backtests_dir=tmp_path)


def _write_run(root: Path, run_id: str = "20260628T120000Z-test") -> Path:
    run_dir = root / run_id
    payload_dir = run_dir / "payloads"
    screenshot_dir = run_dir / "screenshots"
    payload_dir.mkdir(parents=True)
    screenshot_dir.mkdir(parents=True)

    (payload_dir / "ui_network_summary.json").write_text(
        json.dumps({"network_errors": []}),
        encoding="utf-8",
    )
    (screenshot_dir / "upload.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run_dir / "summary.md").write_text("# Backtest Run\n\n- Status: success\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-06-28T12:00:00+00:00",
                "metadata": {"suite": "ui"},
                "status": "success",
                "steps": [
                    {
                        "name": "ui_open_upload",
                        "status": "success",
                        "duration_seconds": 0.25,
                        "payload_path": "payloads/ui_network_summary.json",
                        "error": None,
                        "metadata": {"route": "data-upload"},
                    }
                ],
                "assertions": [
                    {
                        "name": "ui_has_no_network_errors",
                        "status": "success",
                        "expected": [],
                        "actual": [],
                        "message": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    return run_dir


def test_backtest_service_lists_and_reads_run_detail(tmp_path: Path) -> None:
    _write_run(tmp_path)
    service = BacktestArtifactService(_settings(tmp_path))

    runs = service.list_runs()
    detail = service.get_run_detail(runs[0].run_id)

    assert len(runs) == 1
    assert runs[0].status == "success"
    assert runs[0].step_count == 1
    assert runs[0].assertion_count == 1
    assert runs[0].payload_count == 1
    assert runs[0].screenshot_count == 1
    assert detail.steps[0].name == "ui_open_upload"
    assert detail.assertions[0].name == "ui_has_no_network_errors"
    assert detail.summary_markdown and "Backtest Run" in detail.summary_markdown


def test_backtest_service_reads_payload_and_rejects_unsafe_names(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    service = BacktestArtifactService(_settings(tmp_path))

    payload = service.get_payload(run_dir.name, "ui_network_summary.json")

    assert payload.payload["network_errors"] == []

    with pytest.raises(BacktestArtifactValidationError):
        service.get_payload(run_dir.name, "../run.json")

    with pytest.raises(BacktestArtifactValidationError):
        service.get_screenshot_path(run_dir.name, "upload.json")


def test_backtest_service_lists_corrupt_run_as_invalid(tmp_path: Path) -> None:
    run_dir = tmp_path / "broken-run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("{not-json", encoding="utf-8")

    runs = BacktestArtifactService(_settings(tmp_path)).list_runs()

    assert runs[0].run_id == "broken-run"
    assert runs[0].status == "invalid"
    assert runs[0].error


def test_backtest_service_reads_suite_summary(tmp_path: Path) -> None:
    (tmp_path / "scenario_backtest_suite_summary.json").write_text(
        json.dumps(
            {
                "suite": "scenario_backtest",
                "created_at": "2026-06-28T12:30:00+00:00",
                "status": "failed",
                "runs": [{"scenario": "missing_values", "status": "failed"}],
            }
        ),
        encoding="utf-8",
    )

    service = BacktestArtifactService(_settings(tmp_path))

    suites = service.list_suites()
    suite = service.get_suite("scenario_backtest")

    assert suites[0].suite_id == "scenario_backtest"
    assert suite.status == "failed"
    assert suite.runs[0]["scenario"] == "missing_values"


def test_backtest_routes_use_configured_artifact_directory(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    (tmp_path / "scenario_backtest_suite_summary.json").write_text(
        json.dumps({"suite": "scenario_backtest", "status": "success", "runs": []}),
        encoding="utf-8",
    )

    settings = _settings(tmp_path)
    app = create_app()
    app.dependency_overrides[get_backtest_service] = lambda: BacktestArtifactService(settings)
    client = TestClient(app)

    list_response = client.get("/api/backtests/runs")
    detail_response = client.get(f"/api/backtests/runs/{run_dir.name}")
    payload_response = client.get(
        f"/api/backtests/runs/{run_dir.name}/payloads/ui_network_summary.json"
    )
    screenshot_response = client.get(f"/api/backtests/runs/{run_dir.name}/screenshots/upload.png")
    suite_response = client.get("/api/backtests/suites/scenario_backtest")

    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1
    assert detail_response.status_code == 200
    assert detail_response.json()["steps"][0]["name"] == "ui_open_upload"
    assert payload_response.status_code == 200
    assert payload_response.json()["payload"]["network_errors"] == []
    assert screenshot_response.status_code == 200
    assert screenshot_response.headers["content-type"] == "image/png"
    assert suite_response.status_code == 200
    assert suite_response.json()["suite"] == "scenario_backtest"

    app.dependency_overrides.clear()
