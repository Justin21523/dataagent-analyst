import argparse
import json
from contextlib import suppress
from pathlib import Path

from scripts.backtest_scenarios import build_churn_rows, write_rows
from scripts.backtest_support import (
    ArtifactStore,
    StepResult,
    add_common_arguments,
    ensure_demo_dataset,
    safe_name,
    write_suite_summary,
)


def test_artifact_store_writes_run_and_summary(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, run_id="test-run")
    payload_path = store.write_payload("step one", {"ok": True})
    store.add_step(
        StepResult(
            name="step one",
            status="success",
            duration_seconds=0.1,
            payload_path=payload_path,
        )
    )

    run_path = store.write_run_json({"dataset_id": "dataset-1"})
    summary_path = store.write_summary({"dataset_id": "dataset-1"})

    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert run_payload["run_id"] == "test-run"
    assert run_payload["status"] == "success"
    assert run_payload["assertions"] == []
    assert "step one" in summary_path.read_text(encoding="utf-8")


def test_artifact_store_failed_assertion_marks_run_failed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, run_id="test-run")

    with suppress(AssertionError):
        store.assert_that(
            "metric_floor",
            False,
            expected=">= 0.45",
            actual=0.2,
            message="metric below floor",
        )

    run_path = store.write_run_json()
    summary_path = store.write_summary()
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))

    assert run_payload["status"] == "failed"
    assert run_payload["assertions"][0]["name"] == "metric_floor"
    assert run_payload["assertions"][0]["status"] == "failed"
    assert "Failed assertions: 1" in summary_path.read_text(encoding="utf-8")


def test_common_arguments_have_expected_defaults() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)

    args = parser.parse_args([])

    assert args.base_url == "http://127.0.0.1:8000"
    assert args.frontend_url == "http://127.0.0.1:5173"
    assert args.artifact_dir == Path("data/backtests")


def test_safe_name_replaces_path_characters() -> None:
    assert safe_name("ui/model route") == "ui_model_route"


def test_ensure_demo_dataset_uses_existing_file(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sample.write_text("a,b\n1,2\n", encoding="utf-8")

    ensure_demo_dataset(sample, ".venv/bin/python")

    assert sample.read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_write_suite_summary_marks_any_failed_run_failed(tmp_path: Path) -> None:
    path = write_suite_summary(
        tmp_path,
        "scenario_backtest",
        [
            {"scenario": "happy", "status": "success"},
            {"scenario": "negative", "status": "failed"},
        ],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "failed"
    assert len(payload["runs"]) == 2


def test_scenario_dataset_factory_is_deterministic(tmp_path: Path) -> None:
    rows = build_churn_rows(seed=7, row_count=3)
    path = write_rows(tmp_path / "scenario.csv", rows)

    assert list(rows[0]) == [
        "customer_id",
        "age",
        "gender",
        "region",
        "signup_date",
        "monthly_charges",
        "tenure_months",
        "contract_type",
        "payment_method",
        "is_active",
        "support_tickets",
        "satisfaction_score",
        "notes",
        "churn",
    ]
    assert path.read_text(encoding="utf-8").startswith("customer_id,age,gender")
