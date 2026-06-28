#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_full_workflow import (  # noqa: E402
    assert_baseline_training,
    assert_dataset_schema,
    assert_drift_report,
    assert_ml_plan,
    assert_non_empty_mapping,
    run_workflow,
    upload_dataset,
)
from scripts.backtest_support import (  # noqa: E402
    ArtifactStore,
    BacktestHttpClient,
    ManagedServices,
    StepResult,
    add_common_arguments,
    ensure_demo_dataset,
    write_suite_summary,
)

ScenarioRunner = Callable[[str, Path, ArtifactStore, dict[str, Any]], None]


SCENARIOS: dict[str, ScenarioRunner] = {
    "happy_path_churn": lambda base_url, sample_file, store, metadata: run_workflow(
        base_url, sample_file, store, metadata
    ),
    "dirty_missing_churn": lambda base_url, _sample_file, store, metadata: run_dirty_missing_churn(
        base_url, store, metadata
    ),
    "high_cardinality_categories": (
        lambda base_url, _sample_file, store, metadata: run_high_cardinality_categories(
            base_url, store, metadata
        )
    ),
    "schema_migration_drift": lambda base_url, _sample_file, store, metadata: (
        run_schema_migration_drift(base_url, store, metadata)
    ),
    "missing_target_negative": lambda base_url, _sample_file, store, metadata: (
        run_missing_target_negative(base_url, store, metadata)
    ),
    "drift_retrain_challenger": lambda base_url, _sample_file, store, metadata: (
        run_drift_retrain_challenger(base_url, store, metadata)
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario-based workflow backtests.")
    add_common_arguments(parser)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
        help="Scenario id to run. Can be provided multiple times.",
    )
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args()

    if args.list_scenarios:
        for scenario_id in sorted(SCENARIOS):
            print(scenario_id)
        return

    ensure_demo_dataset(args.sample_file, args.python)
    scenario_ids = args.scenario or list(SCENARIOS)
    runs: list[dict[str, Any]] = []

    with ManagedServices(
        base_url=args.base_url,
        frontend_url=args.frontend_url,
        python=args.python,
        no_start=args.no_start,
        keep_servers=args.keep_servers,
    ):
        for scenario_id in scenario_ids:
            store = ArtifactStore(args.artifact_dir)
            metadata: dict[str, Any] = {
                "scenario": scenario_id,
                "base_url": args.base_url,
                "sample_file": str(args.sample_file),
            }
            started_at = time.perf_counter()
            try:
                SCENARIOS[scenario_id](args.base_url, args.sample_file, store, metadata)
            except Exception as exc:
                if not any(step.status == "failed" for step in store.steps):
                    store.add_step(
                        StepResult(
                            name="scenario_unhandled_exception",
                            status="failed",
                            duration_seconds=time.perf_counter() - started_at,
                            error=str(exc),
                        )
                    )
            finally:
                run_path = store.write_run_json(metadata)
                store.write_summary(metadata)
                runs.append(
                    {
                        "scenario": scenario_id,
                        "run_id": store.run_id,
                        "run_path": str(run_path),
                        "status": store.status(),
                        "duration_seconds": round(time.perf_counter() - started_at, 4),
                    }
                )

    summary_path = write_suite_summary(args.artifact_dir, "scenario_backtest", runs)
    print(f"Scenario suite summary: {summary_path}")

    if any(run["status"] == "failed" for run in runs):
        raise SystemExit(1)


def run_dirty_missing_churn(
    base_url: str,
    store: ArtifactStore,
    metadata: dict[str, Any],
) -> None:
    sample_file = write_dirty_missing_churn_dataset(store)
    metadata["sample_file"] = str(sample_file)
    client = BacktestHttpClient(base_url, store)

    try:
        dataset = upload_dataset(client, sample_file)
        dataset_id = dataset["id"]
        schema = client.request_json("dirty_schema", "GET", f"/api/datasets/{dataset_id}/schema")
        assert_dataset_schema(store, schema)
        client.request_json("dirty_eda", "GET", f"/api/eda/{dataset_id}/summary")
        transformed = client.request_json(
            "dirty_transform",
            "POST",
            f"/api/datasets/{dataset_id}/transform",
            json_payload={
                "drop_duplicate_rows": True,
                "fill_missing": [
                    {"column": "age", "strategy": "median"},
                    {"column": "monthly_charges", "strategy": "median"},
                    {"column": "satisfaction_score", "strategy": "median"},
                ],
                "iqr_clip": [{"column": "monthly_charges", "factor": 1.5}],
            },
            expected_status={201},
        )
        store.assert_that(
            "dirty_transform_created_version",
            bool(transformed.get("version", {}).get("version_id")),
            expected="version id",
            actual=transformed.get("version"),
        )
        training = client.request_json(
            "dirty_baseline_training",
            "POST",
            f"/api/ml/{dataset_id}/train",
            json_payload={
                "target_column": "churn",
                "task_type": "auto",
                "test_size": 0.25,
                "random_state": 42,
            },
        )
        assert_baseline_training(store, training, metric_floor=0.3)
    finally:
        client.close()


def run_high_cardinality_categories(
    base_url: str,
    store: ArtifactStore,
    metadata: dict[str, Any],
) -> None:
    sample_file = write_high_cardinality_dataset(store)
    metadata["sample_file"] = str(sample_file)
    client = BacktestHttpClient(base_url, store)

    try:
        dataset = upload_dataset(client, sample_file)
        dataset_id = dataset["id"]
        schema = client.request_json(
            "high_cardinality_schema", "GET", f"/api/datasets/{dataset_id}/schema"
        )
        assert_dataset_schema(store, schema)
        eda = client.request_json("high_cardinality_eda", "GET", f"/api/eda/{dataset_id}/summary")
        assert_non_empty_mapping(store, "high_cardinality_eda_payload", eda)
        visualization = client.request_json(
            "high_cardinality_visualization",
            "POST",
            f"/api/visualizations/{dataset_id}/lab",
            json_payload={
                "target_column": "churn",
                "sample_rows": 80,
                "max_numeric_columns": 8,
                "max_categories": 12,
            },
        )
        assert_non_empty_mapping(store, "high_cardinality_visualization_payload", visualization)
        plan = client.request_json(
            "high_cardinality_ml_plan",
            "POST",
            f"/api/ml-workbench/{dataset_id}/plan",
            json_payload={
                "target_column": "churn",
                "task_type": "auto",
                "include_datetime": True,
                "include_text": False,
            },
        )
        assert_ml_plan(store, plan)
    finally:
        client.close()


def run_schema_migration_drift(
    base_url: str,
    store: ArtifactStore,
    metadata: dict[str, Any],
) -> None:
    sample_file = write_clean_churn_dataset(store, "schema_migration.csv")
    metadata["sample_file"] = str(sample_file)
    client = BacktestHttpClient(base_url, store)

    try:
        dataset = upload_dataset(client, sample_file)
        dataset_id = dataset["id"]
        training = train_churn_model(client, dataset_id, "schema_migration_training")
        model_id = training["best_model_id"]
        transformed = client.request_json(
            "schema_migration_transform",
            "POST",
            f"/api/datasets/{dataset_id}/transform",
            json_payload={
                "drop_columns": ["notes"],
                "datetime_parts": [{"column": "signup_date", "parts": ["year", "month"]}],
            },
            expected_status={201},
        )
        drift = client.request_json(
            "schema_migration_drift_report",
            "POST",
            "/api/drift/reports",
            json_payload={
                "dataset_id": dataset_id,
                "reference_version_id": "v1",
                "current_version_id": transformed["version"]["version_id"],
                "model_id": model_id,
                "target_column": "churn",
            },
        )
        store.assert_that(
            "schema_migration_detected_schema_drift",
            drift.get("schema_drift", {}).get("status") == "drift",
            expected="drift",
            actual=drift.get("schema_drift"),
        )
        store.assert_that(
            "schema_migration_has_recommendations",
            bool(drift.get("recommendations")),
            expected="non-empty recommendations",
            actual=drift.get("recommendations"),
        )
    finally:
        client.close()


def run_missing_target_negative(
    base_url: str,
    store: ArtifactStore,
    metadata: dict[str, Any],
) -> None:
    sample_file = write_clean_churn_dataset(store, "missing_target.csv")
    metadata["sample_file"] = str(sample_file)
    client = BacktestHttpClient(base_url, store)

    try:
        dataset = upload_dataset(client, sample_file)
        dataset_id = dataset["id"]
        error = client.request_json_expected_error(
            "missing_target_train_error",
            "POST",
            f"/api/ml/{dataset_id}/train",
            json_payload={
                "target_column": "not_a_column",
                "task_type": "auto",
                "test_size": 0.25,
                "random_state": 42,
            },
        )
        store.assert_that(
            "missing_target_error_is_json",
            isinstance(error, dict) and bool(error),
            expected="json error payload",
            actual=error,
        )
    finally:
        client.close()


def run_drift_retrain_challenger(
    base_url: str,
    store: ArtifactStore,
    metadata: dict[str, Any],
) -> None:
    sample_file = write_clean_churn_dataset(store, "drift_retrain.csv")
    metadata["sample_file"] = str(sample_file)
    client = BacktestHttpClient(base_url, store)

    try:
        dataset = upload_dataset(client, sample_file)
        dataset_id = dataset["id"]
        champion_training = train_churn_model(client, dataset_id, "champion_training")
        champion_id = champion_training["best_model_id"]
        client.request_json(
            "champion_promote",
            "PATCH",
            f"/api/ml/models/{champion_id}/status",
            json_payload={"lifecycle_status": "production"},
        )
        transformed = client.request_json(
            "challenger_drift_transform",
            "POST",
            f"/api/datasets/{dataset_id}/transform",
            json_payload={
                "fill_missing": [{"column": "age", "strategy": "constant", "value": 72}],
                "drop_duplicate_rows": True,
            },
            expected_status={201},
        )
        current_version_id = transformed["version"]["version_id"]
        drift = client.request_json(
            "challenger_drift_report",
            "POST",
            "/api/drift/reports",
            json_payload={
                "dataset_id": dataset_id,
                "reference_version_id": "v1",
                "current_version_id": current_version_id,
                "model_id": champion_id,
                "target_column": "churn",
            },
        )
        assert_drift_report(store, drift)
        plan = client.request_json(
            "challenger_retrain_plan",
            "POST",
            f"/api/ml/models/{champion_id}/retrain-plan",
            json_payload={
                "current_version_id": current_version_id,
                "reference_version_id": "v1",
                "drift_report_id": drift["report_id"],
            },
        )
        store.assert_that(
            "challenger_plan_has_model",
            bool(plan.get("selected_model")),
            expected="selected model",
            actual=plan,
        )
        retrain = client.request_json(
            "challenger_retrain",
            "POST",
            f"/api/ml/models/{champion_id}/retrain",
            json_payload={
                "current_version_id": current_version_id,
                "reference_version_id": "v1",
                "drift_report_id": drift["report_id"],
            },
        )
        challenger = retrain.get("challenger_model", {})
        store.assert_that(
            "challenger_model_is_candidate",
            challenger.get("lifecycle_status") == "candidate",
            expected="candidate",
            actual=challenger,
        )
        store.assert_that(
            "challenger_comparison_has_models",
            bool(retrain.get("comparison", {}).get("models")),
            expected="comparison models",
            actual=retrain.get("comparison"),
        )
        migration = client.request_json(
            "challenger_migration_check",
            "POST",
            f"/api/ml/models/{champion_id}/migration-check",
        )
        store.assert_that(
            "challenger_migration_check_has_checks",
            bool(migration.get("checks")),
            expected="checks",
            actual=migration,
        )
    finally:
        client.close()


def train_churn_model(
    client: BacktestHttpClient,
    dataset_id: str,
    step_name: str,
) -> dict[str, Any]:
    training = client.request_json(
        step_name,
        "POST",
        f"/api/ml/{dataset_id}/train",
        json_payload={
            "target_column": "churn",
            "selected_models": ["logistic_regression"],
            "test_size": 0.25,
            "random_state": 42,
        },
        expected_status={201},
    )
    assert_baseline_training(client.store, training, metric_floor=0.3)
    return training


def write_clean_churn_dataset(store: ArtifactStore, filename: str) -> Path:
    return write_rows(store.run_dir / filename, build_churn_rows(seed=121, row_count=120))


def write_dirty_missing_churn_dataset(store: ArtifactStore) -> Path:
    rows = build_churn_rows(seed=212, row_count=140)
    for index, row in enumerate(rows):
        if index % 4 == 0:
            row["age"] = ""
        if index % 5 == 0:
            row["monthly_charges"] = ""
        if index % 6 == 0:
            row["satisfaction_score"] = ""
        if index % 17 == 0:
            row["monthly_charges"] = 9800
        if index % 19 == 0:
            row["contract_type"] = f"Custom-{index}"
    rows.extend(rows[:5])
    return write_rows(store.run_dir / "dirty_missing_churn.csv", rows)


def write_high_cardinality_dataset(store: ArtifactStore) -> Path:
    rows = build_churn_rows(seed=313, row_count=120)
    for index, row in enumerate(rows):
        row["region"] = f"Region-{index:03d}"
        row["payment_method"] = f"Payment-{index % 55:02d}"
        row["notes"] = f"Free-form account note {index:03d} with segment {index % 11}"
    return write_rows(store.run_dir / "high_cardinality.csv", rows)


def build_churn_rows(seed: int, row_count: int) -> list[dict[str, Any]]:
    random.seed(seed)
    rows = []
    for index in range(1, row_count + 1):
        age = random.randint(19, 72)
        tenure = random.randint(1, 60)
        contract_type = random.choices(
            ["Monthly", "One Year", "Two Year"],
            weights=[0.55, 0.28, 0.17],
            k=1,
        )[0]
        support_tickets = random.randint(0, 5)
        satisfaction = random.randint(1, 5)
        monthly_charges = 480 + support_tickets * 65 + random.randint(0, 750)
        is_active = random.random() > 0.22
        churn_score = 0
        churn_score += 2 if contract_type == "Monthly" else 0
        churn_score += support_tickets
        churn_score += 2 if satisfaction <= 2 else 0
        churn_score += 2 if not is_active else 0
        churn_score += 1 if monthly_charges >= 1100 else 0
        churn_score -= 1 if tenure >= 24 else 0
        churn = "Yes" if churn_score >= 4 else "No"
        rows.append(
            {
                "customer_id": f"S{index:05d}",
                "age": age,
                "gender": random.choice(["Female", "Male"]),
                "region": random.choice(["North", "Central", "South", "East"]),
                "signup_date": f"2024-{(index % 12) + 1:02d}-{(index % 27) + 1:02d}",
                "monthly_charges": monthly_charges,
                "tenure_months": tenure,
                "contract_type": contract_type,
                "payment_method": random.choice(
                    ["Credit Card", "Bank Transfer", "Electronic Wallet"]
                ),
                "is_active": str(is_active).lower(),
                "support_tickets": support_tickets,
                "satisfaction_score": satisfaction,
                "notes": random.choice(["Regular customer", "Asked about pricing", ""]),
                "churn": churn,
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    main()
