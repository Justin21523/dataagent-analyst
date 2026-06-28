#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_support import (  # noqa: E402
    ArtifactStore,
    BacktestHttpClient,
    ManagedServices,
    add_common_arguments,
    ensure_demo_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the API-level full workflow backtest.")
    add_common_arguments(parser)
    args = parser.parse_args()

    ensure_demo_dataset(args.sample_file, args.python)
    store = ArtifactStore(args.artifact_dir, args.run_id)

    metadata: dict[str, Any] = {
        "base_url": args.base_url,
        "sample_file": str(args.sample_file),
    }

    try:
        with ManagedServices(
            base_url=args.base_url,
            frontend_url=args.frontend_url,
            python=args.python,
            no_start=args.no_start,
            keep_servers=args.keep_servers,
        ):
            run_workflow(args.base_url, args.sample_file, store, metadata)
    finally:
        store.write_run_json(metadata)
        store.write_summary(metadata)

    print(f"Backtest artifacts: {store.run_dir}")


def run_workflow(
    base_url: str,
    sample_file: Path,
    store: ArtifactStore,
    metadata: dict[str, Any],
) -> None:
    client = BacktestHttpClient(base_url, store)

    try:
        health = client.request_json("health", "GET", "/api/health")
        store.assert_that(
            "health_status_present",
            "status" in health,
            expected="status key",
            actual=sorted(health.keys()),
        )
        dataset = upload_dataset(client, sample_file)
        dataset_id = dataset["id"]
        metadata["dataset_id"] = dataset_id
        store.assert_that(
            "dataset_upload_has_id",
            bool(dataset_id),
            expected="non-empty dataset id",
            actual=dataset_id,
        )

        preview = client.request_json(
            "dataset_preview", "GET", f"/api/datasets/{dataset_id}/preview?max_rows=20"
        )
        assert_dataset_preview(store, preview)
        schema = client.request_json("dataset_schema", "GET", f"/api/datasets/{dataset_id}/schema")
        assert_dataset_schema(store, schema)
        target_column = choose_target(schema)
        metadata["target_column"] = target_column
        store.assert_that(
            "target_column_selected",
            bool(target_column),
            expected="selected target",
            actual=target_column,
        )

        eda = client.request_json("eda_summary", "GET", f"/api/eda/{dataset_id}/summary")
        assert_non_empty_mapping(store, "eda_summary_payload", eda)
        visualization = client.request_json(
            "visualization_lab",
            "POST",
            f"/api/visualizations/{dataset_id}/lab",
            json_payload={
                "target_column": target_column,
                "sample_rows": 80,
                "max_numeric_columns": 8,
                "max_categories": 12,
            },
        )
        assert_non_empty_mapping(store, "visualization_lab_payload", visualization)

        ml_plan = client.request_json(
            "ml_workbench_plan",
            "POST",
            f"/api/ml-workbench/{dataset_id}/plan",
            json_payload={
                "target_column": target_column,
                "task_type": "auto",
                "include_datetime": True,
                "include_text": False,
            },
        )
        assert_ml_plan(store, ml_plan)
        selected_model = choose_workbench_model(ml_plan)
        experiment = client.request_json(
            "ml_workbench_experiment",
            "POST",
            f"/api/ml-workbench/{dataset_id}/experiments",
            json_payload={
                "target_column": target_column,
                "task_type": ml_plan["detected_task_type"],
                "selected_models": [selected_model],
                "include_datetime": True,
                "include_text": False,
                "cv_folds": 3,
                "test_size": 0.2,
            },
        )
        assert_ml_experiment(store, experiment, metric_floor=0.45)

        training = client.request_json(
            "baseline_training",
            "POST",
            f"/api/ml/{dataset_id}/train",
            json_payload={
                "target_column": target_column,
                "task_type": "auto",
                "test_size": 0.25,
                "random_state": 42,
            },
        )
        assert_baseline_training(store, training, metric_floor=0.45)
        model_id = training["best_model_id"]
        metadata["model_id"] = model_id

        evaluation = client.request_json(
            "model_evaluation", "GET", f"/api/ml/models/{model_id}/evaluation"
        )
        assert_model_evaluation(store, evaluation)
        explainability = client.request_json(
            "model_explainability",
            "POST",
            f"/api/explainability/models/{model_id}/analyze",
            json_payload={
                "sample_size": 30,
                "background_size": 15,
                "permutation_repeats": 3,
                "local_row_position": 0,
                "include_permutation": True,
                "include_shap": True,
            },
        )
        assert_model_explainability(store, explainability)
        prediction = client.request_json(
            "single_prediction",
            "POST",
            f"/api/ml/models/{model_id}/predict",
            json_payload={"records": [prediction_record()]},
        )
        assert_prediction(store, prediction)

        report = client.request_json(
            "report_generate", "POST", f"/api/reports/{dataset_id}/generate"
        )
        assert_report(store, report)
        metadata["report_id"] = report["report"]["id"]

        maybe_run_drift(client, dataset_id, model_id, target_column)
        run_workspace_round_trip(client, dataset_id, model_id, target_column)
        run_agent_job(client, dataset_id, target_column, metadata)
    finally:
        client.close()


def upload_dataset(client: BacktestHttpClient, sample_file: Path) -> dict[str, Any]:
    with sample_file.open("rb") as file:
        payload = client.request_json(
            "dataset_upload",
            "POST",
            "/api/datasets/upload",
            files={"file": (sample_file.name, file, "text/csv")},
            expected_status={201},
        )
    return payload["dataset"]


def choose_target(schema: dict[str, Any]) -> str:
    candidates = schema["summary"].get("target_candidates") or []
    if "churn" in candidates:
        return "churn"
    if candidates:
        return str(candidates[0])
    columns = [column["name"] for column in schema["columns"]]
    if "churn" in columns:
        return "churn"
    return str(columns[-1])


def choose_workbench_model(plan: dict[str, Any]) -> str:
    models = plan["available_models"]
    recommended = next((model for model in models if model.get("recommended")), None)
    if recommended:
        return str(recommended["id"])
    return str(models[0]["id"])


def maybe_run_drift(
    client: BacktestHttpClient,
    dataset_id: str,
    model_id: str,
    target_column: str,
) -> None:
    transformed = client.request_json(
        "dataset_transform_for_drift",
        "POST",
        f"/api/datasets/{dataset_id}/transform",
        json_payload={
            "drop_duplicate_rows": True,
            "fill_missing": [{"column": "age", "strategy": "median"}],
        },
        expected_status={201},
    )
    client.store.assert_that(
        "drift_transform_created_version",
        bool(transformed.get("version", {}).get("version_id")),
        expected="derived version id",
        actual=transformed.get("version"),
    )

    current_version_id = transformed["version"]["version_id"]
    drift = client.request_json(
        "drift_report",
        "POST",
        "/api/drift/reports",
        json_payload={
            "dataset_id": dataset_id,
            "reference_version_id": "v1",
            "current_version_id": current_version_id,
            "model_id": model_id,
            "target_column": target_column,
        },
    )
    assert_drift_report(client.store, drift)


def run_workspace_round_trip(
    client: BacktestHttpClient,
    dataset_id: str,
    model_id: str,
    target_column: str,
) -> None:
    client.request_json(
        "workspace_state_patch",
        "PATCH",
        "/api/workspaces/default/state",
        json_payload={
            "active_route": "model-workbench",
            "dataset_id": dataset_id,
            "dataset_version_id": "v1",
            "target_column": target_column,
            "selected_model_id": model_id,
            "drift_status": "Not checked",
        },
    )
    state = client.request_json("workspace_state_get", "GET", "/api/workspaces/default/state")
    client.store.assert_that(
        "workspace_dataset_round_trip",
        state["dataset_id"] == dataset_id,
        expected=dataset_id,
        actual=state["dataset_id"],
    )
    client.store.assert_that(
        "workspace_model_round_trip",
        state["selected_model_id"] == model_id,
        expected=model_id,
        actual=state["selected_model_id"],
    )


def run_agent_job(
    client: BacktestHttpClient,
    dataset_id: str,
    target_column: str,
    metadata: dict[str, Any],
) -> None:
    response = client.request_json(
        "agent_job_start",
        "POST",
        f"/api/agent-jobs/{dataset_id}/start",
        json_payload={
            "user_goal": "Run automated backtest workflow.",
            "target_column": target_column,
            "run_ml": False,
            "generate_report": False,
            "generate_ai_insight": False,
        },
        expected_status={202},
    )
    job_id = response["job"]["job_id"]
    metadata["agent_job_id"] = job_id

    terminal_status = ""
    for _ in range(80):
        detail = client.request_json("agent_job_poll", "GET", f"/api/agent-jobs/{job_id}")
        terminal_status = detail["status"]
        if terminal_status in {"success", "completed_with_warnings"}:
            client.store.assert_that(
                "agent_job_terminal_success",
                True,
                expected="success or completed_with_warnings",
                actual=terminal_status,
            )
            return
        if terminal_status == "failed":
            raise RuntimeError("Agent job failed.")
        time.sleep(0.5)

    raise RuntimeError(f"Agent job did not complete, last status={terminal_status}")


def prediction_record() -> dict[str, Any]:
    return {
        "age": 34,
        "gender": "Female",
        "region": "North",
        "monthly_charges": 980,
        "tenure_months": 14,
        "contract_type": "Monthly",
        "payment_method": "Credit Card",
        "is_active": True,
        "support_tickets": 1,
        "satisfaction_score": 4,
    }


def assert_non_empty_mapping(store: ArtifactStore, name: str, payload: dict[str, Any]) -> None:
    store.assert_that(
        name,
        isinstance(payload, dict) and bool(payload),
        expected="non-empty object",
        actual=sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
    )


def assert_dataset_preview(store: ArtifactStore, preview: dict[str, Any]) -> None:
    store.assert_that(
        "dataset_preview_has_rows",
        any(isinstance(value, list) and value for value in preview.values()),
        expected="preview contains row list",
        actual=sorted(preview.keys()),
    )


def assert_dataset_schema(store: ArtifactStore, schema: dict[str, Any]) -> None:
    columns = schema.get("columns")
    store.assert_that(
        "dataset_schema_has_columns",
        isinstance(columns, list) and bool(columns),
        expected="non-empty columns list",
        actual=len(columns) if isinstance(columns, list) else type(columns).__name__,
    )
    store.assert_that(
        "dataset_schema_has_summary",
        isinstance(schema.get("summary"), dict),
        expected="summary object",
        actual=type(schema.get("summary")).__name__,
    )


def assert_ml_plan(store: ArtifactStore, plan: dict[str, Any]) -> None:
    models = plan.get("available_models")
    store.assert_that(
        "ml_plan_has_available_models",
        isinstance(models, list) and bool(models),
        expected="available model list",
        actual=len(models) if isinstance(models, list) else type(models).__name__,
    )
    store.assert_that(
        "ml_plan_detected_task_type",
        plan.get("detected_task_type")
        in {"classification", "binary_classification", "multiclass_classification", "regression"},
        expected="classification-like or regression",
        actual=plan.get("detected_task_type"),
    )


def assert_ml_experiment(
    store: ArtifactStore,
    experiment: dict[str, Any],
    *,
    metric_floor: float,
) -> None:
    store.assert_that(
        "ml_experiment_completed",
        experiment.get("status") in {"completed", "completed_with_warnings"},
        expected="completed or completed_with_warnings",
        actual=experiment.get("status"),
    )
    metric_value = experiment.get("best_metric_value")
    store.assert_that(
        "ml_experiment_metric_floor",
        isinstance(metric_value, int | float) and metric_value >= metric_floor,
        expected=f">= {metric_floor}",
        actual=metric_value,
    )


def assert_baseline_training(
    store: ArtifactStore,
    training: dict[str, Any],
    *,
    metric_floor: float,
) -> None:
    metric_value = training.get("best_metric_value")
    store.assert_that(
        "baseline_training_metric_name",
        training.get("best_metric_name") == "f1_macro",
        expected="f1_macro",
        actual=training.get("best_metric_name"),
    )
    store.assert_that(
        "baseline_training_metric_floor",
        isinstance(metric_value, int | float) and metric_value >= metric_floor,
        expected=f">= {metric_floor}",
        actual=metric_value,
    )
    store.assert_that(
        "baseline_training_has_best_model",
        bool(training.get("best_model_id")),
        expected="best model id",
        actual=training.get("best_model_id"),
    )


def assert_model_evaluation(store: ArtifactStore, evaluation: dict[str, Any]) -> None:
    model = evaluation.get("model")
    store.assert_that(
        "model_evaluation_has_model",
        isinstance(model, dict) and bool(model.get("id")),
        expected="model object with id",
        actual=model,
    )


def assert_model_explainability(store: ArtifactStore, explainability: dict[str, Any]) -> None:
    store.assert_that(
        "model_explainability_payload",
        isinstance(explainability, dict) and bool(explainability),
        expected="non-empty explainability payload",
        actual=sorted(explainability.keys()) if isinstance(explainability, dict) else None,
    )


def assert_prediction(store: ArtifactStore, prediction: dict[str, Any]) -> None:
    predictions = prediction.get("predictions")
    store.assert_that(
        "single_prediction_has_result",
        isinstance(predictions, list) and bool(predictions),
        expected="prediction list",
        actual=predictions,
    )


def assert_report(store: ArtifactStore, report: dict[str, Any]) -> None:
    report_payload = report.get("report")
    store.assert_that(
        "report_generate_has_markdown_path",
        isinstance(report_payload, dict) and bool(report_payload.get("markdown_path")),
        expected="markdown path",
        actual=report_payload,
    )


def assert_drift_report(store: ArtifactStore, drift: dict[str, Any]) -> None:
    feature_drift = drift.get("feature_drift")
    store.assert_that(
        "drift_report_has_status",
        drift.get("status") in {"stable", "warning", "drift"},
        expected="stable/warning/drift",
        actual=drift.get("status"),
    )
    store.assert_that(
        "drift_report_has_feature_drift",
        isinstance(feature_drift, list) and bool(feature_drift),
        expected="non-empty feature drift list",
        actual=len(feature_drift) if isinstance(feature_drift, list) else None,
    )
    for key in ("target_drift", "prediction_drift", "retraining_recommendation"):
        store.assert_that(
            f"drift_report_has_{key}",
            isinstance(drift.get(key), dict) and bool(drift.get(key)),
            expected=f"{key} object",
            actual=drift.get(key),
        )


if __name__ == "__main__":
    main()
