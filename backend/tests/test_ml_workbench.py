import csv
from io import StringIO

from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _classification_csv() -> bytes:
    output = StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "customer_id",
            "signup_date",
            "age",
            "monthly_charges",
            "contract_type",
            "notes",
            "churn",
        ]
    )

    for index in range(1, 61):
        churn = "Yes" if index % 3 == 0 else "No"

        writer.writerow(
            [
                f"C{index:04d}",
                f"2024-01-{((index - 1) % 28) + 1:02d}",
                20 + index % 45,
                500 + index * 22,
                ("Monthly" if index % 2 == 0 else "One Year"),
                ("requested support" if churn == "Yes" else "regular customer"),
                churn,
            ]
        )

    return output.getvalue().encode()


def _regression_csv() -> bytes:
    output = StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "record_id",
            "age",
            "tenure",
            "support_tickets",
            "monthly_revenue",
        ]
    )

    for index in range(1, 81):
        age = 18 + index % 50
        tenure = 1 + index % 36
        tickets = index % 6
        revenue = 400 + age * 7 + tenure * 12 - tickets * 15

        writer.writerow(
            [
                f"R{index:04d}",
                age,
                tenure,
                tickets,
                revenue,
            ]
        )

    return output.getvalue().encode()


def _upload_dataset(
    filename: str,
    content: bytes,
) -> str:
    response = client.post(
        "/api/datasets/upload",
        files={
            "file": (
                filename,
                content,
                "text/csv",
            )
        },
    )

    assert response.status_code == 201

    return response.json()["dataset"]["id"]


def test_ml_workbench_plan_detects_binary_classification() -> None:
    dataset_id = _upload_dataset(
        "workbench_classification.csv",
        _classification_csv(),
    )

    response = client.post(
        f"/api/ml-workbench/{dataset_id}/plan",
        json={
            "target_column": "churn",
            "task_type": "auto",
            "include_datetime": True,
            "include_text": True,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["detected_task_type"] == "binary_classification"
    assert payload["target_column"] == "churn"
    assert "age" in payload["feature_groups"]["numeric"]
    assert "signup_date" in payload["feature_groups"]["datetime"]
    assert "notes" in payload["feature_groups"]["text"]
    assert "customer_id" in payload["feature_groups"]["excluded"]

    available_models = {model["id"] for model in payload["available_models"]}

    assert "logistic_regression" in available_models
    assert "random_forest_classifier" in available_models


def test_ml_workbench_runs_supervised_experiment() -> None:
    dataset_id = _upload_dataset(
        "workbench_supervised.csv",
        _classification_csv(),
    )

    response = client.post(
        f"/api/ml-workbench/{dataset_id}/experiments",
        json={
            "target_column": "churn",
            "task_type": "binary_classification",
            "selected_models": [
                "logistic_regression",
                "random_forest_classifier",
            ],
            "include_datetime": True,
            "include_text": False,
            "cv_folds": 3,
            "test_size": 0.2,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] in {
        "success",
        "completed_with_warnings",
    }
    assert payload["cv_folds"] == 3
    assert len(payload["model_results"]) == 2
    assert payload["best_model_id"]
    assert payload["best_model_name"]

    successful = [result for result in payload["model_results"] if result["status"] == "success"]

    assert successful
    assert successful[0]["cv_metrics"]
    assert "f1_macro" in successful[0]["holdout_metrics"]


def test_ml_workbench_detects_regression() -> None:
    dataset_id = _upload_dataset(
        "workbench_regression.csv",
        _regression_csv(),
    )

    response = client.post(
        f"/api/ml-workbench/{dataset_id}/plan",
        json={
            "target_column": "monthly_revenue",
            "task_type": "auto",
        },
    )

    assert response.status_code == 200
    assert response.json()["detected_task_type"] == "regression"


def test_ml_workbench_runs_clustering() -> None:
    dataset_id = _upload_dataset(
        "workbench_clustering.csv",
        _classification_csv(),
    )

    response = client.post(
        f"/api/ml-workbench/{dataset_id}/experiments",
        json={
            "task_type": "clustering",
            "selected_models": [
                "kmeans",
            ],
            "include_datetime": False,
            "include_text": False,
            "n_clusters": 3,
        },
    )

    assert response.status_code == 200

    payload = response.json()
    result = payload["model_results"][0]

    assert payload["task_type"] == "clustering"
    assert result["status"] == "success"
    assert "cluster_count" in result["holdout_metrics"]
    assert result["artifact"]["projection_points"]


def test_ml_workbench_runs_anomaly_detection() -> None:
    dataset_id = _upload_dataset(
        "workbench_anomaly.csv",
        _regression_csv(),
    )

    response = client.post(
        f"/api/ml-workbench/{dataset_id}/experiments",
        json={
            "task_type": "anomaly_detection",
            "selected_models": [
                "isolation_forest",
            ],
            "contamination": 0.1,
        },
    )

    assert response.status_code == 200

    result = response.json()["model_results"][0]

    assert result["status"] == "success"
    assert "anomaly_ratio" in result["holdout_metrics"]
    assert result["artifact"]["projection_points"]


def test_ml_workbench_experiment_history() -> None:
    dataset_id = _upload_dataset(
        "workbench_history.csv",
        _classification_csv(),
    )

    run_response = client.post(
        f"/api/ml-workbench/{dataset_id}/experiments",
        json={
            "target_column": "churn",
            "selected_models": [
                "logistic_regression",
            ],
            "cv_folds": 3,
        },
    )

    assert run_response.status_code == 200

    list_response = client.get(f"/api/ml-workbench/{dataset_id}/experiments")

    assert list_response.status_code == 200
    assert list_response.json()["total"] >= 1
