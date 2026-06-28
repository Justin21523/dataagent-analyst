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
            "age",
            "monthly_charges",
            "tenure_months",
            "contract_type",
            "support_tickets",
            "churn",
        ]
    )

    for index in range(1, 101):
        contract_type = "Monthly" if index % 3 == 0 else "One Year"

        support_tickets = index % 6

        churn = "Yes" if (contract_type == "Monthly" and support_tickets >= 2) else "No"

        writer.writerow(
            [
                f"C{index:05d}",
                20 + index % 48,
                550 + index * 14,
                1 + index % 48,
                contract_type,
                support_tickets,
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

    for index in range(1, 101):
        age = 18 + index % 50
        tenure = 1 + index % 48
        tickets = index % 7

        revenue = 350 + age * 8 + tenure * 13 - tickets * 17

        writer.writerow(
            [
                f"R{index:05d}",
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


def _train_model(
    dataset_id: str,
    target_column: str,
    task_type: str,
    model_name: str,
) -> str:
    response = client.post(
        f"/api/ml-workbench/{dataset_id}/experiments",
        json={
            "target_column": target_column,
            "task_type": task_type,
            "selected_models": [
                model_name,
            ],
            "include_datetime": False,
            "include_text": False,
            "cv_folds": 3,
            "test_size": 0.2,
            "random_state": 42,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["best_model_id"]

    return payload["best_model_id"]


def test_tree_model_explainability_returns_shap_and_curves() -> None:
    dataset_id = _upload_dataset(
        "explainability_classification.csv",
        _classification_csv(),
    )

    model_id = _train_model(
        dataset_id=dataset_id,
        target_column="churn",
        task_type="binary_classification",
        model_name="random_forest_classifier",
    )

    response = client.post(
        f"/api/explainability/models/{model_id}/analyze",
        json={
            "sample_size": 30,
            "background_size": 20,
            "permutation_repeats": 3,
            "local_row_position": 0,
            "include_permutation": True,
            "include_shap": True,
            "force_recompute": True,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["model_id"] == model_id
    assert payload["task_type"] == "classification"
    assert payload["holdout"]["source"] == "saved_split_context"

    assert payload["classification"] is not None
    assert payload["classification"]["curves"]
    assert payload["permutation_importance"]

    assert payload["shap"]["available"] is True
    assert payload["shap"]["explainer_type"] == "TreeExplainer"
    assert payload["shap"]["source_feature_importance"]
    assert payload["shap"]["beeswarm_points"]
    assert payload["shap"]["local_explanation"]


def test_linear_regression_explainability_returns_residuals() -> None:
    dataset_id = _upload_dataset(
        "explainability_regression.csv",
        _regression_csv(),
    )

    model_id = _train_model(
        dataset_id=dataset_id,
        target_column="monthly_revenue",
        task_type="regression",
        model_name="ridge_regression",
    )

    response = client.post(
        f"/api/explainability/models/{model_id}/analyze",
        json={
            "sample_size": 30,
            "background_size": 20,
            "permutation_repeats": 3,
            "include_shap": True,
            "force_recompute": True,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["task_type"] == "regression"
    assert payload["regression"] is not None
    assert payload["regression"]["points"]
    assert payload["permutation_importance"]
    assert payload["shap"]["available"] is True
    assert payload["shap"]["explainer_type"] == "LinearExplainer"


def test_explainability_uses_cache() -> None:
    dataset_id = _upload_dataset(
        "explainability_cache.csv",
        _regression_csv(),
    )

    model_id = _train_model(
        dataset_id=dataset_id,
        target_column="monthly_revenue",
        task_type="regression",
        model_name="ridge_regression",
    )

    request_payload = {
        "sample_size": 20,
        "background_size": 15,
        "permutation_repeats": 3,
        "include_shap": False,
    }

    first_response = client.post(
        f"/api/explainability/models/{model_id}/analyze",
        json=request_payload,
    )

    second_response = client.post(
        f"/api/explainability/models/{model_id}/analyze",
        json=request_payload,
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert first_response.json()["cache_hit"] is False
    assert second_response.json()["cache_hit"] is True


def test_explainability_llm_endpoint_returns_fallback() -> None:
    dataset_id = _upload_dataset(
        "explainability_llm.csv",
        _classification_csv(),
    )

    model_id = _train_model(
        dataset_id=dataset_id,
        target_column="churn",
        task_type="binary_classification",
        model_name="logistic_regression",
    )

    response = client.post(
        (f"/api/ai-insights/{dataset_id}/models/{model_id}/explainability"),
        json={"user_goal": ("Explain reliability and important drivers.")},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["title"] == "Model Explainability Interpretation"
    assert payload["source"] == "fallback"
