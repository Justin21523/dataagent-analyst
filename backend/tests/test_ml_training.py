from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_ml_sample_dataset() -> str:
    # 測試資料維持小型但類別平衡，避免 train/test split 後只剩單一類別。
    csv_content = (
        b"customer_id,age,monthly_charges,tenure_months,contract_type,payment_method,churn\n"
        b"C0001,29,799,12,Monthly,Credit Card,No\n"
        b"C0002,41,1299,4,Monthly,Bank Transfer,Yes\n"
        b"C0003,35,999,24,One Year,Credit Card,No\n"
        b"C0004,52,1599,2,Monthly,Electronic Wallet,Yes\n"
        b"C0005,46,899,36,Two Year,Bank Transfer,No\n"
        b"C0006,31,1099,7,Monthly,Credit Card,Yes\n"
        b"C0007,27,699,18,One Year,Electronic Wallet,No\n"
        b"C0008,63,1899,3,Monthly,Credit Card,Yes\n"
        b"C0009,38,1199,15,One Year,Bank Transfer,No\n"
        b"C0010,44,1399,6,Monthly,Electronic Wallet,Yes\n"
        b"C0011,33,850,16,One Year,Credit Card,No\n"
        b"C0012,58,1750,5,Monthly,Bank Transfer,Yes\n"
    )

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("ml_sample.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201

    return response.json()["dataset"]["id"]


def test_train_classification_models_returns_results() -> None:
    dataset_id = _upload_ml_sample_dataset()

    response = client.post(
        f"/api/ml/{dataset_id}/train",
        json={
            "target_column": "churn",
            "task_type": "auto",
            "test_size": 0.25,
            "random_state": 42,
        },
    )

    assert response.status_code == 201

    payload = response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["task_type"] == "classification"
    assert payload["target_column"] == "churn"
    assert payload["model_count"] >= 2
    assert payload["best_model_id"]
    assert payload["best_metric_name"] == "f1_macro"

    first_model = payload["models"][0]
    assert first_model["status"] == "trained"
    assert "accuracy" in first_model["metrics"]
    assert "f1_macro" in first_model["metrics"]


def test_list_dataset_models_returns_trained_models() -> None:
    dataset_id = _upload_ml_sample_dataset()

    train_response = client.post(
        f"/api/ml/{dataset_id}/train",
        json={"target_column": "churn"},
    )

    assert train_response.status_code == 201

    list_response = client.get(f"/api/ml/{dataset_id}/models")

    assert list_response.status_code == 200

    payload = list_response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["total"] >= 2
    assert len(payload["models"]) >= 2


def test_model_metrics_endpoint_returns_metrics() -> None:
    dataset_id = _upload_ml_sample_dataset()

    train_response = client.post(
        f"/api/ml/{dataset_id}/train",
        json={"target_column": "churn"},
    )

    model_id = train_response.json()["best_model_id"]

    metrics_response = client.get(f"/api/ml/models/{model_id}/metrics")

    assert metrics_response.status_code == 200

    payload = metrics_response.json()

    assert payload["model_id"] == model_id
    assert "f1_macro" in payload["metrics"]
