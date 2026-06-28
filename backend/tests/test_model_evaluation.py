from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _train_classification_model() -> str:
    # 測試資料用分類 target，確保會產生 confusion matrix。
    csv_content = (
        b"customer_id,age,monthly_charges,tenure_months,contract_type,churn\n"
        b"C0001,29,799,12,Monthly,No\n"
        b"C0002,41,1299,4,Monthly,Yes\n"
        b"C0003,35,999,24,One Year,No\n"
        b"C0004,52,1599,2,Monthly,Yes\n"
        b"C0005,46,899,36,Two Year,No\n"
        b"C0006,31,1099,7,Monthly,Yes\n"
        b"C0007,27,699,18,One Year,No\n"
        b"C0008,63,1899,3,Monthly,Yes\n"
        b"C0009,38,1199,15,One Year,No\n"
        b"C0010,44,1399,6,Monthly,Yes\n"
        b"C0011,33,850,16,One Year,No\n"
        b"C0012,58,1750,5,Monthly,Yes\n"
    )

    upload_response = client.post(
        "/api/datasets/upload",
        files={"file": ("evaluation_sample.csv", csv_content, "text/csv")},
    )

    assert upload_response.status_code == 201

    dataset_id = upload_response.json()["dataset"]["id"]

    train_response = client.post(
        f"/api/ml/{dataset_id}/train",
        json={
            "target_column": "churn",
            "task_type": "auto",
            "test_size": 0.25,
            "random_state": 42,
        },
    )

    assert train_response.status_code == 201

    return train_response.json()["best_model_id"]


def test_model_evaluation_endpoint_returns_confusion_matrix() -> None:
    model_id = _train_classification_model()

    response = client.get(f"/api/ml/models/{model_id}/evaluation")

    assert response.status_code == 200

    payload = response.json()

    assert payload["model"]["id"] == model_id
    assert payload["confusion_matrix"] is not None
    assert "labels" in payload["confusion_matrix"]
    assert "matrix" in payload["confusion_matrix"]
    assert isinstance(payload["confusion_matrix"]["matrix"], list)
    assert "feature_importance" in payload


def test_model_evaluation_metrics_still_available() -> None:
    model_id = _train_classification_model()

    response = client.get(f"/api/ml/models/{model_id}/metrics")

    assert response.status_code == 200

    payload = response.json()

    assert payload["model_id"] == model_id
    assert "f1_macro" in payload["metrics"]
