from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _train_prediction_model() -> str:
    # 測試資料用分類 target，並保留多種 feature type。
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

    upload_response = client.post(
        "/api/datasets/upload",
        files={"file": ("prediction_training.csv", csv_content, "text/csv")},
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


def test_predict_with_json_records_returns_predictions() -> None:
    model_id = _train_prediction_model()

    response = client.post(
        f"/api/ml/models/{model_id}/predict",
        json={
            "records": [
                {
                    "age": 30,
                    "monthly_charges": 900,
                    "tenure_months": 10,
                    "contract_type": "Monthly",
                    "payment_method": "Credit Card",
                }
            ]
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["model_id"] == model_id
    assert payload["total"] == 1
    assert payload["predictions"][0]["row_index"] == 0
    assert "prediction" in payload["predictions"][0]


def test_predict_with_batch_csv_returns_predictions() -> None:
    model_id = _train_prediction_model()

    prediction_csv = (
        b"age,monthly_charges,tenure_months,contract_type,payment_method\n"
        b"30,900,10,Monthly,Credit Card\n"
        b"55,1700,3,Monthly,Bank Transfer\n"
    )

    response = client.post(
        f"/api/ml/models/{model_id}/predict-csv",
        files={"file": ("prediction_batch.csv", prediction_csv, "text/csv")},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["model_id"] == model_id
    assert payload["original_filename"] == "prediction_batch.csv"
    assert payload["total"] == 2
    assert len(payload["predictions"]) == 2
