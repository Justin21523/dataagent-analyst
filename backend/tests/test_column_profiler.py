from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_profile_sample_dataset() -> str:
    # 建立同時包含數值、類別、日期、布林、文字與 target 的測試資料。
    csv_content = (
        b"customer_id,age,signup_date,is_active,contract_type,notes,churn\n"
        b"C0001,29,2024-01-15,true,Monthly,Called support once,No\n"
        b"C0002,41,2024-02-20,false,Monthly,Requested cancellation,Yes\n"
        b"C0003,,2024-03-05,true,One Year,,No\n"
        b"C0004,52,2024-03-19,false,Monthly,High monthly usage,Yes\n"
    )

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("profile_sample.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201

    return response.json()["dataset"]["id"]


def test_columns_endpoint_returns_column_profiles() -> None:
    dataset_id = _upload_profile_sample_dataset()

    response = client.get(f"/api/datasets/{dataset_id}/columns")

    assert response.status_code == 200

    payload = response.json()
    columns = {column["name"]: column for column in payload["columns"]}

    assert payload["dataset_id"] == dataset_id
    assert payload["total"] == 7

    assert columns["customer_id"]["semantic_role"] == "identifier"
    assert columns["age"]["inferred_type"] == "numeric"
    assert columns["age"]["missing_count"] == 1
    assert columns["signup_date"]["inferred_type"] == "datetime"
    assert columns["is_active"]["inferred_type"] == "boolean"
    assert columns["contract_type"]["inferred_type"] == "categorical"
    assert columns["churn"]["semantic_role"] == "target_candidate"


def test_schema_endpoint_returns_summary() -> None:
    dataset_id = _upload_profile_sample_dataset()

    response = client.get(f"/api/datasets/{dataset_id}/schema")

    assert response.status_code == 200

    payload = response.json()
    summary = payload["summary"]

    assert summary["dataset_id"] == dataset_id
    assert summary["row_count"] == 4
    assert summary["column_count"] == 7
    assert summary["missing_cell_count"] == 2
    assert "numeric" in summary["type_counts"]
    assert "datetime" in summary["type_counts"]
    assert "churn" in summary["target_candidates"]
    assert len(payload["columns"]) == 7
