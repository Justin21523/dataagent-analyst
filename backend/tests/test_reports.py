from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_report_sample_dataset() -> str:
    # 測試資料包含 target 與數值欄位，讓報告能涵蓋 schema、EDA、ML 區塊。
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

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("report_sample.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201

    dataset_id = response.json()["dataset"]["id"]

    train_response = client.post(
        f"/api/ml/{dataset_id}/train",
        json={"target_column": "churn"},
    )

    assert train_response.status_code == 201

    return dataset_id


def test_generate_report_returns_markdown_content() -> None:
    dataset_id = _upload_report_sample_dataset()

    response = client.post(f"/api/reports/{dataset_id}/generate")

    assert response.status_code == 200

    payload = response.json()
    report = payload["report"]

    assert payload["message"] == "Report generated successfully."
    assert report["dataset_id"] == dataset_id
    assert report["status"] == "ready"
    assert "# Data Analysis Report" in report["markdown_content"]
    assert "## 3. Data Quality Summary" in report["markdown_content"]
    assert "## 8. Machine Learning Results" in report["markdown_content"]


def test_list_reports_returns_generated_report() -> None:
    dataset_id = _upload_report_sample_dataset()

    generate_response = client.post(f"/api/reports/{dataset_id}/generate")
    assert generate_response.status_code == 200

    list_response = client.get(f"/api/reports/dataset/{dataset_id}")

    assert list_response.status_code == 200

    payload = list_response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["total"] >= 1
    assert payload["reports"][0]["dataset_id"] == dataset_id


def test_get_report_returns_markdown_content() -> None:
    dataset_id = _upload_report_sample_dataset()

    generate_response = client.post(f"/api/reports/{dataset_id}/generate")
    report_id = generate_response.json()["report"]["id"]

    response = client.get(f"/api/reports/{report_id}")

    assert response.status_code == 200

    payload = response.json()

    assert payload["id"] == report_id
    assert "## 1. Dataset Overview" in payload["markdown_content"]
