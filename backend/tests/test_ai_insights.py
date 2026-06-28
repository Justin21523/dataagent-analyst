from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_ai_insight_sample_dataset() -> str:
    # AI insight 測試不依賴真實 LLM，預設會走 fallback。
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
        files={"file": ("ai_insight_sample.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201

    dataset_id = response.json()["dataset"]["id"]

    train_response = client.post(
        f"/api/ml/{dataset_id}/train",
        json={"target_column": "churn"},
    )

    assert train_response.status_code == 201

    return dataset_id


def test_llm_status_returns_disabled_by_default() -> None:
    response = client.get("/api/ai-insights/status")

    assert response.status_code == 200

    payload = response.json()

    assert payload["enabled"] is False
    assert payload["online"] is False
    assert "LLM integration is disabled" in payload["message"]


def test_eda_explanation_returns_fallback_content() -> None:
    dataset_id = _upload_ai_insight_sample_dataset()

    response = client.post(
        f"/api/ai-insights/{dataset_id}/eda",
        json={"user_goal": "Explain data quality"},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["title"] == "EDA Explanation"
    assert payload["source"] == "fallback"
    assert "EDA fallback explanation" in payload["content"]


def test_report_summary_returns_fallback_content() -> None:
    dataset_id = _upload_ai_insight_sample_dataset()

    response = client.post(
        f"/api/ai-insights/{dataset_id}/report-summary",
        json={"user_goal": "Summarize report"},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["title"] == "Report Executive Summary"
    assert payload["source"] == "fallback"
