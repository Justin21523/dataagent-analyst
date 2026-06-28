from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_agent_sample_dataset() -> str:
    # Agent workflow 測試資料要支援 profile、EDA、visualization、ML、report。
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
        files={"file": ("agent_sample.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201

    return response.json()["dataset"]["id"]


def test_agent_workflow_runs_end_to_end() -> None:
    dataset_id = _upload_agent_sample_dataset()

    response = client.post(
        f"/api/agents/{dataset_id}/run",
        json={
            "user_goal": "Run a complete analysis workflow for a portfolio demo.",
            "target_column": "churn",
            "run_ml": True,
            "generate_report": True,
            "generate_ai_insight": True,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["status"] in {"success", "completed_with_warnings"}
    assert payload["workflow_id"]
    assert len(payload["steps"]) >= 7
    assert payload["outputs"]["schema"] is not None
    assert payload["outputs"]["eda_summary"] is not None
    assert payload["outputs"]["visualizations"] is not None
    assert payload["outputs"]["report"] is not None
    assert payload["final_summary"]


def test_agent_workflow_can_skip_optional_steps() -> None:
    dataset_id = _upload_agent_sample_dataset()

    response = client.post(
        f"/api/agents/{dataset_id}/run",
        json={
            "user_goal": "Run EDA only.",
            "run_ml": False,
            "generate_report": False,
            "generate_ai_insight": False,
        },
    )

    assert response.status_code == 200

    payload = response.json()
    step_statuses = {step["name"]: step["status"] for step in payload["steps"]}

    assert step_statuses["ML Trainer"] == "skipped"
    assert step_statuses["Report Writer"] == "skipped"
    assert step_statuses["AI Insight Generator"] == "skipped"
