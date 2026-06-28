from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_agent_history_sample_dataset() -> str:
    # Run history 測試使用可支援完整 workflow 的小型分類資料。
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
        files={"file": ("agent_history_sample.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201

    return response.json()["dataset"]["id"]


def _run_agent_workflow(dataset_id: str) -> str:
    response = client.post(
        f"/api/agents/{dataset_id}/run",
        json={
            "user_goal": "Run workflow history test.",
            "target_column": "churn",
            "run_ml": True,
            "generate_report": True,
            "generate_ai_insight": False,
        },
    )

    assert response.status_code == 200

    return response.json()["workflow_id"]


def test_agent_run_history_lists_completed_runs() -> None:
    dataset_id = _upload_agent_history_sample_dataset()
    workflow_id = _run_agent_workflow(dataset_id)

    response = client.get(f"/api/agents/{dataset_id}/runs")

    assert response.status_code == 200

    payload = response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["total"] >= 1
    assert any(run["workflow_id"] == workflow_id for run in payload["runs"])


def test_agent_run_detail_returns_saved_run() -> None:
    dataset_id = _upload_agent_history_sample_dataset()
    workflow_id = _run_agent_workflow(dataset_id)

    response = client.get(f"/api/agents/runs/{workflow_id}")

    assert response.status_code == 200

    payload = response.json()

    assert payload["workflow_id"] == workflow_id
    assert payload["dataset_id"] == dataset_id
    assert len(payload["steps"]) >= 7
    assert payload["outputs"]["schema"] is not None


def test_agent_run_steps_returns_timeline_steps() -> None:
    dataset_id = _upload_agent_history_sample_dataset()
    workflow_id = _run_agent_workflow(dataset_id)

    response = client.get(f"/api/agents/runs/{workflow_id}/steps")

    assert response.status_code == 200

    payload = response.json()

    assert payload["workflow_id"] == workflow_id
    assert payload["total"] >= 7
    assert payload["steps"][0]["name"] == "Planner"
