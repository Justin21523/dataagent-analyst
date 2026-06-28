import time

from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_agent_job_sample_dataset() -> str:
    # Agent job 測試資料需要能完整支援 profiling、EDA、visualization、ML、report。
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
        files={"file": ("agent_job_sample.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201

    return response.json()["dataset"]["id"]


def _start_agent_job(dataset_id: str) -> str:
    response = client.post(
        f"/api/agent-jobs/{dataset_id}/start",
        json={
            "user_goal": "Run background agent job test.",
            "target_column": "churn",
            "run_ml": True,
            "generate_report": True,
            "generate_ai_insight": False,
        },
    )

    assert response.status_code == 202

    payload = response.json()

    assert payload["message"] == "Agent job accepted."
    assert payload["job"]["status"] in {"queued", "running"}

    return payload["job"]["job_id"]


def _wait_for_terminal_job(job_id: str) -> dict:
    for _ in range(30):
        response = client.get(f"/api/agent-jobs/{job_id}")
        assert response.status_code == 200

        payload = response.json()

        if payload["status"] in {"success", "completed_with_warnings", "failed"}:
            return payload

        time.sleep(0.2)

    raise AssertionError("Agent job did not reach terminal status in time.")


def test_start_agent_job_returns_job_id() -> None:
    dataset_id = _upload_agent_job_sample_dataset()
    job_id = _start_agent_job(dataset_id)

    response = client.get(f"/api/agent-jobs/{job_id}")

    assert response.status_code == 200

    payload = response.json()

    assert payload["job_id"] == job_id
    assert payload["dataset_id"] == dataset_id
    assert payload["events"]


def test_agent_job_completes_and_saves_events() -> None:
    dataset_id = _upload_agent_job_sample_dataset()
    job_id = _start_agent_job(dataset_id)

    job = _wait_for_terminal_job(job_id)

    assert job["status"] in {"success", "completed_with_warnings"}
    assert job["workflow_id"]
    assert job["result"] is not None
    assert len(job["events"]) >= 5


def test_list_agent_jobs_returns_started_job() -> None:
    dataset_id = _upload_agent_job_sample_dataset()
    job_id = _start_agent_job(dataset_id)
    _wait_for_terminal_job(job_id)

    response = client.get(f"/api/agent-jobs/dataset/{dataset_id}")

    assert response.status_code == 200

    payload = response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["total"] >= 1
    assert any(job["job_id"] == job_id for job in payload["jobs"])


def test_agent_job_events_endpoint_returns_events() -> None:
    dataset_id = _upload_agent_job_sample_dataset()
    job_id = _start_agent_job(dataset_id)
    _wait_for_terminal_job(job_id)

    response = client.get(f"/api/agent-jobs/{job_id}/events")

    assert response.status_code == 200

    payload = response.json()

    assert payload["job_id"] == job_id
    assert payload["total"] >= 5
