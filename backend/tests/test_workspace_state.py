from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def test_get_default_workspace_state_returns_empty_state() -> None:
    response = client.get("/api/workspaces/test-default-empty/state")

    assert response.status_code == 200

    payload = response.json()
    assert payload["workspace_id"] == "test-default-empty"
    assert payload["active_route"] == "data-upload"
    assert payload["dataset_id"] is None
    assert payload["drift_status"] == "Not checked"
    assert payload["workflow_flags"] == {}
    assert payload["retrain_candidate_id"] is None


def test_patch_workspace_state_round_trips_partial_update() -> None:
    response = client.patch(
        "/api/workspaces/test-patch-state/state",
        json={
            "active_route": "model-workbench",
            "dataset_id": "dataset-1",
            "dataset_version_id": "v2",
            "target_column": "churn",
            "selected_model_id": "model-1",
            "drift_status": "warning",
            "workflow_flags": {
                "schemaReviewed": True,
                "trainingCompleted": True,
            },
            "retrain_candidate_id": "candidate-1",
        },
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["active_route"] == "model-workbench"
    assert payload["dataset_id"] == "dataset-1"
    assert payload["dataset_version_id"] == "v2"
    assert payload["target_column"] == "churn"
    assert payload["selected_model_id"] == "model-1"
    assert payload["drift_status"] == "warning"
    assert payload["workflow_flags"]["schemaReviewed"] is True
    assert payload["workflow_flags"]["trainingCompleted"] is True
    assert payload["retrain_candidate_id"] == "candidate-1"

    read_response = client.get("/api/workspaces/test-patch-state/state")

    assert read_response.status_code == 200
    assert read_response.json()["selected_model_id"] == "model-1"
    assert read_response.json()["workflow_flags"]["schemaReviewed"] is True
    assert read_response.json()["retrain_candidate_id"] == "candidate-1"
