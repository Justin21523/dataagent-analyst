from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def test_not_found_error_returns_structured_response() -> None:
    response = client.get("/api/datasets/not-existing-dataset")

    assert response.status_code == 404

    payload = response.json()

    assert payload["detail"] == "Dataset not found: not-existing-dataset"
    assert payload["error"]["code"] == "http_404"
    assert payload["error"]["message"] == "Dataset not found: not-existing-dataset"
    assert payload["request_id"]
    assert payload["path"] == "/api/datasets/not-existing-dataset"
    assert response.headers["X-Request-ID"] == payload["request_id"]


def test_validation_error_returns_structured_response() -> None:
    response = client.post(
        "/api/ml/example-dataset/train",
        json={},
    )

    assert response.status_code == 422

    payload = response.json()

    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Request validation failed."
    assert isinstance(payload["error"]["details"], list)
    assert payload["request_id"]


def test_client_request_id_is_propagated() -> None:
    response = client.get(
        "/api/health",
        headers={"X-Request-ID": "portfolio-test-request"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "portfolio-test-request"
    assert "X-Process-Time-Ms" in response.headers
