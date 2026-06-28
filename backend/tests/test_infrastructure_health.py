from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def test_infrastructure_status_returns_service_checks() -> None:
    # 本機測試預設 checks disabled；Docker 環境會改成 enabled。
    response = client.get("/api/health/infrastructure")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] in {
        "disabled",
        "healthy",
        "degraded",
    }
    assert "checks_enabled" in payload
    assert len(payload["services"]) == 3

    service_names = {service["service"] for service in payload["services"]}

    assert service_names == {
        "postgresql",
        "qdrant",
        "llama.cpp",
    }
