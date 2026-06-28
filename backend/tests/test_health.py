from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def test_root_endpoint_returns_basic_navigation() -> None:
    # 測試 root endpoint 是否提供基本 API 導覽資訊。
    response = client.get("/")

    assert response.status_code == 200

    payload = response.json()
    assert payload["message"] == "DataAgent Analyst API"
    assert payload["docs_url"] == "/docs"
    assert payload["health_url"] == "/api/health"


def test_health_endpoint_returns_ok_status() -> None:
    # 測試 health endpoint 是否回傳服務狀態與版本資訊。
    response = client.get("/api/health")

    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "DataAgent Analyst API"
    assert payload["environment"] == "development"
    assert payload["version"] == "0.1.0"
    assert "timestamp" in payload


def test_readiness_endpoint_returns_runtime_checks() -> None:
    # 測試 readiness endpoint 是否回傳必要資料夾檢查結果。
    response = client.get("/api/health/ready")

    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["checks"]["data_dir"] is True
    assert payload["checks"]["raw_data_dir"] is True
    assert payload["checks"]["processed_data_dir"] is True
    assert payload["checks"]["reports_dir"] is True
    assert payload["checks"]["models_dir"] is True
    assert payload["checks"]["vector_store_dir"] is True
    assert "timestamp" in payload
