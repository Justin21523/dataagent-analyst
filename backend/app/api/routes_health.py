from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.infrastructure_schema import InfrastructureStatusResponse
from backend.app.services.infrastructure_service import InfrastructureService

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    status: str = Field(description="API health status")
    service: str = Field(description="Service name")
    environment: str = Field(description="Current runtime environment")
    version: str = Field(description="Application version")
    timestamp: datetime = Field(description="Server response timestamp")


class ReadyResponse(BaseModel):
    status: str = Field(description="Readiness status")
    checks: dict[str, bool] = Field(description="Runtime dependency checks")
    timestamp: datetime = Field(description="Server response timestamp")


@router.get("", response_model=HealthResponse)
def health_check() -> HealthResponse:
    # Health check 用於確認 API server 是否正常回應。
    settings = get_settings()

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_env,
        version=settings.app_version,
        timestamp=datetime.now(UTC),
    )


@router.get("/ready", response_model=ReadyResponse)
def readiness_check() -> ReadyResponse:
    # Readiness check 用於確認後端需要的基本資料夾是否可用。
    settings = get_settings()

    checks = {
        "data_dir": settings.data_dir.exists(),
        "raw_data_dir": settings.raw_data_dir.exists(),
        "processed_data_dir": settings.processed_data_dir.exists(),
        "reports_dir": settings.reports_dir.exists(),
        "models_dir": settings.models_dir.exists(),
        "vector_store_dir": settings.vector_store_dir.exists(),
    }

    is_ready = all(checks.values())

    return ReadyResponse(
        status="ready" if is_ready else "not_ready",
        checks=checks,
        timestamp=datetime.now(UTC),
    )


def get_infrastructure_service(
    settings: Settings = Depends(get_settings),
) -> InfrastructureService:
    # Infrastructure service 集中檢查 PostgreSQL、Qdrant 與 local LLM。
    return InfrastructureService(settings)


@router.get("/infrastructure", response_model=InfrastructureStatusResponse)
def infrastructure_check(
    infrastructure_service: InfrastructureService = Depends(get_infrastructure_service),
) -> InfrastructureStatusResponse:
    return infrastructure_service.check_status()
