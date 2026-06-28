from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class InfrastructureServiceCheck(BaseModel):
    service: str = Field(description="Infrastructure service name")
    status: str = Field(description="healthy, unhealthy, or disabled")
    healthy: bool | None = Field(description="Health state")
    message: str = Field(description="Health check message")
    latency_ms: float | None = Field(description="Health check latency")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional service details",
    )


class InfrastructureStatusResponse(BaseModel):
    status: str = Field(description="healthy, degraded, or disabled")
    checks_enabled: bool = Field(description="Whether checks are enabled")
    services: list[InfrastructureServiceCheck] = Field(description="Infrastructure service checks")
    checked_at: datetime = Field(description="Infrastructure check time")
