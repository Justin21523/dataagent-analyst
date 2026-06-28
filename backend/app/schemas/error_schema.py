from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: Any | None = Field(default=None, description="Optional error details")


class ErrorResponse(BaseModel):
    detail: Any = Field(description="Backward-compatible error detail")
    error: ErrorBody = Field(description="Structured error body")
    request_id: str = Field(description="Request trace ID")
    path: str = Field(description="Request path")
    timestamp: datetime = Field(description="Error timestamp")
