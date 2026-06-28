from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from backend.app.schemas.agent_schema import AgentRunRequest, AgentRunResponse


class AgentJobEvent(BaseModel):
    event_id: str = Field(description="Event ID")
    job_id: str = Field(description="Agent job ID")
    workflow_id: str | None = Field(default=None, description="Workflow run ID")
    event_type: str = Field(description="queued, running, step, completed, or failed")
    step_name: str | None = Field(default=None, description="Workflow step name")
    status: str = Field(description="Event status")
    message: str = Field(description="Event message")
    payload: dict[str, Any] = Field(default_factory=dict, description="Event payload")
    created_at: datetime = Field(description="Event creation time")


class AgentJobSummary(BaseModel):
    job_id: str = Field(description="Agent job ID")
    dataset_id: str = Field(description="Dataset ID")
    workflow_id: str | None = Field(default=None, description="Workflow run ID")
    status: str = Field(description="queued, running, success, failed")
    user_goal: str | None = Field(description="User goal")
    created_at: datetime = Field(description="Job creation time")
    started_at: datetime | None = Field(default=None, description="Job start time")
    finished_at: datetime | None = Field(default=None, description="Job finish time")
    event_count: int = Field(description="Job event count")
    error_message: str | None = Field(default=None, description="Error message")


class AgentJobDetail(AgentJobSummary):
    request: AgentRunRequest = Field(description="Original agent run request")
    events: list[AgentJobEvent] = Field(description="Job events")
    result: AgentRunResponse | None = Field(default=None, description="Workflow result")


class AgentJobStartResponse(BaseModel):
    message: str = Field(description="Job start message")
    job: AgentJobSummary = Field(description="Started job summary")


class AgentJobListResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    jobs: list[AgentJobSummary] = Field(description="Agent jobs")
    total: int = Field(description="Total job count")


class AgentJobEventListResponse(BaseModel):
    job_id: str = Field(description="Agent job ID")
    events: list[AgentJobEvent] = Field(description="Agent job events")
    total: int = Field(description="Total event count")
