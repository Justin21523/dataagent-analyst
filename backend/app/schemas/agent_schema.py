from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    user_goal: str | None = Field(
        default=None,
        description="Optional user goal for the analysis workflow",
    )
    target_column: str | None = Field(
        default=None,
        description="Optional target column for ML training",
    )
    run_ml: bool = Field(
        default=True,
        description="Whether the workflow should train baseline ML models",
    )
    generate_report: bool = Field(
        default=True,
        description="Whether the workflow should generate a Markdown report",
    )
    generate_ai_insight: bool = Field(
        default=True,
        description="Whether the workflow should generate AI insights",
    )


class AgentStep(BaseModel):
    name: str = Field(description="Workflow step name")
    status: str = Field(description="success, skipped, warning, or error")
    message: str = Field(description="Step message")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Step output payload",
    )
    created_at: datetime = Field(description="Step creation timestamp")


class AgentRunResponse(BaseModel):
    workflow_id: str = Field(description="Workflow run ID")
    dataset_id: str = Field(description="Dataset ID")
    status: str = Field(description="success or completed_with_warnings")
    user_goal: str | None = Field(description="User goal")
    steps: list[AgentStep] = Field(description="Workflow execution steps")
    outputs: dict[str, Any] = Field(description="Final workflow outputs")
    errors: list[str] = Field(description="Workflow warnings or errors")
    final_summary: str = Field(description="Final workflow summary")


class AgentRunSummary(BaseModel):
    workflow_id: str = Field(description="Workflow run ID")
    dataset_id: str = Field(description="Dataset ID")
    status: str = Field(description="Run status")
    user_goal: str | None = Field(description="User goal")
    step_count: int = Field(description="Workflow step count")
    error_count: int = Field(description="Workflow error count")
    started_at: datetime = Field(description="Workflow start time")
    finished_at: datetime = Field(description="Workflow finish time")
    final_summary: str = Field(description="Final workflow summary")


class AgentRunListResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    runs: list[AgentRunSummary] = Field(description="Agent run summaries")
    total: int = Field(description="Total run count")


class AgentStepListResponse(BaseModel):
    workflow_id: str = Field(description="Workflow run ID")
    steps: list[AgentStep] = Field(description="Workflow steps")
    total: int = Field(description="Total step count")
