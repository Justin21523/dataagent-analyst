from datetime import datetime

from pydantic import BaseModel, Field


class WorkspaceState(BaseModel):
    workspace_id: str = Field(description="Workspace identifier")
    active_route: str = Field(default="data-upload", description="Last active UI route")
    dataset_id: str | None = Field(default=None, description="Selected dataset ID")
    dataset_version_id: str | None = Field(default=None, description="Selected dataset version ID")
    target_column: str | None = Field(default=None, description="Selected target column")
    selected_model_id: str | None = Field(default=None, description="Selected model ID")
    drift_report_id: str | None = Field(default=None, description="Latest drift report ID")
    drift_status: str = Field(default="Not checked", description="Latest drift status")
    workflow_flags: dict[str, bool] = Field(
        default_factory=dict,
        description="UI workflow completion flags",
    )
    retrain_candidate_id: str | None = Field(
        default=None,
        description="Latest challenger model candidate ID",
    )
    created_at: datetime = Field(description="State creation time")
    updated_at: datetime = Field(description="State update time")


class WorkspaceStateUpdate(BaseModel):
    active_route: str | None = Field(default=None, description="Last active UI route")
    dataset_id: str | None = Field(default=None, description="Selected dataset ID")
    dataset_version_id: str | None = Field(default=None, description="Selected dataset version ID")
    target_column: str | None = Field(default=None, description="Selected target column")
    selected_model_id: str | None = Field(default=None, description="Selected model ID")
    drift_report_id: str | None = Field(default=None, description="Latest drift report ID")
    drift_status: str | None = Field(default=None, description="Latest drift status")
    workflow_flags: dict[str, bool] | None = Field(
        default=None,
        description="UI workflow completion flags",
    )
    retrain_candidate_id: str | None = Field(
        default=None,
        description="Latest challenger model candidate ID",
    )
