from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DriftReportRequest(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    reference_version_id: str = Field(description="Reference dataset version")
    current_version_id: str = Field(description="Current dataset version")
    model_id: str | None = Field(
        default=None,
        description="Optional model used for prediction/performance drift",
    )
    target_column: str | None = Field(
        default=None,
        description="Optional target column",
    )


class DriftMetric(BaseModel):
    column: str = Field(description="Column name")
    drift_type: str = Field(description="numeric or categorical")
    status: str = Field(description="stable, warning, or drift")
    psi: float | None = Field(default=None)
    ks_statistic: float | None = Field(default=None)
    js_distance: float | None = Field(default=None)
    details: dict[str, Any] = Field(default_factory=dict)


class DriftReportResponse(BaseModel):
    report_id: str = Field(description="Drift report ID")
    dataset_id: str = Field(description="Dataset ID")
    reference_version_id: str = Field(description="Reference version")
    current_version_id: str = Field(description="Current version")
    model_id: str | None = Field(default=None)
    target_column: str | None = Field(default=None)
    status: str = Field(description="stable, warning, drift, or failed")
    schema_drift: dict[str, Any] = Field(description="Schema drift summary")
    feature_drift: list[DriftMetric] = Field(description="Feature drift metrics")
    target_drift: DriftMetric | None = Field(default=None)
    prediction_drift: DriftMetric | None = Field(default=None)
    performance_drift: dict[str, Any] | None = Field(default=None)
    retraining_recommendation: dict[str, Any] = Field(
        default_factory=dict,
        description="Retraining recommendation summary",
    )
    recommendations: list[str] = Field(description="Recommended next actions")
    warnings: list[str] = Field(description="Warnings")
    created_at: datetime = Field(description="Creation time")


class DriftReportSummary(BaseModel):
    report_id: str = Field(description="Drift report ID")
    dataset_id: str = Field(description="Dataset ID")
    reference_version_id: str = Field(description="Reference version")
    current_version_id: str = Field(description="Current version")
    status: str = Field(description="stable, warning, drift, or failed")
    model_id: str | None = Field(default=None)
    target_column: str | None = Field(default=None)
    created_at: datetime = Field(description="Creation time")


class DriftReportListResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    reports: list[DriftReportSummary] = Field(description="Drift reports")
    total: int = Field(description="Report count")
