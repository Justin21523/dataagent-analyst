from typing import Any

from pydantic import BaseModel, Field


class ChartData(BaseModel):
    labels: list[Any] = Field(default_factory=list, description="Chart labels")
    datasets: list[dict[str, Any]] = Field(description="Chart.js datasets")
    x_label: str | None = Field(default=None, description="X axis label")
    y_label: str | None = Field(default=None, description="Y axis label")


class ChartRecommendation(BaseModel):
    id: str = Field(description="Unique chart recommendation ID")
    title: str = Field(description="Chart title")
    description: str = Field(description="Chart description")
    chart_type: str = Field(description="Chart.js chart type")
    chart_family: str = Field(description="High-level chart family")
    priority: int = Field(description="Display priority")
    columns: list[str] = Field(description="Columns used in this chart")
    reason: str = Field(description="Why this chart is recommended")
    data: ChartData = Field(description="Chart.js-compatible chart data")


class VisualizationRecommendationsResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    recommendations: list[ChartRecommendation] = Field(description="Recommended charts")
    total: int = Field(description="Total recommendation count")
