from typing import Any, Literal

from pydantic import BaseModel, Field

VisualizationChartType = Literal[
    "missing_bar",
    "missing_heatmap",
    "correlation_heatmap",
    "histogram",
    "boxplot",
    "category_bar",
    "line",
    "scatter",
    "target_distribution",
    "target_boxplot",
    "target_stacked_bar",
    "feature_target_scatter",
]

CustomChartType = Literal[
    "histogram",
    "boxplot",
    "category_bar",
    "scatter",
    "line",
    "correlation_heatmap",
    "feature_target",
]


class VisualizationLabRequest(BaseModel):
    target_column: str | None = Field(
        default=None,
        description="Optional target column",
    )
    selected_columns: list[str] = Field(
        default_factory=list,
        description="Optional columns included in visualization analysis",
    )
    sample_rows: int = Field(
        default=80,
        ge=20,
        le=200,
        description="Maximum rows used by heatmaps and scatter charts",
    )
    max_numeric_columns: int = Field(
        default=8,
        ge=2,
        le=12,
        description="Maximum numeric columns used in matrix charts",
    )
    max_categories: int = Field(
        default=12,
        ge=3,
        le=30,
        description="Maximum displayed categories",
    )


class VisualizationChartBuilderRequest(BaseModel):
    chart_type: CustomChartType = Field(description="Requested custom chart type")
    x_column: str | None = Field(
        default=None,
        description="Primary column",
    )
    y_column: str | None = Field(
        default=None,
        description="Secondary column",
    )
    group_column: str | None = Field(
        default=None,
        description="Optional grouping column",
    )
    target_column: str | None = Field(
        default=None,
        description="Optional target column",
    )
    sample_rows: int = Field(
        default=200,
        ge=20,
        le=1000,
    )
    max_categories: int = Field(
        default=15,
        ge=3,
        le=50,
    )


class VisualizationColumnOption(BaseModel):
    name: str = Field(description="Column name")
    inferred_type: str = Field(description="Inferred column type")
    semantic_role: str = Field(description="Semantic role")


class VisualizationSpec(BaseModel):
    id: str = Field(description="Visualization ID")
    title: str = Field(description="Visualization title")
    description: str = Field(description="Visualization description")
    insight: str = Field(description="Deterministic chart insight")
    chart_type: VisualizationChartType = Field(description="Visualization renderer type")
    chart_family: str = Field(description="Visualization family")
    priority: int = Field(description="Display priority")
    columns: list[str] = Field(description="Columns used by chart")
    data: dict[str, Any] = Field(description="Renderer-independent visualization data")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Visualization configuration",
    )


class VisualizationLabSummary(BaseModel):
    row_count: int = Field(description="Dataset row count")
    column_count: int = Field(description="Dataset column count")
    sampled_rows: int = Field(description="Rows sampled for dense charts")
    target_column: str | None = Field(description="Selected target")
    target_candidates: list[str] = Field(description="Detected target candidates")
    numeric_columns: list[str] = Field(description="Numeric columns")
    categorical_columns: list[str] = Field(description="Categorical columns")
    datetime_columns: list[str] = Field(description="Datetime columns")
    column_options: list[VisualizationColumnOption] = Field(
        description="Columns available to custom chart builder"
    )
    chart_count: int = Field(description="Generated chart count")


class VisualizationLabResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    summary: VisualizationLabSummary = Field(description="Visualization lab summary")
    charts: list[VisualizationSpec] = Field(description="Visualization specifications")
    warnings: list[str] = Field(description="Visualization warnings")
