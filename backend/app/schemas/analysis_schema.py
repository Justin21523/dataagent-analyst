from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

QualityStatus = Literal[
    "excellent",
    "good",
    "fair",
    "poor",
]

FindingSeverity = Literal[
    "info",
    "warning",
    "critical",
]

FindingCategory = Literal[
    "data_quality",
    "schema",
    "target",
    "relationship",
    "modeling",
    "visualization",
]


class AnalysisContextRequest(BaseModel):
    target_column: str | None = Field(
        default=None,
        description="Optional target column selected by the user",
    )
    user_goal: str | None = Field(
        default=None,
        description="Optional business or analysis goal",
    )


class AnalysisColumnSummary(BaseModel):
    name: str = Field(description="Column name")
    inferred_type: str = Field(description="Inferred column type")
    semantic_role: str = Field(description="Suggested semantic role")
    role_confidence: float = Field(description="Semantic role confidence")
    missing_ratio: float = Field(description="Missing value ratio")
    unique_count: int = Field(description="Unique value count")
    unique_ratio: float = Field(description="Unique value ratio")
    cardinality: str = Field(description="binary, low, medium, or high")
    sample_values: list[Any] = Field(description="Sample non-null values")
    recommended_preprocessing: list[str] = Field(description="Recommended preprocessing steps")
    warnings: list[str] = Field(description="Column-level warnings")


class DataQualityDimension(BaseModel):
    key: str = Field(description="Quality dimension key")
    label: str = Field(description="Quality dimension label")
    score: float = Field(description="Dimension score from 0 to 100")
    weight: float = Field(description="Weight used in overall score")
    status: QualityStatus = Field(description="Dimension status")
    evidence: list[str] = Field(description="Evidence used for scoring")
    recommended_action: str = Field(description="Recommended action")


class DataQualityAssessment(BaseModel):
    overall_score: float = Field(description="Overall quality score")
    grade: str = Field(description="Overall quality grade")
    dimensions: list[DataQualityDimension] = Field(description="Quality dimensions")
    primary_risks: list[str] = Field(description="Main data quality risks")


class TargetClassDistribution(BaseModel):
    label: str = Field(description="Target class label")
    count: int = Field(description="Target class count")
    ratio: float = Field(description="Target class ratio")


class TargetAnalysis(BaseModel):
    target_column: str = Field(description="Target column")
    task_type: Literal["classification", "regression"] = Field(description="Detected ML task type")
    inferred_type: str = Field(description="Target inferred type")
    non_null_count: int = Field(description="Non-null target count")
    missing_ratio: float = Field(description="Target missing ratio")
    unique_count: int = Field(description="Target unique count")
    readiness_score: float = Field(description="Modeling readiness score")
    recommended_primary_metric: str = Field(description="Recommended primary evaluation metric")
    class_distribution: list[TargetClassDistribution] = Field(
        default_factory=list,
        description="Classification target distribution",
    )
    majority_class: str | None = Field(
        default=None,
        description="Majority class label",
    )
    majority_ratio: float | None = Field(
        default=None,
        description="Majority class ratio",
    )
    imbalance_ratio: float | None = Field(
        default=None,
        description="Majority to minority count ratio",
    )
    numeric_summary: dict[str, float | None] | None = Field(
        default=None,
        description="Regression target statistics",
    )
    warnings: list[str] = Field(description="Target-level warnings")


class AnalysisEdaSummary(BaseModel):
    missing_cell_count: int = Field(description="Total missing cells")
    missing_cell_ratio: float = Field(description="Missing cell ratio")
    duplicate_row_count: int = Field(description="Duplicate row count")
    duplicate_row_ratio: float = Field(description="Duplicate row ratio")
    top_missing_columns: list[dict[str, Any]] = Field(
        description="Columns with the most missing values"
    )
    outlier_columns: list[dict[str, Any]] = Field(description="Columns with detected outliers")
    strongest_correlations: list[dict[str, Any]] = Field(
        description="Strongest numeric correlation pairs"
    )
    numeric_column_count: int = Field(description="Numeric column count")


class AnalysisChartSummary(BaseModel):
    id: str = Field(description="Chart recommendation ID")
    title: str = Field(description="Chart title")
    chart_type: str = Field(description="Chart type")
    chart_family: str = Field(description="Chart family")
    columns: list[str] = Field(description="Columns used by chart")
    reason: str = Field(description="Recommendation reason")


class AnalysisModelSummary(BaseModel):
    id: str = Field(description="Model ID")
    model_name: str = Field(description="Model name")
    task_type: str = Field(description="Model task type")
    target_column: str = Field(description="Model target column")
    metrics: dict[str, float | None] = Field(description="Model metrics")
    feature_importance: list[dict[str, Any]] = Field(description="Top feature importance values")
    created_at: datetime = Field(description="Model creation time")


class AnalysisFinding(BaseModel):
    id: str = Field(description="Stable finding ID")
    category: FindingCategory = Field(description="Finding category")
    severity: FindingSeverity = Field(description="Finding severity")
    title: str = Field(description="Finding title")
    summary: str = Field(description="Finding summary")
    evidence: dict[str, Any] = Field(description="Finding evidence")
    recommended_action: str = Field(description="Recommended action")


class AnalysisContextResponse(BaseModel):
    analysis_context_version: str = Field(description="Analysis context schema version")
    dataset_id: str = Field(description="Dataset ID")
    user_goal: str | None = Field(description="User analysis goal")
    generated_at: datetime = Field(description="Context generation time")
    dataset_summary: dict[str, Any] = Field(description="Dataset summary")
    column_profiles: list[AnalysisColumnSummary] = Field(description="Enriched column profiles")
    data_quality: DataQualityAssessment = Field(description="Data quality assessment")
    target_analysis: TargetAnalysis | None = Field(
        description="Target analysis if a target is available"
    )
    eda_summary: AnalysisEdaSummary = Field(description="Compact EDA summary")
    chart_recommendations: list[AnalysisChartSummary] = Field(
        description="Compact chart recommendations"
    )
    ml_results: list[AnalysisModelSummary] = Field(description="Existing model results")
    feature_importance: list[dict[str, Any]] = Field(
        description="Reference model feature importance"
    )
    rag_context: list[dict[str, Any]] = Field(description="Reserved retrieved RAG context")
    findings: list[AnalysisFinding] = Field(description="Actionable analysis findings")
    recommended_next_actions: list[str] = Field(description="Recommended next analysis actions")
