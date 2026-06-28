from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class FeatureImportanceItem(BaseModel):
    feature: str = Field(description="Feature name")
    importance: float = Field(description="Feature importance score")


class MLTrainRequest(BaseModel):
    target_column: str = Field(description="Target column for model training")
    feature_columns: list[str] | None = Field(
        default=None,
        description="Optional selected feature columns",
    )
    task_type: str = Field(
        default="auto",
        description="auto, classification, or regression",
    )
    test_size: float = Field(default=0.25, ge=0.1, le=0.5)
    random_state: int = Field(default=42)
    selected_models: list[str] | None = Field(
        default=None,
        description="Optional model names to train",
    )
    dataset_version_id: str | None = Field(
        default=None,
        description="Dataset version used for training; defaults to latest",
    )


class MLModelResult(BaseModel):
    id: str = Field(description="Model ID")
    dataset_id: str = Field(description="Dataset ID")
    dataset_version_id: str | None = Field(
        default=None,
        description="Dataset version used for training",
    )
    model_name: str = Field(description="Model name")
    task_type: str = Field(description="classification or regression")
    target_column: str = Field(description="Target column")
    feature_columns: list[str] = Field(description="Feature columns")
    metrics: dict[str, float | None] = Field(description="Evaluation metrics")
    feature_importance: list[FeatureImportanceItem] = Field(
        default_factory=list,
        description="Feature importance values",
    )
    model_path: str = Field(description="Saved model path")
    evaluation_artifacts_path: str | None = Field(
        default=None,
        description="Saved evaluation artifacts path",
    )
    status: str = Field(description="Model status")
    lifecycle_status: str = Field(
        default="candidate",
        description="candidate, staging, production, or archived",
    )
    training_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Training configuration snapshot",
    )
    feature_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="Feature schema and runtime contract",
    )
    preprocessing_recipe: dict[str, Any] = Field(
        default_factory=dict,
        description="Preprocessing recipe or pipeline description",
    )
    created_at: datetime = Field(description="Model creation time")


class MLTrainResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    task_type: str = Field(description="Detected or selected task type")
    target_column: str = Field(description="Target column")
    feature_columns: list[str] = Field(description="Used feature columns")
    model_count: int = Field(description="Number of trained models")
    best_model_id: str = Field(description="Best model ID")
    best_metric_name: str = Field(description="Metric used for best model selection")
    best_metric_value: float | None = Field(description="Best metric value")
    models: list[MLModelResult] = Field(description="Training results")


class MLModelListResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    models: list[MLModelResult] = Field(description="Model list")
    total: int = Field(description="Total model count")


class MLModelMetricsResponse(BaseModel):
    model_id: str = Field(description="Model ID")
    metrics: dict[str, float | None] = Field(description="Model metrics")


class ConfusionMatrixArtifact(BaseModel):
    labels: list[str] = Field(description="Class labels")
    matrix: list[list[int]] = Field(description="Confusion matrix values")


class RegressionResidualSample(BaseModel):
    actual: float | None = Field(description="Actual target value")
    predicted: float | None = Field(description="Predicted target value")
    residual: float | None = Field(description="Actual minus predicted")


class MLModelEvaluationResponse(BaseModel):
    model: MLModelResult = Field(description="Model metadata")
    confusion_matrix: ConfusionMatrixArtifact | None = Field(
        default=None,
        description="Classification confusion matrix",
    )
    regression_residuals: list[RegressionResidualSample] = Field(
        default_factory=list,
        description="Regression residual samples",
    )
    feature_importance: list[FeatureImportanceItem] = Field(
        default_factory=list,
        description="Feature importance values",
    )


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(description="Input records for prediction")


class PredictionResult(BaseModel):
    row_index: int = Field(description="Input row index")
    prediction: Any = Field(description="Predicted value")
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Classification probabilities if available",
    )


class PredictionResponse(BaseModel):
    model_id: str = Field(description="Model ID")
    dataset_id: str = Field(description="Dataset ID")
    model_name: str = Field(description="Model name")
    task_type: str = Field(description="classification or regression")
    target_column: str = Field(description="Target column")
    feature_columns: list[str] = Field(description="Required feature columns")
    predictions: list[PredictionResult] = Field(description="Prediction results")
    total: int = Field(description="Total prediction count")


class BatchPredictionResponse(PredictionResponse):
    original_filename: str = Field(description="Uploaded batch prediction filename")


class ModelStatusUpdateRequest(BaseModel):
    lifecycle_status: str = Field(
        description="candidate, staging, production, or archived",
    )


class ModelCompareRequest(BaseModel):
    model_ids: list[str] = Field(description="Models to compare")
    primary_metric: str | None = Field(
        default=None,
        description="Metric used for sorting; defaults by task type",
    )


class ModelCompareItem(BaseModel):
    model_id: str = Field(description="Model ID")
    model_name: str = Field(description="Model name")
    lifecycle_status: str = Field(description="Model lifecycle status")
    task_type: str = Field(description="Task type")
    target_column: str = Field(description="Target column")
    metrics: dict[str, float | None] = Field(description="Model metrics")
    primary_metric_value: float | None = Field(description="Primary metric value")
    rank: int = Field(description="Comparison rank")


class ModelCompareResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    primary_metric: str = Field(description="Primary metric")
    models: list[ModelCompareItem] = Field(description="Ranked models")


class ThresholdAnalysisRequest(BaseModel):
    dataset_version_id: str | None = Field(
        default=None,
        description="Dataset version to evaluate; defaults to latest",
    )
    positive_class: str | None = Field(
        default=None,
        description="Positive class label; defaults to last class",
    )
    thresholds: list[float] = Field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7],
        description="Thresholds to evaluate",
    )


class ThresholdAnalysisPoint(BaseModel):
    threshold: float = Field(description="Classification threshold")
    precision: float | None = Field(description="Precision")
    recall: float | None = Field(description="Recall")
    f1: float | None = Field(description="F1")
    confusion_matrix: dict[str, int] = Field(description="TP/FP/TN/FN counts")


class ThresholdAnalysisResponse(BaseModel):
    model_id: str = Field(description="Model ID")
    dataset_id: str = Field(description="Dataset ID")
    dataset_version_id: str = Field(description="Dataset version")
    positive_class: str = Field(description="Positive class")
    points: list[ThresholdAnalysisPoint] = Field(description="Threshold metrics")


class SegmentMetricsRequest(BaseModel):
    segment_column: str = Field(description="Column used for segment metrics")
    dataset_version_id: str | None = Field(default=None)
    max_bins: int = Field(default=5, ge=2, le=20)


class SegmentMetricItem(BaseModel):
    segment: str = Field(description="Segment label")
    row_count: int = Field(description="Rows in segment")
    metrics: dict[str, float | None] = Field(description="Segment metrics")


class SegmentMetricsResponse(BaseModel):
    model_id: str = Field(description="Model ID")
    dataset_id: str = Field(description="Dataset ID")
    dataset_version_id: str = Field(description="Dataset version")
    segment_column: str = Field(description="Segment column")
    segments: list[SegmentMetricItem] = Field(description="Segment metrics")


class WhatIfScenario(BaseModel):
    name: str = Field(description="Scenario name")
    changes: dict[str, Any] = Field(description="Feature changes")


class WhatIfRequest(BaseModel):
    base_record: dict[str, Any] = Field(description="Base prediction record")
    scenarios: list[WhatIfScenario] = Field(description="What-if scenarios")


class WhatIfResult(BaseModel):
    scenario: str = Field(description="Scenario name")
    record: dict[str, Any] = Field(description="Scenario record")
    prediction: Any = Field(description="Prediction")
    probabilities: dict[str, float] | None = Field(default=None)


class WhatIfResponse(BaseModel):
    model_id: str = Field(description="Model ID")
    dataset_id: str = Field(description="Dataset ID")
    results: list[WhatIfResult] = Field(description="Scenario results")


class ModelMigrationCheckResponse(BaseModel):
    model_id: str = Field(description="Model ID")
    compatible: bool = Field(description="Whether the model passed migration checks")
    checks: list[dict[str, Any]] = Field(description="Individual checks")
    warnings: list[str] = Field(description="Migration warnings")


class ModelRetrainRequest(BaseModel):
    current_version_id: str = Field(description="Dataset version used to train the challenger")
    reference_version_id: str | None = Field(
        default=None,
        description="Reference version used for context",
    )
    drift_report_id: str | None = Field(
        default=None,
        description="Optional drift report that triggered retraining",
    )
    auto_promote: bool = Field(
        default=False,
        description="Whether to promote a clearly better challenger automatically",
    )


class ModelRetrainPlanResponse(BaseModel):
    champion_model_id: str = Field(description="Champion model ID")
    dataset_id: str = Field(description="Dataset ID")
    current_version_id: str = Field(description="Current dataset version")
    target_column: str = Field(description="Target column")
    task_type: str = Field(description="classification or regression")
    selected_model: str = Field(description="Model family to retrain")
    feature_columns: list[str] = Field(description="Features reused from champion")
    primary_metric: str = Field(description="Metric used for comparison")
    warnings: list[str] = Field(description="Planning warnings")


class ModelRetrainResponse(BaseModel):
    champion_model: MLModelResult = Field(description="Original champion model")
    challenger_model: MLModelResult = Field(description="New challenger model")
    comparison: ModelCompareResponse = Field(description="Champion/challenger comparison")
    recommendation: str = Field(description="promote, review, or keep_champion")
    reasons: list[str] = Field(description="Recommendation reasons")
    promoted: bool = Field(description="Whether challenger was promoted")


class PromoteChallengerRequest(BaseModel):
    challenger_model_id: str = Field(description="Challenger model to promote")


class PromoteChallengerResponse(BaseModel):
    promoted_model: MLModelResult = Field(description="Promoted challenger model")
    archived_model_id: str | None = Field(
        default=None,
        description="Previous production model archived by the promotion",
    )
