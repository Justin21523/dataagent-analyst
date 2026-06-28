from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ModelExplainabilityRequest(BaseModel):
    sample_size: int = Field(
        default=100,
        ge=10,
        le=300,
        description="Rows used for SHAP explanations",
    )
    background_size: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Rows used as SHAP background",
    )
    permutation_repeats: int = Field(
        default=8,
        ge=3,
        le=30,
        description="Permutation importance repeats",
    )
    random_state: int = Field(default=42)
    local_row_position: int = Field(
        default=0,
        ge=0,
        description="Holdout row position for local explanation",
    )
    positive_class: str | None = Field(
        default=None,
        description="Positive class for classification explanations",
    )
    include_permutation: bool = Field(default=True)
    include_shap: bool = Field(default=True)
    force_recompute: bool = Field(default=False)


class ExplainabilityImportanceItem(BaseModel):
    feature: str = Field(description="Feature name")
    source_feature: str = Field(description="Original input feature")
    importance_mean: float = Field(description="Mean importance")
    importance_std: float | None = Field(
        default=None,
        description="Importance standard deviation",
    )


class ExplainabilityCurvePoint(BaseModel):
    x: float = Field(description="X coordinate")
    y: float = Field(description="Y coordinate")
    threshold: float | None = Field(
        default=None,
        description="Decision threshold",
    )


class ClassificationCurve(BaseModel):
    class_label: str = Field(description="One-vs-rest class")
    score_source: str = Field(description="predict_proba or decision_function")
    positive_support: int = Field(description="Positive sample count")
    negative_support: int = Field(description="Negative sample count")
    roc_auc: float | None = Field(description="ROC AUC")
    average_precision: float | None = Field(description="Average precision")
    roc_points: list[ExplainabilityCurvePoint] = Field(description="ROC points")
    precision_recall_points: list[ExplainabilityCurvePoint] = Field(
        description="Precision-recall points"
    )


class CalibrationPoint(BaseModel):
    mean_predicted_probability: float = Field(description="Mean predicted probability")
    fraction_of_positives: float = Field(description="Observed positive ratio")


class ClassificationDiagnostics(BaseModel):
    labels: list[str] = Field(description="Class labels")
    confusion_matrix: list[list[int]] = Field(description="Confusion matrix")
    curves: list[ClassificationCurve] = Field(description="One-vs-rest curves")
    calibration_class: str | None = Field(
        default=None,
        description="Class used for calibration",
    )
    calibration_points: list[CalibrationPoint] = Field(
        default_factory=list,
        description="Binary calibration curve",
    )


class RegressionDiagnosticPoint(BaseModel):
    row_position: int = Field(description="Holdout row position")
    dataset_index: str = Field(description="Original row index")
    actual: float = Field(description="Actual target")
    predicted: float = Field(description="Predicted target")
    residual: float = Field(description="Actual minus predicted")
    absolute_error: float = Field(description="Absolute error")


class RegressionDiagnostics(BaseModel):
    points: list[RegressionDiagnosticPoint] = Field(
        description="Predicted/actual and residual points"
    )
    residual_mean: float | None = Field(description="Mean residual")
    residual_std: float | None = Field(description="Residual standard deviation")
    residual_median: float | None = Field(description="Median residual")
    maximum_absolute_error: float | None = Field(description="Maximum absolute error")


class ShapBeeswarmPoint(BaseModel):
    feature: str = Field(description="Transformed feature")
    source_feature: str = Field(description="Original source feature")
    row_position: int = Field(description="Explanation sample position")
    shap_value: float = Field(description="SHAP contribution")
    feature_value: float | None = Field(description="Transformed feature value")


class ShapLocalContribution(BaseModel):
    feature: str = Field(description="Transformed feature")
    source_feature: str = Field(description="Original source feature")
    feature_value: float | None = Field(description="Transformed feature value")
    shap_value: float = Field(description="SHAP contribution")
    direction: str = Field(description="positive or negative")


class ShapLocalExplanation(BaseModel):
    holdout_row_position: int = Field(description="Position in holdout data")
    dataset_index: str = Field(description="Original dataset index")
    output_name: str = Field(description="Explained model output")
    base_value: float | None = Field(description="Expected model output")
    explained_output_value: float | None = Field(description="Base value plus SHAP contributions")
    prediction: Any = Field(description="Predicted value")
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Prediction probabilities",
    )
    raw_record: dict[str, Any] = Field(description="Raw input record")
    contributions: list[ShapLocalContribution] = Field(description="Top local contributions")


class ShapSummary(BaseModel):
    available: bool = Field(description="Whether SHAP was generated")
    explainer_type: str | None = Field(
        default=None,
        description="TreeExplainer or LinearExplainer",
    )
    output_name: str | None = Field(
        default=None,
        description="Explained class or output",
    )
    transformed_feature_importance: list[ExplainabilityImportanceItem] = Field(
        default_factory=list,
        description="Transformed feature SHAP importance",
    )
    source_feature_importance: list[ExplainabilityImportanceItem] = Field(
        default_factory=list,
        description="Aggregated raw feature SHAP importance",
    )
    beeswarm_points: list[ShapBeeswarmPoint] = Field(
        default_factory=list,
        description="SHAP beeswarm points",
    )
    local_explanation: ShapLocalExplanation | None = Field(
        default=None,
        description="Single-row SHAP explanation",
    )
    warning: str | None = Field(
        default=None,
        description="SHAP warning",
    )


class ExplainabilityHoldoutSummary(BaseModel):
    source: str = Field(description="saved_split_context or deterministic_fallback")
    row_count: int = Field(description="Holdout row count")
    sampled_row_count: int = Field(description="Rows used for SHAP")
    feature_count: int = Field(description="Raw feature count")
    target_non_null_count: int = Field(description="Non-null target count")


class ModelExplainabilityResponse(BaseModel):
    model_id: str = Field(description="Model ID")
    dataset_id: str = Field(description="Dataset ID")
    model_name: str = Field(description="Model name")
    task_type: str = Field(description="classification or regression")
    target_column: str = Field(description="Target column")
    generated_at: datetime = Field(description="Generation time")
    cache_hit: bool = Field(description="Whether cached output was used")
    holdout: ExplainabilityHoldoutSummary = Field(description="Holdout summary")
    classification: ClassificationDiagnostics | None = Field(
        default=None,
        description="Classification diagnostics",
    )
    regression: RegressionDiagnostics | None = Field(
        default=None,
        description="Regression diagnostics",
    )
    permutation_scoring: str | None = Field(
        default=None,
        description="Permutation importance scorer",
    )
    permutation_importance: list[ExplainabilityImportanceItem] = Field(
        default_factory=list,
        description="Raw feature permutation importance",
    )
    shap: ShapSummary = Field(description="SHAP explanation")
    error_samples: list[dict[str, Any]] = Field(description="Misclassified or high-error rows")
    warnings: list[str] = Field(description="Explainability warnings")
