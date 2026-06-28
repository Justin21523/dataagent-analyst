from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

MLTaskType = Literal[
    "auto",
    "binary_classification",
    "multiclass_classification",
    "regression",
    "clustering",
    "anomaly_detection",
]

DetectedMLTaskType = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "clustering",
    "anomaly_detection",
]

ScalerType = Literal[
    "standard",
    "robust",
    "minmax",
    "none",
]

NumericImputerType = Literal[
    "median",
    "mean",
]

ClassWeightMode = Literal[
    "auto",
    "balanced",
    "none",
]


class MLWorkbenchPlanRequest(BaseModel):
    target_column: str | None = Field(
        default=None,
        description="Optional supervised learning target",
    )
    dataset_version_id: str | None = Field(
        default=None,
        description="Dataset version used for planning; defaults to latest",
    )
    task_type: MLTaskType = Field(
        default="auto",
        description="Requested task type",
    )
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Optional selected feature columns",
    )
    excluded_columns: list[str] = Field(
        default_factory=list,
        description="Columns excluded from the experiment",
    )
    include_datetime: bool = Field(
        default=True,
        description="Include engineered datetime features",
    )
    include_text: bool = Field(
        default=False,
        description="Include TF-IDF text features",
    )
    numeric_imputer: NumericImputerType = Field(
        default="median",
    )
    scaler: ScalerType = Field(
        default="standard",
    )
    one_hot_min_frequency: int = Field(
        default=1,
        ge=1,
        le=1000,
    )
    text_max_features: int = Field(
        default=300,
        ge=50,
        le=3000,
    )


class MLWorkbenchExperimentRequest(MLWorkbenchPlanRequest):
    selected_models: list[str] = Field(
        default_factory=list,
        description="Selected model IDs",
    )
    cv_folds: int = Field(
        default=5,
        ge=2,
        le=10,
    )
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
    )
    random_state: int = Field(default=42)
    class_weight_mode: ClassWeightMode = Field(
        default="auto",
    )
    n_clusters: int = Field(
        default=3,
        ge=2,
        le=30,
    )
    dbscan_eps: float = Field(
        default=0.7,
        gt=0,
        le=20,
    )
    dbscan_min_samples: int = Field(
        default=5,
        ge=2,
        le=100,
    )
    contamination: float = Field(
        default=0.05,
        gt=0,
        le=0.5,
    )


class MLModelOption(BaseModel):
    id: str = Field(description="Model ID")
    label: str = Field(description="Display label")
    description: str = Field(description="Model description")
    recommended: bool = Field(description="Recommended by default")
    supports_probability: bool = Field(description="Whether predict_proba is available")


class MLFeatureGroups(BaseModel):
    numeric: list[str] = Field(description="Numeric features")
    categorical: list[str] = Field(description="Categorical and boolean features")
    datetime: list[str] = Field(description="Datetime features")
    text: list[str] = Field(description="Text features")
    excluded: list[str] = Field(description="Excluded columns")


class MLPipelineStep(BaseModel):
    id: str = Field(description="Pipeline step ID")
    label: str = Field(description="Pipeline step label")
    columns: list[str] = Field(description="Affected columns")
    operations: list[str] = Field(description="Operations")


class MLWorkbenchPlanResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    dataset_version_id: str | None = Field(
        default=None,
        description="Dataset version used for planning",
    )
    detected_task_type: DetectedMLTaskType = Field(description="Detected task type")
    target_column: str | None = Field(description="Selected target")
    target_readiness_score: float | None = Field(description="Target readiness score")
    feature_groups: MLFeatureGroups = Field(description="Selected feature groups")
    preprocessing_steps: list[MLPipelineStep] = Field(description="Preprocessing pipeline")
    estimated_feature_count: int = Field(description="Estimated transformed feature count")
    available_models: list[MLModelOption] = Field(description="Available model options")
    recommended_metrics: list[str] = Field(description="Recommended evaluation metrics")
    primary_metric: str = Field(description="Primary model selection metric")
    warnings: list[str] = Field(description="Planning warnings")


class CrossValidationMetric(BaseModel):
    name: str = Field(description="Metric name")
    mean: float | None = Field(description="Mean cross-validation score")
    std: float | None = Field(description="Cross-validation standard deviation")
    fold_values: list[float | None] = Field(description="Individual fold scores")
    direction: Literal["maximize", "minimize"] = Field(description="Metric optimization direction")


class MLWorkbenchModelResult(BaseModel):
    model_id: str | None = Field(
        default=None,
        description="Saved model ID for supervised models",
    )
    model_name: str = Field(description="Model ID")
    model_label: str = Field(description="Model display label")
    status: Literal["success", "failed"] = Field(description="Model run status")
    cv_metrics: list[CrossValidationMetric] = Field(
        default_factory=list,
        description="Cross-validation metrics",
    )
    holdout_metrics: dict[str, float | None] = Field(
        default_factory=dict,
        description="Holdout or unsupervised metrics",
    )
    feature_count: int = Field(
        default=0,
        description="Transformed feature count",
    )
    training_seconds: float = Field(
        default=0,
        description="Training duration",
    )
    feature_importance: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top feature importance",
    )
    error_samples: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Misclassified or high-residual samples",
    )
    artifact: dict[str, Any] = Field(
        default_factory=dict,
        description="Clustering or anomaly artifacts",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Model warnings",
    )
    error_message: str | None = Field(
        default=None,
        description="Failure message",
    )


class MLWorkbenchExperimentResponse(BaseModel):
    experiment_id: str = Field(description="Experiment ID")
    dataset_id: str = Field(description="Dataset ID")
    dataset_version_id: str | None = Field(
        default=None,
        description="Dataset version used for the experiment",
    )
    status: Literal[
        "success",
        "completed_with_warnings",
        "failed",
    ] = Field(description="Experiment status")
    task_type: DetectedMLTaskType = Field(description="Experiment task type")
    target_column: str | None = Field(description="Target column")
    feature_groups: MLFeatureGroups = Field(description="Experiment feature groups")
    preprocessing_steps: list[MLPipelineStep] = Field(description="Preprocessing pipeline")
    cv_folds: int | None = Field(description="Actual cross-validation folds")
    primary_metric: str = Field(description="Primary metric")
    best_model_id: str | None = Field(description="Best saved model ID")
    best_model_name: str | None = Field(description="Best model option ID")
    best_metric_value: float | None = Field(description="Best primary metric value")
    model_results: list[MLWorkbenchModelResult] = Field(description="Model experiment results")
    warnings: list[str] = Field(description="Experiment warnings")
    created_at: datetime = Field(description="Experiment creation time")


class MLWorkbenchExperimentSummary(BaseModel):
    experiment_id: str = Field(description="Experiment ID")
    dataset_id: str = Field(description="Dataset ID")
    status: str = Field(description="Experiment status")
    task_type: str = Field(description="Task type")
    target_column: str | None = Field(description="Target column")
    model_count: int = Field(description="Model result count")
    best_model_name: str | None = Field(description="Best model")
    best_metric_value: float | None = Field(description="Best metric value")
    created_at: datetime = Field(description="Creation time")


class MLWorkbenchExperimentListResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    experiments: list[MLWorkbenchExperimentSummary] = Field(description="Experiment history")
    total: int = Field(description="Experiment count")
