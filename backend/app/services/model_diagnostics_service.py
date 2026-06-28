import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from backend.app.core.config import Settings
from backend.app.schemas.ml_schema import (
    ModelMigrationCheckResponse,
    SegmentMetricItem,
    SegmentMetricsRequest,
    SegmentMetricsResponse,
    ThresholdAnalysisPoint,
    ThresholdAnalysisRequest,
    ThresholdAnalysisResponse,
    WhatIfRequest,
    WhatIfResponse,
    WhatIfResult,
)
from backend.app.services.dataset_service import DatasetService
from backend.app.services.model_prediction_service import ModelPredictionError
from backend.app.services.model_registry_service import ModelRegistryService


class ModelDiagnosticsError(Exception):
    """Raised when model diagnostics cannot be generated."""


class ModelDiagnosticsService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings)
        self.model_registry_service = ModelRegistryService(settings)

    def threshold_analysis(
        self,
        model_id: str,
        request: ThresholdAnalysisRequest,
    ) -> ThresholdAnalysisResponse:
        model = self.model_registry_service.get_model(model_id)

        if model.task_type != "classification":
            raise ModelDiagnosticsError("Threshold analysis only supports classification models.")

        pipeline = self._load_pipeline(model.model_path)
        estimator = pipeline.named_steps.get("model")

        if (
            estimator is None
            or not hasattr(estimator, "classes_")
            or not hasattr(
                pipeline,
                "predict_proba",
            )
        ):
            raise ModelDiagnosticsError("Model does not expose class probabilities.")

        dataframe, version_id = self._load_labeled_dataframe(model, request.dataset_version_id)
        x_data = dataframe[model.feature_columns]
        y_true = dataframe[model.target_column].astype(str)
        classes = [str(label) for label in estimator.classes_.tolist()]
        positive_class = request.positive_class or classes[-1]

        if positive_class not in classes:
            raise ModelDiagnosticsError(f"Positive class not found: {positive_class}")

        positive_index = classes.index(positive_class)
        probabilities = pipeline.predict_proba(x_data)[:, positive_index]
        actual_positive = y_true == positive_class
        points = []

        for threshold in request.thresholds:
            threshold = round(float(threshold), 4)
            predicted_positive = probabilities >= threshold
            tp = int(np.sum(predicted_positive & actual_positive))
            fp = int(np.sum(predicted_positive & ~actual_positive))
            tn = int(np.sum(~predicted_positive & ~actual_positive))
            fn = int(np.sum(~predicted_positive & actual_positive))

            precision = precision_score(
                actual_positive,
                predicted_positive,
                zero_division=0,
            )
            recall = recall_score(
                actual_positive,
                predicted_positive,
                zero_division=0,
            )
            f1 = f1_score(
                actual_positive,
                predicted_positive,
                zero_division=0,
            )

            points.append(
                ThresholdAnalysisPoint(
                    threshold=threshold,
                    precision=self._safe_float(precision),
                    recall=self._safe_float(recall),
                    f1=self._safe_float(f1),
                    confusion_matrix={
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                    },
                )
            )

        return ThresholdAnalysisResponse(
            model_id=model.id,
            dataset_id=model.dataset_id,
            dataset_version_id=version_id,
            positive_class=positive_class,
            points=points,
        )

    def segment_metrics(
        self,
        model_id: str,
        request: SegmentMetricsRequest,
    ) -> SegmentMetricsResponse:
        model = self.model_registry_service.get_model(model_id)
        pipeline = self._load_pipeline(model.model_path)
        dataframe, version_id = self._load_labeled_dataframe(model, request.dataset_version_id)

        if request.segment_column not in dataframe.columns:
            raise ModelDiagnosticsError(f"Segment column not found: {request.segment_column}")

        segments = self._build_segments(
            dataframe=dataframe,
            segment_column=request.segment_column,
            max_bins=request.max_bins,
        )
        items = []

        for segment_label, segment_frame in segments:
            if segment_frame.empty:
                continue

            predictions = pipeline.predict(segment_frame[model.feature_columns])
            metrics = self._metrics(
                task_type=model.task_type,
                y_true=segment_frame[model.target_column],
                predictions=predictions,
            )
            items.append(
                SegmentMetricItem(
                    segment=segment_label,
                    row_count=int(segment_frame.shape[0]),
                    metrics=metrics,
                )
            )

        return SegmentMetricsResponse(
            model_id=model.id,
            dataset_id=model.dataset_id,
            dataset_version_id=version_id,
            segment_column=request.segment_column,
            segments=items,
        )

    def what_if(
        self,
        model_id: str,
        request: WhatIfRequest,
    ) -> WhatIfResponse:
        if not request.scenarios:
            raise ModelDiagnosticsError("At least one scenario is required.")

        model = self.model_registry_service.get_model(model_id)
        pipeline = self._load_pipeline(model.model_path)
        records = []
        names = []

        for scenario in request.scenarios:
            record = dict(request.base_record)
            record.update(scenario.changes)
            records.append(record)
            names.append(scenario.name)

        dataframe = pd.DataFrame(records).reindex(columns=model.feature_columns)

        try:
            predictions = pipeline.predict(dataframe)
        except Exception as exc:
            raise ModelPredictionError(f"What-if prediction failed: {exc}") from exc

        probabilities = self._probabilities(model.task_type, pipeline, dataframe)
        results = []

        for index, prediction in enumerate(predictions.tolist()):
            results.append(
                WhatIfResult(
                    scenario=names[index],
                    record=records[index],
                    prediction=self._json_value(prediction),
                    probabilities=probabilities[index] if probabilities else None,
                )
            )

        return WhatIfResponse(
            model_id=model.id,
            dataset_id=model.dataset_id,
            results=results,
        )

    def migration_check(self, model_id: str) -> ModelMigrationCheckResponse:
        model = self.model_registry_service.get_model(model_id)
        checks = []
        warnings = []
        model_path = self.settings.project_root / model.model_path

        checks.append(
            {
                "name": "artifact_exists",
                "passed": model_path.exists(),
                "detail": str(model_path),
            }
        )

        pipeline = None

        if model_path.exists():
            try:
                pipeline = joblib.load(model_path)
                checks.append(
                    {
                        "name": "artifact_loads",
                        "passed": isinstance(pipeline, Pipeline),
                        "detail": type(pipeline).__name__,
                    }
                )
            except Exception as exc:
                checks.append(
                    {
                        "name": "artifact_loads",
                        "passed": False,
                        "detail": str(exc),
                    }
                )

        recorded_sklearn = model.feature_schema.get("sklearn_version")

        if recorded_sklearn and recorded_sklearn != sklearn.__version__:
            warnings.append(
                "Model was trained with sklearn "
                f"{recorded_sklearn}; current is {sklearn.__version__}."
            )

        checks.append(
            {
                "name": "runtime_versions",
                "passed": True,
                "detail": {
                    "python": platform.python_version(),
                    "sklearn": sklearn.__version__,
                    "recorded_sklearn": recorded_sklearn,
                },
            }
        )

        if pipeline is not None:
            sample = pd.DataFrame([{column: np.nan for column in model.feature_columns}])

            try:
                pipeline.predict(sample)
                checks.append(
                    {
                        "name": "sample_prediction",
                        "passed": True,
                        "detail": "Prediction smoke test succeeded.",
                    }
                )
            except Exception as exc:
                checks.append(
                    {
                        "name": "sample_prediction",
                        "passed": False,
                        "detail": str(exc),
                    }
                )

        compatible = all(bool(check["passed"]) for check in checks)

        return ModelMigrationCheckResponse(
            model_id=model.id,
            compatible=compatible,
            checks=checks,
            warnings=warnings,
        )

    def _load_pipeline(self, relative_path: str) -> Pipeline:
        model_path = self.settings.project_root / relative_path

        if not model_path.exists():
            raise ModelDiagnosticsError("Saved model file is missing.")

        pipeline = joblib.load(model_path)

        if not isinstance(pipeline, Pipeline):
            raise ModelDiagnosticsError("Saved model is not a scikit-learn Pipeline.")

        return pipeline

    def _load_labeled_dataframe(
        self,
        model,
        version_id: str | None,
    ) -> tuple[pd.DataFrame, str]:
        resolved_version_id = version_id or model.dataset_version_id
        version = (
            self.dataset_service.get_dataset_version(model.dataset_id, resolved_version_id)
            if resolved_version_id
            else self.dataset_service.list_dataset_versions(model.dataset_id).versions[-1]
        )
        dataframe = self.dataset_service.load_dataset_dataframe(
            model.dataset_id,
            version_id=version.version_id,
        )

        if model.target_column not in dataframe.columns:
            raise ModelDiagnosticsError("Dataset version does not contain the model target column.")

        missing_features = [column for column in model.feature_columns if column not in dataframe]

        if missing_features:
            raise ModelDiagnosticsError(
                "Dataset version is missing model feature columns: " + ", ".join(missing_features)
            )

        dataframe = dataframe[model.feature_columns + [model.target_column]].dropna(
            subset=[model.target_column]
        )

        if dataframe.empty:
            raise ModelDiagnosticsError("No labeled rows are available for diagnostics.")

        return dataframe, version.version_id

    def _build_segments(
        self,
        dataframe: pd.DataFrame,
        segment_column: str,
        max_bins: int,
    ) -> list[tuple[str, pd.DataFrame]]:
        series = dataframe[segment_column]

        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > max_bins:
            binned = pd.qcut(
                series,
                q=min(max_bins, series.nunique(dropna=True)),
                duplicates="drop",
            )
            return [
                (str(label), dataframe.loc[binned == label])
                for label in binned.dropna().unique().tolist()
            ]

        return [
            (str(label), dataframe.loc[series.astype(str) == str(label)])
            for label in series.astype(str).value_counts().head(max_bins).index.tolist()
        ]

    def _metrics(
        self,
        task_type: str,
        y_true: pd.Series,
        predictions: np.ndarray,
    ) -> dict[str, float | None]:
        if task_type == "classification":
            return {
                "accuracy": self._safe_float(accuracy_score(y_true, predictions)),
                "precision_macro": self._safe_float(
                    precision_score(y_true, predictions, average="macro", zero_division=0)
                ),
                "recall_macro": self._safe_float(
                    recall_score(y_true, predictions, average="macro", zero_division=0)
                ),
                "f1_macro": self._safe_float(
                    f1_score(y_true, predictions, average="macro", zero_division=0)
                ),
            }

        mse = mean_squared_error(y_true, predictions)

        return {
            "mae": self._safe_float(mean_absolute_error(y_true, predictions)),
            "rmse": self._safe_float(np.sqrt(mse)),
            "r2": self._safe_float(r2_score(y_true, predictions)),
        }

    def _probabilities(
        self,
        task_type: str,
        pipeline: Pipeline,
        dataframe: pd.DataFrame,
    ) -> list[dict[str, float]] | None:
        if task_type != "classification" or not hasattr(pipeline, "predict_proba"):
            return None

        estimator = pipeline.named_steps.get("model")

        if estimator is None or not hasattr(estimator, "classes_"):
            return None

        probability_matrix = pipeline.predict_proba(dataframe)
        labels = [str(label) for label in estimator.classes_.tolist()]

        return [
            {labels[index]: round(float(probability), 4) for index, probability in enumerate(row)}
            for row in probability_matrix
        ]

    def _safe_float(self, value: Any) -> float | None:
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            return None

        if np.isnan(float_value) or np.isinf(float_value):
            return None

        return round(float_value, 4)

    def _json_value(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value
