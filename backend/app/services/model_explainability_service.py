import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from backend.app.core.config import Settings
from backend.app.schemas.explainability_schema import (
    CalibrationPoint,
    ClassificationCurve,
    ClassificationDiagnostics,
    ExplainabilityCurvePoint,
    ExplainabilityHoldoutSummary,
    ExplainabilityImportanceItem,
    ModelExplainabilityRequest,
    ModelExplainabilityResponse,
    RegressionDiagnosticPoint,
    RegressionDiagnostics,
    ShapSummary,
)
from backend.app.schemas.ml_schema import MLModelResult
from backend.app.services.dataset_service import DatasetService
from backend.app.services.model_registry_service import (
    ModelRegistryService,
)
from backend.app.services.shap_explainability_service import (
    ShapExplainabilityService,
)


class ModelExplainabilityError(Exception):
    """Raised when model explainability cannot be generated."""


class ModelExplainabilityService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings)
        self.model_registry_service = ModelRegistryService(settings)
        self.shap_service = ShapExplainabilityService(settings)

    def analyze(
        self,
        model_id: str,
        request: ModelExplainabilityRequest,
    ) -> ModelExplainabilityResponse:
        model = self.model_registry_service.get_model(model_id)

        cache_path = self._cache_path(
            model=model,
            request=request,
        )

        if cache_path.exists() and not request.force_recompute:
            cached = ModelExplainabilityResponse.model_validate_json(
                cache_path.read_text(encoding="utf-8")
            )

            return cached.model_copy(
                update={
                    "cache_hit": True,
                }
            )

        pipeline = self._load_pipeline(model)
        artifacts = self._load_artifacts(model)

        (
            x_holdout,
            y_holdout,
            holdout_source,
            holdout_warnings,
        ) = self._prepare_holdout(
            model=model,
            artifacts=artifacts,
            request=request,
        )

        predictions = pipeline.predict(x_holdout)

        warnings = list(holdout_warnings)

        classification = None
        regression = None

        if model.task_type == "classification":
            (
                classification,
                diagnostic_warnings,
                error_samples,
            ) = self._classification_diagnostics(
                pipeline=pipeline,
                x_holdout=x_holdout,
                y_holdout=y_holdout,
                predictions=predictions,
                positive_class=request.positive_class,
            )
        else:
            (
                regression,
                diagnostic_warnings,
                error_samples,
            ) = self._regression_diagnostics(
                x_holdout=x_holdout,
                y_holdout=y_holdout,
                predictions=predictions,
            )

        warnings.extend(diagnostic_warnings)

        permutation_items: list[ExplainabilityImportanceItem] = []
        permutation_scoring: str | None = None

        if request.include_permutation:
            (
                permutation_items,
                permutation_scoring,
                permutation_warning,
            ) = self._permutation_importance(
                model=model,
                pipeline=pipeline,
                x_holdout=x_holdout,
                y_holdout=y_holdout,
                repeats=request.permutation_repeats,
                random_state=request.random_state,
            )

            if permutation_warning:
                warnings.append(permutation_warning)

        if request.include_shap:
            shap_summary = self.shap_service.explain(
                pipeline=pipeline,
                x_holdout=x_holdout,
                local_row_position=(request.local_row_position),
                sample_size=request.sample_size,
                background_size=(request.background_size),
                random_state=request.random_state,
                positive_class=request.positive_class,
            )
        else:
            shap_summary = ShapSummary(
                available=False,
                warning="SHAP was disabled by request.",
            )

        if shap_summary.warning:
            warnings.append(shap_summary.warning)

        sampled_row_count = min(
            len(x_holdout),
            request.sample_size,
            self.settings.explainability_max_sample_size,
        )

        response = ModelExplainabilityResponse(
            model_id=model.id,
            dataset_id=model.dataset_id,
            model_name=model.model_name,
            task_type=model.task_type,
            target_column=model.target_column,
            generated_at=datetime.now(UTC),
            cache_hit=False,
            holdout=ExplainabilityHoldoutSummary(
                source=holdout_source,
                row_count=len(x_holdout),
                sampled_row_count=sampled_row_count,
                feature_count=len(model.feature_columns),
                target_non_null_count=int(y_holdout.notna().sum()),
            ),
            classification=classification,
            regression=regression,
            permutation_scoring=permutation_scoring,
            permutation_importance=(permutation_items),
            shap=shap_summary,
            error_samples=error_samples,
            warnings=list(dict.fromkeys(warnings)),
        )

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cache_path.write_text(
            response.model_dump_json(
                indent=2,
            ),
            encoding="utf-8",
        )

        return response

    def _prepare_holdout(
        self,
        model: MLModelResult,
        artifacts: dict[str, Any],
        request: ModelExplainabilityRequest,
    ) -> tuple[
        pd.DataFrame,
        pd.Series,
        str,
        list[str],
    ]:
        dataframe = self.dataset_service.load_dataset_dataframe(model.dataset_id)

        required_columns = model.feature_columns + [model.target_column]

        missing_columns = [column for column in required_columns if column not in dataframe.columns]

        if missing_columns:
            raise ModelExplainabilityError(
                "Dataset no longer contains required columns: " + ", ".join(missing_columns)
            )

        working_dataframe = dataframe[required_columns].copy()

        if model.task_type == "regression":
            working_dataframe[model.target_column] = pd.to_numeric(
                working_dataframe[model.target_column]
                .astype(str)
                .str.replace(
                    ",",
                    "",
                    regex=False,
                ),
                errors="coerce",
            )

        working_dataframe = working_dataframe.dropna(subset=[model.target_column])

        split_context = artifacts.get(
            "split_context",
            {},
        )

        holdout_indices = split_context.get(
            "holdout_row_indices",
            [],
        )

        matched_indices = [index for index in holdout_indices if index in working_dataframe.index]

        warnings = []

        if len(matched_indices) >= 2:
            holdout_dataframe = working_dataframe.loc[matched_indices]
            source = "saved_split_context"
        else:
            warnings.append(
                "Model does not contain an exact R4 holdout context. "
                "A deterministic fallback split was created."
            )

            x_data = working_dataframe[model.feature_columns]
            y_data = working_dataframe[model.target_column]

            test_size = split_context.get(
                "test_size",
                0.2,
            )

            random_state = split_context.get(
                "random_state",
                request.random_state,
            )

            stratify = None

            if model.task_type == "classification":
                class_counts = y_data.value_counts()

                if len(class_counts) > 1 and class_counts.min() >= 2:
                    stratify = y_data

            try:
                (
                    _,
                    x_holdout,
                    _,
                    y_holdout,
                ) = train_test_split(
                    x_data,
                    y_data,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify,
                )
            except ValueError:
                (
                    _,
                    x_holdout,
                    _,
                    y_holdout,
                ) = train_test_split(
                    x_data,
                    y_data,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=None,
                )

            return (
                x_holdout,
                y_holdout,
                "deterministic_fallback",
                warnings,
            )

        return (
            holdout_dataframe[model.feature_columns],
            holdout_dataframe[model.target_column],
            source,
            warnings,
        )

    def _classification_diagnostics(
        self,
        pipeline: Pipeline,
        x_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        predictions: np.ndarray,
        positive_class: str | None,
    ) -> tuple[
        ClassificationDiagnostics,
        list[str],
        list[dict[str, Any]],
    ]:
        estimator = pipeline.named_steps.get("model")

        classes = [
            str(value)
            for value in getattr(
                estimator,
                "classes_",
                sorted({str(value) for value in y_holdout.tolist()}),
            )
        ]

        y_string = y_holdout.astype(str)
        prediction_strings = np.asarray([str(value) for value in predictions.tolist()])

        matrix = confusion_matrix(
            y_string,
            prediction_strings,
            labels=classes,
        )

        (
            score_matrix,
            score_source,
            probability_matrix,
        ) = self._classification_scores(
            pipeline=pipeline,
            x_holdout=x_holdout,
            class_count=len(classes),
        )

        warnings = []
        curves = []
        calibration_points = []
        calibration_class = None

        if score_matrix is None:
            warnings.append(
                "The model does not expose probability or decision scores for ROC/PR curves."
            )
        else:
            for class_index, class_label in enumerate(classes):
                binary_target = (y_string == class_label).astype(int)

                positive_support = int(binary_target.sum())
                negative_support = int(len(binary_target) - positive_support)

                if positive_support == 0 or negative_support == 0:
                    continue

                class_scores = score_matrix[
                    :,
                    class_index,
                ]

                false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
                    binary_target,
                    class_scores,
                )

                precision, recall, pr_thresholds = precision_recall_curve(
                    binary_target,
                    class_scores,
                )

                curves.append(
                    ClassificationCurve(
                        class_label=class_label,
                        score_source=score_source,
                        positive_support=positive_support,
                        negative_support=negative_support,
                        roc_auc=self._safe_float(
                            auc(
                                false_positive_rate,
                                true_positive_rate,
                            )
                        ),
                        average_precision=self._safe_float(
                            average_precision_score(
                                binary_target,
                                class_scores,
                            )
                        ),
                        roc_points=self._curve_points(
                            x_values=false_positive_rate,
                            y_values=true_positive_rate,
                            thresholds=roc_thresholds,
                        ),
                        precision_recall_points=(
                            self._curve_points(
                                x_values=recall,
                                y_values=precision,
                                thresholds=pr_thresholds,
                            )
                        ),
                    )
                )

            if probability_matrix is not None and len(classes) == 2:
                requested_class = positive_class if positive_class in classes else classes[-1]

                class_index = classes.index(requested_class)

                binary_target = (y_string == requested_class).astype(int)

                (
                    fraction_of_positives,
                    mean_predicted_probability,
                ) = calibration_curve(
                    binary_target,
                    probability_matrix[
                        :,
                        class_index,
                    ],
                    n_bins=min(
                        10,
                        max(
                            3,
                            len(y_holdout) // 5,
                        ),
                    ),
                    strategy="quantile",
                )

                calibration_class = requested_class

                calibration_points = [
                    CalibrationPoint(
                        mean_predicted_probability=round(
                            float(probability),
                            6,
                        ),
                        fraction_of_positives=round(
                            float(fraction),
                            6,
                        ),
                    )
                    for probability, fraction in zip(
                        mean_predicted_probability,
                        fraction_of_positives,
                        strict=False,
                    )
                ]

        confidence_values = self._prediction_confidence(
            probability_matrix=probability_matrix,
            predictions=prediction_strings,
            classes=classes,
        )

        error_samples = []

        for row_position, (
            dataset_index,
            actual,
            predicted,
        ) in enumerate(
            zip(
                x_holdout.index,
                y_string.tolist(),
                prediction_strings.tolist(),
                strict=False,
            )
        ):
            if actual == predicted:
                continue

            error_samples.append(
                {
                    "row_position": row_position,
                    "dataset_index": str(dataset_index),
                    "actual": actual,
                    "predicted": predicted,
                    "confidence": (confidence_values[row_position] if confidence_values else None),
                    "record": self._record_snapshot(x_holdout.iloc[row_position]),
                }
            )

            if len(error_samples) >= 25:
                break

        return (
            ClassificationDiagnostics(
                labels=classes,
                confusion_matrix=matrix.astype(int).tolist(),
                curves=curves,
                calibration_class=calibration_class,
                calibration_points=(calibration_points),
            ),
            warnings,
            error_samples,
        )

    def _regression_diagnostics(
        self,
        x_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        predictions: np.ndarray,
    ) -> tuple[
        RegressionDiagnostics,
        list[str],
        list[dict[str, Any]],
    ]:
        points = []

        for row_position, (
            dataset_index,
            actual,
            predicted,
        ) in enumerate(
            zip(
                x_holdout.index,
                y_holdout.tolist(),
                predictions.tolist(),
                strict=False,
            )
        ):
            actual_value = self._safe_float(actual)
            predicted_value = self._safe_float(predicted)

            if actual_value is None or predicted_value is None:
                continue

            residual = actual_value - predicted_value

            points.append(
                RegressionDiagnosticPoint(
                    row_position=row_position,
                    dataset_index=str(dataset_index),
                    actual=actual_value,
                    predicted=predicted_value,
                    residual=round(
                        residual,
                        6,
                    ),
                    absolute_error=round(
                        abs(residual),
                        6,
                    ),
                )
            )

        sampled_points = self._sample_points(
            points,
            maximum=250,
        )

        residual_values = np.asarray(
            [point.residual for point in points],
            dtype=float,
        )

        sorted_errors = sorted(
            points,
            key=lambda point: point.absolute_error,
            reverse=True,
        )

        error_samples = [
            {
                "row_position": point.row_position,
                "dataset_index": point.dataset_index,
                "actual": point.actual,
                "predicted": point.predicted,
                "residual": point.residual,
                "absolute_error": (point.absolute_error),
                "record": self._record_snapshot(x_holdout.iloc[point.row_position]),
            }
            for point in sorted_errors[:25]
        ]

        return (
            RegressionDiagnostics(
                points=sampled_points,
                residual_mean=self._safe_float(
                    np.mean(residual_values) if len(residual_values) else None
                ),
                residual_std=self._safe_float(
                    np.std(residual_values) if len(residual_values) else None
                ),
                residual_median=self._safe_float(
                    np.median(residual_values) if len(residual_values) else None
                ),
                maximum_absolute_error=(
                    self._safe_float(
                        max(
                            (point.absolute_error for point in points),
                            default=0,
                        )
                    )
                ),
            ),
            [],
            error_samples,
        )

    def _permutation_importance(
        self,
        model: MLModelResult,
        pipeline: Pipeline,
        x_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        repeats: int,
        random_state: int,
    ) -> tuple[
        list[ExplainabilityImportanceItem],
        str,
        str | None,
    ]:
        scoring = (
            "f1_macro" if model.task_type == "classification" else "neg_root_mean_squared_error"
        )

        actual_repeats = min(
            repeats,
            self.settings.explainability_max_permutation_repeats,
        )

        try:
            result = permutation_importance(
                estimator=pipeline,
                X=x_holdout,
                y=y_holdout,
                scoring=scoring,
                n_repeats=actual_repeats,
                random_state=random_state,
                n_jobs=1,
            )
        except Exception as exc:
            return (
                [],
                scoring,
                f"Permutation importance failed: {exc}",
            )

        items = [
            ExplainabilityImportanceItem(
                feature=feature,
                source_feature=feature,
                importance_mean=round(
                    float(result.importances_mean[index]),
                    6,
                ),
                importance_std=round(
                    float(result.importances_std[index]),
                    6,
                ),
            )
            for index, feature in enumerate(x_holdout.columns)
        ]

        items.sort(
            key=lambda item: item.importance_mean,
            reverse=True,
        )

        warning = "Permutation importance can distribute importance across correlated features."

        return items, scoring, warning

    def _classification_scores(
        self,
        pipeline: Pipeline,
        x_holdout: pd.DataFrame,
        class_count: int,
    ) -> tuple[
        np.ndarray | None,
        str,
        np.ndarray | None,
    ]:
        if hasattr(pipeline, "predict_proba"):
            try:
                probabilities = np.asarray(
                    pipeline.predict_proba(x_holdout),
                    dtype=float,
                )

                if probabilities.ndim == 1:
                    probabilities = np.column_stack(
                        [
                            1 - probabilities,
                            probabilities,
                        ]
                    )

                return (
                    probabilities,
                    "predict_proba",
                    probabilities,
                )
            except Exception:
                pass

        if hasattr(
            pipeline,
            "decision_function",
        ):
            try:
                decision_values = np.asarray(
                    pipeline.decision_function(x_holdout),
                    dtype=float,
                )

                if decision_values.ndim == 1:
                    decision_values = np.column_stack(
                        [
                            -decision_values,
                            decision_values,
                        ]
                    )

                if decision_values.shape[1] != class_count:
                    return None, "", None

                return (
                    decision_values,
                    "decision_function",
                    None,
                )
            except Exception:
                pass

        return None, "", None

    def _prediction_confidence(
        self,
        probability_matrix: np.ndarray | None,
        predictions: np.ndarray,
        classes: list[str],
    ) -> list[float] | None:
        if probability_matrix is None:
            return None

        class_index = {label: index for index, label in enumerate(classes)}

        confidence_values = []

        for row_position, prediction in enumerate(predictions):
            index = class_index.get(str(prediction))

            if index is None:
                confidence_values.append(0.0)
                continue

            confidence_values.append(
                round(
                    float(
                        probability_matrix[
                            row_position,
                            index,
                        ]
                    ),
                    6,
                )
            )

        return confidence_values

    def _curve_points(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        thresholds: np.ndarray,
    ) -> list[ExplainabilityCurvePoint]:
        total_points = len(x_values)

        if total_points > 250:
            positions = np.linspace(
                0,
                total_points - 1,
                250,
                dtype=int,
            )
        else:
            positions = np.arange(total_points)

        points = []

        for position in positions:
            threshold = None

            if position < len(thresholds):
                threshold = self._safe_float(thresholds[position])

            points.append(
                ExplainabilityCurvePoint(
                    x=round(
                        float(x_values[position]),
                        6,
                    ),
                    y=round(
                        float(y_values[position]),
                        6,
                    ),
                    threshold=threshold,
                )
            )

        return points

    def _load_pipeline(
        self,
        model: MLModelResult,
    ) -> Pipeline:
        model_path = self.settings.project_root / model.model_path

        if not model_path.exists():
            raise ModelExplainabilityError("Saved model file is missing.")

        try:
            pipeline = joblib.load(model_path)
        except Exception as exc:
            raise ModelExplainabilityError(f"Failed to load model: {exc}") from exc

        if not isinstance(pipeline, Pipeline):
            raise ModelExplainabilityError("Saved model is not a scikit-learn Pipeline.")

        return pipeline

    def _load_artifacts(
        self,
        model: MLModelResult,
    ) -> dict[str, Any]:
        if not model.evaluation_artifacts_path:
            return {}

        path = self.settings.project_root / model.evaluation_artifacts_path

        if not path.exists():
            return {}

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _cache_path(
        self,
        model: MLModelResult,
        request: ModelExplainabilityRequest,
    ) -> Path:
        signature_payload = {
            **request.model_dump(
                exclude={
                    "force_recompute",
                }
            ),
            "model_id": model.id,
            "model_created_at": (model.created_at.isoformat()),
        }

        signature = hashlib.sha256(
            json.dumps(
                signature_payload,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]

        return self.settings.models_dir / "explainability" / f"{model.id}_{signature}.json"

    def _sample_points(
        self,
        points: list[RegressionDiagnosticPoint],
        maximum: int,
    ) -> list[RegressionDiagnosticPoint]:
        if len(points) <= maximum:
            return points

        positions = np.linspace(
            0,
            len(points) - 1,
            maximum,
            dtype=int,
        )

        return [points[position] for position in positions]

    def _record_snapshot(
        self,
        row: pd.Series,
    ) -> dict[str, Any]:
        return {str(key): self._json_value(value) for key, value in row.to_dict().items()}

    def _json_value(
        self,
        value: Any,
    ) -> Any:
        if pd.isna(value):
            return None

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        return value

    def _safe_float(
        self,
        value: object,
    ) -> float | None:
        if value is None:
            return None

        try:
            float_value = float(cast(Any, value))
        except (TypeError, ValueError):
            return None

        if not math.isfinite(float_value):
            return None

        return round(float_value, 6)
