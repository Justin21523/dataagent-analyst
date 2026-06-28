from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from backend.app.core.config import Settings
from backend.app.repositories.metadata_repository import (
    MetadataRepositoryError,
    create_metadata_repository,
)
from backend.app.schemas.drift_schema import (
    DriftMetric,
    DriftReportRequest,
    DriftReportResponse,
    DriftReportSummary,
)
from backend.app.services.dataset_service import DatasetService
from backend.app.services.model_registry_service import ModelRegistryService


class DriftServiceError(Exception):
    """Raised when drift analysis fails."""


class DriftReportNotFoundError(DriftServiceError):
    """Raised when a drift report cannot be found."""


class DriftService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metadata_repository = create_metadata_repository(settings)
        self.dataset_service = DatasetService(settings)
        self.model_registry_service = ModelRegistryService(settings)

    def create_report(self, request: DriftReportRequest) -> DriftReportResponse:
        reference = self.dataset_service.load_dataset_dataframe(
            request.dataset_id,
            version_id=request.reference_version_id,
        )
        current = self.dataset_service.load_dataset_dataframe(
            request.dataset_id,
            version_id=request.current_version_id,
        )

        schema_drift = self._schema_drift(reference, current)
        target_column = request.target_column
        model = None

        if request.model_id:
            model = self.model_registry_service.get_model(request.model_id)
            target_column = target_column or model.target_column

        feature_columns = [
            column
            for column in sorted(set(reference.columns) & set(current.columns))
            if column != target_column
        ]
        feature_drift = [
            self._column_drift(column, reference[column], current[column])
            for column in feature_columns
        ]

        target_drift = None

        if (
            target_column
            and target_column in reference.columns
            and target_column in current.columns
        ):
            target_drift = self._column_drift(
                target_column,
                reference[target_column],
                current[target_column],
            )

        prediction_drift = None
        performance_drift = None
        warnings = []

        if model is not None:
            try:
                pipeline = self._load_pipeline(model.model_path)
                x_reference = reference.reindex(columns=model.feature_columns)
                x_current = current.reindex(columns=model.feature_columns)
                reference_predictions = pd.Series(pipeline.predict(x_reference), name="prediction")
                current_predictions = pd.Series(pipeline.predict(x_current), name="prediction")
                prediction_drift = self._column_drift(
                    "prediction",
                    reference_predictions,
                    current_predictions,
                )

                if (
                    model.target_column in reference.columns
                    and model.target_column in current.columns
                ):
                    performance_drift = self._performance_drift(
                        task_type=model.task_type,
                        reference_y=reference[model.target_column],
                        reference_predictions=reference_predictions,
                        current_y=current[model.target_column],
                        current_predictions=current_predictions,
                    )
            except Exception as exc:
                warnings.append(f"Model drift section failed: {exc}")

        status = self._overall_status(
            schema_drift=schema_drift,
            metrics=[*feature_drift, target_drift, prediction_drift],
            performance_drift=performance_drift,
        )
        recommendations = self._recommendations(status, schema_drift, feature_drift)
        retraining_recommendation = self._retraining_recommendation(
            schema_drift=schema_drift,
            feature_drift=feature_drift,
            prediction_drift=prediction_drift,
            performance_drift=performance_drift,
            model_id=request.model_id,
        )
        report = DriftReportResponse(
            report_id=uuid4().hex,
            dataset_id=request.dataset_id,
            reference_version_id=request.reference_version_id,
            current_version_id=request.current_version_id,
            model_id=request.model_id,
            target_column=target_column,
            status=status,
            schema_drift=schema_drift,
            feature_drift=feature_drift,
            target_drift=target_drift,
            prediction_drift=prediction_drift,
            performance_drift=performance_drift,
            retraining_recommendation=retraining_recommendation,
            recommendations=recommendations,
            warnings=warnings,
            created_at=datetime.now(UTC),
        )
        self._add_report(report)

        return report

    def get_report(self, report_id: str) -> DriftReportResponse:
        registry = self._load_registry()

        for record in registry["reports"]:
            if record["report_id"] == report_id:
                return DriftReportResponse.model_validate(record)

        raise DriftReportNotFoundError(f"Drift report not found: {report_id}")

    def list_reports(self, dataset_id: str) -> list[DriftReportSummary]:
        registry = self._load_registry()
        summaries = [
            DriftReportSummary.model_validate(record)
            for record in registry["reports"]
            if record["dataset_id"] == dataset_id
        ]

        return sorted(summaries, key=lambda report: report.created_at, reverse=True)

    def _schema_drift(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
        reference_columns = set(reference.columns)
        current_columns = set(current.columns)
        common_columns = sorted(reference_columns & current_columns)
        type_changes = []

        for column in common_columns:
            reference_type = self._simple_dtype(reference[column])
            current_type = self._simple_dtype(current[column])

            if reference_type != current_type:
                type_changes.append(
                    {
                        "column": column,
                        "reference_type": reference_type,
                        "current_type": current_type,
                    }
                )

        return {
            "added_columns": sorted(current_columns - reference_columns),
            "removed_columns": sorted(reference_columns - current_columns),
            "type_changes": type_changes,
            "status": "drift"
            if (current_columns - reference_columns)
            or (reference_columns - current_columns)
            or type_changes
            else "stable",
        }

    def _column_drift(
        self,
        column: str,
        reference: pd.Series,
        current: pd.Series,
    ) -> DriftMetric:
        if self._is_boolean_like(reference, current):
            return self._categorical_column_drift(column, reference, current)

        reference_numeric = pd.to_numeric(reference, errors="coerce")
        current_numeric = pd.to_numeric(current, errors="coerce")

        if reference_numeric.notna().mean() >= 0.8 and current_numeric.notna().mean() >= 0.8:
            psi = self._numeric_psi(reference_numeric.dropna(), current_numeric.dropna())
            ks = ks_2samp(reference_numeric.dropna(), current_numeric.dropna()).statistic
            return DriftMetric(
                column=column,
                drift_type="numeric",
                status=self._psi_status(psi),
                psi=psi,
                ks_statistic=self._safe_float(ks),
                details={
                    "reference_mean": self._safe_float(reference_numeric.mean()),
                    "current_mean": self._safe_float(current_numeric.mean()),
                },
            )

        return self._categorical_column_drift(column, reference, current)

    def _numeric_psi(self, reference: pd.Series, current: pd.Series) -> float | None:
        if reference.empty or current.empty:
            return None

        quantiles = np.linspace(0, 1, 11)
        bins = np.unique(np.quantile(reference, quantiles))

        if len(bins) < 3:
            return None

        bins[0] = -np.inf
        bins[-1] = np.inf
        reference_counts, _ = np.histogram(reference, bins=bins)
        current_counts, _ = np.histogram(current, bins=bins)

        return self._psi_from_counts(reference_counts, current_counts)

    def _categorical_distances(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> tuple[float | None, float | None]:
        reference_counts = self._categorical_labels(reference).value_counts(dropna=False)
        current_counts = self._categorical_labels(current).value_counts(dropna=False)
        labels = sorted(set(reference_counts.index) | set(current_counts.index))
        reference_array = np.array([reference_counts.get(label, 0) for label in labels])
        current_array = np.array([current_counts.get(label, 0) for label in labels])
        psi = self._psi_from_counts(reference_array, current_array)
        reference_prob = self._smooth_prob(reference_array)
        current_prob = self._smooth_prob(current_array)
        js = float(jensenshannon(reference_prob, current_prob))

        return psi, round(js, 4)

    def _categorical_labels(self, series: pd.Series) -> pd.Series:
        return series.map(lambda value: "<missing>" if pd.isna(value) else str(value))

    def _categorical_column_drift(
        self,
        column: str,
        reference: pd.Series,
        current: pd.Series,
    ) -> DriftMetric:
        psi, js = self._categorical_distances(reference, current)

        return DriftMetric(
            column=column,
            drift_type="categorical",
            status=self._psi_status(psi),
            psi=psi,
            js_distance=js,
            details={
                "reference_unique": int(reference.nunique(dropna=True)),
                "current_unique": int(current.nunique(dropna=True)),
            },
        )

    def _is_boolean_like(self, reference: pd.Series, current: pd.Series) -> bool:
        if pd.api.types.is_bool_dtype(reference) or pd.api.types.is_bool_dtype(current):
            return True

        combined = pd.concat([reference.dropna(), current.dropna()])
        if combined.empty:
            return False

        normalized = combined.astype(str).str.strip().str.lower()
        return normalized.isin({"true", "false", "0", "1"}).all() and normalized.nunique() <= 2

    def _psi_from_counts(
        self,
        reference_counts: np.ndarray,
        current_counts: np.ndarray,
    ) -> float | None:
        if reference_counts.sum() == 0 or current_counts.sum() == 0:
            return None

        reference_percents = self._smooth_prob(reference_counts)
        current_percents = self._smooth_prob(current_counts)
        psi = np.sum(
            (current_percents - reference_percents) * np.log(current_percents / reference_percents)
        )

        return round(float(psi), 4)

    def _performance_drift(
        self,
        task_type: str,
        reference_y: pd.Series,
        reference_predictions: pd.Series,
        current_y: pd.Series,
        current_predictions: pd.Series,
    ) -> dict[str, Any]:
        reference_metrics = self._metrics(task_type, reference_y, reference_predictions)
        current_metrics = self._metrics(task_type, current_y, current_predictions)

        return {
            "reference_metrics": reference_metrics,
            "current_metrics": current_metrics,
            "metric_delta": self._metric_delta(reference_metrics, current_metrics),
        }

    def _metric_delta(
        self,
        reference_metrics: dict[str, float | None],
        current_metrics: dict[str, float | None],
    ) -> dict[str, float | None]:
        deltas: dict[str, float | None] = {}

        for metric in set(reference_metrics) | set(current_metrics):
            reference_value = reference_metrics.get(metric)
            current_value = current_metrics.get(metric)

            if reference_value is None or current_value is None:
                deltas[metric] = None
            else:
                deltas[metric] = round(current_value - reference_value, 4)

        return deltas

    def _metrics(
        self,
        task_type: str,
        y_true: pd.Series,
        predictions: pd.Series,
    ) -> dict[str, float | None]:
        if task_type == "classification":
            return {
                "accuracy": self._safe_float(accuracy_score(y_true, predictions)),
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

    def _overall_status(
        self,
        schema_drift: dict[str, Any],
        metrics: list[DriftMetric | None],
        performance_drift: dict[str, Any] | None,
    ) -> str:
        statuses = [metric.status for metric in metrics if metric is not None]

        if schema_drift["status"] == "drift" or "drift" in statuses:
            return "drift"
        if "warning" in statuses:
            return "warning"

        if performance_drift:
            deltas = [
                abs(value)
                for value in performance_drift.get("metric_delta", {}).values()
                if isinstance(value, int | float)
            ]

            if any(delta >= 0.1 for delta in deltas):
                return "warning"

        return "stable"

    def _recommendations(
        self,
        status: str,
        schema_drift: dict[str, Any],
        feature_drift: list[DriftMetric],
    ) -> list[str]:
        recommendations = []

        if schema_drift["status"] == "drift":
            recommendations.append("Review schema changes before reusing existing models.")

        drifted_features = [metric.column for metric in feature_drift if metric.status == "drift"]

        if drifted_features:
            recommendations.append(
                "Investigate drifted features and consider retraining: "
                + ", ".join(drifted_features[:8])
            )

        if status == "stable":
            recommendations.append("No immediate retraining action is required.")

        return recommendations

    def _retraining_recommendation(
        self,
        schema_drift: dict[str, Any],
        feature_drift: list[DriftMetric],
        prediction_drift: DriftMetric | None,
        performance_drift: dict[str, Any] | None,
        model_id: str | None,
    ) -> dict[str, Any]:
        score = 0
        reasons = []

        if schema_drift["status"] == "drift":
            score += 30
            reasons.append("Schema changed between reference and current versions.")

        drifted_features = [metric for metric in feature_drift if metric.status == "drift"]
        warning_features = [metric for metric in feature_drift if metric.status == "warning"]

        if drifted_features:
            score += min(35, 10 + 5 * len(drifted_features))
            reasons.append(f"{len(drifted_features)} feature(s) show material drift.")

        if warning_features:
            score += min(15, 3 * len(warning_features))
            reasons.append(f"{len(warning_features)} feature(s) show warning-level drift.")

        if prediction_drift and prediction_drift.status == "drift":
            score += 20
            reasons.append("Model prediction distribution changed materially.")
        elif prediction_drift and prediction_drift.status == "warning":
            score += 10
            reasons.append("Model prediction distribution changed moderately.")

        if performance_drift:
            deltas = [
                abs(float(value))
                for value in performance_drift.get("metric_delta", {}).values()
                if isinstance(value, int | float)
            ]

            if any(delta >= 0.1 for delta in deltas):
                score += 25
                reasons.append("Observed model performance changed by at least 0.10.")

        score = min(score, 100)

        if not model_id:
            action = "monitor"
            recommended = False
            reasons.append("No model was supplied, so retraining cannot be automated.")
        elif score >= 50:
            action = "retrain_challenger"
            recommended = True
        elif score >= 25:
            action = "review"
            recommended = False
        else:
            action = "monitor"
            recommended = False

        if not reasons:
            reasons.append("No material drift signal was detected.")

        return {
            "recommended": recommended,
            "score": score,
            "action": action,
            "reasons": reasons,
        }

    def _load_pipeline(self, relative_path: str) -> Pipeline:
        model_path = self.settings.project_root / relative_path

        if not model_path.exists():
            raise DriftServiceError("Saved model file is missing.")

        pipeline = joblib.load(model_path)

        if not isinstance(pipeline, Pipeline):
            raise DriftServiceError("Saved model is not a scikit-learn Pipeline.")

        return pipeline

    def _simple_dtype(self, series: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        return "categorical"

    def _psi_status(self, psi: float | None) -> str:
        if psi is None:
            return "warning"
        if psi >= 0.25:
            return "drift"
        if psi >= 0.1:
            return "warning"
        return "stable"

    def _smooth_prob(self, counts: np.ndarray) -> np.ndarray:
        smoothed = counts.astype(float) + 1e-6
        return smoothed / smoothed.sum()

    def _safe_float(self, value: Any) -> float | None:
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            return None

        if np.isnan(float_value) or np.isinf(float_value):
            return None

        return round(float_value, 4)

    def _add_report(self, report: DriftReportResponse) -> None:
        registry = self._load_registry()
        registry["reports"].append(report.model_dump(mode="json"))
        self._save_registry(registry)

    def _load_registry(self) -> dict[str, list[dict[str, Any]]]:
        try:
            return self.metadata_repository.load_registry("drift_reports")
        except MetadataRepositoryError as exc:
            raise DriftServiceError("Drift report registry format is invalid.") from exc

    def _save_registry(self, registry: dict[str, list[dict[str, Any]]]) -> None:
        try:
            self.metadata_repository.save_registry("drift_reports", registry)
        except MetadataRepositoryError as exc:
            raise DriftServiceError("Drift report registry format is invalid.") from exc
