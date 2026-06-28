import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.app.core.config import Settings
from backend.app.schemas.dataset_schema import ColumnProfile
from backend.app.schemas.ml_schema import (
    FeatureImportanceItem,
    MLModelResult,
    MLTrainRequest,
    MLTrainResponse,
)
from backend.app.services.column_profiler_service import ColumnProfilerService
from backend.app.services.dataset_service import DatasetService
from backend.app.services.model_registry_service import ModelRegistryService


class MLTrainingError(Exception):
    """Raised when model training fails."""


class MLTrainingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings)
        self.column_profiler_service = ColumnProfilerService(settings)
        self.model_registry_service = ModelRegistryService(settings)

    def train_models(self, dataset_id: str, request: MLTrainRequest) -> MLTrainResponse:
        # Training service 負責完整 ML pipeline：資料準備、訓練、評估、儲存模型。
        dataset_version = (
            self.dataset_service.get_dataset_version(
                dataset_id,
                request.dataset_version_id,
            )
            if request.dataset_version_id
            else self.dataset_service.list_dataset_versions(dataset_id).versions[-1]
        )
        dataframe = self.dataset_service.load_dataset_dataframe(
            dataset_id,
            version_id=dataset_version.version_id,
        )
        profiles = self.column_profiler_service.profile_columns(dataset_id).columns

        target_profile = self._get_target_profile(profiles, request.target_column)
        task_type = self._detect_task_type(target_profile, dataframe, request.task_type)
        feature_columns = self._select_feature_columns(profiles, request)

        training_dataframe = dataframe[feature_columns + [request.target_column]].dropna(
            subset=[request.target_column]
        )

        if len(training_dataframe) < 8:
            raise MLTrainingError("At least 8 rows with non-missing target values are required.")

        x_data = training_dataframe[feature_columns]
        y_data = training_dataframe[request.target_column]

        if task_type == "classification" and y_data.nunique(dropna=True) < 2:
            raise MLTrainingError("Classification requires at least two target classes.")

        x_train, x_test, y_train, y_test = self._split_dataset(
            x_data=x_data,
            y_data=y_data,
            task_type=task_type,
            test_size=request.test_size,
            random_state=request.random_state,
        )

        preprocessor = self._build_preprocessor(feature_columns, profiles)
        model_factories = self._get_model_factories(task_type, request.selected_models)

        trained_models = []

        for model_name, estimator in model_factories.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )

            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)

            metrics = self._evaluate_model(task_type, y_test, predictions)
            feature_importance = self._extract_feature_importance(pipeline)
            evaluation_artifacts = self._build_evaluation_artifacts(
                task_type=task_type,
                y_true=y_test,
                predictions=predictions,
                feature_importance=feature_importance,
            )

            model_result = self._save_model_result(
                dataset_id=dataset_id,
                dataset_version_id=dataset_version.version_id,
                model_name=model_name,
                task_type=task_type,
                target_column=request.target_column,
                feature_columns=feature_columns,
                training_config=request.model_dump(mode="json"),
                metrics=metrics,
                feature_importance=feature_importance,
                evaluation_artifacts=evaluation_artifacts,
                pipeline=pipeline,
            )

            trained_models.append(model_result)

        if not trained_models:
            raise MLTrainingError("No models were trained.")

        best_model, best_metric_name, best_metric_value = self._select_best_model(
            task_type,
            trained_models,
        )

        return MLTrainResponse(
            dataset_id=dataset_id,
            task_type=task_type,
            target_column=request.target_column,
            feature_columns=feature_columns,
            model_count=len(trained_models),
            best_model_id=best_model.id,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            models=trained_models,
        )

    def _get_target_profile(
        self,
        profiles: list[ColumnProfile],
        target_column: str,
    ) -> ColumnProfile:
        for profile in profiles:
            if profile.name == target_column:
                return profile

        raise MLTrainingError(f"Target column not found: {target_column}")

    def _detect_task_type(
        self,
        target_profile: ColumnProfile,
        dataframe: pd.DataFrame,
        requested_task_type: str,
    ) -> str:
        if requested_task_type in {"classification", "regression"}:
            return requested_task_type

        if requested_task_type != "auto":
            raise MLTrainingError("task_type must be auto, classification, or regression.")

        target_series = dataframe[target_profile.name].dropna()
        unique_count = int(target_series.nunique(dropna=True))

        if target_profile.inferred_type == "numeric" and unique_count > max(
            10, len(target_series) // 10
        ):
            return "regression"

        return "classification"

    def _select_feature_columns(
        self,
        profiles: list[ColumnProfile],
        request: MLTrainRequest,
    ) -> list[str]:
        allowed_types = {"numeric", "categorical", "boolean"}
        requested_features = set(request.feature_columns or [])

        feature_columns = []

        for profile in profiles:
            if profile.name == request.target_column:
                continue

            if profile.semantic_role == "identifier":
                continue

            if profile.inferred_type not in allowed_types:
                continue

            if requested_features and profile.name not in requested_features:
                continue

            feature_columns.append(profile.name)

        if not feature_columns:
            raise MLTrainingError("No valid feature columns are available for training.")

        return feature_columns

    def _split_dataset(
        self,
        x_data: pd.DataFrame,
        y_data: pd.Series,
        task_type: str,
        test_size: float,
        random_state: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        stratify = None

        if task_type == "classification":
            class_counts = y_data.value_counts()

            if len(class_counts) > 1 and class_counts.min() >= 2:
                stratify = y_data

        return train_test_split(
            x_data,
            y_data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    def _build_preprocessor(
        self,
        feature_columns: list[str],
        profiles: list[ColumnProfile],
    ) -> ColumnTransformer:
        profile_map = {profile.name: profile for profile in profiles}

        numeric_features = [
            column for column in feature_columns if profile_map[column].inferred_type == "numeric"
        ]

        categorical_features = [
            column
            for column in feature_columns
            if profile_map[column].inferred_type in {"categorical", "boolean"}
        ]

        transformers = []

        if numeric_features:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", numeric_pipeline, numeric_features))

        if categorical_features:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("categorical", categorical_pipeline, categorical_features))

        if not transformers:
            raise MLTrainingError("No preprocessing transformers can be built.")

        return ColumnTransformer(transformers=transformers)

    def _get_model_factories(
        self,
        task_type: str,
        selected_models: list[str] | None,
    ) -> dict[str, Any]:
        if task_type == "classification":
            models = {
                "logistic_regression": LogisticRegression(max_iter=1000),
                "random_forest_classifier": RandomForestClassifier(
                    n_estimators=120,
                    random_state=42,
                ),
            }
        else:
            models = {
                "linear_regression": LinearRegression(),
                "random_forest_regressor": RandomForestRegressor(
                    n_estimators=120,
                    random_state=42,
                ),
            }

        if selected_models:
            selected = set(selected_models)
            models = {name: estimator for name, estimator in models.items() if name in selected}

        if not models:
            raise MLTrainingError("No supported models were selected.")

        return models

    def _evaluate_model(
        self,
        task_type: str,
        y_true: pd.Series,
        predictions: np.ndarray,
    ) -> dict[str, float | None]:
        if task_type == "classification":
            return {
                "accuracy": self._json_float(accuracy_score(y_true, predictions)),
                "precision_macro": self._json_float(
                    precision_score(y_true, predictions, average="macro", zero_division=0)
                ),
                "recall_macro": self._json_float(
                    recall_score(y_true, predictions, average="macro", zero_division=0)
                ),
                "f1_macro": self._json_float(
                    f1_score(y_true, predictions, average="macro", zero_division=0)
                ),
            }

        mse = mean_squared_error(y_true, predictions)

        return {
            "mae": self._json_float(mean_absolute_error(y_true, predictions)),
            "rmse": self._json_float(np.sqrt(mse)),
            "r2": self._json_float(r2_score(y_true, predictions)),
        }

    def _extract_feature_importance(
        self,
        pipeline: Pipeline,
    ) -> list[FeatureImportanceItem]:
        feature_names = self._get_feature_names(pipeline)
        estimator = pipeline.named_steps["model"]

        importances = None

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_

        if importances is None and hasattr(estimator, "coef_"):
            coefficients = np.abs(estimator.coef_)
            importances = coefficients.mean(axis=0) if coefficients.ndim == 2 else coefficients

        if importances is None:
            return []

        items = [
            FeatureImportanceItem(
                feature=feature_names[index] if index < len(feature_names) else f"feature_{index}",
                importance=self._json_float(value) or 0.0,
            )
            for index, value in enumerate(importances)
        ]

        items.sort(key=lambda item: item.importance, reverse=True)

        return items[:15]

    def _get_feature_names(self, pipeline: Pipeline) -> list[str]:
        preprocessor = pipeline.named_steps["preprocessor"]

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            return []

        return [
            str(feature_name).replace("numeric__", "").replace("categorical__", "")
            for feature_name in feature_names
        ]

    def _build_evaluation_artifacts(
        self,
        task_type: str,
        y_true: pd.Series,
        predictions: np.ndarray,
        feature_importance: list[FeatureImportanceItem],
    ) -> dict[str, Any]:
        artifacts: dict[str, Any] = {
            "feature_importance": [item.model_dump() for item in feature_importance],
        }

        if task_type == "classification":
            labels = sorted({str(value) for value in y_true.tolist() + predictions.tolist()})
            matrix = confusion_matrix(
                y_true.astype(str),
                pd.Series(predictions).astype(str),
                labels=labels,
            )
            artifacts["confusion_matrix"] = {
                "labels": labels,
                "matrix": matrix.astype(int).tolist(),
            }
            return artifacts

        artifacts["regression_residuals"] = [
            {
                "actual": self._json_float(actual),
                "predicted": self._json_float(predicted),
                "residual": self._json_float(float(actual) - float(predicted)),
            }
            for actual, predicted in zip(y_true.tolist(), predictions.tolist(), strict=False)
        ][:50]

        return artifacts

    def _save_model_result(
        self,
        dataset_id: str,
        dataset_version_id: str,
        model_name: str,
        task_type: str,
        target_column: str,
        feature_columns: list[str],
        training_config: dict[str, Any],
        metrics: dict[str, float | None],
        feature_importance: list[FeatureImportanceItem],
        evaluation_artifacts: dict[str, Any],
        pipeline: Pipeline,
    ) -> MLModelResult:
        model_id = uuid4().hex
        model_filename = f"{model_id}_{model_name}.joblib"
        model_path = self.settings.models_dir / model_filename

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)

        evaluation_artifacts_path = self._save_evaluation_artifacts(
            model_id=model_id,
            evaluation_artifacts=evaluation_artifacts,
        )

        record = {
            "id": model_id,
            "dataset_id": dataset_id,
            "dataset_version_id": dataset_version_id,
            "model_name": model_name,
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "feature_importance": [item.model_dump() for item in feature_importance],
            "model_path": str(model_path.relative_to(self.settings.project_root)),
            "evaluation_artifacts_path": str(
                evaluation_artifacts_path.relative_to(self.settings.project_root)
            ),
            "status": "trained",
            "lifecycle_status": "candidate",
            "training_config": training_config,
            "feature_schema": {
                "required_columns": feature_columns,
                "target_column": target_column,
                "task_type": task_type,
                "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
            },
            "preprocessing_recipe": {
                "pipeline": "baseline_ml_training",
                "preprocessor": "numeric_median_standard_scaler_categorical_one_hot",
            },
            "created_at": datetime.now(UTC).isoformat(),
        }

        return self.model_registry_service.add_model_record(record)

    def _save_evaluation_artifacts(
        self,
        model_id: str,
        evaluation_artifacts: dict[str, Any],
    ) -> Path:
        artifacts_path = self.settings.models_dir / f"{model_id}_evaluation.json"
        artifacts_path.parent.mkdir(parents=True, exist_ok=True)

        with artifacts_path.open("w", encoding="utf-8") as file:
            json.dump(evaluation_artifacts, file, ensure_ascii=False, indent=2)

        return artifacts_path

    def _select_best_model(
        self,
        task_type: str,
        models: list[MLModelResult],
    ) -> tuple[MLModelResult, str, float | None]:
        if task_type == "classification":
            metric_name = "f1_macro"
            best_model = max(
                models,
                key=lambda model: model.metrics.get(metric_name) or -1,
            )
            return best_model, metric_name, best_model.metrics.get(metric_name)

        metric_name = "rmse"
        best_model = min(
            models,
            key=lambda model: model.metrics.get(metric_name) or float("inf"),
        )
        return best_model, metric_name, best_model.metrics.get(metric_name)

    def _json_float(self, value: object) -> float | None:
        try:
            float_value = float(cast(Any, value))
        except (TypeError, ValueError):
            return None

        if np.isnan(float_value) or np.isinf(float_value):
            return None

        return round(float_value, 4)
