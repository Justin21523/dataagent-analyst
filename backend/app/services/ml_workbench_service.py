import json
import math
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal, cast
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, LinearSVR

from backend.app.core.config import Settings
from backend.app.schemas.analysis_schema import (
    AnalysisContextRequest,
)
from backend.app.schemas.ml_workbench_schema import (
    CrossValidationMetric,
    DetectedMLTaskType,
    MLModelOption,
    MLWorkbenchExperimentRequest,
    MLWorkbenchExperimentResponse,
    MLWorkbenchModelResult,
    MLWorkbenchPlanRequest,
    MLWorkbenchPlanResponse,
)
from backend.app.services.analysis_context_service import (
    AnalysisContextService,
)
from backend.app.services.dataset_service import DatasetService
from backend.app.services.ml_experiment_registry_service import (
    MLExperimentRegistryService,
)
from backend.app.services.ml_feature_pipeline_service import (
    MLFeaturePipelineBundle,
    MLFeaturePipelineError,
    MLFeaturePipelineService,
)
from backend.app.services.model_registry_service import (
    ModelRegistryService,
)


class MLWorkbenchError(Exception):
    """Raised when an ML Workbench operation fails."""


class MLWorkbenchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings)
        self.analysis_context_service = AnalysisContextService(settings)
        self.feature_pipeline_service = MLFeaturePipelineService()
        self.experiment_registry_service = MLExperimentRegistryService(settings)
        self.model_registry_service = ModelRegistryService(settings)

    def build_plan(
        self,
        dataset_id: str,
        request: MLWorkbenchPlanRequest,
    ) -> MLWorkbenchPlanResponse:
        dataset_version = (
            self.dataset_service.get_dataset_version(dataset_id, request.dataset_version_id)
            if request.dataset_version_id
            else self.dataset_service.list_dataset_versions(dataset_id).versions[-1]
        )
        dataframe = self.dataset_service.load_dataset_dataframe(
            dataset_id,
            version_id=dataset_version.version_id,
        )

        context = self.analysis_context_service.build_context(
            dataset_id=dataset_id,
            request=AnalysisContextRequest(
                target_column=self._context_target(request),
                user_goal="Plan an ML Workbench experiment.",
            ),
        )

        detected_task_type = self._detect_task_type(
            request=request,
            context=context,
        )

        target_column = self._resolve_target_column(
            request=request,
            context=context,
            detected_task_type=detected_task_type,
        )

        try:
            pipeline_bundle = self.feature_pipeline_service.build_bundle(
                dataframe=dataframe,
                profiles=context.column_profiles,
                request=request,
                target_column=target_column,
            )
        except MLFeaturePipelineError as exc:
            raise MLWorkbenchError(str(exc)) from exc

        model_options = self._model_options(detected_task_type)

        warnings = list(pipeline_bundle.warnings)

        if context.target_analysis and target_column:
            warnings.extend(context.target_analysis.warnings)

        if int(dataframe.shape[0]) < 50:
            warnings.append(
                "The dataset contains fewer than 50 rows. Treat model metrics as exploratory."
            )

        if (
            detected_task_type
            in {
                "binary_classification",
                "multiclass_classification",
            }
            and context.target_analysis
            and context.target_analysis.majority_ratio
            and context.target_analysis.majority_ratio >= 0.65
        ):
            warnings.append("Class imbalance detected. Balanced class weights are recommended.")

        primary_metric, metrics = self._recommended_metrics(detected_task_type)

        return MLWorkbenchPlanResponse(
            dataset_id=dataset_id,
            dataset_version_id=dataset_version.version_id,
            detected_task_type=detected_task_type,
            target_column=target_column,
            target_readiness_score=(
                context.target_analysis.readiness_score
                if context.target_analysis and target_column
                else None
            ),
            feature_groups=(pipeline_bundle.feature_groups),
            preprocessing_steps=(pipeline_bundle.preprocessing_steps),
            estimated_feature_count=(pipeline_bundle.estimated_feature_count),
            available_models=model_options,
            recommended_metrics=metrics,
            primary_metric=primary_metric,
            warnings=list(dict.fromkeys(warnings)),
        )

    def run_experiment(
        self,
        dataset_id: str,
        request: MLWorkbenchExperimentRequest,
    ) -> MLWorkbenchExperimentResponse:
        experiment_id = uuid4().hex
        created_at = datetime.now(UTC)

        # 儲存 supervised model 時需要對應目前 dataset。
        self._current_dataset_id = dataset_id

        plan_request = MLWorkbenchPlanRequest(
            **request.model_dump(
                exclude={
                    "selected_models",
                    "cv_folds",
                    "test_size",
                    "random_state",
                    "class_weight_mode",
                    "n_clusters",
                    "dbscan_eps",
                    "dbscan_min_samples",
                    "contamination",
                }
            )
        )

        plan = self.build_plan(
            dataset_id=dataset_id,
            request=plan_request,
        )
        self._current_dataset_version_id = plan.dataset_version_id

        dataframe = self.dataset_service.load_dataset_dataframe(
            dataset_id,
            version_id=plan.dataset_version_id,
        )

        context = self.analysis_context_service.build_context(
            dataset_id=dataset_id,
            request=AnalysisContextRequest(
                target_column=plan.target_column,
                user_goal="Run an ML Workbench experiment.",
            ),
        )

        try:
            bundle = self.feature_pipeline_service.build_bundle(
                dataframe=dataframe,
                profiles=context.column_profiles,
                request=request,
                target_column=plan.target_column,
            )
        except MLFeaturePipelineError as exc:
            raise MLWorkbenchError(str(exc)) from exc

        selected_models = self._selected_models(
            request=request,
            plan=plan,
        )

        if plan.detected_task_type in {
            "binary_classification",
            "multiclass_classification",
            "regression",
        }:
            (
                model_results,
                actual_cv_folds,
            ) = self._run_supervised_experiment(
                experiment_id=experiment_id,
                dataframe=dataframe,
                task_type=plan.detected_task_type,
                target_column=plan.target_column,
                bundle=bundle,
                selected_models=selected_models,
                request=request,
                target_analysis=context.target_analysis,
            )
        elif plan.detected_task_type == "clustering":
            model_results = self._run_clustering_experiment(
                dataframe=dataframe,
                bundle=bundle,
                selected_models=selected_models,
                request=request,
            )
            actual_cv_folds = None
        else:
            model_results = self._run_anomaly_experiment(
                dataframe=dataframe,
                bundle=bundle,
                selected_models=selected_models,
                request=request,
            )
            actual_cv_folds = None

        successful_results = [result for result in model_results if result.status == "success"]

        warnings = list(plan.warnings)

        failed_results = [result for result in model_results if result.status == "failed"]

        if failed_results:
            warnings.append(f"{len(failed_results)} model(s) failed.")

        if not successful_results:
            status: Literal["success", "completed_with_warnings", "failed"] = "failed"
            best_model_id = None
            best_model_name = None
            best_metric_value = None
        else:
            best_result = self._select_best_result(
                task_type=plan.detected_task_type,
                primary_metric=plan.primary_metric,
                results=successful_results,
            )

            status = "completed_with_warnings" if warnings or failed_results else "success"

            best_model_id = best_result.model_id
            best_model_name = best_result.model_name
            best_metric_value = self._result_metric_value(
                result=best_result,
                metric_name=plan.primary_metric,
            )

        response = MLWorkbenchExperimentResponse(
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            dataset_version_id=plan.dataset_version_id,
            status=status,
            task_type=plan.detected_task_type,
            target_column=plan.target_column,
            feature_groups=bundle.feature_groups,
            preprocessing_steps=bundle.preprocessing_steps,
            cv_folds=actual_cv_folds,
            primary_metric=plan.primary_metric,
            best_model_id=best_model_id,
            best_model_name=best_model_name,
            best_metric_value=best_metric_value,
            model_results=model_results,
            warnings=list(dict.fromkeys(warnings)),
            created_at=created_at,
        )

        self.experiment_registry_service.add_experiment(response)

        return response

    def _run_supervised_experiment(
        self,
        experiment_id: str,
        dataframe: pd.DataFrame,
        task_type: str,
        target_column: str | None,
        bundle: MLFeaturePipelineBundle,
        selected_models: list[str],
        request: MLWorkbenchExperimentRequest,
        target_analysis,
    ) -> tuple[list[MLWorkbenchModelResult], int]:
        if not target_column:
            raise MLWorkbenchError("Supervised experiment requires a target column.")

        feature_columns = self._bundle_feature_columns(bundle)

        working_dataframe = dataframe[feature_columns + [target_column]].copy()

        if task_type == "regression":
            working_dataframe[target_column] = pd.to_numeric(
                working_dataframe[target_column]
                .astype(str)
                .str.replace(
                    ",",
                    "",
                    regex=False,
                ),
                errors="coerce",
            )

        working_dataframe = working_dataframe.dropna(subset=[target_column])

        if len(working_dataframe) < 12:
            raise MLWorkbenchError("Supervised experiment requires at least 12 usable target rows.")

        x_data = working_dataframe[feature_columns]
        y_data = working_dataframe[target_column]

        if (
            task_type
            in {
                "binary_classification",
                "multiclass_classification",
            }
            and y_data.nunique() < 2
        ):
            raise MLWorkbenchError("Classification requires at least two classes.")

        (
            x_train,
            x_test,
            y_train,
            y_test,
        ) = self._supervised_split(
            x_data=x_data,
            y_data=y_data,
            task_type=task_type,
            test_size=request.test_size,
            random_state=request.random_state,
        )

        cv_splitter, actual_cv_folds = self._build_cv_splitter(
            task_type=task_type,
            y_train=y_train,
            requested_folds=request.cv_folds,
            random_state=request.random_state,
        )

        model_factories = self._supervised_model_factories(
            task_type=task_type,
            selected_models=selected_models,
            class_weight_mode=request.class_weight_mode,
            target_analysis=target_analysis,
            random_state=request.random_state,
        )

        scoring = self._scoring_map(task_type)

        results = []

        for model_name, model_definition in model_factories.items():
            started_at = perf_counter()

            try:
                pipeline = Pipeline(
                    steps=[
                        (
                            "preprocessor",
                            clone(bundle.preprocessor),
                        ),
                        (
                            "model",
                            model_definition["estimator"],
                        ),
                    ]
                )

                cv_output = cross_validate(
                    estimator=pipeline,
                    X=x_train,
                    y=y_train,
                    scoring=scoring,
                    cv=cv_splitter,
                    n_jobs=1,
                    error_score="raise",
                    return_train_score=False,
                )

                pipeline.fit(
                    x_train,
                    y_train,
                )

                predictions = pipeline.predict(x_test)

                holdout_metrics = self._holdout_metrics(
                    task_type=task_type,
                    pipeline=pipeline,
                    x_test=x_test,
                    y_test=y_test,
                    predictions=predictions,
                )

                feature_importance = self._feature_importance(pipeline)

                error_samples = self._error_samples(
                    task_type=task_type,
                    x_test=x_test,
                    y_test=y_test,
                    predictions=predictions,
                )

                artifacts = self._evaluation_artifacts(
                    task_type=task_type,
                    pipeline=pipeline,
                    x_test=x_test,
                    y_test=y_test,
                    predictions=predictions,
                    feature_importance=(feature_importance),
                    error_samples=error_samples,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    test_size=request.test_size,
                    random_state=request.random_state,
                )

                model_id = self._save_supervised_model(
                    experiment_id=experiment_id,
                    dataset_id=(self._current_dataset_id),
                    dataset_version_id=(self._current_dataset_version_id),
                    model_name=model_name,
                    task_type=self._registry_task_type(task_type),
                    target_column=target_column,
                    feature_columns=feature_columns,
                    holdout_metrics=holdout_metrics,
                    feature_importance=(feature_importance),
                    evaluation_artifacts=artifacts,
                    pipeline=pipeline,
                )

                feature_count = self._fitted_feature_count(pipeline)

                results.append(
                    MLWorkbenchModelResult(
                        model_id=model_id,
                        model_name=model_name,
                        model_label=(model_definition["label"]),
                        status="success",
                        cv_metrics=(
                            self._summarize_cv_metrics(
                                cv_output=cv_output,
                                scoring=scoring,
                            )
                        ),
                        holdout_metrics=holdout_metrics,
                        feature_count=feature_count,
                        training_seconds=round(
                            perf_counter() - started_at,
                            4,
                        ),
                        feature_importance=(feature_importance),
                        error_samples=error_samples,
                        warnings=(model_definition["warnings"]),
                    )
                )
            except Exception as exc:
                results.append(
                    MLWorkbenchModelResult(
                        model_name=model_name,
                        model_label=(model_definition["label"]),
                        status="failed",
                        training_seconds=round(
                            perf_counter() - started_at,
                            4,
                        ),
                        warnings=(model_definition["warnings"]),
                        error_message=str(exc),
                    )
                )

        return results, actual_cv_folds

    @property
    def _current_dataset_id(self) -> str:
        if not hasattr(
            self,
            "_current_dataset_id_value",
        ):
            raise MLWorkbenchError("Current dataset ID is not initialized.")

        return self._current_dataset_id_value

    @_current_dataset_id.setter
    def _current_dataset_id(
        self,
        value: str,
    ) -> None:
        self._current_dataset_id_value = value

    def _run_clustering_experiment(
        self,
        dataframe: pd.DataFrame,
        bundle: MLFeaturePipelineBundle,
        selected_models: list[str],
        request: MLWorkbenchExperimentRequest,
    ) -> list[MLWorkbenchModelResult]:
        feature_columns = self._bundle_feature_columns(bundle)
        x_data = dataframe[feature_columns]

        transformed = bundle.preprocessor.fit_transform(x_data)
        dense_matrix = self._dense_matrix(transformed)
        projection = self._pca_projection(dense_matrix)

        results = []

        for model_name in selected_models:
            started_at = perf_counter()

            try:
                if model_name == "kmeans":
                    cluster_count = min(
                        request.n_clusters,
                        max(
                            2,
                            len(dense_matrix) - 1,
                        ),
                    )

                    estimator = KMeans(
                        n_clusters=cluster_count,
                        n_init="auto",
                        random_state=(request.random_state),
                    )

                    labels = estimator.fit_predict(dense_matrix)

                    metrics = self._cluster_metrics(
                        dense_matrix,
                        labels,
                    )
                    metrics["inertia"] = self._safe_float(estimator.inertia_)

                    model_label = "KMeans"
                elif model_name == "dbscan":
                    estimator = DBSCAN(
                        eps=request.dbscan_eps,
                        min_samples=(request.dbscan_min_samples),
                    )

                    labels = estimator.fit_predict(dense_matrix)

                    metrics = self._cluster_metrics(
                        dense_matrix,
                        labels,
                    )

                    model_label = "DBSCAN"
                else:
                    raise MLWorkbenchError(f"Unsupported clustering model: {model_name}")

                artifact = self._projection_artifact(
                    projection=projection,
                    labels=labels,
                )

                results.append(
                    MLWorkbenchModelResult(
                        model_name=model_name,
                        model_label=model_label,
                        status="success",
                        holdout_metrics=metrics,
                        feature_count=(dense_matrix.shape[1]),
                        training_seconds=round(
                            perf_counter() - started_at,
                            4,
                        ),
                        artifact=artifact,
                    )
                )
            except Exception as exc:
                results.append(
                    MLWorkbenchModelResult(
                        model_name=model_name,
                        model_label=model_name,
                        status="failed",
                        training_seconds=round(
                            perf_counter() - started_at,
                            4,
                        ),
                        error_message=str(exc),
                    )
                )

        return results

    def _run_anomaly_experiment(
        self,
        dataframe: pd.DataFrame,
        bundle: MLFeaturePipelineBundle,
        selected_models: list[str],
        request: MLWorkbenchExperimentRequest,
    ) -> list[MLWorkbenchModelResult]:
        feature_columns = self._bundle_feature_columns(bundle)
        x_data = dataframe[feature_columns]

        transformed = bundle.preprocessor.fit_transform(x_data)
        dense_matrix = self._dense_matrix(transformed)
        projection = self._pca_projection(dense_matrix)

        results = []

        for model_name in selected_models:
            started_at = perf_counter()

            try:
                if model_name == "isolation_forest":
                    estimator = IsolationForest(
                        contamination=(request.contamination),
                        random_state=(request.random_state),
                        n_estimators=200,
                    )

                    labels = estimator.fit_predict(dense_matrix)
                    scores = estimator.decision_function(dense_matrix)
                    model_label = "Isolation Forest"
                elif model_name == "local_outlier_factor":
                    neighbor_count = min(
                        20,
                        max(
                            2,
                            len(dense_matrix) - 1,
                        ),
                    )

                    estimator = LocalOutlierFactor(
                        n_neighbors=neighbor_count,
                        contamination=(request.contamination),
                    )

                    labels = estimator.fit_predict(dense_matrix)
                    scores = estimator.negative_outlier_factor_
                    model_label = "Local Outlier Factor"
                else:
                    raise MLWorkbenchError(f"Unsupported anomaly model: {model_name}")

                anomaly_count = int(np.sum(labels == -1))

                artifact = self._projection_artifact(
                    projection=projection,
                    labels=np.where(
                        labels == -1,
                        "anomaly",
                        "normal",
                    ),
                    scores=scores,
                )

                results.append(
                    MLWorkbenchModelResult(
                        model_name=model_name,
                        model_label=model_label,
                        status="success",
                        holdout_metrics={
                            "anomaly_count": (float(anomaly_count)),
                            "anomaly_ratio": round(
                                anomaly_count
                                / max(
                                    len(labels),
                                    1,
                                ),
                                4,
                            ),
                        },
                        feature_count=(dense_matrix.shape[1]),
                        training_seconds=round(
                            perf_counter() - started_at,
                            4,
                        ),
                        artifact=artifact,
                    )
                )
            except Exception as exc:
                results.append(
                    MLWorkbenchModelResult(
                        model_name=model_name,
                        model_label=model_name,
                        status="failed",
                        training_seconds=round(
                            perf_counter() - started_at,
                            4,
                        ),
                        error_message=str(exc),
                    )
                )

        return results

    def _context_target(
        self,
        request: MLWorkbenchPlanRequest,
    ) -> str | None:
        if request.task_type in {
            "clustering",
            "anomaly_detection",
        }:
            return None

        return request.target_column

    def _detect_task_type(
        self,
        request: MLWorkbenchPlanRequest,
        context,
    ) -> DetectedMLTaskType:
        if request.task_type != "auto":
            return request.task_type

        target_analysis = context.target_analysis

        if target_analysis is None:
            return "clustering"

        if target_analysis.task_type == "regression":
            return "regression"

        if target_analysis.unique_count == 2:
            return "binary_classification"

        return "multiclass_classification"

    def _resolve_target_column(
        self,
        request: MLWorkbenchPlanRequest,
        context,
        detected_task_type: str,
    ) -> str | None:
        if detected_task_type in {
            "clustering",
            "anomaly_detection",
        }:
            return None

        target_column = request.target_column or (
            context.target_analysis.target_column if context.target_analysis else None
        )

        if not target_column:
            raise MLWorkbenchError("Supervised task requires a target column.")

        return target_column

    def _model_options(
        self,
        task_type: str,
    ) -> list[MLModelOption]:
        definitions = self._model_definitions(task_type)

        return [
            MLModelOption(
                id=model_id,
                label=definition["label"],
                description=definition["description"],
                recommended=definition["recommended"],
                supports_probability=definition["supports_probability"],
            )
            for model_id, definition in definitions.items()
        ]

    def _model_definitions(
        self,
        task_type: str,
    ) -> dict[str, dict[str, Any]]:
        if task_type in {
            "binary_classification",
            "multiclass_classification",
        }:
            return {
                "logistic_regression": {
                    "label": "Logistic Regression",
                    "description": ("Regularized linear classification baseline."),
                    "recommended": True,
                    "supports_probability": True,
                },
                "random_forest_classifier": {
                    "label": "Random Forest",
                    "description": ("Non-linear tree ensemble with feature importance."),
                    "recommended": True,
                    "supports_probability": True,
                },
                "extra_trees_classifier": {
                    "label": "Extra Trees",
                    "description": ("Highly randomized tree ensemble."),
                    "recommended": False,
                    "supports_probability": True,
                },
                "gradient_boosting_classifier": {
                    "label": "Gradient Boosting",
                    "description": ("Stage-wise boosted decision trees."),
                    "recommended": False,
                    "supports_probability": True,
                },
                "linear_svm": {
                    "label": "Linear SVM",
                    "description": ("Linear margin classifier for high-dimensional data."),
                    "recommended": False,
                    "supports_probability": False,
                },
                "knn_classifier": {
                    "label": "KNN Classifier",
                    "description": ("Distance-based local classification baseline."),
                    "recommended": False,
                    "supports_probability": True,
                },
            }

        if task_type == "regression":
            return {
                "ridge_regression": {
                    "label": "Ridge Regression",
                    "description": ("L2-regularized linear regression."),
                    "recommended": True,
                    "supports_probability": False,
                },
                "lasso_regression": {
                    "label": "Lasso Regression",
                    "description": ("L1-regularized sparse linear regression."),
                    "recommended": False,
                    "supports_probability": False,
                },
                "random_forest_regressor": {
                    "label": "Random Forest Regressor",
                    "description": ("Non-linear tree ensemble regressor."),
                    "recommended": True,
                    "supports_probability": False,
                },
                "extra_trees_regressor": {
                    "label": "Extra Trees Regressor",
                    "description": ("Randomized tree ensemble regressor."),
                    "recommended": False,
                    "supports_probability": False,
                },
                "gradient_boosting_regressor": {
                    "label": "Gradient Boosting Regressor",
                    "description": ("Boosted decision tree regression."),
                    "recommended": False,
                    "supports_probability": False,
                },
                "linear_svr": {
                    "label": "Linear SVR",
                    "description": ("Linear support vector regression."),
                    "recommended": False,
                    "supports_probability": False,
                },
                "knn_regressor": {
                    "label": "KNN Regressor",
                    "description": ("Distance-based local regression baseline."),
                    "recommended": False,
                    "supports_probability": False,
                },
            }

        if task_type == "clustering":
            return {
                "kmeans": {
                    "label": "KMeans",
                    "description": ("Centroid-based clustering."),
                    "recommended": True,
                    "supports_probability": False,
                },
                "dbscan": {
                    "label": "DBSCAN",
                    "description": ("Density-based clustering with noise detection."),
                    "recommended": False,
                    "supports_probability": False,
                },
            }

        return {
            "isolation_forest": {
                "label": "Isolation Forest",
                "description": ("Tree-based unsupervised anomaly detection."),
                "recommended": True,
                "supports_probability": False,
            },
            "local_outlier_factor": {
                "label": "Local Outlier Factor",
                "description": ("Local density-based anomaly detection."),
                "recommended": False,
                "supports_probability": False,
            },
        }

    def _recommended_metrics(
        self,
        task_type: str,
    ) -> tuple[str, list[str]]:
        if task_type == "binary_classification":
            return (
                "f1_macro",
                [
                    "accuracy",
                    "balanced_accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                    "roc_auc",
                ],
            )

        if task_type == "multiclass_classification":
            return (
                "f1_macro",
                [
                    "accuracy",
                    "balanced_accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                ],
            )

        if task_type == "regression":
            return (
                "rmse",
                [
                    "mae",
                    "rmse",
                    "r2",
                ],
            )

        if task_type == "clustering":
            return (
                "silhouette",
                [
                    "silhouette",
                    "davies_bouldin",
                    "cluster_count",
                    "noise_ratio",
                ],
            )

        return (
            "anomaly_ratio",
            [
                "anomaly_count",
                "anomaly_ratio",
            ],
        )

    def _selected_models(
        self,
        request: MLWorkbenchExperimentRequest,
        plan: MLWorkbenchPlanResponse,
    ) -> list[str]:
        available_ids = {model.id for model in plan.available_models}

        if request.selected_models:
            unknown_models = [
                model_id for model_id in request.selected_models if model_id not in available_ids
            ]

            if unknown_models:
                raise MLWorkbenchError("Unsupported models: " + ", ".join(unknown_models))

            return list(dict.fromkeys(request.selected_models))

        recommended = [model.id for model in plan.available_models if model.recommended]

        if not recommended:
            raise MLWorkbenchError("No recommended models are available.")

        return recommended

    def _supervised_model_factories(
        self,
        task_type: str,
        selected_models: list[str],
        class_weight_mode: str,
        target_analysis,
        random_state: int,
    ) -> dict[str, dict[str, Any]]:
        class_weight = self._class_weight(
            mode=class_weight_mode,
            target_analysis=target_analysis,
        )

        if task_type in {
            "binary_classification",
            "multiclass_classification",
        }:
            all_models = {
                "logistic_regression": {
                    "label": "Logistic Regression",
                    "estimator": LogisticRegression(
                        max_iter=2000,
                        class_weight=class_weight,
                    ),
                    "warnings": [],
                },
                "random_forest_classifier": {
                    "label": "Random Forest",
                    "estimator": RandomForestClassifier(
                        n_estimators=200,
                        random_state=random_state,
                        class_weight=class_weight,
                        n_jobs=-1,
                    ),
                    "warnings": [],
                },
                "extra_trees_classifier": {
                    "label": "Extra Trees",
                    "estimator": ExtraTreesClassifier(
                        n_estimators=200,
                        random_state=random_state,
                        class_weight=class_weight,
                        n_jobs=-1,
                    ),
                    "warnings": [],
                },
                "gradient_boosting_classifier": {
                    "label": "Gradient Boosting",
                    "estimator": GradientBoostingClassifier(
                        n_estimators=120,
                        random_state=random_state,
                    ),
                    "warnings": (
                        ["Gradient Boosting does not use class_weight in this implementation."]
                        if class_weight
                        else []
                    ),
                },
                "linear_svm": {
                    "label": "Linear SVM",
                    "estimator": LinearSVC(
                        class_weight=class_weight,
                        random_state=random_state,
                    ),
                    "warnings": ["Linear SVM does not expose predict_proba."],
                },
                "knn_classifier": {
                    "label": "KNN Classifier",
                    "estimator": KNeighborsClassifier(
                        n_neighbors=5,
                    ),
                    "warnings": ["KNN can be slow with high-dimensional features."],
                },
            }
        else:
            all_models = {
                "ridge_regression": {
                    "label": "Ridge Regression",
                    "estimator": Ridge(alpha=1.0),
                    "warnings": [],
                },
                "lasso_regression": {
                    "label": "Lasso Regression",
                    "estimator": Lasso(
                        alpha=0.01,
                        max_iter=5000,
                    ),
                    "warnings": [],
                },
                "random_forest_regressor": {
                    "label": "Random Forest Regressor",
                    "estimator": RandomForestRegressor(
                        n_estimators=200,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                    "warnings": [],
                },
                "extra_trees_regressor": {
                    "label": "Extra Trees Regressor",
                    "estimator": ExtraTreesRegressor(
                        n_estimators=200,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                    "warnings": [],
                },
                "gradient_boosting_regressor": {
                    "label": "Gradient Boosting Regressor",
                    "estimator": GradientBoostingRegressor(
                        n_estimators=120,
                        random_state=random_state,
                    ),
                    "warnings": [],
                },
                "linear_svr": {
                    "label": "Linear SVR",
                    "estimator": LinearSVR(
                        random_state=random_state,
                        max_iter=5000,
                    ),
                    "warnings": [],
                },
                "knn_regressor": {
                    "label": "KNN Regressor",
                    "estimator": KNeighborsRegressor(
                        n_neighbors=5,
                    ),
                    "warnings": ["KNN can be slow with high-dimensional features."],
                },
            }

        return {model_name: all_models[model_name] for model_name in selected_models}

    def _class_weight(
        self,
        mode: str,
        target_analysis,
    ) -> str | None:
        if mode == "balanced":
            return "balanced"

        if mode == "none":
            return None

        if (
            target_analysis
            and target_analysis.majority_ratio
            and target_analysis.majority_ratio >= 0.6
        ):
            return "balanced"

        return None

    def _scoring_map(
        self,
        task_type: str,
    ) -> dict[str, str]:
        if task_type == "binary_classification":
            return {
                "accuracy": "accuracy",
                "balanced_accuracy": ("balanced_accuracy"),
                "precision_macro": ("precision_macro"),
                "recall_macro": "recall_macro",
                "f1_macro": "f1_macro",
                "roc_auc": "roc_auc",
            }

        if task_type == "multiclass_classification":
            return {
                "accuracy": "accuracy",
                "balanced_accuracy": ("balanced_accuracy"),
                "precision_macro": ("precision_macro"),
                "recall_macro": "recall_macro",
                "f1_macro": "f1_macro",
            }

        return {
            "mae": "neg_mean_absolute_error",
            "rmse": ("neg_root_mean_squared_error"),
            "r2": "r2",
        }

    def _supervised_split(
        self,
        x_data: pd.DataFrame,
        y_data: pd.Series,
        task_type: str,
        test_size: float,
        random_state: int,
    ):
        stratify = None
        actual_test_size = test_size

        if task_type in {
            "binary_classification",
            "multiclass_classification",
        }:
            class_counts = y_data.value_counts()
            class_count = len(class_counts)

            minimum_test_rows = class_count
            requested_test_rows = max(
                int(math.ceil(len(y_data) * test_size)),
                minimum_test_rows,
            )

            actual_test_size = requested_test_rows / len(y_data)

            if actual_test_size >= 0.5:
                actual_test_size = 0.5

            if class_counts.min() >= 2:
                stratify = y_data

        try:
            return train_test_split(
                x_data,
                y_data,
                test_size=actual_test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            return train_test_split(
                x_data,
                y_data,
                test_size=actual_test_size,
                random_state=random_state,
                stratify=None,
            )

    def _build_cv_splitter(
        self,
        task_type: str,
        y_train: pd.Series,
        requested_folds: int,
        random_state: int,
    ):
        if task_type in {
            "binary_classification",
            "multiclass_classification",
        }:
            minimum_class_count = int(y_train.value_counts().min())

            actual_folds = min(
                requested_folds,
                minimum_class_count,
            )

            if actual_folds < 2:
                raise MLWorkbenchError(
                    "Cross-validation requires at least two training samples per class."
                )

            return (
                StratifiedKFold(
                    n_splits=actual_folds,
                    shuffle=True,
                    random_state=random_state,
                ),
                actual_folds,
            )

        actual_folds = min(
            requested_folds,
            len(y_train),
        )

        if actual_folds < 2:
            raise MLWorkbenchError("Cross-validation requires at least two rows.")

        return (
            KFold(
                n_splits=actual_folds,
                shuffle=True,
                random_state=random_state,
            ),
            actual_folds,
        )

    def _summarize_cv_metrics(
        self,
        cv_output: dict[str, Any],
        scoring: dict[str, str],
    ) -> list[CrossValidationMetric]:
        results = []

        for metric_name in scoring:
            values = np.asarray(
                cv_output[f"test_{metric_name}"],
                dtype=float,
            )

            if metric_name in {
                "mae",
                "rmse",
            }:
                values = -values
                direction: Literal["maximize", "minimize"] = "minimize"
            else:
                direction = "maximize"

            results.append(
                CrossValidationMetric(
                    name=metric_name,
                    mean=self._safe_float(np.mean(values)),
                    std=self._safe_float(np.std(values)),
                    fold_values=[self._safe_float(value) for value in values.tolist()],
                    direction=direction,
                )
            )

        return results

    def _holdout_metrics(
        self,
        task_type: str,
        pipeline: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        predictions: np.ndarray,
    ) -> dict[str, float | None]:
        if task_type in {
            "binary_classification",
            "multiclass_classification",
        }:
            metrics = {
                "accuracy": self._safe_float(
                    accuracy_score(
                        y_test,
                        predictions,
                    )
                ),
                "balanced_accuracy": self._safe_float(
                    balanced_accuracy_score(
                        y_test,
                        predictions,
                    )
                ),
                "precision_macro": self._safe_float(
                    precision_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "recall_macro": self._safe_float(
                    recall_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "f1_macro": self._safe_float(
                    f1_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }

            roc_auc = self._roc_auc(
                task_type=task_type,
                pipeline=pipeline,
                x_test=x_test,
                y_test=y_test,
            )

            if roc_auc is not None:
                metrics["roc_auc"] = roc_auc

            return metrics

        mse = mean_squared_error(
            y_test,
            predictions,
        )

        return {
            "mae": self._safe_float(
                mean_absolute_error(
                    y_test,
                    predictions,
                )
            ),
            "rmse": self._safe_float(math.sqrt(mse)),
            "r2": self._safe_float(
                r2_score(
                    y_test,
                    predictions,
                )
            ),
        }

    def _roc_auc(
        self,
        task_type: str,
        pipeline: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> float | None:
        try:
            if hasattr(
                pipeline,
                "predict_proba",
            ):
                probabilities = pipeline.predict_proba(x_test)

                if task_type == "binary_classification":
                    scores = probabilities[:, 1]

                    return self._safe_float(
                        roc_auc_score(
                            y_test,
                            scores,
                        )
                    )

                return self._safe_float(
                    roc_auc_score(
                        y_test,
                        probabilities,
                        multi_class="ovr",
                        average="weighted",
                    )
                )

            if task_type == "binary_classification" and hasattr(
                pipeline,
                "decision_function",
            ):
                scores = pipeline.decision_function(x_test)

                return self._safe_float(
                    roc_auc_score(
                        y_test,
                        scores,
                    )
                )
        except Exception:
            return None

        return None

    def _feature_importance(
        self,
        pipeline: Pipeline,
    ) -> list[dict[str, Any]]:
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]

        try:
            feature_names = [str(name) for name in (preprocessor.get_feature_names_out())]
        except Exception:
            feature_names = []

        importance_values = None

        if hasattr(
            estimator,
            "feature_importances_",
        ):
            importance_values = estimator.feature_importances_
        elif hasattr(
            estimator,
            "coef_",
        ):
            coefficients = np.abs(estimator.coef_)

            if coefficients.ndim == 2:
                importance_values = coefficients.mean(axis=0)
            else:
                importance_values = coefficients

        if importance_values is None:
            return []

        items = []

        for index, value in enumerate(importance_values):
            feature_name = (
                feature_names[index] if index < len(feature_names) else f"feature_{index}"
            )

            items.append(
                {
                    "feature": feature_name,
                    "importance": (self._safe_float(value) or 0.0),
                }
            )

        items.sort(
            key=lambda item: float(cast(Any, item["importance"])),
            reverse=True,
        )

        return items[:20]

    def _error_samples(
        self,
        task_type: str,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        predictions: np.ndarray,
    ) -> list[dict[str, Any]]:
        if task_type in {
            "binary_classification",
            "multiclass_classification",
        }:
            samples = []

            for position, (
                row_index,
                actual,
                predicted,
            ) in enumerate(
                zip(
                    y_test.index,
                    y_test.tolist(),
                    predictions.tolist(),
                    strict=False,
                )
            ):
                if actual == predicted:
                    continue

                samples.append(
                    {
                        "position": position,
                        "row_index": str(row_index),
                        "actual": self._json_value(actual),
                        "predicted": (self._json_value(predicted)),
                    }
                )

                if len(samples) >= 20:
                    break

            return samples

        rows = []

        for position, (
            row_index,
            actual,
            predicted,
        ) in enumerate(
            zip(
                y_test.index,
                y_test.tolist(),
                predictions.tolist(),
                strict=False,
            )
        ):
            actual_value = self._safe_float(actual)
            predicted_value = self._safe_float(predicted)

            if actual_value is None or predicted_value is None:
                continue

            residual = actual_value - predicted_value

            rows.append(
                {
                    "position": position,
                    "row_index": str(row_index),
                    "actual": actual_value,
                    "predicted": predicted_value,
                    "residual": self._safe_float(residual),
                    "absolute_error": (self._safe_float(abs(residual))),
                }
            )

        rows.sort(
            key=lambda item: float(cast(Any, item["absolute_error"] or 0)),
            reverse=True,
        )

        return rows[:20]

    def _evaluation_artifacts(
        self,
        task_type: str,
        pipeline: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        predictions: np.ndarray,
        feature_importance: list[dict[str, Any]],
        error_samples: list[dict[str, Any]],
        feature_columns: list[str],
        target_column: str,
        test_size: float,
        random_state: int,
    ) -> dict[str, Any]:
        artifacts: dict[str, Any] = {
            "feature_importance": (feature_importance),
            "error_samples": error_samples,
            "split_context": {
                "feature_columns": feature_columns,
                "target_column": target_column,
                "holdout_row_indices": [self._json_value(index) for index in x_test.index.tolist()],
                "test_size": test_size,
                "random_state": random_state,
            },
        }

        if task_type in {
            "binary_classification",
            "multiclass_classification",
        }:
            labels = sorted({str(value) for value in (y_test.tolist() + predictions.tolist())})

            matrix = confusion_matrix(
                [str(value) for value in y_test.tolist()],
                [str(value) for value in predictions.tolist()],
                labels=labels,
            )

            artifacts["confusion_matrix"] = {
                "labels": labels,
                "matrix": matrix.astype(int).tolist(),
            }
        else:
            artifacts["regression_residuals"] = [
                {
                    "actual": item["actual"],
                    "predicted": item["predicted"],
                    "residual": item["residual"],
                }
                for item in error_samples
            ]

        return artifacts

    def _save_supervised_model(
        self,
        experiment_id: str,
        dataset_id: str,
        dataset_version_id: str | None,
        model_name: str,
        task_type: str,
        target_column: str,
        feature_columns: list[str],
        holdout_metrics: dict[str, float | None],
        feature_importance: list[dict[str, Any]],
        evaluation_artifacts: dict[str, Any],
        pipeline: Pipeline,
    ) -> str:
        model_id = uuid4().hex
        model_filename = f"{model_id}_{model_name}.joblib"
        model_path = self.settings.models_dir / model_filename

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            pipeline,
            model_path,
        )

        evaluation_dir = self.settings.models_dir / "evaluations"
        evaluation_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        evaluation_path = evaluation_dir / f"{model_id}_evaluation.json"

        with evaluation_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                evaluation_artifacts,
                file,
                ensure_ascii=False,
                indent=2,
            )

        record = {
            "id": model_id,
            "dataset_id": dataset_id,
            "dataset_version_id": dataset_version_id,
            "model_name": model_name,
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "metrics": holdout_metrics,
            "feature_importance": (feature_importance),
            "model_path": str(model_path.relative_to(self.settings.project_root)),
            "evaluation_artifacts_path": str(
                evaluation_path.relative_to(self.settings.project_root)
            ),
            "status": "trained",
            "lifecycle_status": "candidate",
            "training_config": {
                "experiment_id": experiment_id,
                "source": "ml_workbench",
            },
            "feature_schema": {
                "required_columns": feature_columns,
                "target_column": target_column,
                "task_type": task_type,
            },
            "preprocessing_recipe": {
                "experiment_id": experiment_id,
                "source": "ml_workbench",
            },
            "created_at": datetime.now(UTC).isoformat(),
        }

        self.model_registry_service.add_model_record(record)

        return model_id

    def _cluster_metrics(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float | None]:
        unique_labels = set(labels.tolist())
        non_noise_labels = {label for label in unique_labels if label != -1}

        cluster_count = len(non_noise_labels)
        noise_count = int(np.sum(labels == -1))

        metrics = {
            "cluster_count": float(cluster_count),
            "noise_ratio": round(
                noise_count / max(len(labels), 1),
                4,
            ),
            "silhouette": None,
            "davies_bouldin": None,
        }

        if cluster_count >= 2 and cluster_count < len(matrix):
            valid_mask = labels != -1
            valid_matrix = matrix[valid_mask]
            valid_labels = labels[valid_mask]

            if len(set(valid_labels.tolist())) >= 2 and len(valid_matrix) > len(
                set(valid_labels.tolist())
            ):
                metrics["silhouette"] = self._safe_float(
                    silhouette_score(
                        valid_matrix,
                        valid_labels,
                    )
                )
                metrics["davies_bouldin"] = self._safe_float(
                    davies_bouldin_score(
                        valid_matrix,
                        valid_labels,
                    )
                )

        return metrics

    def _pca_projection(
        self,
        matrix: np.ndarray,
    ) -> np.ndarray:
        if matrix.shape[1] == 0:
            raise MLWorkbenchError("No transformed features are available.")

        component_count = min(
            2,
            matrix.shape[0],
            matrix.shape[1],
        )

        projection = PCA(
            n_components=component_count,
            random_state=42,
        ).fit_transform(matrix)

        if component_count == 1:
            projection = np.column_stack(
                [
                    projection[:, 0],
                    np.zeros(projection.shape[0]),
                ]
            )

        return projection

    def _projection_artifact(
        self,
        projection: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> dict[str, Any]:
        point_count = min(
            len(projection),
            500,
        )

        positions = np.linspace(
            0,
            len(projection) - 1,
            point_count,
            dtype=int,
        )

        points = []

        for position in positions:
            point = {
                "row_index": int(position),
                "x": self._safe_float(projection[position, 0]),
                "y": self._safe_float(projection[position, 1]),
                "label": str(labels[position]),
            }

            if scores is not None:
                point["score"] = self._safe_float(scores[position])

            points.append(point)

        label_counts = pd.Series(labels).astype(str).value_counts().to_dict()

        return {
            "projection_method": "pca",
            "projection_points": points,
            "label_counts": {str(label): int(count) for label, count in label_counts.items()},
        }

    def _dense_matrix(
        self,
        matrix,
    ) -> np.ndarray:
        if sparse.issparse(matrix):
            estimated_cells = matrix.shape[0] * matrix.shape[1]

            if estimated_cells > 8_000_000:
                raise MLWorkbenchError(
                    "Transformed sparse matrix is too large to convert to dense format."
                )

            matrix = matrix.toarray()

        dense_matrix = np.asarray(
            matrix,
            dtype=float,
        )

        if dense_matrix.ndim != 2:
            raise MLWorkbenchError("Transformed feature matrix must be two-dimensional.")

        return dense_matrix

    def _bundle_feature_columns(
        self,
        bundle: MLFeaturePipelineBundle,
    ) -> list[str]:
        return (
            bundle.feature_groups.numeric
            + bundle.feature_groups.categorical
            + bundle.feature_groups.datetime
            + bundle.feature_groups.text
        )

    def _fitted_feature_count(
        self,
        pipeline: Pipeline,
    ) -> int:
        preprocessor = pipeline.named_steps["preprocessor"]

        try:
            return len(preprocessor.get_feature_names_out())
        except Exception:
            return 0

    def _select_best_result(
        self,
        task_type: str,
        primary_metric: str,
        results: list[MLWorkbenchModelResult],
    ) -> MLWorkbenchModelResult:
        minimize_metrics = {
            "rmse",
            "mae",
            "davies_bouldin",
        }

        valid_results = [
            result
            for result in results
            if self._result_metric_value(
                result,
                primary_metric,
            )
            is not None
        ]

        if not valid_results:
            return results[0]

        if primary_metric in minimize_metrics:
            return min(
                valid_results,
                key=lambda result: (
                    self._result_metric_value(
                        result,
                        primary_metric,
                    )
                    or math.inf
                ),
            )

        return max(
            valid_results,
            key=lambda result: (
                self._result_metric_value(
                    result,
                    primary_metric,
                )
                or -math.inf
            ),
        )

    def _result_metric_value(
        self,
        result: MLWorkbenchModelResult,
        metric_name: str,
    ) -> float | None:
        for metric in result.cv_metrics:
            if metric.name == metric_name:
                return metric.mean

        return result.holdout_metrics.get(metric_name)

    def _registry_task_type(
        self,
        task_type: str,
    ) -> str:
        if task_type in {
            "binary_classification",
            "multiclass_classification",
        }:
            return "classification"

        return "regression"

    def _json_value(
        self,
        value: Any,
    ) -> Any:
        if pd.isna(value):
            return None

        if isinstance(value, np.generic):
            return value.item()

        return value

    def _safe_float(
        self,
        value: object,
    ) -> float | None:
        try:
            float_value = float(cast(Any, value))
        except (TypeError, ValueError):
            return None

        if not math.isfinite(float_value):
            return None

        return round(float_value, 4)
