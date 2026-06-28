from backend.app.core.config import Settings
from backend.app.schemas.ml_schema import (
    MLModelResult,
    MLTrainRequest,
    ModelCompareItem,
    ModelCompareResponse,
    ModelRetrainPlanResponse,
    ModelRetrainRequest,
    ModelRetrainResponse,
    PromoteChallengerResponse,
)
from backend.app.services.dataset_service import DatasetService
from backend.app.services.ml_training_service import MLTrainingService
from backend.app.services.model_registry_service import ModelRegistryService


class ModelRetrainingError(Exception):
    """Raised when challenger retraining cannot be completed."""


class ModelRetrainingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings)
        self.ml_training_service = MLTrainingService(settings)
        self.model_registry_service = ModelRegistryService(settings)

    def build_plan(
        self,
        champion_model_id: str,
        request: ModelRetrainRequest,
    ) -> ModelRetrainPlanResponse:
        champion = self.model_registry_service.get_model(champion_model_id)
        self._validate_champion(champion)
        current_version = self.dataset_service.get_dataset_version(
            champion.dataset_id,
            request.current_version_id,
        )
        warnings = []
        dataframe = self.dataset_service.load_dataset_dataframe(
            champion.dataset_id,
            version_id=current_version.version_id,
            nrows=25,
        )
        missing_features = [
            column for column in champion.feature_columns if column not in dataframe.columns
        ]

        if missing_features:
            warnings.append(
                "Current dataset version is missing champion feature columns: "
                + ", ".join(missing_features)
            )

        if champion.target_column not in dataframe.columns:
            warnings.append("Current dataset version is missing the champion target column.")

        return ModelRetrainPlanResponse(
            champion_model_id=champion.id,
            dataset_id=champion.dataset_id,
            current_version_id=current_version.version_id,
            target_column=champion.target_column,
            task_type=champion.task_type,
            selected_model=champion.model_name,
            feature_columns=champion.feature_columns,
            primary_metric=self._primary_metric(champion.task_type),
            warnings=warnings,
        )

    def retrain(
        self,
        champion_model_id: str,
        request: ModelRetrainRequest,
    ) -> ModelRetrainResponse:
        plan = self.build_plan(champion_model_id, request)

        if plan.warnings:
            raise ModelRetrainingError("Cannot retrain challenger: " + " ".join(plan.warnings))

        champion = self.model_registry_service.get_model(champion_model_id)
        training_config = champion.training_config or {}
        train_response = self.ml_training_service.train_models(
            champion.dataset_id,
            MLTrainRequest(
                target_column=champion.target_column,
                feature_columns=champion.feature_columns,
                task_type=champion.task_type,
                test_size=float(training_config.get("test_size", 0.25)),
                random_state=int(training_config.get("random_state", 42)),
                selected_models=[champion.model_name],
                dataset_version_id=plan.current_version_id,
            ),
        )
        challenger = self.model_registry_service.get_model(train_response.best_model_id)
        comparison = self._compare_models(champion, challenger)
        recommendation, reasons = self._recommend(comparison)
        promoted = False

        if request.auto_promote and recommendation == "promote":
            challenger = self.model_registry_service.update_model_status(
                challenger.id,
                "production",
            )
            promoted = True
            comparison = self._compare_models(champion, challenger)

        return ModelRetrainResponse(
            champion_model=champion,
            challenger_model=challenger,
            comparison=comparison,
            recommendation=recommendation,
            reasons=reasons,
            promoted=promoted,
        )

    def promote_challenger(
        self,
        champion_model_id: str,
        challenger_model_id: str,
    ) -> PromoteChallengerResponse:
        champion = self.model_registry_service.get_model(champion_model_id)
        challenger = self.model_registry_service.get_model(challenger_model_id)

        if champion.dataset_id != challenger.dataset_id:
            raise ModelRetrainingError("Champion and challenger must belong to the same dataset.")

        if (
            champion.target_column != challenger.target_column
            or champion.task_type != challenger.task_type
        ):
            raise ModelRetrainingError("Champion and challenger must share target and task type.")

        promoted = self.model_registry_service.update_model_status(
            challenger.id,
            "production",
        )

        return PromoteChallengerResponse(
            promoted_model=promoted,
            archived_model_id=champion.id if champion.lifecycle_status == "production" else None,
        )

    def _validate_champion(self, champion: MLModelResult) -> None:
        if champion.task_type not in {"classification", "regression"}:
            raise ModelRetrainingError("Retraining only supports supervised models.")

        if not champion.feature_columns:
            raise ModelRetrainingError("Champion model does not contain feature columns.")

    def _compare_models(
        self,
        champion: MLModelResult,
        challenger: MLModelResult,
    ) -> ModelCompareResponse:
        primary_metric = self._primary_metric(champion.task_type)
        reverse = primary_metric != "rmse"
        models = [champion, challenger]
        models.sort(
            key=lambda model: self._metric_value(model, primary_metric, reverse),
            reverse=reverse,
        )

        return ModelCompareResponse(
            dataset_id=champion.dataset_id,
            primary_metric=primary_metric,
            models=[
                ModelCompareItem(
                    model_id=model.id,
                    model_name=model.model_name,
                    lifecycle_status=model.lifecycle_status,
                    task_type=model.task_type,
                    target_column=model.target_column,
                    metrics=model.metrics,
                    primary_metric_value=model.metrics.get(primary_metric),
                    rank=index + 1,
                )
                for index, model in enumerate(models)
            ],
        )

    def _recommend(self, comparison: ModelCompareResponse) -> tuple[str, list[str]]:
        ranked_models = comparison.models
        challenger = next(
            (model for model in ranked_models if model.lifecycle_status == "candidate"),
            None,
        )

        if challenger is None:
            return "review", ["No candidate challenger was available for recommendation."]

        if ranked_models[0].model_id == challenger.model_id:
            return "promote", ["Challenger ranks ahead of the current champion."]

        return "keep_champion", ["Current champion still ranks ahead of the challenger."]

    def _primary_metric(self, task_type: str) -> str:
        return "rmse" if task_type == "regression" else "f1_macro"

    def _metric_value(
        self,
        model: MLModelResult,
        metric_name: str,
        maximize: bool,
    ) -> float:
        value = model.metrics.get(metric_name)

        if value is None:
            return -1.0 if maximize else float("inf")

        return float(value)
