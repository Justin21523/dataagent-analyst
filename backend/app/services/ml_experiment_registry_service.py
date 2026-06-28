from threading import Lock
from typing import Any

from backend.app.core.config import Settings
from backend.app.repositories.metadata_repository import (
    MetadataRepositoryError,
    create_metadata_repository,
)
from backend.app.schemas.ml_workbench_schema import (
    MLWorkbenchExperimentResponse,
    MLWorkbenchExperimentSummary,
)

_REGISTRY_LOCK = Lock()


class MLExperimentRegistryError(Exception):
    """ML experiment registry base exception."""


class MLExperimentNotFoundError(
    MLExperimentRegistryError,
):
    """Raised when an ML experiment cannot be found."""


class MLExperimentRegistryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metadata_repository = create_metadata_repository(settings)

    def add_experiment(
        self,
        experiment: MLWorkbenchExperimentResponse,
    ) -> MLWorkbenchExperimentResponse:
        with _REGISTRY_LOCK:
            registry = self._load_registry()
            registry["experiments"].append(experiment.model_dump(mode="json"))
            self._save_registry(registry)

        return experiment

    def list_experiments(
        self,
        dataset_id: str,
    ) -> list[MLWorkbenchExperimentSummary]:
        registry = self._load_registry()

        summaries = [
            self._build_summary(record)
            for record in registry["experiments"]
            if record["dataset_id"] == dataset_id
        ]

        return sorted(
            summaries,
            key=lambda experiment: experiment.created_at,
            reverse=True,
        )

    def get_experiment(
        self,
        experiment_id: str,
    ) -> MLWorkbenchExperimentResponse:
        registry = self._load_registry()

        for record in registry["experiments"]:
            if record["experiment_id"] == experiment_id:
                return MLWorkbenchExperimentResponse.model_validate(record)

        raise MLExperimentNotFoundError(f"ML experiment not found: {experiment_id}")

    def _build_summary(
        self,
        record: dict[str, Any],
    ) -> MLWorkbenchExperimentSummary:
        return MLWorkbenchExperimentSummary(
            experiment_id=record["experiment_id"],
            dataset_id=record["dataset_id"],
            status=record["status"],
            task_type=record["task_type"],
            target_column=record.get("target_column"),
            model_count=len(record.get("model_results", [])),
            best_model_name=record.get("best_model_name"),
            best_metric_value=record.get("best_metric_value"),
            created_at=record["created_at"],
        )

    def _load_registry(
        self,
    ) -> dict[str, list[dict[str, Any]]]:
        try:
            return self.metadata_repository.load_registry("ml_experiments")
        except MetadataRepositoryError as exc:
            raise MLExperimentRegistryError("ML experiment registry is corrupted.") from exc

    def _save_registry(
        self,
        registry: dict[str, list[dict[str, Any]]],
    ) -> None:
        try:
            self.metadata_repository.save_registry("ml_experiments", registry)
        except MetadataRepositoryError as exc:
            raise MLExperimentRegistryError("ML experiment registry format is invalid.") from exc
