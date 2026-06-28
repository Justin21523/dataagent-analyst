import json
from threading import Lock
from typing import Any

from backend.app.core.config import Settings
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
        self.registry_path = settings.processed_data_dir / settings.ml_experiment_registry_filename

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
        if not self.registry_path.exists():
            return {
                "experiments": [],
            }

        try:
            with self.registry_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                registry = json.load(file)
        except json.JSONDecodeError as exc:
            raise MLExperimentRegistryError("ML experiment registry is corrupted.") from exc

        if "experiments" not in registry or not isinstance(
            registry["experiments"],
            list,
        ):
            raise MLExperimentRegistryError("ML experiment registry format is invalid.")

        return registry

    def _save_registry(
        self,
        registry: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = self.registry_path.with_suffix(".tmp")

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                registry,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(self.registry_path)
