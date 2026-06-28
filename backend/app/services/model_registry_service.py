from typing import Any

from backend.app.core.config import Settings
from backend.app.repositories.metadata_repository import (
    MetadataRepositoryError,
    create_metadata_repository,
)
from backend.app.schemas.ml_schema import MLModelResult


class ModelRegistryError(Exception):
    """Model registry base exception."""


class ModelNotFoundError(ModelRegistryError):
    """Raised when a model record cannot be found."""


class ModelRegistryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metadata_repository = create_metadata_repository(settings)

    def add_model_record(self, record: dict[str, Any]) -> MLModelResult:
        # 模型註冊表集中管理，避免訓練 service 和 API route 直接操作 JSON 檔。
        registry = self._load_registry()
        registry["models"].append(record)
        self._save_registry(registry)

        return MLModelResult.model_validate(record)

    def list_models(self, dataset_id: str | None = None) -> list[MLModelResult]:
        registry = self._load_registry()
        records = registry["models"]

        if dataset_id is not None:
            records = [record for record in records if record["dataset_id"] == dataset_id]

        models = [MLModelResult.model_validate(record) for record in records]

        return sorted(models, key=lambda model: model.created_at, reverse=True)

    def get_production_model(
        self,
        dataset_id: str,
        target_column: str | None = None,
        task_type: str | None = None,
    ) -> MLModelResult:
        models = [
            model
            for model in self.list_models(dataset_id=dataset_id)
            if model.lifecycle_status == "production"
            and (target_column is None or model.target_column == target_column)
            and (task_type is None or model.task_type == task_type)
        ]

        if not models:
            raise ModelNotFoundError("Production model not found.")

        return models[0]

    def get_model(self, model_id: str) -> MLModelResult:
        registry = self._load_registry()

        for record in registry["models"]:
            if record["id"] == model_id:
                return MLModelResult.model_validate(record)

        raise ModelNotFoundError(f"Model not found: {model_id}")

    def update_model_status(
        self,
        model_id: str,
        lifecycle_status: str,
    ) -> MLModelResult:
        allowed_statuses = {
            "candidate",
            "staging",
            "production",
            "archived",
        }

        if lifecycle_status not in allowed_statuses:
            raise ModelRegistryError(
                "lifecycle_status must be candidate, staging, production, or archived."
            )

        registry = self._load_registry()
        selected_record: dict[str, Any] | None = None

        for record in registry["models"]:
            if record["id"] == model_id:
                selected_record = record
                break

        if selected_record is None:
            raise ModelNotFoundError(f"Model not found: {model_id}")

        if lifecycle_status == "production":
            for record in registry["models"]:
                same_scope = (
                    record.get("dataset_id") == selected_record.get("dataset_id")
                    and record.get("target_column") == selected_record.get("target_column")
                    and record.get("task_type") == selected_record.get("task_type")
                    and record.get("id") != model_id
                    and record.get("lifecycle_status", "candidate") == "production"
                )

                if same_scope:
                    record["lifecycle_status"] = "archived"

        selected_record["lifecycle_status"] = lifecycle_status
        self._save_registry(registry)

        return MLModelResult.model_validate(selected_record)

    def _load_registry(self) -> dict[str, list[dict[str, Any]]]:
        try:
            return self.metadata_repository.load_registry("models")
        except MetadataRepositoryError as exc:
            raise ModelRegistryError("Model registry file is corrupted.") from exc

    def _save_registry(self, registry: dict[str, list[dict[str, Any]]]) -> None:
        try:
            self.metadata_repository.save_registry("models", registry)
        except MetadataRepositoryError as exc:
            raise ModelRegistryError("Model registry format is invalid.") from exc
