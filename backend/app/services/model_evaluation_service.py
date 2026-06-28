import json
from typing import Any

from backend.app.core.config import Settings
from backend.app.schemas.ml_schema import MLModelEvaluationResponse
from backend.app.services.model_registry_service import ModelRegistryService


class ModelEvaluationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_registry_service = ModelRegistryService(settings)

    def get_model_evaluation(self, model_id: str) -> MLModelEvaluationResponse:
        # 先從 registry 找模型，再讀取模型對應的 evaluation artifacts。
        model = self.model_registry_service.get_model(model_id)
        artifacts = self._load_artifacts(model.evaluation_artifacts_path)

        return MLModelEvaluationResponse(
            model=model,
            confusion_matrix=artifacts.get("confusion_matrix"),
            regression_residuals=artifacts.get("regression_residuals", []),
            feature_importance=artifacts.get("feature_importance", model.feature_importance),
        )

    def _load_artifacts(self, relative_path: str | None) -> dict[str, Any]:
        # 舊模型可能沒有 artifacts path，因此要安全回傳空 dict。
        if not relative_path:
            return {}

        artifact_path = self.settings.project_root / relative_path

        if not artifact_path.exists():
            return {}

        with artifact_path.open("r", encoding="utf-8") as file:
            return json.load(file)
