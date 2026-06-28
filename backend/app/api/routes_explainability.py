from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.explainability_schema import (
    ModelExplainabilityRequest,
    ModelExplainabilityResponse,
)
from backend.app.services.dataset_service import (
    DatasetNotFoundError,
    DatasetValidationError,
)
from backend.app.services.model_explainability_service import (
    ModelExplainabilityError,
    ModelExplainabilityService,
)
from backend.app.services.model_registry_service import (
    ModelNotFoundError,
)

router = APIRouter(
    prefix="/explainability",
    tags=["Model Explainability"],
)


def get_model_explainability_service(
    settings: Settings = Depends(get_settings),
) -> ModelExplainabilityService:
    return ModelExplainabilityService(settings)


@router.post(
    "/models/{model_id}/analyze",
    response_model=ModelExplainabilityResponse,
)
def analyze_model_explainability(
    model_id: str,
    request: ModelExplainabilityRequest,
    service: ModelExplainabilityService = Depends(get_model_explainability_service),
) -> ModelExplainabilityResponse:
    try:
        return service.analyze(
            model_id=model_id,
            request=request,
        )
    except (
        ModelNotFoundError,
        DatasetNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (
        DatasetValidationError,
        ModelExplainabilityError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
