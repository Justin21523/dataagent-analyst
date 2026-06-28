from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.analysis_schema import (
    AnalysisContextRequest,
    AnalysisContextResponse,
)
from backend.app.services.analysis_context_service import (
    AnalysisContextError,
    AnalysisContextService,
)
from backend.app.services.dataset_service import (
    DatasetNotFoundError,
    DatasetValidationError,
)

router = APIRouter(
    prefix="/analysis",
    tags=["Analysis Context"],
)


def get_analysis_context_service(
    settings: Settings = Depends(get_settings),
) -> AnalysisContextService:
    # Canonical context service 統一提供 LLM、RAG、Agent 所需資料。
    return AnalysisContextService(settings)


@router.post(
    "/{dataset_id}/context",
    response_model=AnalysisContextResponse,
)
def build_analysis_context(
    dataset_id: str,
    request: AnalysisContextRequest,
    analysis_context_service: AnalysisContextService = Depends(get_analysis_context_service),
) -> AnalysisContextResponse:
    try:
        return analysis_context_service.build_context(
            dataset_id=dataset_id,
            request=request,
        )
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (
        DatasetValidationError,
        AnalysisContextError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
