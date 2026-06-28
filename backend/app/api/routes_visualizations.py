from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.visualization_lab_schema import (
    VisualizationChartBuilderRequest,
    VisualizationLabRequest,
    VisualizationLabResponse,
    VisualizationSpec,
)
from backend.app.schemas.visualization_schema import VisualizationRecommendationsResponse
from backend.app.services.analysis_context_service import (
    AnalysisContextError,
)
from backend.app.services.dataset_service import DatasetNotFoundError, DatasetValidationError
from backend.app.services.visualization_lab_service import (
    VisualizationLabError,
    VisualizationLabService,
)
from backend.app.services.visualization_service import VisualizationService

router = APIRouter(prefix="/visualizations", tags=["Visualizations"])


def get_visualization_lab_service(
    settings: Settings = Depends(get_settings),
) -> VisualizationLabService:
    # Visualization Lab v2 使用 canonical analysis context 建立圖表規格。
    return VisualizationLabService(settings)


def get_visualization_service(
    settings: Settings = Depends(get_settings),
) -> VisualizationService:
    # Visualization service 專責圖表推薦與圖表資料轉換。
    return VisualizationService(settings)


@router.get(
    "/{dataset_id}/recommendations",
    response_model=VisualizationRecommendationsResponse,
)
def get_visualization_recommendations(
    dataset_id: str,
    visualization_service: VisualizationService = Depends(get_visualization_service),
) -> VisualizationRecommendationsResponse:
    try:
        return visualization_service.get_recommendations(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/{dataset_id}/lab",
    response_model=VisualizationLabResponse,
)
def get_visualization_lab(
    dataset_id: str,
    request: VisualizationLabRequest,
    visualization_lab_service: VisualizationLabService = Depends(get_visualization_lab_service),
) -> VisualizationLabResponse:
    try:
        return visualization_lab_service.build_lab(
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
        VisualizationLabError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/build",
    response_model=VisualizationSpec,
)
def build_custom_visualization(
    dataset_id: str,
    request: VisualizationChartBuilderRequest,
    visualization_lab_service: VisualizationLabService = Depends(get_visualization_lab_service),
) -> VisualizationSpec:
    try:
        return visualization_lab_service.build_custom_chart(
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
        VisualizationLabError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
