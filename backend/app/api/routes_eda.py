from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.eda_schema import (
    CorrelationAnalysisResponse,
    DuplicateAnalysisResponse,
    EdaSummaryResponse,
    MissingAnalysisResponse,
    NumericStatisticsResponse,
    OutlierAnalysisResponse,
)
from backend.app.services.dataset_service import DatasetNotFoundError, DatasetValidationError
from backend.app.services.eda_service import EdaService

router = APIRouter(prefix="/eda", tags=["EDA"])


def get_eda_service(settings: Settings = Depends(get_settings)) -> EdaService:
    # EDA service 集中處理資料探索分析，route 不直接操作 DataFrame。
    return EdaService(settings)


@router.get("/{dataset_id}/summary", response_model=EdaSummaryResponse)
def get_eda_summary(
    dataset_id: str,
    eda_service: EdaService = Depends(get_eda_service),
) -> EdaSummaryResponse:
    try:
        return eda_service.get_summary(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/{dataset_id}/missing", response_model=MissingAnalysisResponse)
def get_missing_analysis(
    dataset_id: str,
    eda_service: EdaService = Depends(get_eda_service),
) -> MissingAnalysisResponse:
    try:
        return eda_service.get_missing_analysis(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/{dataset_id}/duplicates", response_model=DuplicateAnalysisResponse)
def get_duplicate_analysis(
    dataset_id: str,
    eda_service: EdaService = Depends(get_eda_service),
) -> DuplicateAnalysisResponse:
    try:
        return eda_service.get_duplicate_analysis(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/{dataset_id}/statistics", response_model=NumericStatisticsResponse)
def get_numeric_statistics(
    dataset_id: str,
    eda_service: EdaService = Depends(get_eda_service),
) -> NumericStatisticsResponse:
    try:
        return eda_service.get_numeric_statistics(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/{dataset_id}/outliers", response_model=OutlierAnalysisResponse)
def get_outlier_analysis(
    dataset_id: str,
    eda_service: EdaService = Depends(get_eda_service),
) -> OutlierAnalysisResponse:
    try:
        return eda_service.get_outlier_analysis(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/{dataset_id}/correlation", response_model=CorrelationAnalysisResponse)
def get_correlation_analysis(
    dataset_id: str,
    eda_service: EdaService = Depends(get_eda_service),
) -> CorrelationAnalysisResponse:
    try:
        return eda_service.get_correlation_analysis(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
