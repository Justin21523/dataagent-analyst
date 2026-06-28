from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.ai_insight_schema import (
    AIInsightRequest,
    AIInsightResponse,
    LLMStatusResponse,
)
from backend.app.services.ai_insight_service import AIInsightService
from backend.app.services.dataset_service import DatasetNotFoundError, DatasetValidationError
from backend.app.services.model_registry_service import ModelNotFoundError
from backend.app.services.report_service import ReportServiceError

router = APIRouter(prefix="/ai-insights", tags=["AI Insights"])


def get_ai_insight_service(settings: Settings = Depends(get_settings)) -> AIInsightService:
    # AI Insight service 負責整合 deterministic results 與 optional local LLM。
    return AIInsightService(settings)


@router.get("/status", response_model=LLMStatusResponse)
def get_llm_status(
    ai_insight_service: AIInsightService = Depends(get_ai_insight_service),
) -> LLMStatusResponse:
    return ai_insight_service.get_llm_status()


@router.post("/{dataset_id}/eda", response_model=AIInsightResponse)
def explain_eda(
    dataset_id: str,
    request: AIInsightRequest,
    ai_insight_service: AIInsightService = Depends(get_ai_insight_service),
) -> AIInsightResponse:
    try:
        return ai_insight_service.explain_eda(dataset_id, request.user_goal)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/{dataset_id}/models/{model_id}", response_model=AIInsightResponse)
def explain_model(
    dataset_id: str,
    model_id: str,
    request: AIInsightRequest,
    ai_insight_service: AIInsightService = Depends(get_ai_insight_service),
) -> AIInsightResponse:
    try:
        return ai_insight_service.explain_model(dataset_id, model_id, request.user_goal)
    except (DatasetNotFoundError, ModelNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/{dataset_id}/report-summary", response_model=AIInsightResponse)
def summarize_report(
    dataset_id: str,
    request: AIInsightRequest,
    ai_insight_service: AIInsightService = Depends(get_ai_insight_service),
) -> AIInsightResponse:
    try:
        return ai_insight_service.summarize_report(dataset_id, request.user_goal)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (DatasetValidationError, ReportServiceError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/{dataset_id}/models/{model_id}/explainability",
    response_model=AIInsightResponse,
)
def explain_model_explainability(
    dataset_id: str,
    model_id: str,
    request: AIInsightRequest,
    ai_insight_service: AIInsightService = Depends(get_ai_insight_service),
) -> AIInsightResponse:
    try:
        return ai_insight_service.explain_model_explainability(
            dataset_id=dataset_id,
            model_id=model_id,
            user_goal=request.user_goal,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
