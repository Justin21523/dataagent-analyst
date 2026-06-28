from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.drift_schema import (
    DriftReportListResponse,
    DriftReportRequest,
    DriftReportResponse,
)
from backend.app.services.dataset_service import DatasetNotFoundError, DatasetValidationError
from backend.app.services.drift_service import (
    DriftReportNotFoundError,
    DriftService,
    DriftServiceError,
)
from backend.app.services.model_registry_service import ModelNotFoundError

router = APIRouter(prefix="/drift", tags=["Drift Center"])


def get_drift_service(settings: Settings = Depends(get_settings)) -> DriftService:
    return DriftService(settings)


@router.post("/reports", response_model=DriftReportResponse)
def create_drift_report(
    request: DriftReportRequest,
    service: DriftService = Depends(get_drift_service),
) -> DriftReportResponse:
    try:
        return service.create_report(request)
    except (DatasetNotFoundError, ModelNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (DatasetValidationError, DriftServiceError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/reports/{report_id}", response_model=DriftReportResponse)
def get_drift_report(
    report_id: str,
    service: DriftService = Depends(get_drift_service),
) -> DriftReportResponse:
    try:
        return service.get_report(report_id)
    except DriftReportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/dataset/{dataset_id}", response_model=DriftReportListResponse)
def list_drift_reports(
    dataset_id: str,
    service: DriftService = Depends(get_drift_service),
) -> DriftReportListResponse:
    reports = service.list_reports(dataset_id)

    return DriftReportListResponse(
        dataset_id=dataset_id,
        reports=reports,
        total=len(reports),
    )
