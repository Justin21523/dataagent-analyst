from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.report_schema import (
    ReportDetail,
    ReportGenerateResponse,
    ReportListResponse,
)
from backend.app.services.dataset_service import DatasetNotFoundError, DatasetValidationError
from backend.app.services.report_service import (
    ReportNotFoundError,
    ReportService,
    ReportServiceError,
)

router = APIRouter(prefix="/reports", tags=["Reports"])


def get_report_service(settings: Settings = Depends(get_settings)) -> ReportService:
    # Report service 負責整合 dataset、EDA、visualization、ML 結果。
    return ReportService(settings)


@router.post("/{dataset_id}/generate", response_model=ReportGenerateResponse)
def generate_report(
    dataset_id: str,
    report_service: ReportService = Depends(get_report_service),
) -> ReportGenerateResponse:
    try:
        report = report_service.generate_report(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (DatasetValidationError, ReportServiceError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return ReportGenerateResponse(
        message="Report generated successfully.",
        report=report,
    )


@router.get("/dataset/{dataset_id}", response_model=ReportListResponse)
def list_reports(
    dataset_id: str,
    report_service: ReportService = Depends(get_report_service),
) -> ReportListResponse:
    reports = report_service.list_reports(dataset_id)

    return ReportListResponse(
        dataset_id=dataset_id,
        reports=reports,
        total=len(reports),
    )


@router.get("/{report_id}", response_model=ReportDetail)
def get_report(
    report_id: str,
    report_service: ReportService = Depends(get_report_service),
) -> ReportDetail:
    try:
        return report_service.get_report(report_id)
    except ReportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
