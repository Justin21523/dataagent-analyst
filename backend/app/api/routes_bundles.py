from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.bundle_schema import (
    BundleExportRequest,
    BundleExportResponse,
    BundleImportResponse,
)
from backend.app.services.bundle_service import BundleService, BundleServiceError
from backend.app.services.dataset_service import DatasetNotFoundError

router = APIRouter(prefix="/bundles", tags=["Bundles"])


def get_bundle_service(settings: Settings = Depends(get_settings)) -> BundleService:
    return BundleService(settings)


@router.post("/export", response_model=BundleExportResponse)
def export_bundle(
    request: BundleExportRequest,
    service: BundleService = Depends(get_bundle_service),
) -> BundleExportResponse:
    try:
        return service.export_bundle(request)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except BundleServiceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/import", response_model=BundleImportResponse)
async def import_bundle(
    file: UploadFile = File(...),
    service: BundleService = Depends(get_bundle_service),
) -> BundleImportResponse:
    try:
        file_content = await file.read()
        return service.import_bundle(file_content)
    except BundleServiceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
