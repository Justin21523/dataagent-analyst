from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.dataset_schema import (
    DatasetColumnsResponse,
    DatasetDetail,
    DatasetListResponse,
    DatasetPreviewResponse,
    DatasetSchemaResponse,
    DatasetTransformPreviewResponse,
    DatasetTransformRequest,
    DatasetTransformResponse,
    DatasetUploadResponse,
    DatasetVersionListResponse,
    DatasetVersionSummary,
)
from backend.app.services.column_profiler_service import ColumnProfilerService
from backend.app.services.dataset_service import (
    DatasetNotFoundError,
    DatasetService,
    DatasetValidationError,
)

router = APIRouter(prefix="/datasets", tags=["Datasets"])


def get_dataset_service(settings: Settings = Depends(get_settings)) -> DatasetService:
    # 使用 dependency injection，方便日後測試或替換資料儲存方式。
    return DatasetService(settings)


def get_column_profiler_service(
    settings: Settings = Depends(get_settings),
) -> ColumnProfilerService:
    # 欄位分析獨立成 service，讓 EDA / ML / Agent 可重複使用。
    return ColumnProfilerService(settings)


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetUploadResponse:
    # Route 只負責接收 HTTP 檔案；實際驗證與儲存交給 service。
    try:
        file_content = await file.read()
        dataset = dataset_service.create_dataset(
            original_filename=file.filename or "dataset.csv",
            file_content=file_content,
        )
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return DatasetUploadResponse(
        message="Dataset uploaded successfully.",
        dataset=dataset,
    )


@router.get("", response_model=DatasetListResponse)
def list_datasets(
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetListResponse:
    # 回傳所有已上傳資料集，供前端列表使用。
    datasets = dataset_service.list_datasets()

    return DatasetListResponse(
        datasets=datasets,
        total=len(datasets),
    )


@router.get("/{dataset_id}/columns", response_model=DatasetColumnsResponse)
def get_dataset_columns(
    dataset_id: str,
    column_profiler_service: ColumnProfilerService = Depends(get_column_profiler_service),
) -> DatasetColumnsResponse:
    # 回傳每個欄位的 profile，供前端 Column Cards 使用。
    try:
        return column_profiler_service.profile_columns(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("/{dataset_id}/schema", response_model=DatasetSchemaResponse)
def get_dataset_schema(
    dataset_id: str,
    column_profiler_service: ColumnProfilerService = Depends(get_column_profiler_service),
) -> DatasetSchemaResponse:
    # 回傳 schema summary 與 column profiles，供 Agent / Dashboard 使用。
    try:
        return column_profiler_service.summarize_schema(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("/{dataset_id}", response_model=DatasetDetail)
def get_dataset_detail(
    dataset_id: str,
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetDetail:
    # 回傳單一 dataset 的完整 metadata。
    try:
        return dataset_service.get_dataset_detail(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def get_dataset_preview(
    dataset_id: str,
    max_rows: int = 50,
    version_id: str | None = None,
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetPreviewResponse:
    # Preview 只讀取前 N 筆，避免大型 CSV 讓前端卡住。
    try:
        return dataset_service.get_dataset_preview(
            dataset_id,
            max_rows=max_rows,
            version_id=version_id,
        )
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("/{dataset_id}/versions", response_model=DatasetVersionListResponse)
def list_dataset_versions(
    dataset_id: str,
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetVersionListResponse:
    try:
        return dataset_service.list_dataset_versions(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/{dataset_id}/versions/{version_id}",
    response_model=DatasetVersionSummary,
)
def get_dataset_version(
    dataset_id: str,
    version_id: str,
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetVersionSummary:
    try:
        return dataset_service.get_dataset_version(dataset_id, version_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/transform/preview",
    response_model=DatasetTransformPreviewResponse,
)
def preview_dataset_transform(
    dataset_id: str,
    request: DatasetTransformRequest,
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetTransformPreviewResponse:
    try:
        return dataset_service.preview_transform(dataset_id, request)
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/transform",
    response_model=DatasetTransformResponse,
    status_code=status.HTTP_201_CREATED,
)
def apply_dataset_transform(
    dataset_id: str,
    request: DatasetTransformRequest,
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetTransformResponse:
    try:
        return dataset_service.apply_transform(dataset_id, request)
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
