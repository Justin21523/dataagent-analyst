from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    id: str = Field(description="Unique dataset ID")
    name: str = Field(description="Dataset display name")
    original_filename: str = Field(description="Original uploaded filename")
    stored_filename: str = Field(description="Stored filename on server")
    file_size_bytes: int = Field(description="Uploaded file size in bytes")
    row_count: int = Field(description="Total number of rows")
    column_count: int = Field(description="Total number of columns")
    encoding: str = Field(description="Detected CSV encoding")
    status: str = Field(description="Dataset processing status")
    created_at: datetime = Field(description="Dataset creation time")
    updated_at: datetime = Field(description="Dataset update time")
    latest_version_id: str | None = Field(
        default=None,
        description="Latest dataset version ID",
    )


class DatasetDetail(DatasetSummary):
    columns: list[str] = Field(description="Column names")
    preview_rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Preview rows for quick frontend rendering",
    )


class DatasetListResponse(BaseModel):
    datasets: list[DatasetSummary] = Field(description="Dataset list")
    total: int = Field(description="Total dataset count")


class DatasetUploadResponse(BaseModel):
    message: str = Field(description="Upload result message")
    dataset: DatasetDetail = Field(description="Uploaded dataset detail")


class DatasetPreviewResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    version_id: str | None = Field(
        default=None,
        description="Dataset version ID used for the preview",
    )
    columns: list[str] = Field(description="Column names")
    rows: list[dict[str, Any]] = Field(description="Preview rows")
    row_count: int = Field(description="Total dataset rows")
    preview_row_count: int = Field(description="Returned preview row count")


class ColumnProfile(BaseModel):
    name: str = Field(description="Column name")
    inferred_type: str = Field(description="Inferred data type")
    semantic_role: str = Field(description="Suggested semantic role")
    missing_count: int = Field(description="Missing value count")
    missing_ratio: float = Field(description="Missing value ratio")
    unique_count: int = Field(description="Unique value count")
    unique_ratio: float = Field(description="Unique value ratio")
    sample_values: list[Any] = Field(description="Sample non-null values")
    top_values: list[dict[str, Any]] = Field(description="Most frequent values")
    numeric_stats: dict[str, Any] | None = Field(
        default=None,
        description="Numeric statistics if the column is numeric",
    )
    datetime_stats: dict[str, Any] | None = Field(
        default=None,
        description="Datetime statistics if the column is datetime",
    )


class DatasetColumnsResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    columns: list[ColumnProfile] = Field(description="Column profiles")
    total: int = Field(description="Total column count")


class DatasetSchemaSummary(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    row_count: int = Field(description="Total row count")
    column_count: int = Field(description="Total column count")
    type_counts: dict[str, int] = Field(description="Column type distribution")
    missing_cell_count: int = Field(description="Total missing cell count")
    missing_cell_ratio: float = Field(description="Total missing cell ratio")
    duplicate_row_count: int = Field(description="Duplicate row count")
    duplicate_row_ratio: float = Field(description="Duplicate row ratio")
    target_candidates: list[str] = Field(description="Suggested target columns")


class DatasetSchemaResponse(BaseModel):
    summary: DatasetSchemaSummary = Field(description="Dataset schema summary")
    columns: list[ColumnProfile] = Field(description="Column profiles")


class DatasetVersionSummary(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    version_id: str = Field(description="Dataset version ID")
    source_version_id: str | None = Field(
        default=None,
        description="Source version for derived datasets",
    )
    version_index: int = Field(description="Monotonic version number")
    kind: str = Field(description="original or derived")
    file_path: str = Field(description="Version CSV file path")
    row_count: int = Field(description="Version row count")
    column_count: int = Field(description="Version column count")
    columns: list[str] = Field(description="Version columns")
    content_hash: str = Field(description="Version file hash")
    recipe: dict[str, Any] | None = Field(
        default=None,
        description="Transformation recipe for derived versions",
    )
    profile_diff: dict[str, Any] = Field(
        default_factory=dict,
        description="Before/after profile difference",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Version warnings",
    )
    created_at: datetime = Field(description="Version creation time")


class DatasetVersionListResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    versions: list[DatasetVersionSummary] = Field(description="Dataset versions")
    total: int = Field(description="Version count")
    latest_version_id: str = Field(description="Latest dataset version ID")


class FillMissingRule(BaseModel):
    column: str = Field(description="Column to fill")
    strategy: str = Field(description="mean, median, mode, or constant")
    value: Any | None = Field(
        default=None,
        description="Constant fill value when strategy is constant",
    )


class CastColumnRule(BaseModel):
    column: str = Field(description="Column to cast")
    dtype: str = Field(description="numeric, datetime, string, boolean, or category")


class RenameColumnRule(BaseModel):
    source: str = Field(description="Source column name")
    target: str = Field(description="Target column name")


class IqrClipRule(BaseModel):
    column: str = Field(description="Numeric column to clip")
    factor: float = Field(default=1.5, gt=0, le=10)


class DatetimePartsRule(BaseModel):
    column: str = Field(description="Datetime column")
    parts: list[str] = Field(
        default_factory=lambda: ["year", "month", "day", "dayofweek"],
        description="Datetime parts to derive",
    )


class DatasetTransformRequest(BaseModel):
    source_version_id: str | None = Field(
        default=None,
        description="Source version; defaults to latest",
    )
    drop_columns: list[str] = Field(
        default_factory=list,
        description="Columns to remove",
    )
    rename_columns: list[RenameColumnRule] = Field(
        default_factory=list,
        description="Column rename rules",
    )
    cast_columns: list[CastColumnRule] = Field(
        default_factory=list,
        description="Column cast rules",
    )
    fill_missing: list[FillMissingRule] = Field(
        default_factory=list,
        description="Missing-value fill rules",
    )
    drop_missing_rows: list[str] = Field(
        default_factory=list,
        description="Columns whose missing rows should be removed",
    )
    drop_duplicate_rows: bool = Field(
        default=False,
        description="Drop duplicate rows",
    )
    iqr_clip: list[IqrClipRule] = Field(
        default_factory=list,
        description="IQR clipping rules",
    )
    datetime_parts: list[DatetimePartsRule] = Field(
        default_factory=list,
        description="Datetime feature derivation rules",
    )


class DatasetTransformPreviewResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    source_version_id: str = Field(description="Source dataset version")
    row_count_before: int = Field(description="Input row count")
    row_count_after: int = Field(description="Output row count")
    column_count_before: int = Field(description="Input column count")
    column_count_after: int = Field(description="Output column count")
    columns: list[str] = Field(description="Output columns")
    preview_rows: list[dict[str, Any]] = Field(description="Preview rows")
    profile_diff: dict[str, Any] = Field(description="Before/after profile difference")
    warnings: list[str] = Field(description="Transformation warnings")


class DatasetTransformResponse(DatasetTransformPreviewResponse):
    version: DatasetVersionSummary = Field(description="Created dataset version")
