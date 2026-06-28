import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from backend.app.core.config import Settings
from backend.app.repositories.metadata_repository import (
    MetadataRepositoryError,
    create_metadata_repository,
)
from backend.app.schemas.dataset_schema import (
    DatasetDetail,
    DatasetPreviewResponse,
    DatasetSummary,
    DatasetTransformPreviewResponse,
    DatasetTransformRequest,
    DatasetTransformResponse,
    DatasetVersionListResponse,
    DatasetVersionSummary,
)


class DatasetServiceError(Exception):
    """Dataset service base exception."""


class DatasetValidationError(DatasetServiceError):
    """Raised when uploaded dataset is invalid."""


class DatasetNotFoundError(DatasetServiceError):
    """Raised when dataset cannot be found."""


class DatasetService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metadata_repository = create_metadata_repository(settings)

    def create_dataset(self, original_filename: str, file_content: bytes) -> DatasetDetail:
        # 上傳入口集中在 service，route 只負責 HTTP request / response。
        self._validate_upload(original_filename, file_content)

        dataset_id = uuid4().hex
        safe_filename = self._sanitize_filename(original_filename)
        stored_filename = f"{dataset_id}_{safe_filename}"
        file_path = self.settings.raw_data_dir / stored_filename

        file_path.write_bytes(file_content)

        try:
            dataframe, encoding = self._read_csv_with_supported_encoding(file_path)
        except DatasetServiceError:
            # 解析失敗時刪除已寫入檔案，避免留下無效資料。
            file_path.unlink(missing_ok=True)
            raise

        now = datetime.now(UTC)
        columns = [str(column) for column in dataframe.columns]
        preview_rows = self._dataframe_to_records(dataframe, max_rows=20)

        dataset_record = {
            "id": dataset_id,
            "name": Path(original_filename).stem,
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "file_path": str(file_path.relative_to(self.settings.project_root)),
            "file_size_bytes": len(file_content),
            "content_hash": hashlib.sha256(file_content).hexdigest(),
            "row_count": int(dataframe.shape[0]),
            "column_count": int(dataframe.shape[1]),
            "columns": columns,
            "encoding": encoding,
            "status": "ready",
            "latest_version_id": "v1",
            "preview_rows": preview_rows,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        registry = self._load_registry()
        registry["datasets"].append(dataset_record)
        self._save_registry(registry)

        self._create_initial_version(
            dataset_record=dataset_record,
            dataframe=dataframe,
            created_at=now,
        )

        return DatasetDetail.model_validate(dataset_record)

    def list_datasets(self) -> list[DatasetSummary]:
        # Dataset list 不回傳 preview rows，避免列表頁 payload 太肥。
        registry = self._load_registry()

        datasets = [
            DatasetSummary.model_validate(dataset_record) for dataset_record in registry["datasets"]
        ]

        return sorted(datasets, key=lambda dataset: dataset.created_at, reverse=True)

    def get_dataset_detail(self, dataset_id: str) -> DatasetDetail:
        dataset_record = self._find_dataset_record(dataset_id)
        return DatasetDetail.model_validate(dataset_record)

    def get_dataset_preview(
        self,
        dataset_id: str,
        max_rows: int = 50,
        version_id: str | None = None,
    ) -> DatasetPreviewResponse:
        version_record = self._resolve_version_record(dataset_id, version_id)
        file_path = self._resolve_version_path(version_record)

        if not file_path.exists():
            raise DatasetValidationError("Dataset file is missing on server.")

        try:
            dataframe = pd.read_csv(
                file_path,
                encoding=version_record.get("encoding", "utf-8"),
                nrows=max_rows,
                low_memory=False,
            )
        except Exception as exc:
            raise DatasetValidationError(f"Failed to read dataset preview: {exc}") from exc

        rows = self._dataframe_to_records(dataframe, max_rows=max_rows)

        return DatasetPreviewResponse(
            dataset_id=dataset_id,
            version_id=version_record["version_id"],
            columns=[str(column) for column in dataframe.columns],
            rows=rows,
            row_count=int(version_record["row_count"]),
            preview_row_count=len(rows),
        )

    def get_dataset_file_path(self, dataset_id: str) -> Path:
        # 對外提供安全的 dataset 檔案路徑查詢，避免其他 service 直接讀 registry。
        dataset_record = self._find_dataset_record(dataset_id)
        return self._resolve_dataset_path(dataset_record)

    def load_dataset_dataframe(
        self,
        dataset_id: str,
        nrows: int | None = None,
        version_id: str | None = None,
    ) -> pd.DataFrame:
        # 統一讀取 dataset，後續 EDA / ML / Agent 都從這個方法進資料。
        version_record = self._resolve_version_record(dataset_id, version_id)
        file_path = self._resolve_version_path(version_record)

        if not file_path.exists():
            raise DatasetValidationError("Dataset file is missing on server.")

        try:
            return pd.read_csv(
                file_path,
                encoding=version_record.get("encoding", "utf-8"),
                nrows=nrows,
                low_memory=False,
            )
        except Exception as exc:
            raise DatasetValidationError(f"Failed to load dataset: {exc}") from exc

    def list_dataset_versions(self, dataset_id: str) -> DatasetVersionListResponse:
        self._find_dataset_record(dataset_id)
        versions = self._list_version_records(dataset_id)

        if not versions:
            dataset_record = self._find_dataset_record(dataset_id)
            versions = [self._build_virtual_initial_version(dataset_record)]

        version_summaries = [
            DatasetVersionSummary.model_validate(version_record)
            for version_record in sorted(versions, key=lambda record: record["version_index"])
        ]

        latest_version_id = version_summaries[-1].version_id

        return DatasetVersionListResponse(
            dataset_id=dataset_id,
            versions=version_summaries,
            total=len(version_summaries),
            latest_version_id=latest_version_id,
        )

    def get_dataset_version(
        self,
        dataset_id: str,
        version_id: str,
    ) -> DatasetVersionSummary:
        return DatasetVersionSummary.model_validate(
            self._resolve_version_record(dataset_id, version_id)
        )

    def preview_transform(
        self,
        dataset_id: str,
        request: DatasetTransformRequest,
    ) -> DatasetTransformPreviewResponse:
        source_version = self._resolve_version_record(dataset_id, request.source_version_id)
        dataframe = self.load_dataset_dataframe(
            dataset_id,
            version_id=source_version["version_id"],
        )
        transformed, warnings = self._apply_transform_recipe(dataframe, request)
        profile_diff = self._build_profile_diff(dataframe, transformed)

        return DatasetTransformPreviewResponse(
            dataset_id=dataset_id,
            source_version_id=source_version["version_id"],
            row_count_before=int(dataframe.shape[0]),
            row_count_after=int(transformed.shape[0]),
            column_count_before=int(dataframe.shape[1]),
            column_count_after=int(transformed.shape[1]),
            columns=[str(column) for column in transformed.columns],
            preview_rows=self._dataframe_to_records(transformed, max_rows=20),
            profile_diff=profile_diff,
            warnings=warnings,
        )

    def apply_transform(
        self,
        dataset_id: str,
        request: DatasetTransformRequest,
    ) -> DatasetTransformResponse:
        preview = self.preview_transform(dataset_id, request)
        transformed = self._dataframe_from_preview_source(
            dataset_id=dataset_id,
            source_version_id=preview.source_version_id,
            request=request,
        )

        version_record = self._save_derived_version(
            dataset_id=dataset_id,
            source_version_id=preview.source_version_id,
            dataframe=transformed,
            request=request,
            profile_diff=preview.profile_diff,
            warnings=preview.warnings,
        )

        return DatasetTransformResponse(
            **preview.model_dump(),
            version=DatasetVersionSummary.model_validate(version_record),
        )

    def _validate_upload(self, original_filename: str, file_content: bytes) -> None:
        # Phase 1 先限制 CSV 與大小；後面再補更細的 schema validation。
        if not original_filename:
            raise DatasetValidationError("Filename is required.")

        if Path(original_filename).suffix.lower() != ".csv":
            raise DatasetValidationError("Only CSV files are supported in Phase 1.")

        if not file_content:
            raise DatasetValidationError("Uploaded file is empty.")

        max_size_bytes = self.settings.max_upload_size_mb * 1024 * 1024

        if len(file_content) > max_size_bytes:
            raise DatasetValidationError(
                f"File size exceeds {self.settings.max_upload_size_mb} MB limit."
            )

    def _sanitize_filename(self, filename: str) -> str:
        # 檔名只保留安全字元，避免空白、中文、特殊符號造成路徑問題。
        file_path = Path(filename)
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", file_path.stem).strip("._-")
        safe_stem = safe_stem or "dataset"
        safe_suffix = file_path.suffix.lower()

        return f"{safe_stem}{safe_suffix}"

    def _read_csv_with_supported_encoding(self, file_path: Path) -> tuple[pd.DataFrame, str]:
        # 依序嘗試常見編碼；台灣資料有時會遇到 big5 / cp950。
        supported_encodings = ["utf-8", "utf-8-sig", "big5", "cp950", "latin1"]
        last_error: Exception | None = None

        for encoding in supported_encodings:
            try:
                dataframe = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                self._validate_dataframe(dataframe)
                return dataframe, encoding
            except UnicodeDecodeError as exc:
                last_error = exc
            except pd.errors.ParserError as exc:
                raise DatasetValidationError(f"Failed to parse CSV file: {exc}") from exc
            except Exception as exc:
                last_error = exc

        raise DatasetValidationError(f"Failed to read CSV file: {last_error}")

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        # 空資料集不適合進入後續 EDA / ML pipeline，直接擋下。
        if dataframe.empty:
            raise DatasetValidationError("CSV file does not contain any data rows.")

        if len(dataframe.columns) == 0:
            raise DatasetValidationError("CSV file does not contain any columns.")

    def _create_initial_version(
        self,
        dataset_record: dict[str, Any],
        dataframe: pd.DataFrame,
        created_at: datetime,
    ) -> None:
        record = {
            "dataset_id": dataset_record["id"],
            "version_id": "v1",
            "source_version_id": None,
            "version_index": 1,
            "kind": "original",
            "file_path": dataset_record["file_path"],
            "row_count": int(dataframe.shape[0]),
            "column_count": int(dataframe.shape[1]),
            "columns": [str(column) for column in dataframe.columns],
            "content_hash": dataset_record["content_hash"],
            "encoding": dataset_record["encoding"],
            "recipe": None,
            "profile_diff": {},
            "warnings": [],
            "created_at": created_at.isoformat(),
        }

        registry = self._load_version_registry()
        registry["versions"].append(record)
        self._save_version_registry(registry)

    def _save_derived_version(
        self,
        dataset_id: str,
        source_version_id: str,
        dataframe: pd.DataFrame,
        request: DatasetTransformRequest,
        profile_diff: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any]:
        versions = self._list_version_records(dataset_id)
        next_index = max([record["version_index"] for record in versions] or [1]) + 1
        version_id = f"v{next_index}"
        filename = f"{dataset_id}_{version_id}_derived.csv"
        file_path = self.settings.processed_data_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(file_path, index=False)
        file_content = file_path.read_bytes()
        now = datetime.now(UTC)

        record = {
            "dataset_id": dataset_id,
            "version_id": version_id,
            "source_version_id": source_version_id,
            "version_index": next_index,
            "kind": "derived",
            "file_path": str(file_path.relative_to(self.settings.project_root)),
            "row_count": int(dataframe.shape[0]),
            "column_count": int(dataframe.shape[1]),
            "columns": [str(column) for column in dataframe.columns],
            "content_hash": hashlib.sha256(file_content).hexdigest(),
            "encoding": "utf-8",
            "recipe": request.model_dump(mode="json"),
            "profile_diff": profile_diff,
            "warnings": warnings,
            "created_at": now.isoformat(),
        }

        registry = self._load_version_registry()
        registry["versions"].append(record)
        self._save_version_registry(registry)
        self._update_dataset_latest_version(dataset_id, version_id, dataframe, now)

        return record

    def _update_dataset_latest_version(
        self,
        dataset_id: str,
        version_id: str,
        dataframe: pd.DataFrame,
        updated_at: datetime,
    ) -> None:
        registry = self._load_registry()

        for record in registry["datasets"]:
            if record["id"] == dataset_id:
                record["latest_version_id"] = version_id
                record["row_count"] = int(dataframe.shape[0])
                record["column_count"] = int(dataframe.shape[1])
                record["columns"] = [str(column) for column in dataframe.columns]
                record["preview_rows"] = self._dataframe_to_records(dataframe, max_rows=20)
                record["updated_at"] = updated_at.isoformat()
                self._save_registry(registry)
                return

    def _dataframe_from_preview_source(
        self,
        dataset_id: str,
        source_version_id: str,
        request: DatasetTransformRequest,
    ) -> pd.DataFrame:
        dataframe = self.load_dataset_dataframe(dataset_id, version_id=source_version_id)
        transformed, _ = self._apply_transform_recipe(dataframe, request)
        return transformed

    def _apply_transform_recipe(
        self,
        dataframe: pd.DataFrame,
        request: DatasetTransformRequest,
    ) -> tuple[pd.DataFrame, list[str]]:
        transformed = dataframe.copy()
        warnings: list[str] = []

        for column in request.drop_columns:
            if column in transformed.columns:
                transformed = transformed.drop(columns=[column])
            else:
                warnings.append(f"Drop column skipped because it does not exist: {column}")

        for rename_rule in request.rename_columns:
            if rename_rule.source not in transformed.columns:
                warnings.append(
                    f"Rename skipped because source column does not exist: {rename_rule.source}"
                )
                continue
            if (
                rename_rule.target in transformed.columns
                and rename_rule.target != rename_rule.source
            ):
                raise DatasetValidationError(f"Rename target already exists: {rename_rule.target}")
            transformed = transformed.rename(columns={rename_rule.source: rename_rule.target})

        for cast_rule in request.cast_columns:
            self._require_column(transformed, cast_rule.column)
            transformed[cast_rule.column] = self._cast_series(
                transformed[cast_rule.column],
                cast_rule.dtype,
            )

        for fill_rule in request.fill_missing:
            self._require_column(transformed, fill_rule.column)
            fill_value = self._fill_value(
                transformed[fill_rule.column],
                fill_rule.strategy,
                fill_rule.value,
            )
            transformed[fill_rule.column] = transformed[fill_rule.column].fillna(fill_value)

        if request.drop_missing_rows:
            for column in request.drop_missing_rows:
                self._require_column(transformed, column)
            transformed = transformed.dropna(subset=request.drop_missing_rows)

        if request.drop_duplicate_rows:
            before = len(transformed)
            transformed = transformed.drop_duplicates()
            dropped = before - len(transformed)
            if dropped:
                warnings.append(f"Dropped {dropped} duplicate rows.")

        for clip_rule in request.iqr_clip:
            self._require_column(transformed, clip_rule.column)
            numeric = pd.to_numeric(transformed[clip_rule.column], errors="coerce")
            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                warnings.append(f"IQR clipping skipped for low-variance column: {clip_rule.column}")
                continue
            lower = q1 - clip_rule.factor * iqr
            upper = q3 + clip_rule.factor * iqr
            transformed[clip_rule.column] = numeric.clip(lower=lower, upper=upper)

        for datetime_rule in request.datetime_parts:
            self._require_column(transformed, datetime_rule.column)
            datetimes = pd.to_datetime(transformed[datetime_rule.column], errors="coerce")
            for part in datetime_rule.parts:
                output_column = f"{datetime_rule.column}_{part}"
                if part == "year":
                    transformed[output_column] = datetimes.dt.year
                elif part == "month":
                    transformed[output_column] = datetimes.dt.month
                elif part == "day":
                    transformed[output_column] = datetimes.dt.day
                elif part == "dayofweek":
                    transformed[output_column] = datetimes.dt.dayofweek
                else:
                    warnings.append(f"Datetime part skipped because it is unsupported: {part}")

        self._validate_dataframe(transformed)
        return transformed.reset_index(drop=True), warnings

    def _require_column(self, dataframe: pd.DataFrame, column: str) -> None:
        if column not in dataframe.columns:
            raise DatasetValidationError(f"Column not found: {column}")

    def _cast_series(self, series: pd.Series, dtype: str) -> pd.Series:
        if dtype == "numeric":
            return pd.to_numeric(series, errors="coerce")
        if dtype == "datetime":
            return pd.to_datetime(series, errors="coerce")
        if dtype == "string":
            return series.astype("string")
        if dtype == "boolean":
            return series.map(self._to_boolean)
        if dtype == "category":
            return series.astype("category")

        raise DatasetValidationError(
            "Cast dtype must be numeric, datetime, string, boolean, or category."
        )

    def _to_boolean(self, value: Any) -> bool | None:
        if pd.isna(value):
            return None

        normalized = str(value).strip().lower()

        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False

        return None

    def _fill_value(self, series: pd.Series, strategy: str, value: Any | None) -> Any:
        if strategy == "constant":
            return value
        if strategy == "mean":
            return pd.to_numeric(series, errors="coerce").mean()
        if strategy == "median":
            return pd.to_numeric(series, errors="coerce").median()
        if strategy == "mode":
            modes = series.mode(dropna=True)
            return modes.iloc[0] if not modes.empty else value

        raise DatasetValidationError("Fill strategy must be mean, median, mode, or constant.")

    def _build_profile_diff(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
    ) -> dict[str, Any]:
        before_columns = [str(column) for column in before.columns]
        after_columns = [str(column) for column in after.columns]
        before_missing = int(before.isna().sum().sum())
        after_missing = int(after.isna().sum().sum())

        return {
            "row_delta": int(after.shape[0] - before.shape[0]),
            "column_delta": int(after.shape[1] - before.shape[1]),
            "added_columns": sorted(set(after_columns) - set(before_columns)),
            "removed_columns": sorted(set(before_columns) - set(after_columns)),
            "missing_cell_delta": after_missing - before_missing,
            "before": {
                "row_count": int(before.shape[0]),
                "column_count": int(before.shape[1]),
                "missing_cells": before_missing,
            },
            "after": {
                "row_count": int(after.shape[0]),
                "column_count": int(after.shape[1]),
                "missing_cells": after_missing,
            },
        }

    def _dataframe_to_records(
        self,
        dataframe: pd.DataFrame,
        max_rows: int,
    ) -> list[dict[str, Any]]:
        # 使用 pandas to_json 轉換，可把 NaN 穩定轉成 JSON null。
        preview = dataframe.head(max_rows)
        preview = preview.replace({np.nan: None})
        json_text = preview.to_json(orient="records", date_format="iso")

        return json.loads(json_text)

    def _load_registry(self) -> dict[str, list[dict[str, Any]]]:
        try:
            return self.metadata_repository.load_registry("datasets")
        except MetadataRepositoryError as exc:
            raise DatasetValidationError("Dataset registry file is corrupted.") from exc

    def _save_registry(self, registry: dict[str, list[dict[str, Any]]]) -> None:
        try:
            self.metadata_repository.save_registry("datasets", registry)
        except MetadataRepositoryError as exc:
            raise DatasetValidationError("Dataset registry format is invalid.") from exc

    def _load_version_registry(self) -> dict[str, list[dict[str, Any]]]:
        try:
            return self.metadata_repository.load_registry("dataset_versions")
        except MetadataRepositoryError as exc:
            raise DatasetValidationError("Dataset version registry file is corrupted.") from exc

    def _save_version_registry(self, registry: dict[str, list[dict[str, Any]]]) -> None:
        try:
            self.metadata_repository.save_registry("dataset_versions", registry)
        except MetadataRepositoryError as exc:
            raise DatasetValidationError("Dataset version registry format is invalid.") from exc

    def _find_dataset_record(self, dataset_id: str) -> dict[str, Any]:
        registry = self._load_registry()

        for dataset_record in registry["datasets"]:
            if dataset_record["id"] == dataset_id:
                return dataset_record

        raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")

    def _resolve_dataset_path(self, dataset_record: dict[str, Any]) -> Path:
        # Registry 存相對路徑，實際讀取時再轉回專案內的絕對路徑。
        return self.settings.project_root / dataset_record["file_path"]

    def _list_version_records(self, dataset_id: str) -> list[dict[str, Any]]:
        registry = self._load_version_registry()
        records = [record for record in registry["versions"] if record["dataset_id"] == dataset_id]

        if not records:
            dataset_record = self._find_dataset_record(dataset_id)
            return [self._build_virtual_initial_version(dataset_record)]

        return records

    def _resolve_version_record(
        self,
        dataset_id: str,
        version_id: str | None,
    ) -> dict[str, Any]:
        self._find_dataset_record(dataset_id)
        versions = self._list_version_records(dataset_id)

        if version_id is None:
            return max(versions, key=lambda record: record["version_index"])

        for record in versions:
            if record["version_id"] == version_id:
                return record

        raise DatasetNotFoundError(f"Dataset version not found: {dataset_id}/{version_id}")

    def _build_virtual_initial_version(self, dataset_record: dict[str, Any]) -> dict[str, Any]:
        return {
            "dataset_id": dataset_record["id"],
            "version_id": "v1",
            "source_version_id": None,
            "version_index": 1,
            "kind": "original",
            "file_path": dataset_record["file_path"],
            "row_count": int(dataset_record["row_count"]),
            "column_count": int(dataset_record["column_count"]),
            "columns": list(dataset_record["columns"]),
            "content_hash": dataset_record["content_hash"],
            "encoding": dataset_record.get("encoding", "utf-8"),
            "recipe": None,
            "profile_diff": {},
            "warnings": [],
            "created_at": dataset_record["created_at"],
        }

    def _resolve_version_path(self, version_record: dict[str, Any]) -> Path:
        return self.settings.project_root / version_record["file_path"]
