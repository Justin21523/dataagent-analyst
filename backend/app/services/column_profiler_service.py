import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from backend.app.core.config import Settings
from backend.app.schemas.dataset_schema import (
    ColumnProfile,
    DatasetColumnsResponse,
    DatasetSchemaResponse,
    DatasetSchemaSummary,
)
from backend.app.services.dataset_service import DatasetService


class ColumnProfilerService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings)

    def profile_columns(self, dataset_id: str) -> DatasetColumnsResponse:
        # 欄位分析是 EDA、圖表推薦、ML 訓練的共同基礎。
        dataframe = self.dataset_service.load_dataset_dataframe(dataset_id)
        profiles = self._build_column_profiles(dataframe)

        return DatasetColumnsResponse(
            dataset_id=dataset_id,
            columns=profiles,
            total=len(profiles),
        )

    def summarize_schema(self, dataset_id: str) -> DatasetSchemaResponse:
        # Schema summary 提供前端總覽，也讓後續 Agent Planner 快速理解資料集。
        dataframe = self.dataset_service.load_dataset_dataframe(dataset_id)
        profiles = self._build_column_profiles(dataframe)

        type_counts: dict[str, int] = {}

        for profile in profiles:
            type_counts[profile.inferred_type] = type_counts.get(profile.inferred_type, 0) + 1

        total_cells = int(dataframe.shape[0] * dataframe.shape[1])
        missing_cell_count = int(dataframe.isna().sum().sum())
        duplicate_row_count = int(dataframe.duplicated().sum())

        target_candidates = [
            profile.name for profile in profiles if profile.semantic_role == "target_candidate"
        ]

        summary = DatasetSchemaSummary(
            dataset_id=dataset_id,
            row_count=int(dataframe.shape[0]),
            column_count=int(dataframe.shape[1]),
            type_counts=type_counts,
            missing_cell_count=missing_cell_count,
            missing_cell_ratio=self._safe_ratio(missing_cell_count, total_cells),
            duplicate_row_count=duplicate_row_count,
            duplicate_row_ratio=self._safe_ratio(duplicate_row_count, int(dataframe.shape[0])),
            target_candidates=target_candidates,
        )

        return DatasetSchemaResponse(
            summary=summary,
            columns=profiles,
        )

    def _build_column_profiles(self, dataframe: pd.DataFrame) -> list[ColumnProfile]:
        profiles = []

        for column_name in dataframe.columns:
            series = dataframe[column_name]
            inferred_type = self._infer_column_type(series)
            semantic_role = self._infer_semantic_role(str(column_name), inferred_type, series)

            profiles.append(
                ColumnProfile(
                    name=str(column_name),
                    inferred_type=inferred_type,
                    semantic_role=semantic_role,
                    missing_count=int(series.isna().sum()),
                    missing_ratio=self._safe_ratio(int(series.isna().sum()), len(series)),
                    unique_count=int(series.nunique(dropna=True)),
                    unique_ratio=self._safe_ratio(int(series.nunique(dropna=True)), len(series)),
                    sample_values=self._get_sample_values(series),
                    top_values=self._get_top_values(series),
                    numeric_stats=self._get_numeric_stats(series, inferred_type),
                    datetime_stats=self._get_datetime_stats(series, inferred_type),
                )
            )

        return profiles

    def _infer_column_type(self, series: pd.Series) -> str:
        # 先排除空欄位，避免後面型態推論被 NaN 影響。
        non_null_series = series.dropna()

        if non_null_series.empty:
            return "unknown"

        if self._is_boolean_series(non_null_series):
            return "boolean"

        if pd_types.is_numeric_dtype(non_null_series):
            return "numeric"

        if self._can_convert_to_numeric(non_null_series):
            return "numeric"

        if self._can_convert_to_datetime(non_null_series):
            return "datetime"

        unique_count = int(non_null_series.nunique(dropna=True))
        unique_ratio = self._safe_ratio(unique_count, len(non_null_series))
        average_text_length = self._average_text_length(non_null_series)

        if unique_count <= 30 or unique_ratio <= 0.35:
            return "categorical"

        if average_text_length >= 40:
            return "text"

        return "categorical"

    def _infer_semantic_role(
        self,
        column_name: str,
        inferred_type: str,
        series: pd.Series,
    ) -> str:
        # 語意角色使用欄位名稱與型態判斷，不再把所有高唯一值欄位當成 ID。
        normalized_name = column_name.lower().strip()

        target_names = {
            "target",
            "label",
            "class",
            "outcome",
            "result",
            "churn",
            "survived",
            "default",
            "is_fraud",
        }

        target_suffixes = (
            "_target",
            "_label",
            "_class",
            "_outcome",
        )

        identifier_names = {
            "id",
            "uuid",
            "guid",
            "record_id",
            "row_id",
            "customer_id",
            "user_id",
            "transaction_id",
            "order_id",
        }

        identifier_suffixes = (
            "_id",
            "_uuid",
            "_guid",
        )

        if normalized_name in target_names or normalized_name.endswith(target_suffixes):
            return "target_candidate"

        if normalized_name in identifier_names or normalized_name.endswith(identifier_suffixes):
            return "identifier"

        if inferred_type == "datetime":
            return "datetime"

        return "feature_candidate"

    def _is_boolean_series(self, series: pd.Series) -> bool:
        if pd_types.is_bool_dtype(series):
            return True

        normalized_values = {
            str(value).strip().lower() for value in series.dropna().unique().tolist()
        }

        boolean_values = {"true", "false", "yes", "no", "y", "n", "0", "1"}

        return 0 < len(normalized_values) <= 2 and normalized_values.issubset(boolean_values)

    def _can_convert_to_numeric(self, series: pd.Series) -> bool:
        # 將常見逗號移除後嘗試轉數值，支援像 1,200 這類資料。
        normalized = series.astype(str).str.replace(",", "", regex=False)
        converted = pd.to_numeric(normalized, errors="coerce")
        success_ratio = converted.notna().mean()

        return bool(success_ratio >= 0.9)

    def _can_convert_to_datetime(self, series: pd.Series) -> bool:
        # 數值欄位不做 datetime 嘗試，避免年份或編號被誤判成日期。
        if pd_types.is_numeric_dtype(series):
            return False

        sample = series.dropna().astype(str).str.strip().head(100)

        if sample.empty:
            return False

        if not sample.map(self._looks_like_datetime_value).any():
            return False

        converted = pd.to_datetime(sample, errors="coerce", format="mixed")
        success_ratio = converted.notna().mean()

        return bool(success_ratio >= 0.8)

    def _looks_like_datetime_value(self, value: str) -> bool:
        return bool(
            re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", value)
            or re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", value)
        )

    def _get_sample_values(self, series: pd.Series) -> list[Any]:
        # 取少量非空樣本，讓前端欄位卡片快速展示資料長相。
        values = series.dropna().head(5).tolist()
        return [self._to_json_safe_value(value) for value in values]

    def _get_top_values(self, series: pd.Series) -> list[dict[str, Any]]:
        # Top values 對 categorical / boolean / text 欄位都很有用。
        value_counts = series.value_counts(dropna=False).head(5)

        top_values = []

        for value, count in value_counts.items():
            top_values.append(
                {
                    "value": self._to_json_safe_value(value),
                    "count": int(count),
                }
            )

        return top_values

    def _get_numeric_stats(
        self,
        series: pd.Series,
        inferred_type: str,
    ) -> dict[str, Any] | None:
        if inferred_type != "numeric":
            return None

        numeric_series = pd.to_numeric(
            series.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        ).dropna()

        if numeric_series.empty:
            return None

        return {
            "mean": self._round_float(float(numeric_series.mean())),
            "std": self._round_float(float(numeric_series.std())),
            "min": self._round_float(float(numeric_series.min())),
            "q25": self._round_float(float(numeric_series.quantile(0.25))),
            "median": self._round_float(float(numeric_series.median())),
            "q75": self._round_float(float(numeric_series.quantile(0.75))),
            "max": self._round_float(float(numeric_series.max())),
        }

    def _get_datetime_stats(
        self,
        series: pd.Series,
        inferred_type: str,
    ) -> dict[str, Any] | None:
        if inferred_type != "datetime":
            return None

        datetime_series = pd.to_datetime(series, errors="coerce", format="mixed").dropna()

        if datetime_series.empty:
            return None

        return {
            "min": datetime_series.min().isoformat(),
            "max": datetime_series.max().isoformat(),
        }

    def _average_text_length(self, series: pd.Series) -> float:
        text_lengths = series.astype(str).str.len()

        if text_lengths.empty:
            return 0.0

        return float(text_lengths.mean())

    def _safe_ratio(self, numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0

        return round(numerator / denominator, 4)

    def _round_float(self, value: float) -> float | None:
        if np.isnan(value) or np.isinf(value):
            return None

        return round(value, 4)

    def _to_json_safe_value(self, value: Any) -> Any:
        # 將 pandas / numpy 型別轉成 JSON 友善格式，避免 API 序列化失敗。
        if pd.isna(value):
            return None

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        return value
