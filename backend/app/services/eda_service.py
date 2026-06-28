from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from backend.app.core.config import Settings
from backend.app.schemas.dataset_schema import ColumnProfile
from backend.app.schemas.eda_schema import (
    CorrelationAnalysisResponse,
    CorrelationPair,
    DuplicateAnalysisResponse,
    EdaSummaryResponse,
    MissingAnalysisResponse,
    MissingColumnSummary,
    NumericColumnStatistics,
    NumericStatisticsResponse,
    OutlierAnalysisResponse,
    OutlierColumnSummary,
)
from backend.app.services.column_profiler_service import ColumnProfilerService
from backend.app.services.dataset_service import DatasetService


class EdaService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings)
        self.column_profiler_service = ColumnProfilerService(settings)

    def get_summary(self, dataset_id: str) -> EdaSummaryResponse:
        # Summary 是 EDA Dashboard 的主入口，集中回傳前端需要的核心分析結果。
        dataframe = self.dataset_service.load_dataset_dataframe(dataset_id)
        profiles = self.column_profiler_service.profile_columns(dataset_id).columns

        missing = self.get_missing_analysis(dataset_id, dataframe, profiles)
        duplicates = self.get_duplicate_analysis(dataset_id, dataframe)
        numeric_statistics = self.get_numeric_statistics(dataset_id, dataframe, profiles)
        outliers = self.get_outlier_analysis(dataset_id, dataframe, profiles)
        correlation = self.get_correlation_analysis(dataset_id, dataframe, profiles)

        data_quality_score = self._calculate_data_quality_score(
            missing_cell_ratio=missing.missing_cell_ratio,
            duplicate_row_ratio=duplicates.duplicate_row_ratio,
            outliers=outliers,
        )

        return EdaSummaryResponse(
            dataset_id=dataset_id,
            row_count=int(dataframe.shape[0]),
            column_count=int(dataframe.shape[1]),
            data_quality_score=data_quality_score,
            data_quality_grade=self._get_quality_grade(data_quality_score),
            missing=missing,
            duplicates=duplicates,
            numeric_statistics=numeric_statistics,
            outliers=outliers,
            correlation=correlation,
            recommendations=self._build_recommendations(
                missing=missing,
                duplicates=duplicates,
                outliers=outliers,
                correlation=correlation,
            ),
            generated_at=datetime.now(UTC),
        )

    def get_missing_analysis(
        self,
        dataset_id: str,
        dataframe: pd.DataFrame | None = None,
        profiles: list[ColumnProfile] | None = None,
    ) -> MissingAnalysisResponse:
        # 缺失值是資料品質分析的第一層，也是清理策略的依據。
        dataframe = self._get_dataframe(dataset_id, dataframe)
        profiles = self._get_profiles(dataset_id, profiles)

        total_cells = int(dataframe.shape[0] * dataframe.shape[1])
        total_missing_cells = int(dataframe.isna().sum().sum())

        columns = [
            MissingColumnSummary(
                name=profile.name,
                inferred_type=profile.inferred_type,
                missing_count=profile.missing_count,
                missing_ratio=profile.missing_ratio,
            )
            for profile in profiles
        ]

        columns.sort(key=lambda column: column.missing_ratio, reverse=True)

        return MissingAnalysisResponse(
            dataset_id=dataset_id,
            total_missing_cells=total_missing_cells,
            missing_cell_ratio=self._safe_ratio(total_missing_cells, total_cells),
            columns=columns,
        )

    def get_duplicate_analysis(
        self,
        dataset_id: str,
        dataframe: pd.DataFrame | None = None,
    ) -> DuplicateAnalysisResponse:
        # duplicated() 預設只計算第二筆之後的重複列，這比較適合清理時的刪除數量。
        dataframe = self._get_dataframe(dataset_id, dataframe)

        duplicate_row_count = int(dataframe.duplicated().sum())

        return DuplicateAnalysisResponse(
            dataset_id=dataset_id,
            duplicate_row_count=duplicate_row_count,
            duplicate_row_ratio=self._safe_ratio(duplicate_row_count, int(dataframe.shape[0])),
        )

    def get_numeric_statistics(
        self,
        dataset_id: str,
        dataframe: pd.DataFrame | None = None,
        profiles: list[ColumnProfile] | None = None,
    ) -> NumericStatisticsResponse:
        # 數值統計先做成 API，後面圖表推薦與 ML preprocessing 都會用到。
        dataframe = self._get_dataframe(dataset_id, dataframe)
        profiles = self._get_profiles(dataset_id, profiles)

        numeric_columns = self._get_numeric_column_names(profiles)
        statistics = []

        for column_name in numeric_columns:
            numeric_series = self._to_numeric_series(dataframe[column_name])

            if numeric_series.empty:
                continue

            statistics.append(
                NumericColumnStatistics(
                    name=column_name,
                    count=int(numeric_series.count()),
                    mean=self._json_float(numeric_series.mean()),
                    std=self._json_float(numeric_series.std()),
                    min=self._json_float(numeric_series.min()),
                    q25=self._json_float(numeric_series.quantile(0.25)),
                    median=self._json_float(numeric_series.median()),
                    q75=self._json_float(numeric_series.quantile(0.75)),
                    max=self._json_float(numeric_series.max()),
                    skewness=self._json_float(numeric_series.skew()),
                    kurtosis=self._json_float(numeric_series.kurtosis()),
                )
            )

        return NumericStatisticsResponse(
            dataset_id=dataset_id,
            columns=statistics,
        )

    def get_outlier_analysis(
        self,
        dataset_id: str,
        dataframe: pd.DataFrame | None = None,
        profiles: list[ColumnProfile] | None = None,
    ) -> OutlierAnalysisResponse:
        # Phase 3 先用 IQR，因為它穩定、可解釋，也適合一般 EDA 場景。
        dataframe = self._get_dataframe(dataset_id, dataframe)
        profiles = self._get_profiles(dataset_id, profiles)

        numeric_columns = self._get_numeric_column_names(profiles)
        columns = []

        for column_name in numeric_columns:
            numeric_series = self._to_numeric_series(dataframe[column_name])

            if len(numeric_series) < 4:
                columns.append(
                    OutlierColumnSummary(
                        name=column_name,
                        method="iqr",
                        lower_bound=None,
                        upper_bound=None,
                        outlier_count=0,
                        outlier_ratio=0.0,
                    )
                )
                continue

            q25 = numeric_series.quantile(0.25)
            q75 = numeric_series.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            outlier_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
            outlier_count = int(outlier_mask.sum())

            columns.append(
                OutlierColumnSummary(
                    name=column_name,
                    method="iqr",
                    lower_bound=self._json_float(lower_bound),
                    upper_bound=self._json_float(upper_bound),
                    outlier_count=outlier_count,
                    outlier_ratio=self._safe_ratio(outlier_count, int(len(numeric_series))),
                )
            )

        columns.sort(key=lambda column: column.outlier_ratio, reverse=True)

        return OutlierAnalysisResponse(
            dataset_id=dataset_id,
            columns=columns,
        )

    def get_correlation_analysis(
        self,
        dataset_id: str,
        dataframe: pd.DataFrame | None = None,
        profiles: list[ColumnProfile] | None = None,
    ) -> CorrelationAnalysisResponse:
        # Correlation 只對數值欄位計算，避免類別欄位被錯誤轉換。
        dataframe = self._get_dataframe(dataset_id, dataframe)
        profiles = self._get_profiles(dataset_id, profiles)

        numeric_columns = self._get_numeric_column_names(profiles)

        if len(numeric_columns) < 2:
            return CorrelationAnalysisResponse(
                dataset_id=dataset_id,
                method="pearson",
                columns=numeric_columns,
                matrix={},
                strongest_pairs=[],
            )

        numeric_dataframe = pd.DataFrame(
            {
                column_name: self._to_numeric_series(dataframe[column_name])
                for column_name in numeric_columns
            }
        )

        correlation_dataframe = numeric_dataframe.corr(method="pearson")

        matrix = self._correlation_matrix_to_dict(correlation_dataframe)
        strongest_pairs = self._get_strongest_correlation_pairs(correlation_dataframe)

        return CorrelationAnalysisResponse(
            dataset_id=dataset_id,
            method="pearson",
            columns=numeric_columns,
            matrix=matrix,
            strongest_pairs=strongest_pairs,
        )

    def _get_numeric_column_names(self, profiles: list[ColumnProfile]) -> list[str]:
        return [
            profile.name
            for profile in profiles
            if profile.inferred_type == "numeric" and profile.semantic_role != "identifier"
        ]

    def _to_numeric_series(self, series: pd.Series) -> pd.Series:
        # 支援含逗號的數字字串，例如 1,200。
        normalized = series.astype(str).str.replace(",", "", regex=False)
        return pd.to_numeric(normalized, errors="coerce").dropna()

    def _correlation_matrix_to_dict(
        self,
        correlation_dataframe: pd.DataFrame,
    ) -> dict[str, dict[str, float | None]]:
        matrix: dict[str, dict[str, float | None]] = {}

        for row_column in correlation_dataframe.columns:
            matrix[str(row_column)] = {}

            for value_column in correlation_dataframe.columns:
                value = correlation_dataframe.loc[row_column, value_column]
                matrix[str(row_column)][str(value_column)] = self._json_float(value)

        return matrix

    def _get_strongest_correlation_pairs(
        self,
        correlation_dataframe: pd.DataFrame,
    ) -> list[CorrelationPair]:
        pairs = []
        columns = list(correlation_dataframe.columns)

        for index, column_x in enumerate(columns):
            for column_y in columns[index + 1 :]:
                correlation = correlation_dataframe.loc[column_x, column_y]

                if pd.isna(correlation):
                    continue

                pairs.append(
                    CorrelationPair(
                        column_x=str(column_x),
                        column_y=str(column_y),
                        correlation=round(float(correlation), 4),
                    )
                )

        pairs.sort(key=lambda pair: abs(pair.correlation), reverse=True)

        return pairs[:10]

    def _calculate_data_quality_score(
        self,
        missing_cell_ratio: float,
        duplicate_row_ratio: float,
        outliers: OutlierAnalysisResponse,
    ) -> float:
        # 簡單可解釋的品質分數，未來可以再加入型態錯誤、資料漂移等指標。
        average_outlier_ratio = 0.0

        if outliers.columns:
            average_outlier_ratio = sum(column.outlier_ratio for column in outliers.columns) / len(
                outliers.columns
            )

        score = 100.0
        score -= missing_cell_ratio * 40
        score -= duplicate_row_ratio * 20
        score -= average_outlier_ratio * 20

        return round(max(0.0, min(score, 100.0)), 2)

    def _get_quality_grade(self, score: float) -> str:
        if score >= 90:
            return "Excellent"

        if score >= 75:
            return "Good"

        if score >= 60:
            return "Fair"

        return "Needs Attention"

    def _build_recommendations(
        self,
        missing: MissingAnalysisResponse,
        duplicates: DuplicateAnalysisResponse,
        outliers: OutlierAnalysisResponse,
        correlation: CorrelationAnalysisResponse,
    ) -> list[str]:
        # 先用 rule-based recommendation，後面 Phase 7 再交給 LLM 產生自然語言解釋。
        recommendations = []

        if missing.missing_cell_ratio > 0:
            recommendations.append(
                "Review columns with missing values before training machine learning models."
            )

        if duplicates.duplicate_row_count > 0:
            recommendations.append(
                "Check duplicate rows and decide whether they should be removed."
            )

        high_outlier_columns = [
            column.name for column in outliers.columns if column.outlier_ratio >= 0.05
        ]

        if high_outlier_columns:
            recommendations.append(
                "Investigate high-outlier numeric columns: " + ", ".join(high_outlier_columns) + "."
            )

        if correlation.strongest_pairs:
            recommendations.append(
                "Review strongly correlated numeric features to avoid redundant signals."
            )

        if not recommendations:
            recommendations.append(
                "Dataset quality looks stable for initial exploratory data analysis."
            )

        return recommendations

    def _safe_ratio(self, numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0

        return round(numerator / denominator, 4)

    def _get_dataframe(
        self,
        dataset_id: str,
        dataframe: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if dataframe is not None:
            return dataframe

        return self.dataset_service.load_dataset_dataframe(dataset_id)

    def _get_profiles(
        self,
        dataset_id: str,
        profiles: list[ColumnProfile] | None,
    ) -> list[ColumnProfile]:
        if profiles is not None:
            return profiles

        return self.column_profiler_service.profile_columns(dataset_id).columns

    def _json_float(self, value: object) -> float | None:
        # 將 NaN / inf 轉成 None，避免 FastAPI JSON response 出錯。
        if value is None:
            return None

        try:
            float_value = float(cast(Any, value))
        except (TypeError, ValueError):
            return None

        if np.isnan(float_value) or np.isinf(float_value):
            return None

        return round(float_value, 4)
