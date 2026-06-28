import re
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from backend.app.core.config import Settings
from backend.app.schemas.dataset_schema import ColumnProfile
from backend.app.schemas.visualization_schema import (
    ChartData,
    ChartRecommendation,
    VisualizationRecommendationsResponse,
)
from backend.app.services.column_profiler_service import ColumnProfilerService
from backend.app.services.dataset_service import DatasetService


class VisualizationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings)
        self.column_profiler_service = ColumnProfilerService(settings)

    def get_recommendations(self, dataset_id: str) -> VisualizationRecommendationsResponse:
        # Visualization Engine 會根據欄位型態與資料分布產生圖表建議。
        dataframe = self.dataset_service.load_dataset_dataframe(dataset_id)
        profiles = self.column_profiler_service.profile_columns(dataset_id).columns

        recommendations: list[ChartRecommendation] = []

        missing_chart = self._build_missing_values_chart(dataframe, profiles)

        if missing_chart:
            recommendations.append(missing_chart)

        recommendations.extend(self._build_numeric_histograms(dataframe, profiles))
        recommendations.extend(self._build_categorical_bar_charts(dataframe, profiles))
        recommendations.extend(self._build_datetime_trend_charts(dataframe, profiles))
        recommendations.extend(self._build_scatter_charts(dataframe, profiles))
        recommendations.extend(self._build_target_summary_charts(dataframe, profiles))

        recommendations.sort(key=lambda chart: chart.priority)

        return VisualizationRecommendationsResponse(
            dataset_id=dataset_id,
            recommendations=recommendations,
            total=len(recommendations),
        )

    def _build_missing_values_chart(
        self,
        dataframe: pd.DataFrame,
        profiles: list[ColumnProfile],
    ) -> ChartRecommendation | None:
        # 只在資料真的有缺失值時才推薦 missing chart，避免畫出沒資訊量的圖。
        columns_with_missing = [profile for profile in profiles if profile.missing_count > 0]

        if not columns_with_missing:
            return None

        columns_with_missing.sort(key=lambda profile: profile.missing_ratio, reverse=True)

        labels = [profile.name for profile in columns_with_missing]
        values = [profile.missing_count for profile in columns_with_missing]

        return ChartRecommendation(
            id="missing-values",
            title="Missing Values by Column",
            description="Columns with missing values ranked by missing count.",
            chart_type="bar",
            chart_family="data_quality",
            priority=10,
            columns=labels,
            reason="Missing values affect EDA quality and machine learning preprocessing.",
            data=ChartData(
                labels=labels,
                datasets=[
                    {
                        "label": "Missing Count",
                        "data": values,
                    }
                ],
                x_label="Column",
                y_label="Missing Count",
            ),
        )

    def _build_numeric_histograms(
        self,
        dataframe: pd.DataFrame,
        profiles: list[ColumnProfile],
    ) -> list[ChartRecommendation]:
        # 數值欄位用 histogram 快速看分布、偏態與極端值。
        charts = []
        numeric_profiles = self._get_numeric_profiles(profiles)[:3]

        for index, profile in enumerate(numeric_profiles):
            numeric_series = self._to_numeric_series(dataframe[profile.name])

            if numeric_series.nunique() < 2:
                continue

            counts, bin_edges = np.histogram(numeric_series, bins=min(10, numeric_series.nunique()))
            labels = [
                f"{self._format_number(bin_edges[i])} - {self._format_number(bin_edges[i + 1])}"
                for i in range(len(bin_edges) - 1)
            ]

            charts.append(
                ChartRecommendation(
                    id=f"histogram-{self._slugify(profile.name)}",
                    title=f"Distribution of {profile.name}",
                    description=f"Histogram showing the distribution of {profile.name}.",
                    chart_type="bar",
                    chart_family="distribution",
                    priority=20 + index,
                    columns=[profile.name],
                    reason=(
                        "Numeric distributions help detect skewness, outliers, and scale issues."
                    ),
                    data=ChartData(
                        labels=labels,
                        datasets=[
                            {
                                "label": profile.name,
                                "data": [int(value) for value in counts.tolist()],
                            }
                        ],
                        x_label=profile.name,
                        y_label="Frequency",
                    ),
                )
            )

        return charts

    def _build_categorical_bar_charts(
        self,
        dataframe: pd.DataFrame,
        profiles: list[ColumnProfile],
    ) -> list[ChartRecommendation]:
        # 類別欄位用 bar chart 看類別比例，適合觀察 imbalance。
        charts = []
        categorical_profiles = [
            profile
            for profile in profiles
            if profile.inferred_type in {"categorical", "boolean"}
            and profile.semantic_role != "identifier"
            and profile.unique_count <= 20
        ][:3]

        for index, profile in enumerate(categorical_profiles):
            value_counts = dataframe[profile.name].value_counts(dropna=False).head(10)

            labels = [self._label_value(value) for value in value_counts.index.tolist()]
            values = [int(value) for value in value_counts.values.tolist()]

            charts.append(
                ChartRecommendation(
                    id=f"bar-{self._slugify(profile.name)}",
                    title=f"Category Distribution of {profile.name}",
                    description=f"Top categories in {profile.name}.",
                    chart_type="bar",
                    chart_family="category_distribution",
                    priority=30 + index,
                    columns=[profile.name],
                    reason="Categorical distributions reveal imbalance and dominant groups.",
                    data=ChartData(
                        labels=labels,
                        datasets=[
                            {
                                "label": profile.name,
                                "data": values,
                            }
                        ],
                        x_label=profile.name,
                        y_label="Count",
                    ),
                )
            )

        return charts

    def _build_datetime_trend_charts(
        self,
        dataframe: pd.DataFrame,
        profiles: list[ColumnProfile],
    ) -> list[ChartRecommendation]:
        # 如果同時存在日期與數值欄位，就推薦趨勢圖。
        datetime_profiles = [profile for profile in profiles if profile.inferred_type == "datetime"]

        numeric_profiles = self._get_numeric_profiles(profiles)

        if not datetime_profiles or not numeric_profiles:
            return []

        datetime_column = datetime_profiles[0].name
        numeric_column = numeric_profiles[0].name

        date_values = [
            self._parse_datetime_value(value) for value in dataframe[datetime_column].tolist()
        ]
        numeric_values = self._to_numeric_series_with_index(dataframe[numeric_column]).tolist()
        grouped_values: dict[str, list[float]] = {}

        for date_value, numeric_value in zip(date_values, numeric_values, strict=True):
            if date_value is None or pd.isna(numeric_value):
                continue

            date_label = date_value.strftime("%Y-%m-%d")
            grouped_values.setdefault(date_label, []).append(float(numeric_value))

        if not grouped_values:
            return []

        labels = sorted(grouped_values)
        values = [
            self._json_float(sum(grouped_values[label]) / len(grouped_values[label]))
            for label in labels
        ]

        return [
            ChartRecommendation(
                id=f"trend-{self._slugify(datetime_column)}-{self._slugify(numeric_column)}",
                title=f"{numeric_column} Trend over {datetime_column}",
                description=f"Average {numeric_column} grouped by {datetime_column}.",
                chart_type="line",
                chart_family="time_trend",
                priority=40,
                columns=[datetime_column, numeric_column],
                reason="Datetime trend charts help identify temporal changes in numeric values.",
                data=ChartData(
                    labels=labels,
                    datasets=[
                        {
                            "label": f"Average {numeric_column}",
                            "data": values,
                        }
                    ],
                    x_label=datetime_column,
                    y_label=f"Average {numeric_column}",
                ),
            )
        ]

    def _build_scatter_charts(
        self,
        dataframe: pd.DataFrame,
        profiles: list[ColumnProfile],
    ) -> list[ChartRecommendation]:
        # Scatter plot 用於觀察兩個數值欄位之間的關係。
        numeric_profiles = self._get_numeric_profiles(profiles)

        if len(numeric_profiles) < 2:
            return []

        column_x, column_y = self._select_strongest_numeric_pair(dataframe, numeric_profiles)

        if not column_x or not column_y:
            return []

        scatter_dataframe = pd.DataFrame(
            {
                "x": self._to_numeric_series_with_index(dataframe[column_x]),
                "y": self._to_numeric_series_with_index(dataframe[column_y]),
            }
        ).dropna()

        if scatter_dataframe.empty:
            return []

        points = [
            {
                "x": self._json_float(row["x"]),
                "y": self._json_float(row["y"]),
            }
            for _, row in scatter_dataframe.iterrows()
        ]

        return [
            ChartRecommendation(
                id=f"scatter-{self._slugify(column_x)}-{self._slugify(column_y)}",
                title=f"{column_x} vs {column_y}",
                description=f"Scatter plot comparing {column_x} and {column_y}.",
                chart_type="scatter",
                chart_family="relationship",
                priority=50,
                columns=[column_x, column_y],
                reason="Scatter plots help inspect relationships between numeric features.",
                data=ChartData(
                    labels=[],
                    datasets=[
                        {
                            "label": f"{column_x} vs {column_y}",
                            "data": points,
                        }
                    ],
                    x_label=column_x,
                    y_label=column_y,
                ),
            )
        ]

    def _parse_datetime_value(self, value: Any) -> datetime | None:
        if pd.isna(value):
            return None

        normalized = str(value).strip()

        if not normalized:
            return None

        normalized = normalized.replace("Z", "+00:00")

        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            pass

        for date_format in ("%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(normalized, date_format)
            except ValueError:
                continue

        return None

    def _build_target_summary_charts(
        self,
        dataframe: pd.DataFrame,
        profiles: list[ColumnProfile],
    ) -> list[ChartRecommendation]:
        # 如果偵測到 target，先做 target 分組下的數值平均比較。
        target_profiles = [
            profile
            for profile in profiles
            if profile.semantic_role == "target_candidate"
            and profile.inferred_type in {"categorical", "boolean"}
        ]

        numeric_profiles = self._get_numeric_profiles(profiles)

        if not target_profiles or not numeric_profiles:
            return []

        target_column = target_profiles[0].name
        numeric_column = numeric_profiles[0].name

        grouped_series = (
            dataframe.groupby(target_column, dropna=False)[numeric_column]
            .apply(lambda series: self._to_numeric_series(series).mean())
            .dropna()
        )

        if grouped_series.empty:
            return []

        labels = [self._label_value(value) for value in grouped_series.index.tolist()]
        values = [self._json_float(value) for value in grouped_series.values.tolist()]

        return [
            ChartRecommendation(
                id=f"target-summary-{self._slugify(target_column)}-{self._slugify(numeric_column)}",
                title=f"Average {numeric_column} by {target_column}",
                description=f"Compare average {numeric_column} across {target_column} groups.",
                chart_type="bar",
                chart_family="target_analysis",
                priority=60,
                columns=[target_column, numeric_column],
                reason="Target-based summaries help identify feature differences across labels.",
                data=ChartData(
                    labels=labels,
                    datasets=[
                        {
                            "label": f"Average {numeric_column}",
                            "data": values,
                        }
                    ],
                    x_label=target_column,
                    y_label=f"Average {numeric_column}",
                ),
            )
        ]

    def _get_numeric_profiles(self, profiles: list[ColumnProfile]) -> list[ColumnProfile]:
        return [
            profile
            for profile in profiles
            if profile.inferred_type == "numeric" and profile.semantic_role != "identifier"
        ]

    def _select_strongest_numeric_pair(
        self,
        dataframe: pd.DataFrame,
        numeric_profiles: list[ColumnProfile],
    ) -> tuple[str | None, str | None]:
        numeric_columns = [profile.name for profile in numeric_profiles]

        numeric_dataframe = pd.DataFrame(
            {
                column: self._to_numeric_series_with_index(dataframe[column])
                for column in numeric_columns
            }
        )

        correlation_dataframe = numeric_dataframe.corr().abs()

        best_pair: tuple[str | None, str | None] = (None, None)
        best_score = -1.0

        for index, column_x in enumerate(numeric_columns):
            for column_y in numeric_columns[index + 1 :]:
                score = correlation_dataframe.loc[column_x, column_y]

                if pd.isna(score):
                    continue

                if float(score) > best_score:
                    best_score = float(score)
                    best_pair = (column_x, column_y)

        return best_pair

    def _to_numeric_series(self, series: pd.Series) -> pd.Series:
        normalized = series.astype(str).str.replace(",", "", regex=False)
        return pd.to_numeric(normalized, errors="coerce").dropna()

    def _to_numeric_series_with_index(self, series: pd.Series) -> pd.Series:
        normalized = series.astype(str).str.replace(",", "", regex=False)
        return pd.to_numeric(normalized, errors="coerce")

    def _label_value(self, value: Any) -> str:
        if pd.isna(value):
            return "Missing"

        return str(value)

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
        return slug or "chart"

    def _format_number(self, value: float) -> str:
        return f"{value:.2f}"

    def _json_float(self, value: object) -> float | None:
        try:
            float_value = float(cast(Any, value))
        except (TypeError, ValueError):
            return None

        if np.isnan(float_value) or np.isinf(float_value):
            return None

        return round(float_value, 4)
