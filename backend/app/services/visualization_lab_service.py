import math
import re
from typing import Any, cast

import numpy as np
import pandas as pd

from backend.app.core.config import Settings
from backend.app.schemas.analysis_schema import (
    AnalysisColumnSummary,
    AnalysisContextRequest,
    AnalysisContextResponse,
)
from backend.app.schemas.visualization_lab_schema import (
    VisualizationChartBuilderRequest,
    VisualizationChartType,
    VisualizationColumnOption,
    VisualizationLabRequest,
    VisualizationLabResponse,
    VisualizationLabSummary,
    VisualizationSpec,
)
from backend.app.services.analysis_context_service import (
    AnalysisContextService,
)
from backend.app.services.dataset_service import DatasetService


class VisualizationLabError(Exception):
    """Raised when a visualization specification cannot be created."""


class VisualizationLabService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings)
        self.analysis_context_service = AnalysisContextService(settings)

    def build_lab(
        self,
        dataset_id: str,
        request: VisualizationLabRequest,
    ) -> VisualizationLabResponse:
        # R2 使用 R1 canonical context，避免每個模組各自重算語意。
        dataframe = self.dataset_service.load_dataset_dataframe(dataset_id)

        context = self.analysis_context_service.build_context(
            dataset_id=dataset_id,
            request=AnalysisContextRequest(
                target_column=request.target_column,
                user_goal="Generate analytical visualizations.",
            ),
        )

        profile_map = {profile.name: profile for profile in context.column_profiles}

        selected_columns = self._resolve_selected_columns(
            dataframe=dataframe,
            profile_map=profile_map,
            requested_columns=request.selected_columns,
        )

        target_column = context.target_analysis.target_column if context.target_analysis else None

        numeric_columns = [
            column
            for column in selected_columns
            if (
                profile_map[column].inferred_type == "numeric"
                and profile_map[column].semantic_role != "identifier"
                and column != target_column
            )
        ]

        categorical_columns = [
            column
            for column in selected_columns
            if (
                profile_map[column].inferred_type in {"categorical", "boolean"}
                and profile_map[column].semantic_role != "identifier"
                and column != target_column
            )
        ]

        datetime_columns = [
            column for column in selected_columns if profile_map[column].inferred_type == "datetime"
        ]

        charts: list[VisualizationSpec] = []
        warnings: list[str] = []

        missing_bar = self._build_missing_bar(
            dataframe=dataframe,
            columns=selected_columns,
        )

        if missing_bar:
            charts.append(missing_bar)

        missing_heatmap = self._build_missing_heatmap(
            dataframe=dataframe,
            columns=selected_columns,
            sample_rows=request.sample_rows,
        )

        if missing_heatmap:
            charts.append(missing_heatmap)

        correlation_heatmap = self._build_correlation_heatmap(
            dataframe=dataframe,
            columns=numeric_columns[: request.max_numeric_columns],
        )

        if correlation_heatmap:
            charts.append(correlation_heatmap)
        elif len(numeric_columns) < 2:
            warnings.append("Correlation heatmap requires at least two numeric features.")

        boxplot = self._build_boxplot_overview(
            dataframe=dataframe,
            columns=numeric_columns[:6],
        )

        if boxplot:
            charts.append(boxplot)

        for index, column in enumerate(numeric_columns[:3]):
            histogram = self._build_histogram(
                dataframe=dataframe,
                column=column,
                priority=40 + index,
            )

            if histogram:
                charts.append(histogram)

        for index, column in enumerate(categorical_columns[:3]):
            category_chart = self._build_category_bar(
                dataframe=dataframe,
                column=column,
                max_categories=request.max_categories,
                priority=50 + index,
            )

            if category_chart:
                charts.append(category_chart)

        if target_column and context.target_analysis:
            target_chart = self._build_target_distribution(
                dataframe=dataframe,
                context=context,
                target_column=target_column,
            )

            if target_chart:
                charts.append(target_chart)

            relationship_chart = self._build_feature_target_chart(
                dataframe=dataframe,
                context=context,
                target_column=target_column,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                sample_rows=request.sample_rows,
                max_categories=request.max_categories,
            )

            if relationship_chart:
                charts.append(relationship_chart)
        else:
            warnings.append("No target column is available for target analysis charts.")

        charts.sort(key=lambda chart: chart.priority)

        if not charts:
            warnings.append("No supported visualization could be generated.")

        sampled_rows = min(
            int(dataframe.shape[0]),
            request.sample_rows,
        )

        return VisualizationLabResponse(
            dataset_id=dataset_id,
            summary=VisualizationLabSummary(
                row_count=int(dataframe.shape[0]),
                column_count=int(dataframe.shape[1]),
                sampled_rows=sampled_rows,
                target_column=target_column,
                target_candidates=context.dataset_summary.get(
                    "target_candidates",
                    [],
                ),
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                datetime_columns=datetime_columns,
                column_options=[
                    VisualizationColumnOption(
                        name=profile.name,
                        inferred_type=profile.inferred_type,
                        semantic_role=profile.semantic_role,
                    )
                    for profile in context.column_profiles
                ],
                chart_count=len(charts),
            ),
            charts=charts,
            warnings=warnings,
        )

    def build_custom_chart(
        self,
        dataset_id: str,
        request: VisualizationChartBuilderRequest,
    ) -> VisualizationSpec:
        dataframe = self.dataset_service.load_dataset_dataframe(dataset_id)

        context = self.analysis_context_service.build_context(
            dataset_id=dataset_id,
            request=AnalysisContextRequest(
                target_column=request.target_column,
                user_goal="Build a custom analytical visualization.",
            ),
        )

        profile_map = {profile.name: profile for profile in context.column_profiles}

        chart: VisualizationSpec | None = None

        if request.chart_type == "histogram":
            x_column = self._require_column(
                request.x_column,
                profile_map,
                "x_column",
            )
            self._require_numeric(x_column, profile_map)

            chart = self._build_histogram(
                dataframe=dataframe,
                column=x_column,
                priority=1,
                chart_id_prefix="custom",
            )

        elif request.chart_type == "boxplot":
            x_column = self._require_column(
                request.x_column,
                profile_map,
                "x_column",
            )
            self._require_numeric(x_column, profile_map)

            if request.group_column:
                group_column = self._require_column(
                    request.group_column,
                    profile_map,
                    "group_column",
                )

                chart = self._build_grouped_boxplot(
                    dataframe=dataframe,
                    numeric_column=x_column,
                    group_column=group_column,
                    max_categories=request.max_categories,
                    priority=1,
                    chart_id_prefix="custom",
                )
            else:
                chart = self._build_boxplot_overview(
                    dataframe=dataframe,
                    columns=[x_column],
                    priority=1,
                    chart_id_prefix="custom",
                )

        elif request.chart_type == "category_bar":
            x_column = self._require_column(
                request.x_column,
                profile_map,
                "x_column",
            )

            chart = self._build_category_bar(
                dataframe=dataframe,
                column=x_column,
                max_categories=request.max_categories,
                priority=1,
                chart_id_prefix="custom",
            )

        elif request.chart_type == "scatter":
            x_column = self._require_column(
                request.x_column,
                profile_map,
                "x_column",
            )
            y_column = self._require_column(
                request.y_column,
                profile_map,
                "y_column",
            )

            self._require_numeric(x_column, profile_map)
            self._require_numeric(y_column, profile_map)

            chart = self._build_scatter(
                dataframe=dataframe,
                x_column=x_column,
                y_column=y_column,
                sample_rows=request.sample_rows,
                priority=1,
                chart_id_prefix="custom",
            )

        elif request.chart_type == "line":
            x_column = self._require_column(
                request.x_column,
                profile_map,
                "x_column",
            )
            y_column = self._require_column(
                request.y_column,
                profile_map,
                "y_column",
            )

            if profile_map[x_column].inferred_type != "datetime":
                raise VisualizationLabError("Line chart x_column must be datetime.")

            self._require_numeric(y_column, profile_map)

            chart = self._build_line(
                dataframe=dataframe,
                datetime_column=x_column,
                numeric_column=y_column,
                priority=1,
                chart_id_prefix="custom",
            )

        elif request.chart_type == "correlation_heatmap":
            requested_columns = [
                column
                for column in [
                    request.x_column,
                    request.y_column,
                ]
                if column
            ]

            if requested_columns:
                for column in requested_columns:
                    self._require_column(
                        column,
                        profile_map,
                        "correlation column",
                    )
                    self._require_numeric(column, profile_map)

                numeric_columns = requested_columns
            else:
                numeric_columns = [
                    profile.name
                    for profile in context.column_profiles
                    if (
                        profile.inferred_type == "numeric" and profile.semantic_role != "identifier"
                    )
                ][:12]

            chart = self._build_correlation_heatmap(
                dataframe=dataframe,
                columns=numeric_columns,
                priority=1,
                chart_id_prefix="custom",
            )

        elif request.chart_type == "feature_target":
            target_column = request.target_column or (
                context.target_analysis.target_column if context.target_analysis else None
            )

            if not target_column:
                raise VisualizationLabError("feature_target chart requires a target column.")

            selected_numeric = []

            if request.x_column:
                x_column = self._require_column(
                    request.x_column,
                    profile_map,
                    "x_column",
                )
                self._require_numeric(x_column, profile_map)
                selected_numeric = [x_column]
            else:
                selected_numeric = [
                    profile.name
                    for profile in context.column_profiles
                    if (
                        profile.inferred_type == "numeric"
                        and profile.semantic_role != "identifier"
                        and profile.name != target_column
                    )
                ]

            categorical_columns = [
                profile.name
                for profile in context.column_profiles
                if (
                    profile.inferred_type in {"categorical", "boolean"}
                    and profile.semantic_role != "identifier"
                    and profile.name != target_column
                )
            ]

            chart = self._build_feature_target_chart(
                dataframe=dataframe,
                context=context,
                target_column=target_column,
                numeric_columns=selected_numeric,
                categorical_columns=categorical_columns,
                sample_rows=request.sample_rows,
                max_categories=request.max_categories,
                priority=1,
                chart_id_prefix="custom",
            )

        if chart is None:
            raise VisualizationLabError("The requested chart could not be generated.")

        return chart

    def _build_missing_bar(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
        priority: int = 10,
        chart_id_prefix: str = "auto",
    ) -> VisualizationSpec | None:
        rows: list[tuple[str, int, float]] = []

        for column in columns:
            missing_count = int(dataframe[column].isna().sum())

            if missing_count == 0:
                continue

            rows.append(
                (
                    column,
                    missing_count,
                    round(
                        missing_count / max(len(dataframe), 1),
                        4,
                    ),
                )
            )

        if not rows:
            return None

        rows.sort(
            key=lambda row: row[1],
            reverse=True,
        )

        highest = rows[0]

        return VisualizationSpec(
            id=f"{chart_id_prefix}-missing-values",
            title="Missing Values by Column",
            description=("Columns ranked by missing value count and ratio."),
            insight=(
                f"{highest[0]} has the highest missing count ({highest[1]}, {highest[2]:.2%})."
            ),
            chart_type="missing_bar",
            chart_family="data_quality",
            priority=priority,
            columns=[row[0] for row in rows],
            data={
                "dimensions": [
                    "column",
                    "missing_count",
                    "missing_ratio",
                ],
                "source": [
                    [
                        "column",
                        "missing_count",
                        "missing_ratio",
                    ],
                    *[list(row) for row in rows],
                ],
            },
            config={
                "x_dimension": "column",
                "y_dimension": "missing_count",
                "tooltip_dimension": "missing_ratio",
            },
        )

    def _build_missing_heatmap(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
        sample_rows: int,
        priority: int = 11,
        chart_id_prefix: str = "auto",
    ) -> VisualizationSpec | None:
        missing_columns = [column for column in columns if dataframe[column].isna().any()][:18]

        if not missing_columns:
            return None

        sampled_dataframe = self._sample_dataframe(
            dataframe[missing_columns],
            sample_rows,
        )

        x_labels = missing_columns
        y_labels = [f"row-{index + 1}" for index in range(len(sampled_dataframe))]

        values = []

        for y_index, (_, row) in enumerate(sampled_dataframe.iterrows()):
            for x_index, column in enumerate(x_labels):
                values.append(
                    [
                        x_index,
                        y_index,
                        1 if pd.isna(row[column]) else 0,
                    ]
                )

        return VisualizationSpec(
            id=f"{chart_id_prefix}-missing-heatmap",
            title="Missing Value Heatmap",
            description=(
                "Sampled row-level missing value pattern. "
                "Highlighted cells represent missing values."
            ),
            insight=(
                f"{len(missing_columns)} column(s) contain missing values "
                f"within a {len(sampled_dataframe)}-row sample."
            ),
            chart_type="missing_heatmap",
            chart_family="data_quality",
            priority=priority,
            columns=missing_columns,
            data={
                "x_labels": x_labels,
                "y_labels": y_labels,
                "values": values,
            },
            config={
                "minimum": 0,
                "maximum": 1,
            },
        )

    def _build_correlation_heatmap(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
        priority: int = 20,
        chart_id_prefix: str = "auto",
    ) -> VisualizationSpec | None:
        unique_columns = list(dict.fromkeys(columns))

        if len(unique_columns) < 2:
            return None

        numeric_dataframe = pd.DataFrame(
            {column: self._to_numeric_series(dataframe[column]) for column in unique_columns}
        )

        correlation = numeric_dataframe.corr(
            method="pearson",
            min_periods=2,
        )

        values = []
        strongest_pair = None
        strongest_value = -1.0

        for y_index, row_column in enumerate(unique_columns):
            for x_index, value_column in enumerate(unique_columns):
                correlation_value = self._safe_float(
                    correlation.loc[
                        row_column,
                        value_column,
                    ]
                )

                values.append(
                    [
                        x_index,
                        y_index,
                        correlation_value,
                    ]
                )

                if (
                    x_index > y_index
                    and correlation_value is not None
                    and abs(correlation_value) > strongest_value
                ):
                    strongest_value = abs(correlation_value)
                    strongest_pair = (
                        row_column,
                        value_column,
                        correlation_value,
                    )

        if strongest_pair:
            insight = (
                f"The strongest pair is {strongest_pair[0]} and "
                f"{strongest_pair[1]} "
                f"(r={strongest_pair[2]:.4f})."
            )
        else:
            insight = "No valid non-diagonal correlation pair was available."

        return VisualizationSpec(
            id=f"{chart_id_prefix}-correlation-heatmap",
            title="Correlation Heatmap",
            description=("Pearson correlation matrix for numeric features."),
            insight=insight,
            chart_type="correlation_heatmap",
            chart_family="relationship",
            priority=priority,
            columns=unique_columns,
            data={
                "x_labels": unique_columns,
                "y_labels": unique_columns,
                "values": values,
            },
            config={
                "minimum": -1,
                "maximum": 1,
            },
        )

    def _build_boxplot_overview(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
        priority: int = 30,
        chart_id_prefix: str = "auto",
    ) -> VisualizationSpec | None:
        categories: list[str] = []
        boxes: list[list[float]] = []
        outliers: list[list[int | float]] = []

        for column in columns:
            numeric_series = self._to_numeric_series(dataframe[column]).dropna()

            statistics = self._box_statistics(numeric_series)

            if statistics is None:
                continue

            category_index = len(categories)
            categories.append(column)
            boxes.append(statistics["box"])

            for value in statistics["outliers"]:
                outliers.append(
                    [
                        category_index,
                        value,
                    ]
                )

        if not categories:
            return None

        return VisualizationSpec(
            id=f"{chart_id_prefix}-numeric-boxplot",
            title="Numeric Feature Boxplots",
            description=("Distribution, spread, and IQR outliers for numeric features."),
            insight=(
                f"{len(categories)} numeric feature(s) are compared; "
                f"{len(outliers)} sampled IQR outlier point(s) are shown."
            ),
            chart_type="boxplot",
            chart_family="distribution",
            priority=priority,
            columns=categories,
            data={
                "categories": categories,
                "boxes": boxes,
                "outliers": outliers[:300],
            },
            config={},
        )

    def _build_histogram(
        self,
        dataframe: pd.DataFrame,
        column: str,
        priority: int,
        chart_id_prefix: str = "auto",
    ) -> VisualizationSpec | None:
        numeric_series = self._to_numeric_series(dataframe[column]).dropna()

        if numeric_series.nunique() < 2:
            return None

        bin_edges = np.histogram_bin_edges(
            numeric_series.to_numpy(),
            bins="auto",
        )

        if len(bin_edges) > 21:
            bin_edges = np.linspace(
                float(numeric_series.min()),
                float(numeric_series.max()),
                13,
            )

        counts, edges = np.histogram(
            numeric_series.to_numpy(),
            bins=bin_edges,
        )

        rows = [
            [
                (f"{self._format_number(edges[index])} – {self._format_number(edges[index + 1])}"),
                int(count),
                self._safe_float(edges[index]),
                self._safe_float(edges[index + 1]),
            ]
            for index, count in enumerate(counts)
        ]

        skewness = self._safe_float(numeric_series.skew())

        if skewness is None:
            insight = "Distribution skewness could not be calculated."
        elif abs(skewness) >= 1:
            insight = f"{column} is strongly skewed (skewness={skewness:.4f})."
        else:
            insight = f"{column} has limited skew (skewness={skewness:.4f})."

        return VisualizationSpec(
            id=(f"{chart_id_prefix}-histogram-{self._slugify(column)}"),
            title=f"Distribution of {column}",
            description=(f"Histogram showing the distribution of {column}."),
            insight=insight,
            chart_type="histogram",
            chart_family="distribution",
            priority=priority,
            columns=[column],
            data={
                "dimensions": [
                    "bin",
                    "count",
                    "lower_bound",
                    "upper_bound",
                ],
                "source": [
                    [
                        "bin",
                        "count",
                        "lower_bound",
                        "upper_bound",
                    ],
                    *rows,
                ],
            },
            config={
                "x_dimension": "bin",
                "y_dimension": "count",
            },
        )

    def _build_category_bar(
        self,
        dataframe: pd.DataFrame,
        column: str,
        max_categories: int,
        priority: int,
        chart_id_prefix: str = "auto",
    ) -> VisualizationSpec | None:
        labels = dataframe[column].map(self._label_value)

        value_counts = labels.value_counts(dropna=False)

        if value_counts.empty:
            return None

        top_counts = value_counts.head(max_categories)
        remaining_count = int(value_counts.iloc[max_categories:].sum())

        rows = [
            [
                str(label),
                int(count),
                round(
                    int(count) / max(len(dataframe), 1),
                    4,
                ),
            ]
            for label, count in top_counts.items()
        ]

        if remaining_count > 0:
            rows.append(
                [
                    "Other",
                    remaining_count,
                    round(
                        remaining_count / max(len(dataframe), 1),
                        4,
                    ),
                ]
            )

        dominant = rows[0]

        return VisualizationSpec(
            id=(f"{chart_id_prefix}-category-{self._slugify(column)}"),
            title=f"Category Distribution of {column}",
            description=(f"Top category counts and ratios for {column}."),
            insight=(f"{dominant[0]} is the dominant category ({dominant[1]}, {dominant[2]:.2%})."),
            chart_type="category_bar",
            chart_family="category_distribution",
            priority=priority,
            columns=[column],
            data={
                "dimensions": [
                    "category",
                    "count",
                    "ratio",
                ],
                "source": [
                    [
                        "category",
                        "count",
                        "ratio",
                    ],
                    *rows,
                ],
            },
            config={
                "x_dimension": "category",
                "y_dimension": "count",
            },
        )

    def _build_target_distribution(
        self,
        dataframe: pd.DataFrame,
        context: AnalysisContextResponse,
        target_column: str,
        priority: int = 60,
        chart_id_prefix: str = "auto",
    ) -> VisualizationSpec | None:
        target_analysis = context.target_analysis

        if target_analysis is None:
            return None

        if target_analysis.task_type == "classification":
            rows = [
                [
                    item.label,
                    item.count,
                    item.ratio,
                ]
                for item in target_analysis.class_distribution
            ]

            return VisualizationSpec(
                id=f"{chart_id_prefix}-target-distribution",
                title=f"Target Distribution: {target_column}",
                description=("Class counts and ratios for the selected target."),
                insight=(
                    f"The majority class is "
                    f"{target_analysis.majority_class} "
                    f"({target_analysis.majority_ratio or 0:.2%})."
                ),
                chart_type="target_distribution",
                chart_family="target_analysis",
                priority=priority,
                columns=[target_column],
                data={
                    "dimensions": [
                        "class",
                        "count",
                        "ratio",
                    ],
                    "source": [
                        [
                            "class",
                            "count",
                            "ratio",
                        ],
                        *rows,
                    ],
                },
                config={
                    "x_dimension": "class",
                    "y_dimension": "count",
                },
            )

        return self._build_histogram(
            dataframe=dataframe,
            column=target_column,
            priority=priority,
            chart_id_prefix=f"{chart_id_prefix}-target",
        )

    def _build_feature_target_chart(
        self,
        dataframe: pd.DataFrame,
        context: AnalysisContextResponse,
        target_column: str,
        numeric_columns: list[str],
        categorical_columns: list[str],
        sample_rows: int,
        max_categories: int,
        priority: int = 70,
        chart_id_prefix: str = "auto",
    ) -> VisualizationSpec | None:
        target_analysis = context.target_analysis

        if target_analysis is None:
            return None

        if target_analysis.task_type == "regression":
            feature_column = self._select_regression_feature(
                dataframe=dataframe,
                target_column=target_column,
                numeric_columns=numeric_columns,
            )

            if not feature_column:
                return None

            return self._build_scatter(
                dataframe=dataframe,
                x_column=feature_column,
                y_column=target_column,
                sample_rows=sample_rows,
                priority=priority,
                chart_id_prefix=f"{chart_id_prefix}-target",
                chart_type="feature_target_scatter",
                title=(f"{feature_column} vs Target {target_column}"),
            )

        feature_column = self._select_classification_feature(
            dataframe=dataframe,
            target_column=target_column,
            numeric_columns=numeric_columns,
        )

        if feature_column:
            return self._build_grouped_boxplot(
                dataframe=dataframe,
                numeric_column=feature_column,
                group_column=target_column,
                max_categories=max_categories,
                priority=priority,
                chart_id_prefix=f"{chart_id_prefix}-target",
                chart_type="target_boxplot",
            )

        if categorical_columns:
            return self._build_target_stacked_bar(
                dataframe=dataframe,
                feature_column=categorical_columns[0],
                target_column=target_column,
                max_categories=max_categories,
                priority=priority,
                chart_id_prefix=chart_id_prefix,
            )

        return None

    def _build_grouped_boxplot(
        self,
        dataframe: pd.DataFrame,
        numeric_column: str,
        group_column: str,
        max_categories: int,
        priority: int,
        chart_id_prefix: str,
        chart_type: VisualizationChartType = "target_boxplot",
    ) -> VisualizationSpec | None:
        temporary = pd.DataFrame(
            {
                "group": dataframe[group_column].map(self._label_value),
                "value": self._to_numeric_series(dataframe[numeric_column]),
            }
        ).dropna()

        if temporary.empty:
            return None

        top_groups = temporary["group"].value_counts().head(max_categories).index.tolist()

        categories: list[str] = []
        boxes: list[list[float]] = []
        outliers: list[list[int | float]] = []

        for group in top_groups:
            values = temporary.loc[
                temporary["group"] == group,
                "value",
            ].dropna()

            statistics = self._box_statistics(values)

            if statistics is None:
                continue

            category_index = len(categories)
            categories.append(str(group))
            boxes.append(statistics["box"])

            for value in statistics["outliers"]:
                outliers.append(
                    [
                        category_index,
                        value,
                    ]
                )

        if not categories:
            return None

        return VisualizationSpec(
            id=(
                f"{chart_id_prefix}-boxplot-"
                f"{self._slugify(numeric_column)}-by-"
                f"{self._slugify(group_column)}"
            ),
            title=f"{numeric_column} by {group_column}",
            description=(f"Compare {numeric_column} distributions across {group_column} groups."),
            insight=(f"{len(categories)} group distributions are compared."),
            chart_type=chart_type,
            chart_family="target_analysis",
            priority=priority,
            columns=[
                numeric_column,
                group_column,
            ],
            data={
                "categories": categories,
                "boxes": boxes,
                "outliers": outliers[:300],
            },
            config={},
        )

    def _build_target_stacked_bar(
        self,
        dataframe: pd.DataFrame,
        feature_column: str,
        target_column: str,
        max_categories: int,
        priority: int,
        chart_id_prefix: str,
    ) -> VisualizationSpec | None:
        temporary = pd.DataFrame(
            {
                "feature": dataframe[feature_column].map(self._label_value),
                "target": dataframe[target_column].map(self._label_value),
            }
        )

        top_feature_values = temporary["feature"].value_counts().head(max_categories).index.tolist()

        temporary = temporary[temporary["feature"].isin(top_feature_values)]

        cross_table = pd.crosstab(
            temporary["target"],
            temporary["feature"],
            normalize="index",
        )

        if cross_table.empty:
            return None

        categories = [str(value) for value in cross_table.index.tolist()]

        series = [
            {
                "name": str(feature_value),
                "data": [
                    self._safe_float(value) or 0.0 for value in cross_table[feature_value].tolist()
                ],
            }
            for feature_value in cross_table.columns
        ]

        return VisualizationSpec(
            id=(f"{chart_id_prefix}-target-stacked-{self._slugify(feature_column)}"),
            title=f"{feature_column} Composition by {target_column}",
            description=("Normalized categorical feature composition within each target class."),
            insight=("Category proportions differ across target classes."),
            chart_type="target_stacked_bar",
            chart_family="target_analysis",
            priority=priority,
            columns=[
                feature_column,
                target_column,
            ],
            data={
                "categories": categories,
                "series": series,
            },
            config={
                "normalized": True,
            },
        )

    def _build_scatter(
        self,
        dataframe: pd.DataFrame,
        x_column: str,
        y_column: str,
        sample_rows: int,
        priority: int,
        chart_id_prefix: str,
        chart_type: VisualizationChartType = "scatter",
        title: str | None = None,
    ) -> VisualizationSpec | None:
        temporary = pd.DataFrame(
            {
                x_column: self._to_numeric_series(dataframe[x_column]),
                y_column: self._to_numeric_series(dataframe[y_column]),
            }
        ).dropna()

        if temporary.empty:
            return None

        temporary = self._sample_dataframe(
            temporary,
            sample_rows,
        )

        source = [
            [
                x_column,
                y_column,
            ],
            *[
                [
                    self._safe_float(row[x_column]),
                    self._safe_float(row[y_column]),
                ]
                for _, row in temporary.iterrows()
            ],
        ]

        correlation = self._safe_float(temporary[x_column].corr(temporary[y_column]))

        insight = (
            f"Sample Pearson correlation is {correlation:.4f}."
            if correlation is not None
            else "Correlation could not be calculated."
        )

        return VisualizationSpec(
            id=(f"{chart_id_prefix}-scatter-{self._slugify(x_column)}-{self._slugify(y_column)}"),
            title=title or f"{x_column} vs {y_column}",
            description=(f"Sampled relationship between {x_column} and {y_column}."),
            insight=insight,
            chart_type=chart_type,
            chart_family="relationship",
            priority=priority,
            columns=[
                x_column,
                y_column,
            ],
            data={
                "dimensions": [
                    x_column,
                    y_column,
                ],
                "source": source,
            },
            config={
                "x_dimension": x_column,
                "y_dimension": y_column,
            },
        )

    def _build_line(
        self,
        dataframe: pd.DataFrame,
        datetime_column: str,
        numeric_column: str,
        priority: int,
        chart_id_prefix: str,
    ) -> VisualizationSpec | None:
        temporary = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    dataframe[datetime_column],
                    errors="coerce",
                ),
                "value": self._to_numeric_series(dataframe[numeric_column]),
            }
        ).dropna()

        if temporary.empty:
            return None

        temporary["date"] = temporary["date"].dt.floor("D")

        grouped = temporary.groupby("date")["value"].mean().sort_index()

        source = [
            [
                "date",
                "value",
            ],
            *[
                [
                    date_value.isoformat(),
                    self._safe_float(value),
                ]
                for date_value, value in grouped.items()
            ],
        ]

        return VisualizationSpec(
            id=(
                f"{chart_id_prefix}-line-"
                f"{self._slugify(datetime_column)}-"
                f"{self._slugify(numeric_column)}"
            ),
            title=f"{numeric_column} over {datetime_column}",
            description=(f"Daily average {numeric_column} over {datetime_column}."),
            insight=(f"The trend contains {len(grouped)} time point(s)."),
            chart_type="line",
            chart_family="time_trend",
            priority=priority,
            columns=[
                datetime_column,
                numeric_column,
            ],
            data={
                "dimensions": [
                    "date",
                    "value",
                ],
                "source": source,
            },
            config={
                "x_dimension": "date",
                "y_dimension": "value",
            },
        )

    def _select_classification_feature(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        numeric_columns: list[str],
    ) -> str | None:
        best_column = None
        best_score = -1.0

        for column in numeric_columns:
            temporary = pd.DataFrame(
                {
                    "feature": self._to_numeric_series(dataframe[column]),
                    "target": dataframe[target_column].map(self._label_value),
                }
            ).dropna()

            if temporary["target"].nunique() < 2:
                continue

            group_means = temporary.groupby("target")["feature"].mean()

            standard_deviation = temporary["feature"].std()

            if (
                standard_deviation is None
                or not math.isfinite(float(standard_deviation))
                or standard_deviation == 0
            ):
                continue

            score = (float(group_means.max()) - float(group_means.min())) / float(
                standard_deviation
            )

            if score > best_score:
                best_score = score
                best_column = column

        return best_column

    def _select_regression_feature(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        numeric_columns: list[str],
    ) -> str | None:
        target_series = self._to_numeric_series(dataframe[target_column])

        best_column = None
        best_score = -1.0

        for column in numeric_columns:
            feature_series = self._to_numeric_series(dataframe[column])

            correlation = self._safe_float(feature_series.corr(target_series))

            if correlation is not None and abs(correlation) > best_score:
                best_score = abs(correlation)
                best_column = column

        return best_column

    def _box_statistics(
        self,
        series: pd.Series,
    ) -> dict[str, Any] | None:
        numeric_series = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if len(numeric_series) < 4:
            return None

        q25 = float(numeric_series.quantile(0.25))
        median = float(numeric_series.median())
        q75 = float(numeric_series.quantile(0.75))
        iqr = q75 - q25

        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr

        inliers = numeric_series[(numeric_series >= lower_fence) & (numeric_series <= upper_fence)]

        if inliers.empty:
            return None

        outliers = numeric_series[(numeric_series < lower_fence) | (numeric_series > upper_fence)]

        return {
            "box": [
                self._safe_float(inliers.min()),
                self._safe_float(q25),
                self._safe_float(median),
                self._safe_float(q75),
                self._safe_float(inliers.max()),
            ],
            "outliers": [
                self._safe_float(value)
                for value in outliers.tolist()
                if self._safe_float(value) is not None
            ],
        }

    def _resolve_selected_columns(
        self,
        dataframe: pd.DataFrame,
        profile_map: dict[str, AnalysisColumnSummary],
        requested_columns: list[str],
    ) -> list[str]:
        if requested_columns:
            unknown_columns = [
                column for column in requested_columns if column not in dataframe.columns
            ]

            if unknown_columns:
                raise VisualizationLabError(
                    "Unknown visualization columns: " + ", ".join(unknown_columns)
                )

            return list(dict.fromkeys(requested_columns))

        return [
            column
            for column in dataframe.columns
            if (column in profile_map and profile_map[column].semantic_role != "identifier")
        ]

    def _require_column(
        self,
        column: str | None,
        profile_map: dict[str, AnalysisColumnSummary],
        field_name: str,
    ) -> str:
        if not column:
            raise VisualizationLabError(f"{field_name} is required.")

        if column not in profile_map:
            raise VisualizationLabError(f"Column not found: {column}")

        return column

    def _require_numeric(
        self,
        column: str,
        profile_map: dict[str, AnalysisColumnSummary],
    ) -> None:
        if profile_map[column].inferred_type != "numeric":
            raise VisualizationLabError(f"Column must be numeric: {column}")

    def _sample_dataframe(
        self,
        dataframe: pd.DataFrame,
        sample_rows: int,
    ) -> pd.DataFrame:
        if len(dataframe) <= sample_rows:
            return dataframe.copy()

        positions = np.linspace(
            0,
            len(dataframe) - 1,
            sample_rows,
            dtype=int,
        )

        return dataframe.iloc[positions].copy()

    def _to_numeric_series(
        self,
        series: pd.Series,
    ) -> pd.Series:
        normalized = series.astype(str).str.replace(
            ",",
            "",
            regex=False,
        )

        return pd.to_numeric(
            normalized,
            errors="coerce",
        )

    def _label_value(
        self,
        value: Any,
    ) -> str:
        if pd.isna(value):
            return "Missing"

        return str(value)

    def _slugify(
        self,
        value: str,
    ) -> str:
        slug = re.sub(
            r"[^a-zA-Z0-9]+",
            "-",
            value.strip().lower(),
        ).strip("-")

        return slug or "chart"

    def _format_number(
        self,
        value: float,
    ) -> str:
        return f"{value:.2f}"

    def _safe_float(
        self,
        value: object,
    ) -> float | None:
        try:
            float_value = float(cast(Any, value))
        except (TypeError, ValueError):
            return None

        if not math.isfinite(float_value):
            return None

        return round(float_value, 4)
