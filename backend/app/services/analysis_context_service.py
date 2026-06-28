import math
from datetime import UTC, datetime
from typing import Any, Literal, cast

import pandas as pd

from backend.app.core.config import Settings
from backend.app.schemas.analysis_schema import (
    AnalysisChartSummary,
    AnalysisColumnSummary,
    AnalysisContextRequest,
    AnalysisContextResponse,
    AnalysisEdaSummary,
    AnalysisFinding,
    AnalysisModelSummary,
    DataQualityAssessment,
    DataQualityDimension,
    QualityStatus,
    TargetAnalysis,
    TargetClassDistribution,
)
from backend.app.schemas.dataset_schema import ColumnProfile
from backend.app.schemas.ml_schema import MLModelResult
from backend.app.services.column_profiler_service import ColumnProfilerService
from backend.app.services.dataset_service import DatasetService
from backend.app.services.eda_service import EdaService
from backend.app.services.model_registry_service import ModelRegistryService
from backend.app.services.visualization_service import VisualizationService


class AnalysisContextError(Exception):
    """Raised when canonical analysis context cannot be created."""


class AnalysisContextService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings)
        self.column_profiler_service = ColumnProfilerService(settings)
        self.eda_service = EdaService(settings)
        self.visualization_service = VisualizationService(settings)
        self.model_registry_service = ModelRegistryService(settings)

    def build_context(
        self,
        dataset_id: str,
        request: AnalysisContextRequest,
    ) -> AnalysisContextResponse:
        # 所有 LLM、RAG、Agent 都應共用這一份 canonical context。
        dataset = self.dataset_service.get_dataset_detail(dataset_id)
        dataframe = self.dataset_service.load_dataset_dataframe(dataset_id)
        schema = self.column_profiler_service.summarize_schema(dataset_id)
        eda = self.eda_service.get_summary(dataset_id)
        visualizations = self.visualization_service.get_recommendations(dataset_id)

        target_column = self._resolve_target_column(
            dataframe=dataframe,
            requested_target=request.target_column,
            target_candidates=schema.summary.target_candidates,
        )

        target_analysis = self._build_target_analysis(
            dataframe=dataframe,
            profiles=schema.columns,
            target_column=target_column,
        )

        models = self.model_registry_service.list_models(dataset_id=dataset_id)

        if target_column:
            models = [model for model in models if model.target_column == target_column]

        column_profiles = self._build_column_summaries(
            profiles=schema.columns,
            eda=eda,
        )

        data_quality = self._build_data_quality_assessment(
            schema=schema,
            eda=eda,
        )

        eda_summary = self._build_eda_summary(eda)
        chart_summaries = self._build_chart_summaries(visualizations)
        model_summaries = self._build_model_summaries(models)
        reference_model = self._select_reference_model(models)

        feature_importance = []

        if reference_model:
            feature_importance = [
                item.model_dump() for item in reference_model.feature_importance[:15]
            ]

        findings = self._build_findings(
            row_count=int(dataframe.shape[0]),
            columns=column_profiles,
            data_quality=data_quality,
            target_analysis=target_analysis,
            eda_summary=eda_summary,
            models=models,
        )

        recommended_next_actions = self._build_next_actions(
            findings=findings,
            target_analysis=target_analysis,
            models=models,
            chart_count=len(chart_summaries),
        )

        return AnalysisContextResponse(
            analysis_context_version="1.0",
            dataset_id=dataset_id,
            user_goal=request.user_goal,
            generated_at=datetime.now(UTC),
            dataset_summary={
                "name": dataset.name,
                "original_filename": dataset.original_filename,
                "row_count": dataset.row_count,
                "column_count": dataset.column_count,
                "file_size_bytes": dataset.file_size_bytes,
                "encoding": dataset.encoding,
                "status": dataset.status,
                "type_counts": schema.summary.type_counts,
                "target_candidates": schema.summary.target_candidates,
            },
            column_profiles=column_profiles,
            data_quality=data_quality,
            target_analysis=target_analysis,
            eda_summary=eda_summary,
            chart_recommendations=chart_summaries,
            ml_results=model_summaries,
            feature_importance=feature_importance,
            rag_context=[],
            findings=findings,
            recommended_next_actions=recommended_next_actions,
        )

    def _resolve_target_column(
        self,
        dataframe: pd.DataFrame,
        requested_target: str | None,
        target_candidates: list[str],
    ) -> str | None:
        if requested_target:
            if requested_target not in dataframe.columns:
                raise AnalysisContextError(f"Target column not found: {requested_target}")

            return requested_target

        if target_candidates:
            return target_candidates[0]

        return None

    def _build_column_summaries(
        self,
        profiles: list[ColumnProfile],
        eda,
    ) -> list[AnalysisColumnSummary]:
        outlier_map = {column.name: column.outlier_ratio for column in eda.outliers.columns}

        summaries = []

        for profile in profiles:
            outlier_ratio = outlier_map.get(profile.name, 0.0)
            warnings = []

            if profile.missing_ratio >= 0.4:
                warnings.append("High missing value ratio.")
            elif profile.missing_ratio >= 0.1:
                warnings.append("Moderate missing value ratio.")

            if profile.inferred_type in {"categorical", "boolean"} and profile.unique_count > 50:
                warnings.append("High-cardinality categorical column.")

            if profile.inferred_type == "unknown":
                warnings.append("Column type requires manual review.")

            if outlier_ratio >= 0.05:
                warnings.append("Numeric column contains notable IQR outliers.")

            summaries.append(
                AnalysisColumnSummary(
                    name=profile.name,
                    inferred_type=profile.inferred_type,
                    semantic_role=profile.semantic_role,
                    role_confidence=self._role_confidence(profile),
                    missing_ratio=profile.missing_ratio,
                    unique_count=profile.unique_count,
                    unique_ratio=profile.unique_ratio,
                    cardinality=self._cardinality(profile),
                    sample_values=profile.sample_values,
                    recommended_preprocessing=self._recommended_preprocessing(
                        profile=profile,
                        outlier_ratio=outlier_ratio,
                    ),
                    warnings=warnings,
                )
            )

        return summaries

    def _build_data_quality_assessment(
        self,
        schema,
        eda,
    ) -> DataQualityAssessment:
        missing_ratio = schema.summary.missing_cell_ratio
        duplicate_ratio = schema.summary.duplicate_row_ratio
        column_count = max(schema.summary.column_count, 1)
        unknown_count = schema.summary.type_counts.get("unknown", 0)
        unknown_ratio = unknown_count / column_count

        outlier_ratios = [column.outlier_ratio for column in eda.outliers.columns]

        average_outlier_ratio = sum(outlier_ratios) / len(outlier_ratios) if outlier_ratios else 0.0

        dimensions = [
            self._quality_dimension(
                key="completeness",
                label="Completeness",
                score=100 * (1 - min(missing_ratio, 1.0)),
                weight=0.4,
                evidence=[
                    f"Missing cell ratio: {missing_ratio:.2%}",
                    f"Missing cells: {schema.summary.missing_cell_count}",
                ],
                recommended_action=(
                    "Review missing columns and define imputation or exclusion rules."
                    if missing_ratio > 0
                    else "No immediate missing-value action is required."
                ),
            ),
            self._quality_dimension(
                key="consistency",
                label="Row Consistency",
                score=100 * (1 - min(duplicate_ratio, 1.0)),
                weight=0.2,
                evidence=[
                    f"Duplicate row ratio: {duplicate_ratio:.2%}",
                    f"Duplicate rows: {schema.summary.duplicate_row_count}",
                ],
                recommended_action=(
                    "Review and remove unintended duplicate rows."
                    if duplicate_ratio > 0
                    else "No duplicate-row action is required."
                ),
            ),
            self._quality_dimension(
                key="type_stability",
                label="Type Stability",
                score=100 * (1 - min(unknown_ratio, 1.0)),
                weight=0.2,
                evidence=[
                    f"Unknown columns: {unknown_count}",
                    f"Total columns: {schema.summary.column_count}",
                ],
                recommended_action=(
                    "Review columns whose types could not be inferred."
                    if unknown_count > 0
                    else "Column types are ready for initial analysis."
                ),
            ),
            self._quality_dimension(
                key="outlier_stability",
                label="Outlier Stability",
                score=100 * (1 - min(average_outlier_ratio, 1.0)),
                weight=0.2,
                evidence=[(f"Average numeric outlier ratio: {average_outlier_ratio:.2%}")],
                recommended_action=(
                    "Investigate numeric outliers before model training."
                    if average_outlier_ratio >= 0.05
                    else "No broad outlier treatment is currently required."
                ),
            ),
        ]

        overall_score = round(
            sum(dimension.score * dimension.weight for dimension in dimensions),
            2,
        )

        primary_risks = [
            f"{dimension.label}: {dimension.score:.2f}/100"
            for dimension in dimensions
            if dimension.score < 75
        ]

        return DataQualityAssessment(
            overall_score=overall_score,
            grade=self._quality_grade(overall_score),
            dimensions=dimensions,
            primary_risks=primary_risks,
        )

    def _build_target_analysis(
        self,
        dataframe: pd.DataFrame,
        profiles: list[ColumnProfile],
        target_column: str | None,
    ) -> TargetAnalysis | None:
        if not target_column:
            return None

        profile_map = {profile.name: profile for profile in profiles}

        profile = profile_map.get(target_column)

        if profile is None:
            raise AnalysisContextError(f"Target profile not found: {target_column}")

        target_series = dataframe[target_column]
        non_null_series = target_series.dropna()

        if non_null_series.empty:
            raise AnalysisContextError(f"Target column contains no usable values: {target_column}")

        task_type = cast(
            Literal["classification", "regression"],
            self._infer_task_type(
                profile=profile,
                non_null_series=non_null_series,
            ),
        )

        readiness_score = 100.0
        warnings = []

        readiness_score -= profile.missing_ratio * 40

        if len(dataframe) < 50:
            readiness_score -= 25
            warnings.append("Dataset has fewer than 50 rows; model metrics may be unstable.")
        elif len(dataframe) < 200:
            readiness_score -= 10
            warnings.append("Dataset is relatively small; use cross-validation.")

        class_distribution = []
        majority_class = None
        majority_ratio = None
        imbalance_ratio = None
        numeric_summary = None

        if task_type == "classification":
            counts = non_null_series.astype(str).value_counts()
            total_count = int(counts.sum())

            class_distribution = [
                TargetClassDistribution(
                    label=str(label),
                    count=int(count),
                    ratio=round(int(count) / total_count, 4),
                )
                for label, count in counts.items()
            ]

            majority_class = str(counts.index[0])
            majority_count = int(counts.iloc[0])
            minority_count = int(counts.iloc[-1])
            majority_ratio = round(majority_count / total_count, 4)

            if minority_count > 0:
                imbalance_ratio = round(
                    majority_count / minority_count,
                    4,
                )

            if majority_ratio >= 0.8:
                readiness_score -= 25
                warnings.append("Severe class imbalance detected.")
            elif majority_ratio >= 0.65:
                readiness_score -= 10
                warnings.append("Moderate class imbalance detected.")

            if minority_count < 5:
                readiness_score -= 20
                warnings.append("At least one class has fewer than five samples.")

            if profile.unique_count > 20:
                readiness_score -= 15
                warnings.append("Target has many classes; verify that classification is intended.")

            recommended_metric = "f1_macro"
        else:
            numeric_target = pd.to_numeric(
                non_null_series.astype(str).str.replace(
                    ",",
                    "",
                    regex=False,
                ),
                errors="coerce",
            ).dropna()

            numeric_summary = {
                "mean": self._safe_float(numeric_target.mean()),
                "std": self._safe_float(numeric_target.std()),
                "min": self._safe_float(numeric_target.min()),
                "median": self._safe_float(numeric_target.median()),
                "max": self._safe_float(numeric_target.max()),
                "skewness": self._safe_float(numeric_target.skew()),
            }

            skewness = numeric_summary.get("skewness")

            if skewness is not None and abs(skewness) >= 2:
                readiness_score -= 10
                warnings.append("Regression target is strongly skewed.")

            if profile.unique_count < 20:
                readiness_score -= 20
                warnings.append("Regression target has few distinct values.")

            recommended_metric = "rmse"

        return TargetAnalysis(
            target_column=target_column,
            task_type=task_type,
            inferred_type=profile.inferred_type,
            non_null_count=int(non_null_series.shape[0]),
            missing_ratio=profile.missing_ratio,
            unique_count=profile.unique_count,
            readiness_score=round(
                max(0.0, min(readiness_score, 100.0)),
                2,
            ),
            recommended_primary_metric=recommended_metric,
            class_distribution=class_distribution,
            majority_class=majority_class,
            majority_ratio=majority_ratio,
            imbalance_ratio=imbalance_ratio,
            numeric_summary=numeric_summary,
            warnings=warnings,
        )

    def _build_eda_summary(self, eda) -> AnalysisEdaSummary:
        top_missing_columns = [
            {
                "name": column.name,
                "missing_count": column.missing_count,
                "missing_ratio": column.missing_ratio,
            }
            for column in eda.missing.columns
            if column.missing_count > 0
        ][:10]

        outlier_columns = [
            {
                "name": column.name,
                "outlier_count": column.outlier_count,
                "outlier_ratio": column.outlier_ratio,
                "lower_bound": column.lower_bound,
                "upper_bound": column.upper_bound,
            }
            for column in eda.outliers.columns
            if column.outlier_count > 0
        ][:10]

        strongest_correlations = [
            pair.model_dump() for pair in eda.correlation.strongest_pairs[:10]
        ]

        return AnalysisEdaSummary(
            missing_cell_count=eda.missing.total_missing_cells,
            missing_cell_ratio=eda.missing.missing_cell_ratio,
            duplicate_row_count=eda.duplicates.duplicate_row_count,
            duplicate_row_ratio=eda.duplicates.duplicate_row_ratio,
            top_missing_columns=top_missing_columns,
            outlier_columns=outlier_columns,
            strongest_correlations=strongest_correlations,
            numeric_column_count=len(eda.numeric_statistics.columns),
        )

    def _build_chart_summaries(
        self,
        visualizations,
    ) -> list[AnalysisChartSummary]:
        return [
            AnalysisChartSummary(
                id=chart.id,
                title=chart.title,
                chart_type=chart.chart_type,
                chart_family=chart.chart_family,
                columns=chart.columns,
                reason=chart.reason,
            )
            for chart in visualizations.recommendations
        ]

    def _build_model_summaries(
        self,
        models: list[MLModelResult],
    ) -> list[AnalysisModelSummary]:
        return [
            AnalysisModelSummary(
                id=model.id,
                model_name=model.model_name,
                task_type=model.task_type,
                target_column=model.target_column,
                metrics=model.metrics,
                feature_importance=[item.model_dump() for item in model.feature_importance[:10]],
                created_at=model.created_at,
            )
            for model in models[:10]
        ]

    def _build_findings(
        self,
        row_count: int,
        columns: list[AnalysisColumnSummary],
        data_quality: DataQualityAssessment,
        target_analysis: TargetAnalysis | None,
        eda_summary: AnalysisEdaSummary,
        models: list[MLModelResult],
    ) -> list[AnalysisFinding]:
        findings = []

        if row_count < 100:
            findings.append(
                AnalysisFinding(
                    id="small-sample-size",
                    category="modeling",
                    severity="warning",
                    title="Small sample size",
                    summary=(f"The dataset contains only {row_count} rows."),
                    evidence={
                        "row_count": row_count,
                    },
                    recommended_action=(
                        "Use cross-validation and treat model metrics as preliminary."
                    ),
                )
            )

        high_missing_columns = [column for column in columns if column.missing_ratio >= 0.1]

        if high_missing_columns:
            maximum_missing_ratio = max(column.missing_ratio for column in high_missing_columns)

            findings.append(
                AnalysisFinding(
                    id="missing-value-risk",
                    category="data_quality",
                    severity=("critical" if maximum_missing_ratio >= 0.4 else "warning"),
                    title="Missing values require treatment",
                    summary=(
                        f"{len(high_missing_columns)} column(s) have at least 10% missing values."
                    ),
                    evidence={
                        "columns": [
                            {
                                "name": column.name,
                                "missing_ratio": column.missing_ratio,
                            }
                            for column in high_missing_columns
                        ],
                    },
                    recommended_action=(
                        "Define column-specific imputation, exclusion, or data collection rules."
                    ),
                )
            )

        if eda_summary.duplicate_row_count > 0:
            findings.append(
                AnalysisFinding(
                    id="duplicate-row-risk",
                    category="data_quality",
                    severity="warning",
                    title="Duplicate rows detected",
                    summary=(f"{eda_summary.duplicate_row_count} duplicate row(s) were detected."),
                    evidence={
                        "duplicate_row_count": (eda_summary.duplicate_row_count),
                        "duplicate_row_ratio": (eda_summary.duplicate_row_ratio),
                    },
                    recommended_action=(
                        "Confirm whether duplicate rows are valid repeated "
                        "events or accidental duplication."
                    ),
                )
            )

        high_outlier_columns = [
            column for column in eda_summary.outlier_columns if column["outlier_ratio"] >= 0.05
        ]

        if high_outlier_columns:
            findings.append(
                AnalysisFinding(
                    id="outlier-risk",
                    category="data_quality",
                    severity="warning",
                    title="Numeric outliers may affect modeling",
                    summary=(
                        f"{len(high_outlier_columns)} numeric column(s) "
                        "have at least 5% IQR outliers."
                    ),
                    evidence={
                        "columns": high_outlier_columns,
                    },
                    recommended_action=(
                        "Inspect extreme records and compare standard versus robust preprocessing."
                    ),
                )
            )

        strong_correlations = [
            pair for pair in eda_summary.strongest_correlations if abs(pair["correlation"]) >= 0.85
        ]

        if strong_correlations:
            findings.append(
                AnalysisFinding(
                    id="strong-feature-correlation",
                    category="relationship",
                    severity="warning",
                    title="Strong feature correlations detected",
                    summary=(
                        f"{len(strong_correlations)} numeric pair(s) "
                        "have absolute correlation of at least 0.85."
                    ),
                    evidence={
                        "pairs": strong_correlations,
                    },
                    recommended_action=(
                        "Review redundant features and potential "
                        "multicollinearity before interpreting coefficients."
                    ),
                )
            )

        high_cardinality_columns = [
            column.name
            for column in columns
            if (column.inferred_type in {"categorical", "boolean"} and column.cardinality == "high")
        ]

        if high_cardinality_columns:
            findings.append(
                AnalysisFinding(
                    id="high-cardinality-categories",
                    category="schema",
                    severity="warning",
                    title="High-cardinality categorical features",
                    summary=(
                        "Some categorical columns may create too many one-hot encoded features."
                    ),
                    evidence={
                        "columns": high_cardinality_columns,
                    },
                    recommended_action=(
                        "Review rare-category grouping, target encoding, or column exclusion."
                    ),
                )
            )

        if target_analysis is None:
            findings.append(
                AnalysisFinding(
                    id="target-not-confirmed",
                    category="target",
                    severity="info",
                    title="No target column confirmed",
                    summary=(
                        "The system could not safely determine the intended prediction target."
                    ),
                    evidence={},
                    recommended_action=(
                        "Select and confirm a target column before supervised model training."
                    ),
                )
            )
        elif target_analysis.warnings:
            findings.append(
                AnalysisFinding(
                    id="target-readiness-risk",
                    category="target",
                    severity="warning",
                    title="Target requires modeling review",
                    summary=" ".join(target_analysis.warnings),
                    evidence={
                        "target_column": target_analysis.target_column,
                        "readiness_score": (target_analysis.readiness_score),
                        "majority_ratio": (target_analysis.majority_ratio),
                        "imbalance_ratio": (target_analysis.imbalance_ratio),
                    },
                    recommended_action=(
                        "Resolve target warnings before trusting model comparison results."
                    ),
                )
            )

        if target_analysis and not models:
            findings.append(
                AnalysisFinding(
                    id="model-not-trained",
                    category="modeling",
                    severity="info",
                    title="No trained model for current target",
                    summary=(
                        "The analysis context contains no saved model for "
                        f"`{target_analysis.target_column}`."
                    ),
                    evidence={
                        "target_column": (target_analysis.target_column),
                        "task_type": target_analysis.task_type,
                    },
                    recommended_action=(
                        "Run cross-validated baseline models after data "
                        "quality issues are reviewed."
                    ),
                )
            )

        severity_order = {
            "critical": 0,
            "warning": 1,
            "info": 2,
        }

        return sorted(
            findings,
            key=lambda finding: severity_order[finding.severity],
        )

    def _build_next_actions(
        self,
        findings: list[AnalysisFinding],
        target_analysis: TargetAnalysis | None,
        models: list[MLModelResult],
        chart_count: int,
    ) -> list[str]:
        actions = [
            finding.recommended_action
            for finding in findings
            if finding.severity in {"critical", "warning"}
        ]

        if target_analysis is None:
            actions.append("Confirm the business target and expected prediction task.")
        elif not models:
            actions.append("Train baseline models with cross-validation after cleaning.")
        else:
            actions.append("Compare validation metrics, error cases, and feature importance.")

        if chart_count > 0:
            actions.append("Review recommended visualizations before drawing conclusions.")

        actions.append(
            "Use this canonical analysis context as the input for LLM, RAG, and Agent workflows."
        )

        # 保留原始順序並移除重複建議。
        return list(dict.fromkeys(actions))[:8]

    def _quality_dimension(
        self,
        key: str,
        label: str,
        score: float,
        weight: float,
        evidence: list[str],
        recommended_action: str,
    ) -> DataQualityDimension:
        normalized_score = round(
            max(0.0, min(score, 100.0)),
            2,
        )

        return DataQualityDimension(
            key=key,
            label=label,
            score=normalized_score,
            weight=weight,
            status=self._quality_status(normalized_score),
            evidence=evidence,
            recommended_action=recommended_action,
        )

    def _quality_status(self, score: float) -> QualityStatus:
        if score >= 90:
            return "excellent"

        if score >= 75:
            return "good"

        if score >= 60:
            return "fair"

        return "poor"

    def _quality_grade(self, score: float) -> str:
        if score >= 90:
            return "Excellent"

        if score >= 75:
            return "Good"

        if score >= 60:
            return "Fair"

        return "Needs Attention"

    def _infer_task_type(
        self,
        profile: ColumnProfile,
        non_null_series: pd.Series,
    ) -> str:
        if profile.inferred_type != "numeric":
            return "classification"

        unique_count = int(non_null_series.nunique(dropna=True))
        regression_threshold = max(
            10,
            int(len(non_null_series) * 0.05),
        )

        if unique_count > regression_threshold:
            return "regression"

        return "classification"

    def _role_confidence(
        self,
        profile: ColumnProfile,
    ) -> float:
        if profile.semantic_role in {
            "identifier",
            "target_candidate",
        }:
            return 0.98

        if profile.semantic_role == "datetime":
            return 0.95

        if profile.inferred_type == "numeric":
            return 0.9

        if profile.inferred_type in {
            "categorical",
            "boolean",
        }:
            return 0.82

        if profile.inferred_type == "text":
            return 0.75

        return 0.3

    def _cardinality(
        self,
        profile: ColumnProfile,
    ) -> str:
        if profile.unique_count <= 2:
            return "binary"

        if profile.unique_count <= 10:
            return "low"

        if profile.unique_count <= 50 or profile.unique_ratio <= 0.5:
            return "medium"

        return "high"

    def _recommended_preprocessing(
        self,
        profile: ColumnProfile,
        outlier_ratio: float,
    ) -> list[str]:
        if profile.semantic_role == "identifier":
            return ["exclude_from_model"]

        if profile.semantic_role == "target_candidate":
            return [
                "exclude_from_features",
                "validate_target_encoding",
            ]

        if profile.inferred_type == "numeric":
            steps = [
                "median_imputer",
                "standard_scaler",
            ]

            if outlier_ratio >= 0.05:
                steps.append("robust_scaler_candidate")

            return steps

        if profile.inferred_type in {
            "categorical",
            "boolean",
        }:
            return [
                "most_frequent_imputer",
                "one_hot_encoder",
            ]

        if profile.inferred_type == "datetime":
            return ["datetime_feature_extractor"]

        if profile.inferred_type == "text":
            return ["text_vectorizer"]

        return ["manual_type_review"]

    def _select_reference_model(
        self,
        models: list[MLModelResult],
    ) -> MLModelResult | None:
        if not models:
            return None

        classification_models = [model for model in models if model.task_type == "classification"]

        if classification_models:
            return max(
                classification_models,
                key=lambda model: self._metric_value(
                    model,
                    "f1_macro",
                    fallback_metric="accuracy",
                    default=-1.0,
                ),
            )

        regression_models = [model for model in models if model.task_type == "regression"]

        if regression_models:
            return min(
                regression_models,
                key=lambda model: self._metric_value(
                    model,
                    "rmse",
                    default=math.inf,
                ),
            )

        return models[0]

    def _metric_value(
        self,
        model: MLModelResult,
        metric: str,
        fallback_metric: str | None = None,
        default: float = -1.0,
    ) -> float:
        value = model.metrics.get(metric)

        if value is None and fallback_metric is not None:
            value = model.metrics.get(fallback_metric)

        return value if value is not None else default

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
