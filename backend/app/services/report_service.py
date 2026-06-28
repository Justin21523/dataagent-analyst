import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from backend.app.core.config import Settings
from backend.app.schemas.report_schema import ReportDetail, ReportSummary
from backend.app.services.column_profiler_service import ColumnProfilerService
from backend.app.services.dataset_service import DatasetService
from backend.app.services.eda_service import EdaService
from backend.app.services.model_registry_service import ModelRegistryService
from backend.app.services.visualization_service import VisualizationService


class ReportServiceError(Exception):
    """Report service base exception."""


class ReportNotFoundError(ReportServiceError):
    """Raised when report cannot be found."""


class ReportService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry_path = settings.processed_data_dir / settings.report_registry_filename
        self.dataset_service = DatasetService(settings)
        self.column_profiler_service = ColumnProfilerService(settings)
        self.eda_service = EdaService(settings)
        self.visualization_service = VisualizationService(settings)
        self.model_registry_service = ModelRegistryService(settings)

    def generate_report(self, dataset_id: str) -> ReportDetail:
        # Report Center 先整合既有 deterministic results，後面再加入 LLM 解釋層。
        dataset = self.dataset_service.get_dataset_detail(dataset_id)
        schema = self.column_profiler_service.summarize_schema(dataset_id)
        eda_summary = self.eda_service.get_summary(dataset_id)
        visualizations = self.visualization_service.get_recommendations(dataset_id)
        models = self.model_registry_service.list_models(dataset_id=dataset_id)

        report_id = uuid4().hex
        created_at = datetime.now(UTC)
        title = f"Data Analysis Report - {dataset.name}"
        markdown_content = self._build_markdown_report(
            title=title,
            dataset_name=dataset.name,
            dataset_filename=dataset.original_filename,
            schema=schema.model_dump(),
            eda_summary=eda_summary.model_dump(),
            visualization_recommendations=visualizations.model_dump(),
            models=[model.model_dump() for model in models],
            created_at=created_at,
        )

        report_path = self.settings.reports_dir / f"{report_id}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(markdown_content, encoding="utf-8")

        record = {
            "id": report_id,
            "dataset_id": dataset_id,
            "title": title,
            "markdown_path": str(report_path.relative_to(self.settings.project_root)),
            "status": "ready",
            "created_at": created_at.isoformat(),
        }

        registry = self._load_registry()
        registry["reports"].append(record)
        self._save_registry(registry)

        return ReportDetail.model_validate(
            {
                **record,
                "markdown_content": markdown_content,
            }
        )

    def list_reports(self, dataset_id: str) -> list[ReportSummary]:
        registry = self._load_registry()

        reports = [
            ReportSummary.model_validate(record)
            for record in registry["reports"]
            if record["dataset_id"] == dataset_id
        ]

        return sorted(reports, key=lambda report: report.created_at, reverse=True)

    def get_report(self, report_id: str) -> ReportDetail:
        record = self._find_report_record(report_id)
        markdown_path = self.settings.project_root / record["markdown_path"]

        if not markdown_path.exists():
            raise ReportNotFoundError(f"Report markdown file is missing: {report_id}")

        return ReportDetail.model_validate(
            {
                **record,
                "markdown_content": markdown_path.read_text(encoding="utf-8"),
            }
        )

    def _build_markdown_report(
        self,
        title: str,
        dataset_name: str,
        dataset_filename: str,
        schema: dict[str, Any],
        eda_summary: dict[str, Any],
        visualization_recommendations: dict[str, Any],
        models: list[dict[str, Any]],
        created_at: datetime,
    ) -> str:
        # Markdown 採用固定章節，方便後續 LLM 接手改寫每一段。
        lines = [
            f"# {title}",
            "",
            f"Generated at: `{created_at.isoformat()}`",
            "",
            "## 1. Dataset Overview",
            "",
            f"- Dataset name: `{dataset_name}`",
            f"- Original file: `{dataset_filename}`",
            f"- Rows: `{schema['summary']['row_count']}`",
            f"- Columns: `{schema['summary']['column_count']}`",
            "",
            "## 2. Schema Summary",
            "",
            f"- Missing cells: `{schema['summary']['missing_cell_count']}`",
            f"- Missing cell ratio: `{schema['summary']['missing_cell_ratio']}`",
            f"- Duplicate rows: `{schema['summary']['duplicate_row_count']}`",
            f"- Duplicate row ratio: `{schema['summary']['duplicate_row_ratio']}`",
            "",
            "### Column Type Distribution",
            "",
        ]

        for column_type, count in schema["summary"]["type_counts"].items():
            lines.append(f"- {column_type}: `{count}`")

        lines.extend(
            [
                "",
                "### Target Candidates",
                "",
            ]
        )

        target_candidates = schema["summary"]["target_candidates"]

        if target_candidates:
            for target in target_candidates:
                lines.append(f"- `{target}`")
        else:
            lines.append("- No target candidate detected.")

        lines.extend(
            [
                "",
                "## 3. Data Quality Summary",
                "",
                f"- Data quality score: `{eda_summary['data_quality_score']}`",
                f"- Data quality grade: `{eda_summary['data_quality_grade']}`",
                f"- Total missing cells: `{eda_summary['missing']['total_missing_cells']}`",
                f"- Duplicate row count: `{eda_summary['duplicates']['duplicate_row_count']}`",
                "",
                "## 4. Numeric Statistics",
                "",
            ]
        )

        numeric_columns = eda_summary["numeric_statistics"]["columns"]

        if numeric_columns:
            lines.extend(
                [
                    "| Column | Mean | Std | Min | Median | Max |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )

            for column in numeric_columns:
                lines.append(
                    "| "
                    f"{column['name']} | "
                    f"{column['mean']} | "
                    f"{column['std']} | "
                    f"{column['min']} | "
                    f"{column['median']} | "
                    f"{column['max']} |"
                )
        else:
            lines.append("No numeric columns available.")

        lines.extend(
            [
                "",
                "## 5. Outlier Analysis",
                "",
            ]
        )

        outlier_columns = [
            column for column in eda_summary["outliers"]["columns"] if column["outlier_count"] > 0
        ]

        if outlier_columns:
            lines.extend(
                [
                    "| Column | Method | Outlier Count | Outlier Ratio |",
                    "|---|---|---:|---:|",
                ]
            )

            for column in outlier_columns:
                lines.append(
                    "| "
                    f"{column['name']} | "
                    f"{column['method']} | "
                    f"{column['outlier_count']} | "
                    f"{column['outlier_ratio']} |"
                )
        else:
            lines.append("No outlier-heavy numeric columns detected by the current IQR rule.")

        lines.extend(
            [
                "",
                "## 6. Correlation Highlights",
                "",
            ]
        )

        strongest_pairs = eda_summary["correlation"]["strongest_pairs"]

        if strongest_pairs:
            lines.extend(
                [
                    "| Column X | Column Y | Correlation |",
                    "|---|---|---:|",
                ]
            )

            for pair in strongest_pairs[:5]:
                lines.append(f"| {pair['column_x']} | {pair['column_y']} | {pair['correlation']} |")
        else:
            lines.append("No strong numeric correlation pairs available.")

        lines.extend(
            [
                "",
                "## 7. Visualization Recommendations",
                "",
            ]
        )

        chart_recommendations = visualization_recommendations["recommendations"]

        if chart_recommendations:
            for chart in chart_recommendations[:8]:
                lines.append(
                    f"- **{chart['title']}** "
                    f"({chart['chart_family']} / {chart['chart_type']}): "
                    f"{chart['reason']}"
                )
        else:
            lines.append("No chart recommendations available.")

        lines.extend(
            [
                "",
                "## 8. Machine Learning Results",
                "",
            ]
        )

        if models:
            lines.extend(
                [
                    "| Model | Task | Target | Metrics |",
                    "|---|---|---|---|",
                ]
            )

            for model in models[:8]:
                metric_text = ", ".join(
                    f"{name}: {value}" for name, value in model["metrics"].items()
                )
                lines.append(
                    f"| {model['model_name']} | "
                    f"{model['task_type']} | "
                    f"{model['target_column']} | "
                    f"{metric_text} |"
                )
        else:
            lines.append("No trained models are available for this dataset yet.")

        lines.extend(
            [
                "",
                "## 9. Recommendations",
                "",
            ]
        )

        for recommendation in eda_summary["recommendations"]:
            lines.append(f"- {recommendation}")

        lines.extend(
            [
                "",
                "## 10. Current Limitations",
                "",
                "- This report is generated by deterministic rules, not by a local LLM yet.",
                "- The current version does not include SHAP explainability.",
                "- Prediction history is not stored yet.",
                "- Report export currently supports Markdown only.",
                "",
            ]
        )

        return "\n".join(lines)

    def _load_registry(self) -> dict[str, list[dict[str, Any]]]:
        if not self.registry_path.exists():
            return {"reports": []}

        try:
            with self.registry_path.open("r", encoding="utf-8") as file:
                registry = json.load(file)
        except json.JSONDecodeError as exc:
            raise ReportServiceError("Report registry file is corrupted.") from exc

        if "reports" not in registry or not isinstance(registry["reports"], list):
            raise ReportServiceError("Report registry format is invalid.")

        return registry

    def _save_registry(self, registry: dict[str, list[dict[str, Any]]]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.registry_path.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(registry, file, ensure_ascii=False, indent=2)

        temporary_path.replace(self.registry_path)

    def _find_report_record(self, report_id: str) -> dict[str, Any]:
        registry = self._load_registry()

        for record in registry["reports"]:
            if record["id"] == report_id:
                return record

        raise ReportNotFoundError(f"Report not found: {report_id}")
