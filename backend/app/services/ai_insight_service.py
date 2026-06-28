import json

from backend.app.core.config import Settings
from backend.app.schemas.ai_insight_schema import AIInsightResponse, LLMStatusResponse
from backend.app.schemas.explainability_schema import (
    ModelExplainabilityRequest,
)
from backend.app.services.column_profiler_service import ColumnProfilerService
from backend.app.services.eda_service import EdaService
from backend.app.services.local_llm_service import LocalLLMService
from backend.app.services.model_explainability_service import (
    ModelExplainabilityService,
)
from backend.app.services.model_registry_service import ModelRegistryService
from backend.app.services.report_service import ReportService


class AIInsightService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.local_llm_service = LocalLLMService(settings)
        self.column_profiler_service = ColumnProfilerService(settings)
        self.eda_service = EdaService(settings)
        self.model_registry_service = ModelRegistryService(settings)
        self.model_explainability_service = ModelExplainabilityService(settings)
        self.report_service = ReportService(settings)

    def get_llm_status(self) -> LLMStatusResponse:
        return self.local_llm_service.check_status()

    def explain_eda(self, dataset_id: str, user_goal: str | None = None) -> AIInsightResponse:
        # EDA explanation 只把關鍵摘要交給 LLM，不把整份資料塞進 prompt。
        schema = self.column_profiler_service.summarize_schema(dataset_id)
        eda_summary = self.eda_service.get_summary(dataset_id)

        prompt_payload = {
            "user_goal": user_goal,
            "schema_summary": schema.summary.model_dump(),
            "data_quality_score": eda_summary.data_quality_score,
            "data_quality_grade": eda_summary.data_quality_grade,
            "missing": eda_summary.missing.model_dump(),
            "duplicates": eda_summary.duplicates.model_dump(),
            "outliers": eda_summary.outliers.model_dump(),
            "strongest_correlations": [
                pair.model_dump() for pair in eda_summary.correlation.strongest_pairs[:5]
            ],
            "recommendations": eda_summary.recommendations,
        }

        fallback_content = self._build_eda_fallback(
            eda_summary.data_quality_score,
            eda_summary.recommendations,
        )

        result = self.local_llm_service.generate_text(
            system_prompt=self._system_prompt(),
            user_prompt=(
                "Explain this EDA result for a data analyst. "
                "Use concise English bullet points and focus on practical next steps.\n\n"
                f"{json.dumps(prompt_payload, ensure_ascii=False, default=str)}"
            ),
            fallback_content=fallback_content,
        )

        return self._build_response(
            dataset_id=dataset_id,
            title="EDA Explanation",
            result=result,
        )

    def explain_model(
        self,
        dataset_id: str,
        model_id: str,
        user_goal: str | None = None,
    ) -> AIInsightResponse:
        model = self.model_registry_service.get_model(model_id)

        prompt_payload = {
            "user_goal": user_goal,
            "model": model.model_dump(),
        }

        fallback_content = self._build_model_fallback(model.model_name, model.metrics)

        result = self.local_llm_service.generate_text(
            system_prompt=self._system_prompt(),
            user_prompt=(
                "Explain this machine learning model result for a data analyst. "
                "Focus on model quality, metrics, feature importance, and limitations.\n\n"
                f"{json.dumps(prompt_payload, ensure_ascii=False, default=str)}"
            ),
            fallback_content=fallback_content,
        )

        return self._build_response(
            dataset_id=dataset_id,
            title="Model Explanation",
            result=result,
        )

    def summarize_report(self, dataset_id: str, user_goal: str | None = None) -> AIInsightResponse:
        # 直接使用最新 deterministic report 作為 LLM 摘要來源。
        reports = self.report_service.list_reports(dataset_id)

        if reports:
            report = self.report_service.get_report(reports[0].id)
            report_content = report.markdown_content
        else:
            report = self.report_service.generate_report(dataset_id)
            report_content = report.markdown_content

        fallback_content = (
            "Report summary is available in the Markdown report. "
            "Enable the local LLM server to generate an executive summary."
        )

        result = self.local_llm_service.generate_text(
            system_prompt=self._system_prompt(),
            user_prompt=(
                "Summarize this data analysis report for a portfolio project demo. "
                "Use concise English sections: Key Findings, Risks, ML Results, Next Steps.\n\n"
                f"User goal: {user_goal or 'General summary'}\n\n"
                f"{report_content[:12000]}"
            ),
            fallback_content=fallback_content,
        )

        return self._build_response(
            dataset_id=dataset_id,
            title="Report Executive Summary",
            result=result,
        )

    def explain_model_explainability(
        self,
        dataset_id: str,
        model_id: str,
        user_goal: str | None = None,
    ) -> AIInsightResponse:
        model = self.model_registry_service.get_model(model_id)

        if model.dataset_id != dataset_id:
            raise ValueError("Model does not belong to the selected dataset.")

        explainability = self.model_explainability_service.analyze(
            model_id=model_id,
            request=ModelExplainabilityRequest(
                sample_size=100,
                background_size=50,
                permutation_repeats=8,
                include_permutation=True,
                include_shap=True,
            ),
        )

        prompt_payload = {
            "user_goal": user_goal,
            "model": {
                "model_name": explainability.model_name,
                "task_type": explainability.task_type,
                "target_column": (explainability.target_column),
                "holdout_rows": (explainability.holdout.row_count),
            },
            "permutation_importance": [
                item.model_dump() for item in (explainability.permutation_importance[:10])
            ],
            "shap_source_importance": [
                item.model_dump() for item in (explainability.shap.source_feature_importance[:10])
            ],
            "classification_curves": (
                [
                    {
                        "class_label": curve.class_label,
                        "roc_auc": curve.roc_auc,
                        "average_precision": (curve.average_precision),
                    }
                    for curve in (explainability.classification.curves)
                ]
                if explainability.classification
                else []
            ),
            "regression_summary": (
                explainability.regression.model_dump(exclude={"points"})
                if explainability.regression
                else None
            ),
            "error_sample_count": len(explainability.error_samples),
            "warnings": explainability.warnings,
        }

        fallback_content = (
            "Model explainability fallback:\n\n"
            f"- Model: {explainability.model_name}\n"
            f"- Task: {explainability.task_type}\n"
            f"- Holdout rows: "
            f"{explainability.holdout.row_count}\n"
            f"- Error samples: "
            f"{len(explainability.error_samples)}\n"
            "- Review permutation importance, SHAP, "
            "diagnostic curves, and warnings in the "
            "Model Explainability workspace."
        )

        result = self.local_llm_service.generate_text(
            system_prompt=self._system_prompt(),
            user_prompt=(
                "Interpret this model explainability result. "
                "Use concise English sections: Reliability, "
                "Important Drivers, Error Risks, Business Meaning, "
                "Limitations, and Recommended Next Experiments. "
                "Do not claim causality.\n\n"
                f"{json.dumps(prompt_payload, ensure_ascii=False, default=str)}"
            ),
            fallback_content=fallback_content,
        )

        return self._build_response(
            dataset_id=dataset_id,
            title="Model Explainability Interpretation",
            result=result,
        )

    def _system_prompt(self) -> str:
        return (
            "You are DataAgent Analyst, an AI data analysis assistant. "
            "Explain results clearly, avoid unsupported claims, and mention limitations."
        )

    def _build_response(
        self,
        dataset_id: str,
        title: str,
        result,
    ) -> AIInsightResponse:
        return AIInsightResponse(
            dataset_id=dataset_id,
            title=title,
            content=result.content,
            source=result.source,
            model=self.settings.llm_model if result.source == "llm" else "fallback",
            llm_enabled=self.settings.llm_enabled,
            llm_online=result.online,
        )

    def _build_eda_fallback(self, score: float, recommendations: list[str]) -> str:
        recommendation_text = "\n".join(f"- {item}" for item in recommendations)

        return (
            f"EDA fallback explanation:\n\n"
            f"- Data quality score: {score}\n"
            f"- The system generated rule-based recommendations:\n"
            f"{recommendation_text}\n\n"
            f"Enable the local LLM server to generate a richer natural-language explanation."
        )

    def _build_model_fallback(self, model_name: str, metrics: dict[str, float | None]) -> str:
        metric_text = "\n".join(f"- {name}: {value}" for name, value in metrics.items())

        return (
            f"Model fallback explanation:\n\n"
            f"- Model: {model_name}\n"
            f"- Metrics:\n"
            f"{metric_text}\n\n"
            f"Enable the local LLM server to generate a richer model interpretation."
        )
