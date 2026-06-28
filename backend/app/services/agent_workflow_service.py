from datetime import UTC, datetime
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from backend.app.core.config import Settings
from backend.app.schemas.agent_schema import AgentRunRequest, AgentRunResponse, AgentStep
from backend.app.schemas.ml_schema import MLTrainRequest
from backend.app.services.agent_run_registry_service import AgentRunRegistryService
from backend.app.services.ai_insight_service import AIInsightService
from backend.app.services.column_profiler_service import ColumnProfilerService
from backend.app.services.eda_service import EdaService
from backend.app.services.ml_training_service import MLTrainingError, MLTrainingService
from backend.app.services.report_service import ReportService
from backend.app.services.visualization_service import VisualizationService


class AgentWorkflowState(TypedDict, total=False):
    workflow_id: str
    dataset_id: str
    user_goal: str | None
    target_column: str | None
    run_ml: bool
    generate_report: bool
    generate_ai_insight: bool
    steps: list[dict[str, Any]]
    errors: list[str]
    schema: dict[str, Any]
    eda_summary: dict[str, Any]
    visualizations: dict[str, Any]
    ml_result: dict[str, Any] | None
    report: dict[str, Any] | None
    ai_insight: dict[str, Any] | None
    final_summary: str
    event_callback: Any | None


class AgentWorkflowService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.column_profiler_service = ColumnProfilerService(settings)
        self.eda_service = EdaService(settings)
        self.visualization_service = VisualizationService(settings)
        self.ml_training_service = MLTrainingService(settings)
        self.report_service = ReportService(settings)
        self.ai_insight_service = AIInsightService(settings)
        self.agent_run_registry_service = AgentRunRegistryService(settings)
        self.graph = self._build_graph()

    def run_workflow(
        self,
        dataset_id: str,
        request: AgentRunRequest,
        event_callback: Any | None = None,
    ) -> AgentRunResponse:
        # Agent workflow 只負責 orchestration；每個實際工具仍由既有 service 執行。
        initial_state: AgentWorkflowState = {
            "workflow_id": uuid4().hex,
            "dataset_id": dataset_id,
            "user_goal": request.user_goal,
            "target_column": request.target_column,
            "run_ml": request.run_ml,
            "generate_report": request.generate_report,
            "generate_ai_insight": request.generate_ai_insight,
            "steps": [],
            "errors": [],
            "ml_result": None,
            "report": None,
            "ai_insight": None,
            "event_callback": event_callback,
        }

        final_state = self.graph.invoke(initial_state)

        errors = final_state.get("errors", [])
        status = "success" if not errors else "completed_with_warnings"

        outputs = {
            "schema": final_state.get("schema"),
            "eda_summary": self._compact_eda(final_state.get("eda_summary")),
            "visualizations": self._compact_visualizations(final_state.get("visualizations")),
            "ml_result": self._compact_ml(final_state.get("ml_result")),
            "report": self._compact_report(final_state.get("report")),
            "ai_insight": final_state.get("ai_insight"),
        }

        response = AgentRunResponse(
            workflow_id=final_state["workflow_id"],
            dataset_id=dataset_id,
            status=status,
            user_goal=request.user_goal,
            steps=[AgentStep.model_validate(step) for step in final_state.get("steps", [])],
            outputs=outputs,
            errors=errors,
            final_summary=final_state.get("final_summary", ""),
        )

        self.agent_run_registry_service.add_run(response)

        return response

    def _build_graph(self):
        # 使用固定 workflow，先確保可靠，再逐步升級成更動態的 agent。
        graph = StateGraph(AgentWorkflowState)

        graph.add_node("planner", self._planner_node)
        graph.add_node("profiler", self._profiler_node)
        graph.add_node("eda_runner", self._eda_node)
        graph.add_node("visualizer", self._visualization_node)
        graph.add_node("ml_trainer", self._ml_node)
        graph.add_node("reporter", self._report_node)
        graph.add_node("insight_generator", self._insight_node)
        graph.add_node("finalizer", self._finalizer_node)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "profiler")
        graph.add_edge("profiler", "eda_runner")
        graph.add_edge("eda_runner", "visualizer")
        graph.add_edge("visualizer", "ml_trainer")
        graph.add_edge("ml_trainer", "reporter")
        graph.add_edge("reporter", "insight_generator")
        graph.add_edge("insight_generator", "finalizer")
        graph.add_edge("finalizer", END)

        return graph.compile()

    def _planner_node(self, state: AgentWorkflowState) -> dict[str, Any]:
        steps = self._append_step(
            state=state,
            name="Planner",
            status="success",
            message="Analysis workflow planned.",
            payload={
                "run_ml": state.get("run_ml", True),
                "generate_report": state.get("generate_report", True),
                "generate_ai_insight": state.get("generate_ai_insight", True),
            },
        )

        return {"steps": steps}

    def _profiler_node(self, state: AgentWorkflowState) -> dict[str, Any]:
        schema = self.column_profiler_service.summarize_schema(state["dataset_id"])
        target_column = state.get("target_column")

        if not target_column and schema.summary.target_candidates:
            target_column = schema.summary.target_candidates[0]

        steps = self._append_step(
            state=state,
            name="Data Profiler",
            status="success",
            message="Dataset schema profile generated.",
            payload={
                "row_count": schema.summary.row_count,
                "column_count": schema.summary.column_count,
                "target_column": target_column,
                "type_counts": schema.summary.type_counts,
            },
        )

        return {
            "schema": schema.model_dump(),
            "target_column": target_column,
            "steps": steps,
        }

    def _eda_node(self, state: AgentWorkflowState) -> dict[str, Any]:
        eda_summary = self.eda_service.get_summary(state["dataset_id"])

        steps = self._append_step(
            state=state,
            name="EDA Runner",
            status="success",
            message="EDA summary generated.",
            payload={
                "data_quality_score": eda_summary.data_quality_score,
                "data_quality_grade": eda_summary.data_quality_grade,
                "missing_cells": eda_summary.missing.total_missing_cells,
                "duplicate_rows": eda_summary.duplicates.duplicate_row_count,
            },
        )

        return {
            "eda_summary": eda_summary.model_dump(),
            "steps": steps,
        }

    def _visualization_node(self, state: AgentWorkflowState) -> dict[str, Any]:
        visualizations = self.visualization_service.get_recommendations(state["dataset_id"])

        steps = self._append_step(
            state=state,
            name="Visualization Recommender",
            status="success",
            message="Visualization recommendations generated.",
            payload={
                "chart_count": visualizations.total,
                "chart_titles": [chart.title for chart in visualizations.recommendations[:5]],
            },
        )

        return {
            "visualizations": visualizations.model_dump(),
            "steps": steps,
        }

    def _ml_node(self, state: AgentWorkflowState) -> dict[str, Any]:
        if not state.get("run_ml", True):
            steps = self._append_step(
                state=state,
                name="ML Trainer",
                status="skipped",
                message="ML training skipped by request.",
            )
            return {"steps": steps}

        target_column = state.get("target_column")

        if not target_column:
            message = "ML training skipped because no target column was detected."
            steps = self._append_step(
                state=state,
                name="ML Trainer",
                status="warning",
                message=message,
            )
            return {
                "steps": steps,
                "errors": state.get("errors", []) + [message],
            }

        try:
            ml_result = self.ml_training_service.train_models(
                dataset_id=state["dataset_id"],
                request=MLTrainRequest(
                    target_column=target_column,
                    task_type="auto",
                    test_size=0.25,
                    random_state=42,
                ),
            )
        except MLTrainingError as exc:
            message = f"ML training failed: {exc}"
            steps = self._append_step(
                state=state,
                name="ML Trainer",
                status="warning",
                message=message,
            )
            return {
                "steps": steps,
                "errors": state.get("errors", []) + [message],
            }

        steps = self._append_step(
            state=state,
            name="ML Trainer",
            status="success",
            message="Baseline ML models trained.",
            payload={
                "task_type": ml_result.task_type,
                "target_column": ml_result.target_column,
                "model_count": ml_result.model_count,
                "best_model_id": ml_result.best_model_id,
                "best_metric_name": ml_result.best_metric_name,
                "best_metric_value": ml_result.best_metric_value,
            },
        )

        return {
            "ml_result": ml_result.model_dump(),
            "steps": steps,
        }

    def _report_node(self, state: AgentWorkflowState) -> dict[str, Any]:
        if not state.get("generate_report", True):
            steps = self._append_step(
                state=state,
                name="Report Writer",
                status="skipped",
                message="Report generation skipped by request.",
            )
            return {"steps": steps}

        report = self.report_service.generate_report(state["dataset_id"])

        steps = self._append_step(
            state=state,
            name="Report Writer",
            status="success",
            message="Markdown report generated.",
            payload={
                "report_id": report.id,
                "title": report.title,
            },
        )

        return {
            "report": report.model_dump(),
            "steps": steps,
        }

    def _insight_node(self, state: AgentWorkflowState) -> dict[str, Any]:
        if not state.get("generate_ai_insight", True):
            steps = self._append_step(
                state=state,
                name="AI Insight Generator",
                status="skipped",
                message="AI insight generation skipped by request.",
            )
            return {"steps": steps}

        insight = self.ai_insight_service.summarize_report(
            dataset_id=state["dataset_id"],
            user_goal=state.get("user_goal"),
        )

        steps = self._append_step(
            state=state,
            name="AI Insight Generator",
            status="success",
            message="AI insight generated.",
            payload={
                "source": insight.source,
                "llm_enabled": insight.llm_enabled,
                "llm_online": insight.llm_online,
            },
        )

        return {
            "ai_insight": insight.model_dump(),
            "steps": steps,
        }

    def _finalizer_node(self, state: AgentWorkflowState) -> dict[str, Any]:
        step_count = len(state.get("steps", []))
        error_count = len(state.get("errors", []))

        final_summary = (
            f"Agent workflow completed with {step_count} steps and {error_count} warning(s)."
        )

        steps = self._append_step(
            state=state,
            name="Finalizer",
            status="success",
            message=final_summary,
        )

        return {
            "steps": steps,
            "final_summary": final_summary,
        }

    def _append_step(
        self,
        state: AgentWorkflowState,
        name: str,
        status: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        # 每個節點都寫入 step，前端才能顯示完整 workflow timeline。
        step = {
            "name": name,
            "status": status,
            "message": message,
            "payload": payload or {},
            "created_at": datetime.now(UTC).isoformat(),
        }

        event_callback = state.get("event_callback")

        if callable(event_callback):
            # callback 用於 Phase 12 背景任務進度事件，不影響同步 workflow 回傳。
            event_callback(step)

        return state.get("steps", []) + [step]

    def _compact_eda(self, eda_summary: dict[str, Any] | None) -> dict[str, Any] | None:
        if not eda_summary:
            return None

        return {
            "data_quality_score": eda_summary.get("data_quality_score"),
            "data_quality_grade": eda_summary.get("data_quality_grade"),
            "recommendations": eda_summary.get("recommendations"),
        }

    def _compact_visualizations(
        self,
        visualizations: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not visualizations:
            return None

        return {
            "total": visualizations.get("total"),
            "chart_titles": [
                item["title"] for item in visualizations.get("recommendations", [])[:5]
            ],
        }

    def _compact_ml(self, ml_result: dict[str, Any] | None) -> dict[str, Any] | None:
        if not ml_result:
            return None

        return {
            "task_type": ml_result.get("task_type"),
            "target_column": ml_result.get("target_column"),
            "model_count": ml_result.get("model_count"),
            "best_model_id": ml_result.get("best_model_id"),
            "best_metric_name": ml_result.get("best_metric_name"),
            "best_metric_value": ml_result.get("best_metric_value"),
        }

    def _compact_report(self, report: dict[str, Any] | None) -> dict[str, Any] | None:
        if not report:
            return None

        return {
            "id": report.get("id"),
            "title": report.get("title"),
            "status": report.get("status"),
        }
