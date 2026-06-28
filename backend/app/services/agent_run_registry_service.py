import json
from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import Settings
from backend.app.schemas.agent_schema import AgentRunResponse, AgentRunSummary


class AgentRunRegistryError(Exception):
    """Agent run registry base exception."""


class AgentRunNotFoundError(AgentRunRegistryError):
    """Raised when an agent run cannot be found."""


class AgentRunRegistryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry_path = settings.processed_data_dir / settings.agent_run_registry_filename

    def add_run(self, run: AgentRunResponse) -> AgentRunResponse:
        # Agent run 紀錄使用 JSON 儲存，方便 Phase 11 快速建立任務歷史。
        registry = self._load_registry()
        registry["runs"].append(run.model_dump(mode="json"))
        self._save_registry(registry)

        return run

    def list_runs(self, dataset_id: str) -> list[AgentRunSummary]:
        registry = self._load_registry()

        summaries = [
            self._build_run_summary(record)
            for record in registry["runs"]
            if record["dataset_id"] == dataset_id
        ]

        return sorted(summaries, key=lambda run: run.finished_at, reverse=True)

    def get_run(self, workflow_id: str) -> AgentRunResponse:
        record = self._find_run_record(workflow_id)
        return AgentRunResponse.model_validate(record)

    def _build_run_summary(self, record: dict[str, Any]) -> AgentRunSummary:
        steps = record.get("steps", [])
        now = datetime.now(UTC)

        started_at = self._parse_datetime(steps[0]["created_at"]) if steps else now
        finished_at = self._parse_datetime(steps[-1]["created_at"]) if steps else now

        return AgentRunSummary(
            workflow_id=record["workflow_id"],
            dataset_id=record["dataset_id"],
            status=record["status"],
            user_goal=record.get("user_goal"),
            step_count=len(steps),
            error_count=len(record.get("errors", [])),
            started_at=started_at,
            finished_at=finished_at,
            final_summary=record.get("final_summary", ""),
        )

    def _parse_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            return datetime.fromisoformat(value)

        raise AgentRunRegistryError("Agent run timestamp format is invalid.")

    def _load_registry(self) -> dict[str, list[dict[str, Any]]]:
        if not self.registry_path.exists():
            return {"runs": []}

        try:
            with self.registry_path.open("r", encoding="utf-8") as file:
                registry = json.load(file)
        except json.JSONDecodeError as exc:
            raise AgentRunRegistryError("Agent run registry file is corrupted.") from exc

        if "runs" not in registry or not isinstance(registry["runs"], list):
            raise AgentRunRegistryError("Agent run registry format is invalid.")

        return registry

    def _save_registry(self, registry: dict[str, list[dict[str, Any]]]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.registry_path.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(registry, file, ensure_ascii=False, indent=2)

        temporary_path.replace(self.registry_path)

    def _find_run_record(self, workflow_id: str) -> dict[str, Any]:
        registry = self._load_registry()

        for record in registry["runs"]:
            if record["workflow_id"] == workflow_id:
                return record

        raise AgentRunNotFoundError(f"Agent run not found: {workflow_id}")
