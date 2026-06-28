from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

from backend.app.core.config import Settings
from backend.app.repositories.metadata_repository import (
    MetadataRepositoryError,
    create_metadata_repository,
)
from backend.app.schemas.agent_job_schema import AgentJobDetail, AgentJobEvent, AgentJobSummary
from backend.app.schemas.agent_schema import AgentRunRequest, AgentRunResponse

_REGISTRY_LOCK = Lock()


class AgentJobRegistryError(Exception):
    """Agent job registry base exception."""


class AgentJobNotFoundError(AgentJobRegistryError):
    """Raised when an agent job cannot be found."""


class AgentJobRegistryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metadata_repository = create_metadata_repository(settings)

    def create_job(
        self,
        dataset_id: str,
        request: AgentRunRequest,
    ) -> AgentJobDetail:
        now = datetime.now(UTC)
        job_id = uuid4().hex

        event = self._build_event(
            job_id=job_id,
            workflow_id=None,
            event_type="queued",
            step_name=None,
            status="queued",
            message="Agent job queued.",
            payload={},
        )

        record = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "workflow_id": None,
            "status": "queued",
            "user_goal": request.user_goal,
            "request": request.model_dump(mode="json"),
            "events": [event.model_dump(mode="json")],
            "result": None,
            "created_at": now.isoformat(),
            "started_at": None,
            "finished_at": None,
            "error_message": None,
        }

        with _REGISTRY_LOCK:
            registry = self._load_registry()
            registry["jobs"].append(record)
            self._save_registry(registry)

        return self._build_detail(record)

    def mark_running(self, job_id: str) -> AgentJobDetail:
        with _REGISTRY_LOCK:
            registry = self._load_registry()
            record = self._find_job_record_from_registry(registry, job_id)

            record["status"] = "running"
            record["started_at"] = datetime.now(UTC).isoformat()
            record["events"].append(
                self._build_event(
                    job_id=job_id,
                    workflow_id=record.get("workflow_id"),
                    event_type="running",
                    step_name=None,
                    status="running",
                    message="Agent job started.",
                    payload={},
                ).model_dump(mode="json")
            )

            self._save_registry(registry)

        return self._build_detail(record)

    def append_step_event(self, job_id: str, step: dict[str, Any]) -> AgentJobEvent:
        with _REGISTRY_LOCK:
            registry = self._load_registry()
            record = self._find_job_record_from_registry(registry, job_id)

            event = self._build_event(
                job_id=job_id,
                workflow_id=record.get("workflow_id"),
                event_type="step",
                step_name=step.get("name"),
                status=step.get("status", "success"),
                message=step.get("message", ""),
                payload=step.get("payload", {}),
            )

            record["events"].append(event.model_dump(mode="json"))
            self._save_registry(registry)

        return event

    def mark_completed(self, job_id: str, result: AgentRunResponse) -> AgentJobDetail:
        with _REGISTRY_LOCK:
            registry = self._load_registry()
            record = self._find_job_record_from_registry(registry, job_id)

            record["status"] = result.status
            record["workflow_id"] = result.workflow_id
            record["result"] = result.model_dump(mode="json")
            record["finished_at"] = datetime.now(UTC).isoformat()
            record["events"].append(
                self._build_event(
                    job_id=job_id,
                    workflow_id=result.workflow_id,
                    event_type="completed",
                    step_name=None,
                    status=result.status,
                    message=result.final_summary,
                    payload={
                        "workflow_id": result.workflow_id,
                        "step_count": len(result.steps),
                        "error_count": len(result.errors),
                    },
                ).model_dump(mode="json")
            )

            self._save_registry(registry)

        return self._build_detail(record)

    def mark_failed(self, job_id: str, error_message: str) -> AgentJobDetail:
        with _REGISTRY_LOCK:
            registry = self._load_registry()
            record = self._find_job_record_from_registry(registry, job_id)

            record["status"] = "failed"
            record["error_message"] = error_message
            record["finished_at"] = datetime.now(UTC).isoformat()
            record["events"].append(
                self._build_event(
                    job_id=job_id,
                    workflow_id=record.get("workflow_id"),
                    event_type="failed",
                    step_name=None,
                    status="failed",
                    message=error_message,
                    payload={},
                ).model_dump(mode="json")
            )

            self._save_registry(registry)

        return self._build_detail(record)

    def get_job(self, job_id: str) -> AgentJobDetail:
        registry = self._load_registry()
        record = self._find_job_record_from_registry(registry, job_id)
        return self._build_detail(record)

    def list_jobs(self, dataset_id: str) -> list[AgentJobSummary]:
        registry = self._load_registry()

        jobs = [
            self._build_summary(record)
            for record in registry["jobs"]
            if record["dataset_id"] == dataset_id
        ]

        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def list_events(self, job_id: str) -> list[AgentJobEvent]:
        job = self.get_job(job_id)
        return job.events

    def _build_summary(self, record: dict[str, Any]) -> AgentJobSummary:
        return AgentJobSummary(
            job_id=record["job_id"],
            dataset_id=record["dataset_id"],
            workflow_id=record.get("workflow_id"),
            status=record["status"],
            user_goal=record.get("user_goal"),
            created_at=record["created_at"],
            started_at=record.get("started_at"),
            finished_at=record.get("finished_at"),
            event_count=len(record.get("events", [])),
            error_message=record.get("error_message"),
        )

    def _build_detail(self, record: dict[str, Any]) -> AgentJobDetail:
        summary = self._build_summary(record)

        return AgentJobDetail(
            **summary.model_dump(),
            request=AgentRunRequest.model_validate(record["request"]),
            events=[AgentJobEvent.model_validate(event) for event in record.get("events", [])],
            result=(
                AgentRunResponse.model_validate(record["result"])
                if record.get("result") is not None
                else None
            ),
        )

    def _build_event(
        self,
        job_id: str,
        workflow_id: str | None,
        event_type: str,
        step_name: str | None,
        status: str,
        message: str,
        payload: dict[str, Any],
    ) -> AgentJobEvent:
        return AgentJobEvent(
            event_id=uuid4().hex,
            job_id=job_id,
            workflow_id=workflow_id,
            event_type=event_type,
            step_name=step_name,
            status=status,
            message=message,
            payload=payload,
            created_at=datetime.now(UTC),
        )

    def _load_registry(self) -> dict[str, list[dict[str, Any]]]:
        try:
            return self.metadata_repository.load_registry("agent_jobs")
        except MetadataRepositoryError as exc:
            raise AgentJobRegistryError("Agent job registry file is corrupted.") from exc

    def _save_registry(self, registry: dict[str, list[dict[str, Any]]]) -> None:
        try:
            self.metadata_repository.save_registry("agent_jobs", registry)
        except MetadataRepositoryError as exc:
            raise AgentJobRegistryError("Agent job registry format is invalid.") from exc

    def _find_job_record_from_registry(
        self,
        registry: dict[str, list[dict[str, Any]]],
        job_id: str,
    ) -> dict[str, Any]:
        for record in registry["jobs"]:
            if record["job_id"] == job_id:
                return record

        raise AgentJobNotFoundError(f"Agent job not found: {job_id}")
