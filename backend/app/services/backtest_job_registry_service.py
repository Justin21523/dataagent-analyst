from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

from backend.app.core.config import Settings
from backend.app.repositories.metadata_repository import (
    MetadataRepositoryError,
    create_metadata_repository,
)
from backend.app.schemas.backtest_schema import (
    BacktestJobDetail,
    BacktestJobEvent,
    BacktestJobRequest,
    BacktestJobSummary,
)

_REGISTRY_LOCK = Lock()


class BacktestJobRegistryError(Exception):
    """Backtest job registry base exception."""


class BacktestJobNotFoundError(BacktestJobRegistryError):
    """Raised when a backtest job cannot be found."""


class BacktestJobRegistryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metadata_repository = create_metadata_repository(settings)

    def create_job(self, request: BacktestJobRequest) -> BacktestJobDetail:
        now = datetime.now(UTC)
        job_id = uuid4().hex
        event = self._build_event(
            job_id=job_id,
            event_type="queued",
            status="queued",
            message="Backtest job queued.",
            payload={"suite_type": request.suite_type},
        )
        record = {
            "job_id": job_id,
            "suite_type": request.suite_type,
            "status": "queued",
            "request": request.model_dump(mode="json"),
            "events": [event.model_dump(mode="json")],
            "result": None,
            "run_ids": [],
            "suite_ids": [],
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

    def mark_running(self, job_id: str) -> BacktestJobDetail:
        with _REGISTRY_LOCK:
            registry = self._load_registry()
            record = self._find_job_record_from_registry(registry, job_id)
            record["status"] = "running"
            record["started_at"] = datetime.now(UTC).isoformat()
            record["events"].append(
                self._build_event(
                    job_id=job_id,
                    event_type="running",
                    status="running",
                    message="Backtest job started.",
                    payload={},
                ).model_dump(mode="json")
            )
            self._save_registry(registry)

        return self._build_detail(record)

    def append_event(
        self,
        job_id: str,
        *,
        event_type: str,
        status: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> BacktestJobEvent:
        with _REGISTRY_LOCK:
            registry = self._load_registry()
            record = self._find_job_record_from_registry(registry, job_id)
            event = self._build_event(
                job_id=job_id,
                event_type=event_type,
                status=status,
                message=message,
                payload=payload or {},
            )
            record["events"].append(event.model_dump(mode="json"))
            self._save_registry(registry)

        return event

    def mark_completed(self, job_id: str, result: dict[str, Any]) -> BacktestJobDetail:
        with _REGISTRY_LOCK:
            registry = self._load_registry()
            record = self._find_job_record_from_registry(registry, job_id)
            record["status"] = "success"
            record["result"] = result
            record["run_ids"] = result.get("run_ids", [])
            record["suite_ids"] = result.get("suite_ids", [])
            record["finished_at"] = datetime.now(UTC).isoformat()
            record["events"].append(
                self._build_event(
                    job_id=job_id,
                    event_type="completed",
                    status="success",
                    message="Backtest job completed.",
                    payload=result,
                ).model_dump(mode="json")
            )
            self._save_registry(registry)

        return self._build_detail(record)

    def mark_failed(
        self,
        job_id: str,
        error_message: str,
        result: dict[str, Any] | None = None,
    ) -> BacktestJobDetail:
        with _REGISTRY_LOCK:
            registry = self._load_registry()
            record = self._find_job_record_from_registry(registry, job_id)
            record["status"] = "failed"
            record["error_message"] = error_message
            record["result"] = result
            if result:
                record["run_ids"] = result.get("run_ids", [])
                record["suite_ids"] = result.get("suite_ids", [])
            record["finished_at"] = datetime.now(UTC).isoformat()
            record["events"].append(
                self._build_event(
                    job_id=job_id,
                    event_type="failed",
                    status="failed",
                    message=error_message,
                    payload=result or {},
                ).model_dump(mode="json")
            )
            self._save_registry(registry)

        return self._build_detail(record)

    def get_job(self, job_id: str) -> BacktestJobDetail:
        registry = self._load_registry()
        record = self._find_job_record_from_registry(registry, job_id)
        return self._build_detail(record)

    def list_jobs(self, limit: int = 25) -> list[BacktestJobSummary]:
        limit = max(1, min(limit, 100))
        registry = self._load_registry()
        jobs = [self._build_summary(record) for record in registry["jobs"]]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)[:limit]

    def list_events(self, job_id: str) -> list[BacktestJobEvent]:
        return self.get_job(job_id).events

    def _build_summary(self, record: dict[str, Any]) -> BacktestJobSummary:
        return BacktestJobSummary(
            job_id=record["job_id"],
            suite_type=record["suite_type"],
            status=record["status"],
            created_at=record["created_at"],
            started_at=record.get("started_at"),
            finished_at=record.get("finished_at"),
            event_count=len(record.get("events", [])),
            run_ids=record.get("run_ids", []),
            suite_ids=record.get("suite_ids", []),
            error_message=record.get("error_message"),
        )

    def _build_detail(self, record: dict[str, Any]) -> BacktestJobDetail:
        summary = self._build_summary(record)
        return BacktestJobDetail(
            **summary.model_dump(),
            request=BacktestJobRequest.model_validate(record["request"]),
            events=[BacktestJobEvent.model_validate(event) for event in record.get("events", [])],
            result=record.get("result"),
        )

    def _build_event(
        self,
        *,
        job_id: str,
        event_type: str,
        status: str,
        message: str,
        payload: dict[str, Any],
    ) -> BacktestJobEvent:
        return BacktestJobEvent(
            event_id=uuid4().hex,
            job_id=job_id,
            event_type=event_type,
            status=status,
            message=message,
            payload=payload,
            created_at=datetime.now(UTC),
        )

    def _load_registry(self) -> dict[str, list[dict[str, Any]]]:
        try:
            return self.metadata_repository.load_registry("backtest_jobs")
        except MetadataRepositoryError as exc:
            raise BacktestJobRegistryError("Backtest job registry file is corrupted.") from exc

    def _save_registry(self, registry: dict[str, list[dict[str, Any]]]) -> None:
        try:
            self.metadata_repository.save_registry("backtest_jobs", registry)
        except MetadataRepositoryError as exc:
            raise BacktestJobRegistryError("Backtest job registry format is invalid.") from exc

    def _find_job_record_from_registry(
        self,
        registry: dict[str, list[dict[str, Any]]],
        job_id: str,
    ) -> dict[str, Any]:
        for record in registry["jobs"]:
            if record["job_id"] == job_id:
                return record

        raise BacktestJobNotFoundError(f"Backtest job not found: {job_id}")
