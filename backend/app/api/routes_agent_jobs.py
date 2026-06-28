import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.agent_job_schema import (
    AgentJobDetail,
    AgentJobEventListResponse,
    AgentJobListResponse,
    AgentJobStartResponse,
)
from backend.app.schemas.agent_schema import AgentRunRequest
from backend.app.services.agent_job_registry_service import (
    AgentJobNotFoundError,
    AgentJobRegistryService,
)
from backend.app.services.agent_job_runner_service import AgentJobRunnerService
from backend.app.services.dataset_service import DatasetNotFoundError

router = APIRouter(prefix="/agent-jobs", tags=["Agent Jobs"])

TERMINAL_STATUSES = {"success", "completed_with_warnings", "failed"}


def get_agent_job_registry_service(
    settings: Settings = Depends(get_settings),
) -> AgentJobRegistryService:
    # Registry service 提供 job status、events、history 查詢。
    return AgentJobRegistryService(settings)


def get_agent_job_runner_service(
    settings: Settings = Depends(get_settings),
) -> AgentJobRunnerService:
    # Runner service 提供 background workflow execution。
    return AgentJobRunnerService(settings)


def run_agent_job_background(
    settings: Settings,
    job_id: str,
    dataset_id: str,
    request: AgentRunRequest,
) -> None:
    # BackgroundTasks 只接收同步 callable；這裡重新建立 service 避免共用 request scope 物件。
    runner = AgentJobRunnerService(settings)
    runner.run_job(job_id, dataset_id, request)


@router.post(
    "/{dataset_id}/start",
    response_model=AgentJobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_agent_job(
    dataset_id: str,
    request: AgentRunRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    agent_job_runner_service: AgentJobRunnerService = Depends(get_agent_job_runner_service),
) -> AgentJobStartResponse:
    try:
        job = agent_job_runner_service.create_job(dataset_id, request)
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    background_tasks.add_task(
        run_agent_job_background,
        settings,
        job.job_id,
        dataset_id,
        request,
    )

    return AgentJobStartResponse(
        message="Agent job accepted.",
        job=job,
    )


@router.get("/{job_id}", response_model=AgentJobDetail)
def get_agent_job(
    job_id: str,
    agent_job_registry_service: AgentJobRegistryService = Depends(get_agent_job_registry_service),
) -> AgentJobDetail:
    try:
        return agent_job_registry_service.get_job(job_id)
    except AgentJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/dataset/{dataset_id}", response_model=AgentJobListResponse)
def list_agent_jobs(
    dataset_id: str,
    agent_job_registry_service: AgentJobRegistryService = Depends(get_agent_job_registry_service),
) -> AgentJobListResponse:
    jobs = agent_job_registry_service.list_jobs(dataset_id)

    return AgentJobListResponse(
        dataset_id=dataset_id,
        jobs=jobs,
        total=len(jobs),
    )


@router.get("/{job_id}/events", response_model=AgentJobEventListResponse)
def list_agent_job_events(
    job_id: str,
    agent_job_registry_service: AgentJobRegistryService = Depends(get_agent_job_registry_service),
) -> AgentJobEventListResponse:
    try:
        events = agent_job_registry_service.list_events(job_id)
    except AgentJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return AgentJobEventListResponse(
        job_id=job_id,
        events=events,
        total=len(events),
    )


@router.get("/{job_id}/events/stream")
def stream_agent_job_events(
    job_id: str,
    agent_job_registry_service: AgentJobRegistryService = Depends(get_agent_job_registry_service),
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        sent_event_ids: set[str] = set()

        while True:
            try:
                job = agent_job_registry_service.get_job(job_id)
            except AgentJobNotFoundError:
                payload = {
                    "event_type": "failed",
                    "status": "failed",
                    "message": f"Agent job not found: {job_id}",
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break

            for event in job.events:
                if event.event_id in sent_event_ids:
                    continue

                sent_event_ids.add(event.event_id)
                yield f"data: {json.dumps(event.model_dump(mode='json'), ensure_ascii=False)}\n\n"

            if job.status in TERMINAL_STATUSES:
                close_payload: dict[str, Any] = {
                    "event_type": "stream_closed",
                    "status": job.status,
                    "job_id": job.job_id,
                    "workflow_id": job.workflow_id,
                }
                yield f"data: {json.dumps(close_payload, ensure_ascii=False)}\n\n"
                break

            await asyncio.sleep(0.75)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
