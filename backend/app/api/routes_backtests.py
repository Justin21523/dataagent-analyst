import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.backtest_schema import (
    BacktestJobDetail,
    BacktestJobEventListResponse,
    BacktestJobListResponse,
    BacktestJobRequest,
    BacktestJobStartResponse,
    BacktestPayloadDetail,
    BacktestRunDetail,
    BacktestRunListResponse,
    BacktestSuiteListResponse,
    BacktestSuiteSummary,
)
from backend.app.services.backtest_artifact_service import (
    BacktestArtifactNotFoundError,
    BacktestArtifactService,
    BacktestArtifactValidationError,
)
from backend.app.services.backtest_job_registry_service import (
    BacktestJobNotFoundError,
    BacktestJobRegistryService,
)
from backend.app.services.backtest_job_runner_service import (
    TERMINAL_STATUSES,
    BacktestJobRunnerError,
    BacktestJobRunnerService,
)

router = APIRouter(prefix="/backtests", tags=["Backtests"])


def get_backtest_service(settings: Settings = Depends(get_settings)) -> BacktestArtifactService:
    return BacktestArtifactService(settings)


def get_backtest_job_registry_service(
    settings: Settings = Depends(get_settings),
) -> BacktestJobRegistryService:
    return BacktestJobRegistryService(settings)


def get_backtest_job_runner_service(
    settings: Settings = Depends(get_settings),
) -> BacktestJobRunnerService:
    return BacktestJobRunnerService(settings)


def run_backtest_job_background(
    settings: Settings,
    job_id: str,
    request: BacktestJobRequest,
) -> None:
    runner = BacktestJobRunnerService(settings)
    runner.run_job(job_id, request)


@router.post(
    "/jobs",
    response_model=BacktestJobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_backtest_job(
    request: BacktestJobRequest,
    background_tasks: BackgroundTasks,
    runner: BacktestJobRunnerService = Depends(get_backtest_job_runner_service),
) -> BacktestJobStartResponse:
    try:
        job = runner.create_job(request)
    except BacktestJobRunnerError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    background_tasks.add_task(run_backtest_job_background, runner.settings, job.job_id, request)

    return BacktestJobStartResponse(
        message="Backtest job accepted.",
        job=job,
    )


@router.get("/jobs", response_model=BacktestJobListResponse)
def list_backtest_jobs(
    limit: int = 25,
    registry: BacktestJobRegistryService = Depends(get_backtest_job_registry_service),
) -> BacktestJobListResponse:
    jobs = registry.list_jobs(limit=limit)
    return BacktestJobListResponse(jobs=jobs, total=len(jobs))


@router.get("/jobs/{job_id}", response_model=BacktestJobDetail)
def get_backtest_job(
    job_id: str,
    registry: BacktestJobRegistryService = Depends(get_backtest_job_registry_service),
) -> BacktestJobDetail:
    try:
        return registry.get_job(job_id)
    except BacktestJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/events", response_model=BacktestJobEventListResponse)
def list_backtest_job_events(
    job_id: str,
    registry: BacktestJobRegistryService = Depends(get_backtest_job_registry_service),
) -> BacktestJobEventListResponse:
    try:
        events = registry.list_events(job_id)
    except BacktestJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return BacktestJobEventListResponse(job_id=job_id, events=events, total=len(events))


@router.get("/jobs/{job_id}/events/stream")
def stream_backtest_job_events(
    job_id: str,
    registry: BacktestJobRegistryService = Depends(get_backtest_job_registry_service),
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        sent_event_ids: set[str] = set()

        while True:
            try:
                job = registry.get_job(job_id)
            except BacktestJobNotFoundError:
                payload = {
                    "event_type": "failed",
                    "status": "failed",
                    "message": f"Backtest job not found: {job_id}",
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
                    "run_ids": job.run_ids,
                    "suite_ids": job.suite_ids,
                }
                yield f"data: {json.dumps(close_payload, ensure_ascii=False)}\n\n"
                break

            await asyncio.sleep(0.75)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/runs", response_model=BacktestRunListResponse)
def list_backtest_runs(
    limit: int = 50,
    service: BacktestArtifactService = Depends(get_backtest_service),
) -> BacktestRunListResponse:
    runs = service.list_runs(limit=limit)
    return BacktestRunListResponse(runs=runs, total=len(runs))


@router.get("/runs/{run_id}", response_model=BacktestRunDetail)
def get_backtest_run(
    run_id: str,
    service: BacktestArtifactService = Depends(get_backtest_service),
) -> BacktestRunDetail:
    try:
        return service.get_run_detail(run_id)
    except BacktestArtifactNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except BacktestArtifactValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/runs/{run_id}/payloads/{payload_name}", response_model=BacktestPayloadDetail)
def get_backtest_payload(
    run_id: str,
    payload_name: str,
    service: BacktestArtifactService = Depends(get_backtest_service),
) -> BacktestPayloadDetail:
    try:
        return service.get_payload(run_id, payload_name)
    except BacktestArtifactNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except BacktestArtifactValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/runs/{run_id}/screenshots/{screenshot_name}")
def get_backtest_screenshot(
    run_id: str,
    screenshot_name: str,
    service: BacktestArtifactService = Depends(get_backtest_service),
) -> FileResponse:
    try:
        path = service.get_screenshot_path(run_id, screenshot_name)
    except BacktestArtifactNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except BacktestArtifactValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return FileResponse(path, media_type="image/png", filename=path.name)


@router.get("/suites", response_model=BacktestSuiteListResponse)
def list_backtest_suites(
    service: BacktestArtifactService = Depends(get_backtest_service),
) -> BacktestSuiteListResponse:
    suites = service.list_suites()
    return BacktestSuiteListResponse(suites=suites, total=len(suites))


@router.get("/suites/{suite_id}", response_model=BacktestSuiteSummary)
def get_backtest_suite(
    suite_id: str,
    service: BacktestArtifactService = Depends(get_backtest_service),
) -> BacktestSuiteSummary:
    try:
        return service.get_suite(suite_id)
    except BacktestArtifactNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except BacktestArtifactValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
