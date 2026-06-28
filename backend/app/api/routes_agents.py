from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.agent_schema import (
    AgentRunListResponse,
    AgentRunRequest,
    AgentRunResponse,
    AgentStepListResponse,
)
from backend.app.services.agent_run_registry_service import (
    AgentRunNotFoundError,
    AgentRunRegistryService,
)
from backend.app.services.agent_workflow_service import AgentWorkflowService
from backend.app.services.dataset_service import DatasetNotFoundError, DatasetValidationError
from backend.app.services.report_service import ReportServiceError

router = APIRouter(prefix="/agents", tags=["Agents"])


def get_agent_run_registry_service(
    settings: Settings = Depends(get_settings),
) -> AgentRunRegistryService:
    # Agent run registry service 負責查詢 workflow history 與 timeline replay。
    return AgentRunRegistryService(settings)


def get_agent_workflow_service(
    settings: Settings = Depends(get_settings),
) -> AgentWorkflowService:
    # Agent workflow service 負責把既有 tools 串成完整分析流程。
    return AgentWorkflowService(settings)


@router.post("/{dataset_id}/run", response_model=AgentRunResponse)
def run_agent_workflow(
    dataset_id: str,
    request: AgentRunRequest,
    agent_workflow_service: AgentWorkflowService = Depends(get_agent_workflow_service),
) -> AgentRunResponse:
    try:
        return agent_workflow_service.run_workflow(dataset_id, request)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (DatasetValidationError, ReportServiceError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/{dataset_id}/runs", response_model=AgentRunListResponse)
def list_agent_runs(
    dataset_id: str,
    agent_run_registry_service: AgentRunRegistryService = Depends(get_agent_run_registry_service),
) -> AgentRunListResponse:
    runs = agent_run_registry_service.list_runs(dataset_id)

    return AgentRunListResponse(
        dataset_id=dataset_id,
        runs=runs,
        total=len(runs),
    )


@router.get("/runs/{workflow_id}", response_model=AgentRunResponse)
def get_agent_run(
    workflow_id: str,
    agent_run_registry_service: AgentRunRegistryService = Depends(get_agent_run_registry_service),
) -> AgentRunResponse:
    try:
        return agent_run_registry_service.get_run(workflow_id)
    except AgentRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/runs/{workflow_id}/steps", response_model=AgentStepListResponse)
def get_agent_run_steps(
    workflow_id: str,
    agent_run_registry_service: AgentRunRegistryService = Depends(get_agent_run_registry_service),
) -> AgentStepListResponse:
    try:
        run = agent_run_registry_service.get_run(workflow_id)
    except AgentRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return AgentStepListResponse(
        workflow_id=workflow_id,
        steps=run.steps,
        total=len(run.steps),
    )
