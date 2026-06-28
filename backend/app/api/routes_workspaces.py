from fastapi import APIRouter, Depends

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.workspace_schema import WorkspaceState, WorkspaceStateUpdate
from backend.app.services.workspace_state_service import WorkspaceStateService

router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


def get_workspace_state_service(
    settings: Settings = Depends(get_settings),
) -> WorkspaceStateService:
    return WorkspaceStateService(settings)


@router.get("/{workspace_id}/state", response_model=WorkspaceState)
def get_workspace_state(
    workspace_id: str,
    workspace_state_service: WorkspaceStateService = Depends(get_workspace_state_service),
) -> WorkspaceState:
    return workspace_state_service.get_state(workspace_id)


@router.patch("/{workspace_id}/state", response_model=WorkspaceState)
def update_workspace_state(
    workspace_id: str,
    request: WorkspaceStateUpdate,
    workspace_state_service: WorkspaceStateService = Depends(get_workspace_state_service),
) -> WorkspaceState:
    return workspace_state_service.update_state(workspace_id, request)
