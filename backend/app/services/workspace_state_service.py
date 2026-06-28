from datetime import UTC, datetime

from backend.app.core.config import Settings
from backend.app.repositories.metadata_repository import create_metadata_repository
from backend.app.schemas.workspace_schema import WorkspaceState, WorkspaceStateUpdate


class WorkspaceStateService:
    def __init__(self, settings: Settings) -> None:
        self.metadata_repository = create_metadata_repository(settings)

    def get_state(self, workspace_id: str) -> WorkspaceState:
        registry = self._load_registry()
        record = self._find_state_record(registry, workspace_id)

        if record is None:
            record = self._default_state_record(workspace_id)

        return WorkspaceState.model_validate(record)

    def update_state(
        self,
        workspace_id: str,
        update: WorkspaceStateUpdate,
    ) -> WorkspaceState:
        registry = self._load_registry()
        states = registry["states"]
        current = self._find_state_record(registry, workspace_id)

        if current is None:
            current = self._default_state_record(workspace_id)
            states.append(current)

        update_payload = update.model_dump(exclude_unset=True)
        now = datetime.now(UTC).isoformat()

        current.update(update_payload)
        current["workspace_id"] = workspace_id
        current["updated_at"] = now

        self.metadata_repository.save_registry("workspace_states", registry)
        return WorkspaceState.model_validate(current)

    def _load_registry(self) -> dict[str, list[dict]]:
        registry = self.metadata_repository.load_registry("workspace_states")
        states = registry.get("states")

        if not isinstance(states, list):
            return {"states": []}

        return registry

    def _find_state_record(
        self,
        registry: dict[str, list[dict]],
        workspace_id: str,
    ) -> dict | None:
        return next(
            (state for state in registry["states"] if state.get("workspace_id") == workspace_id),
            None,
        )

    def _default_state_record(self, workspace_id: str) -> dict:
        now = datetime.now(UTC).isoformat()
        return {
            "workspace_id": workspace_id,
            "active_route": "data-upload",
            "dataset_id": None,
            "dataset_version_id": None,
            "target_column": None,
            "selected_model_id": None,
            "drift_report_id": None,
            "drift_status": "Not checked",
            "workflow_flags": {},
            "retrain_candidate_id": None,
            "created_at": now,
            "updated_at": now,
        }
