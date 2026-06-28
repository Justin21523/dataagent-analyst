from backend.app.core.config import Settings
from backend.app.schemas.agent_job_schema import AgentJobDetail
from backend.app.schemas.agent_schema import AgentRunRequest
from backend.app.services.agent_job_registry_service import AgentJobRegistryService
from backend.app.services.agent_workflow_service import AgentWorkflowService
from backend.app.services.dataset_service import DatasetService


class AgentJobRunnerService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.agent_job_registry_service = AgentJobRegistryService(settings)
        self.dataset_service = DatasetService(settings)

    def create_job(self, dataset_id: str, request: AgentRunRequest) -> AgentJobDetail:
        # 先驗證 dataset 存在，避免建立永遠不可能成功的背景任務。
        self.dataset_service.get_dataset_detail(dataset_id)

        # 建立 queued job，API 會先回傳 job_id，再由 background task 執行。
        return self.agent_job_registry_service.create_job(dataset_id, request)

    def run_job(self, job_id: str, dataset_id: str, request: AgentRunRequest) -> None:
        # 背景任務入口：更新狀態、執行 workflow、記錄事件與結果。
        self.agent_job_registry_service.mark_running(job_id)

        workflow_service = AgentWorkflowService(self.settings)

        try:
            result = workflow_service.run_workflow(
                dataset_id=dataset_id,
                request=request,
                event_callback=lambda step: self.agent_job_registry_service.append_step_event(
                    job_id=job_id,
                    step=step,
                ),
            )
        except Exception as exc:
            self.agent_job_registry_service.mark_failed(job_id, str(exc))
            return

        self.agent_job_registry_service.mark_completed(job_id, result)
