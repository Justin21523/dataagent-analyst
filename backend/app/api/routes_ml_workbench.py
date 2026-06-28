from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.ml_workbench_schema import (
    MLWorkbenchExperimentListResponse,
    MLWorkbenchExperimentRequest,
    MLWorkbenchExperimentResponse,
    MLWorkbenchPlanRequest,
    MLWorkbenchPlanResponse,
)
from backend.app.services.analysis_context_service import (
    AnalysisContextError,
)
from backend.app.services.dataset_service import (
    DatasetNotFoundError,
    DatasetValidationError,
)
from backend.app.services.ml_experiment_registry_service import (
    MLExperimentNotFoundError,
    MLExperimentRegistryService,
)
from backend.app.services.ml_workbench_service import (
    MLWorkbenchError,
    MLWorkbenchService,
)

router = APIRouter(
    prefix="/ml-workbench",
    tags=["ML Workbench v2"],
)


def get_ml_workbench_service(
    settings: Settings = Depends(get_settings),
) -> MLWorkbenchService:
    return MLWorkbenchService(settings)


def get_ml_experiment_registry_service(
    settings: Settings = Depends(get_settings),
) -> MLExperimentRegistryService:
    return MLExperimentRegistryService(settings)


@router.post(
    "/{dataset_id}/plan",
    response_model=MLWorkbenchPlanResponse,
)
def build_ml_workbench_plan(
    dataset_id: str,
    request: MLWorkbenchPlanRequest,
    service: MLWorkbenchService = Depends(get_ml_workbench_service),
) -> MLWorkbenchPlanResponse:
    try:
        return service.build_plan(
            dataset_id=dataset_id,
            request=request,
        )
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (
        DatasetValidationError,
        AnalysisContextError,
        MLWorkbenchError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/experiments",
    response_model=MLWorkbenchExperimentResponse,
)
def run_ml_workbench_experiment(
    dataset_id: str,
    request: MLWorkbenchExperimentRequest,
    service: MLWorkbenchService = Depends(get_ml_workbench_service),
) -> MLWorkbenchExperimentResponse:
    try:
        return service.run_experiment(
            dataset_id=dataset_id,
            request=request,
        )
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (
        DatasetValidationError,
        AnalysisContextError,
        MLWorkbenchError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "/{dataset_id}/experiments",
    response_model=MLWorkbenchExperimentListResponse,
)
def list_ml_workbench_experiments(
    dataset_id: str,
    registry: MLExperimentRegistryService = Depends(get_ml_experiment_registry_service),
) -> MLWorkbenchExperimentListResponse:
    experiments = registry.list_experiments(dataset_id)

    return MLWorkbenchExperimentListResponse(
        dataset_id=dataset_id,
        experiments=experiments,
        total=len(experiments),
    )


@router.get(
    "/experiments/{experiment_id}",
    response_model=MLWorkbenchExperimentResponse,
)
def get_ml_workbench_experiment(
    experiment_id: str,
    registry: MLExperimentRegistryService = Depends(get_ml_experiment_registry_service),
) -> MLWorkbenchExperimentResponse:
    try:
        return registry.get_experiment(experiment_id)
    except MLExperimentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
