from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.ml_schema import (
    BatchPredictionResponse,
    MLModelEvaluationResponse,
    MLModelListResponse,
    MLModelMetricsResponse,
    MLModelResult,
    MLTrainRequest,
    MLTrainResponse,
    ModelCompareItem,
    ModelCompareRequest,
    ModelCompareResponse,
    ModelMigrationCheckResponse,
    ModelRetrainPlanResponse,
    ModelRetrainRequest,
    ModelRetrainResponse,
    ModelStatusUpdateRequest,
    PredictionRequest,
    PredictionResponse,
    PromoteChallengerRequest,
    PromoteChallengerResponse,
    SegmentMetricsRequest,
    SegmentMetricsResponse,
    ThresholdAnalysisRequest,
    ThresholdAnalysisResponse,
    WhatIfRequest,
    WhatIfResponse,
)
from backend.app.services.dataset_service import DatasetNotFoundError, DatasetValidationError
from backend.app.services.ml_training_service import MLTrainingError, MLTrainingService
from backend.app.services.model_diagnostics_service import (
    ModelDiagnosticsError,
    ModelDiagnosticsService,
)
from backend.app.services.model_evaluation_service import ModelEvaluationService
from backend.app.services.model_prediction_service import (
    ModelPredictionError,
    ModelPredictionService,
)
from backend.app.services.model_registry_service import ModelNotFoundError, ModelRegistryService
from backend.app.services.model_retraining_service import (
    ModelRetrainingError,
    ModelRetrainingService,
)

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


def get_ml_training_service(settings: Settings = Depends(get_settings)) -> MLTrainingService:
    # Training service 負責完整 ML pipeline，不在 route 裡直接寫訓練邏輯。
    return MLTrainingService(settings)


def get_model_prediction_service(
    settings: Settings = Depends(get_settings),
) -> ModelPredictionService:
    # Model prediction service 負責載入已訓練模型並執行推論。
    return ModelPredictionService(settings)


def get_model_evaluation_service(
    settings: Settings = Depends(get_settings),
) -> ModelEvaluationService:
    # Model evaluation service 負責讀取 confusion matrix、residuals、feature importance。
    return ModelEvaluationService(settings)


def get_model_registry_service(settings: Settings = Depends(get_settings)) -> ModelRegistryService:
    # Model registry service 負責查詢已訓練模型紀錄。
    return ModelRegistryService(settings)


def get_model_diagnostics_service(
    settings: Settings = Depends(get_settings),
) -> ModelDiagnosticsService:
    return ModelDiagnosticsService(settings)


def get_model_retraining_service(
    settings: Settings = Depends(get_settings),
) -> ModelRetrainingService:
    return ModelRetrainingService(settings)


@router.post(
    "/{dataset_id}/train",
    response_model=MLTrainResponse,
    status_code=status.HTTP_201_CREATED,
)
def train_models(
    dataset_id: str,
    request: MLTrainRequest,
    ml_training_service: MLTrainingService = Depends(get_ml_training_service),
) -> MLTrainResponse:
    try:
        return ml_training_service.train_models(dataset_id, request)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (DatasetValidationError, MLTrainingError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/{dataset_id}/models", response_model=MLModelListResponse)
def list_dataset_models(
    dataset_id: str,
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service),
) -> MLModelListResponse:
    models = model_registry_service.list_models(dataset_id=dataset_id)

    return MLModelListResponse(
        dataset_id=dataset_id,
        models=models,
        total=len(models),
    )


@router.get("/{dataset_id}/models/production", response_model=MLModelResult)
def get_production_model(
    dataset_id: str,
    target_column: str | None = None,
    task_type: str | None = None,
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service),
) -> MLModelResult:
    try:
        return model_registry_service.get_production_model(
            dataset_id=dataset_id,
            target_column=target_column,
            task_type=task_type,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/{dataset_id}/models/compare", response_model=ModelCompareResponse)
def compare_models(
    dataset_id: str,
    request: ModelCompareRequest,
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service),
) -> ModelCompareResponse:
    models = [
        model
        for model in model_registry_service.list_models(dataset_id=dataset_id)
        if model.id in set(request.model_ids)
    ]

    if not models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No comparable models were found.",
        )

    primary_metric = request.primary_metric or (
        "rmse" if models[0].task_type == "regression" else "f1_macro"
    )
    reverse = primary_metric != "rmse"

    def metric_sort_value(model: MLModelResult) -> float:
        value = model.metrics.get(primary_metric)

        if value is None:
            return -1.0 if reverse else float("inf")

        return float(value)

    models = sorted(
        models,
        key=metric_sort_value,
        reverse=reverse,
    )

    return ModelCompareResponse(
        dataset_id=dataset_id,
        primary_metric=primary_metric,
        models=[
            ModelCompareItem(
                model_id=model.id,
                model_name=model.model_name,
                lifecycle_status=model.lifecycle_status,
                task_type=model.task_type,
                target_column=model.target_column,
                metrics=model.metrics,
                primary_metric_value=model.metrics.get(primary_metric),
                rank=index + 1,
            )
            for index, model in enumerate(models)
        ],
    )


@router.get("/models/{model_id}", response_model=MLModelResult)
def get_model_detail(
    model_id: str,
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service),
) -> MLModelResult:
    try:
        return model_registry_service.get_model(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.patch("/models/{model_id}/status", response_model=MLModelResult)
def update_model_status(
    model_id: str,
    request: ModelStatusUpdateRequest,
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service),
) -> MLModelResult:
    try:
        return model_registry_service.update_model_status(
            model_id=model_id,
            lifecycle_status=request.lifecycle_status,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/models/{model_id}/metrics", response_model=MLModelMetricsResponse)
def get_model_metrics(
    model_id: str,
    model_registry_service: ModelRegistryService = Depends(get_model_registry_service),
) -> MLModelMetricsResponse:
    try:
        model = model_registry_service.get_model(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return MLModelMetricsResponse(
        model_id=model.id,
        metrics=model.metrics,
    )


@router.get("/models/{model_id}/evaluation", response_model=MLModelEvaluationResponse)
def get_model_evaluation(
    model_id: str,
    model_evaluation_service: ModelEvaluationService = Depends(get_model_evaluation_service),
) -> MLModelEvaluationResponse:
    try:
        return model_evaluation_service.get_model_evaluation(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/models/{model_id}/predict", response_model=PredictionResponse)
def predict_with_model(
    model_id: str,
    request: PredictionRequest,
    model_prediction_service: ModelPredictionService = Depends(get_model_prediction_service),
) -> PredictionResponse:
    try:
        return model_prediction_service.predict_records(
            model_id=model_id,
            records=request.records,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ModelPredictionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/models/{model_id}/predict-csv", response_model=BatchPredictionResponse)
async def predict_csv_with_model(
    model_id: str,
    file: UploadFile = File(...),
    model_prediction_service: ModelPredictionService = Depends(get_model_prediction_service),
) -> BatchPredictionResponse:
    try:
        file_content = await file.read()
        return model_prediction_service.predict_csv(
            model_id=model_id,
            original_filename=file.filename or "prediction.csv",
            file_content=file_content,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ModelPredictionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/models/{model_id}/threshold-analysis",
    response_model=ThresholdAnalysisResponse,
)
def analyze_model_thresholds(
    model_id: str,
    request: ThresholdAnalysisRequest,
    service: ModelDiagnosticsService = Depends(get_model_diagnostics_service),
) -> ThresholdAnalysisResponse:
    try:
        return service.threshold_analysis(model_id, request)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (DatasetValidationError, DatasetNotFoundError, ModelDiagnosticsError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/models/{model_id}/segment-metrics",
    response_model=SegmentMetricsResponse,
)
def get_segment_metrics(
    model_id: str,
    request: SegmentMetricsRequest,
    service: ModelDiagnosticsService = Depends(get_model_diagnostics_service),
) -> SegmentMetricsResponse:
    try:
        return service.segment_metrics(model_id, request)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (DatasetValidationError, DatasetNotFoundError, ModelDiagnosticsError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/models/{model_id}/what-if", response_model=WhatIfResponse)
def run_what_if(
    model_id: str,
    request: WhatIfRequest,
    service: ModelDiagnosticsService = Depends(get_model_diagnostics_service),
) -> WhatIfResponse:
    try:
        return service.what_if(model_id, request)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (ModelDiagnosticsError, ModelPredictionError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/models/{model_id}/migration-check",
    response_model=ModelMigrationCheckResponse,
)
def check_model_migration(
    model_id: str,
    service: ModelDiagnosticsService = Depends(get_model_diagnostics_service),
) -> ModelMigrationCheckResponse:
    try:
        return service.migration_check(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/models/{model_id}/retrain-plan",
    response_model=ModelRetrainPlanResponse,
)
def build_model_retrain_plan(
    model_id: str,
    request: ModelRetrainRequest,
    service: ModelRetrainingService = Depends(get_model_retraining_service),
) -> ModelRetrainPlanResponse:
    try:
        return service.build_plan(model_id, request)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (DatasetNotFoundError, DatasetValidationError, ModelRetrainingError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/models/{model_id}/retrain",
    response_model=ModelRetrainResponse,
)
def retrain_model_challenger(
    model_id: str,
    request: ModelRetrainRequest,
    service: ModelRetrainingService = Depends(get_model_retraining_service),
) -> ModelRetrainResponse:
    try:
        return service.retrain(model_id, request)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (
        DatasetNotFoundError,
        DatasetValidationError,
        MLTrainingError,
        ModelRetrainingError,
    ) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/models/{model_id}/promote-challenger",
    response_model=PromoteChallengerResponse,
)
def promote_model_challenger(
    model_id: str,
    request: PromoteChallengerRequest,
    service: ModelRetrainingService = Depends(get_model_retraining_service),
) -> PromoteChallengerResponse:
    try:
        return service.promote_challenger(
            champion_model_id=model_id,
            challenger_model_id=request.challenger_model_id,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ModelRetrainingError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
