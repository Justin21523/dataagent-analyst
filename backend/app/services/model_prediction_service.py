from io import BytesIO
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from backend.app.core.config import Settings
from backend.app.schemas.ml_schema import (
    BatchPredictionResponse,
    MLModelResult,
    PredictionResponse,
    PredictionResult,
)
from backend.app.services.model_registry_service import ModelRegistryService


class ModelPredictionError(Exception):
    """Raised when model prediction fails."""


class ModelPredictionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_registry_service = ModelRegistryService(settings)

    def predict_records(
        self,
        model_id: str,
        records: list[dict[str, Any]],
    ) -> PredictionResponse:
        # 單筆 / 多筆 JSON 預測共用同一條 pipeline，避免前後端行為不一致。
        if not records:
            raise ModelPredictionError("Prediction records cannot be empty.")

        model = self.model_registry_service.get_model(model_id)
        pipeline = self._load_pipeline(model)
        prediction_dataframe = self._build_prediction_dataframe(model, records)

        prediction_results = self._run_predictions(
            model=model,
            pipeline=pipeline,
            prediction_dataframe=prediction_dataframe,
        )

        return PredictionResponse(
            model_id=model.id,
            dataset_id=model.dataset_id,
            model_name=model.model_name,
            task_type=model.task_type,
            target_column=model.target_column,
            feature_columns=model.feature_columns,
            predictions=prediction_results,
            total=len(prediction_results),
        )

    def predict_csv(
        self,
        model_id: str,
        original_filename: str,
        file_content: bytes,
    ) -> BatchPredictionResponse:
        # 批次預測使用 CSV，方便作品集展示「模型可被實際使用」。
        if not file_content:
            raise ModelPredictionError("Prediction CSV file is empty.")

        model = self.model_registry_service.get_model(model_id)
        pipeline = self._load_pipeline(model)
        dataframe = self._read_prediction_csv(file_content)
        prediction_dataframe = self._build_prediction_dataframe_from_dataframe(model, dataframe)

        prediction_results = self._run_predictions(
            model=model,
            pipeline=pipeline,
            prediction_dataframe=prediction_dataframe,
        )

        return BatchPredictionResponse(
            model_id=model.id,
            dataset_id=model.dataset_id,
            model_name=model.model_name,
            task_type=model.task_type,
            target_column=model.target_column,
            feature_columns=model.feature_columns,
            predictions=prediction_results,
            total=len(prediction_results),
            original_filename=original_filename,
        )

    def _load_pipeline(self, model: MLModelResult) -> Pipeline:
        model_path = self.settings.project_root / model.model_path

        if not model_path.exists():
            raise ModelPredictionError("Saved model file is missing.")

        try:
            pipeline = joblib.load(model_path)
        except Exception as exc:
            raise ModelPredictionError(f"Failed to load model: {exc}") from exc

        if not isinstance(pipeline, Pipeline):
            raise ModelPredictionError("Saved model is not a scikit-learn Pipeline.")

        return pipeline

    def _build_prediction_dataframe(
        self,
        model: MLModelResult,
        records: list[dict[str, Any]],
    ) -> pd.DataFrame:
        dataframe = pd.DataFrame(records)
        return self._build_prediction_dataframe_from_dataframe(model, dataframe)

    def _build_prediction_dataframe_from_dataframe(
        self,
        model: MLModelResult,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        if dataframe.empty:
            raise ModelPredictionError("Prediction data cannot be empty.")

        # reindex 會補齊缺少欄位為 NaN，讓訓練時的 imputer 接手處理。
        return dataframe.reindex(columns=model.feature_columns)

    def _read_prediction_csv(self, file_content: bytes) -> pd.DataFrame:
        supported_encodings = ["utf-8", "utf-8-sig", "big5", "cp950", "latin1"]
        last_error: Exception | None = None

        for encoding in supported_encodings:
            try:
                return pd.read_csv(BytesIO(file_content), encoding=encoding, low_memory=False)
            except UnicodeDecodeError as exc:
                last_error = exc
            except pd.errors.ParserError as exc:
                raise ModelPredictionError(f"Failed to parse prediction CSV: {exc}") from exc

        raise ModelPredictionError(f"Failed to read prediction CSV: {last_error}")

    def _run_predictions(
        self,
        model: MLModelResult,
        pipeline: Pipeline,
        prediction_dataframe: pd.DataFrame,
    ) -> list[PredictionResult]:
        try:
            predictions = pipeline.predict(prediction_dataframe)
        except Exception as exc:
            raise ModelPredictionError(f"Prediction failed: {exc}") from exc

        probabilities = self._build_probability_payload(
            model=model,
            pipeline=pipeline,
            prediction_dataframe=prediction_dataframe,
        )

        results = []

        for index, prediction in enumerate(predictions.tolist()):
            probability_payload = probabilities[index] if probabilities else None

            results.append(
                PredictionResult(
                    row_index=index,
                    prediction=self._to_json_safe_value(prediction),
                    probabilities=probability_payload,
                )
            )

        return results

    def _build_probability_payload(
        self,
        model: MLModelResult,
        pipeline: Pipeline,
        prediction_dataframe: pd.DataFrame,
    ) -> list[dict[str, float]] | None:
        if model.task_type != "classification":
            return None

        if not hasattr(pipeline, "predict_proba"):
            return None

        estimator = pipeline.named_steps.get("model")

        if estimator is None or not hasattr(estimator, "classes_"):
            return None

        try:
            probability_matrix = pipeline.predict_proba(prediction_dataframe)
        except Exception:
            return None

        labels = [str(label) for label in estimator.classes_.tolist()]
        probability_payload = []

        for row in probability_matrix:
            probability_payload.append(
                {
                    labels[index]: round(float(probability), 4)
                    for index, probability in enumerate(row)
                }
            )

        return probability_payload

    def _to_json_safe_value(self, value: Any) -> Any:
        if pd.isna(value):
            return None

        if isinstance(value, np.generic):
            return value.item()

        return value
