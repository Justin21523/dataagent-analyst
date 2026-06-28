# Phase 7: Prediction API & Batch Inference

## Goal

Turn trained models into usable prediction services.

This phase adds single-record prediction, batch CSV prediction, frontend prediction UI, and result rendering.

## Backend Deliverables

- Model prediction service
- Single-record prediction API
- Batch CSV prediction API
- Prediction response schema
- Classification probability output
- Regression prediction output
- Prediction API tests

## Frontend Deliverables

- Prediction Lab
- Model selector
- JSON prediction input
- Batch CSV prediction upload
- Prediction result table
- Prediction status panel

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/ml/models/{model_id}/predict` | Predict JSON records |
| POST | `/api/ml/models/{model_id}/predict-csv` | Predict uploaded CSV rows |

## Prediction Flow

```text
Select trained model
→ Load model pipeline from joblib
→ Align input columns with training feature columns
→ Run pipeline.predict
→ Run pipeline.predict_proba if classification
→ Return predictions
→ Render prediction table
