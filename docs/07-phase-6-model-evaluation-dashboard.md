# Phase 6: Model Evaluation Dashboard

## Goal

Improve model result visibility by adding evaluation artifacts and frontend diagnostics.

This phase adds confusion matrix, regression residuals, feature importance display, and a model evaluation dashboard.

## Backend Deliverables

- Evaluation artifacts generated during training
- Confusion matrix for classification models
- Regression residual samples for regression models
- Evaluation artifacts saved as JSON files
- Model evaluation service
- Model evaluation API endpoint

## Frontend Deliverables

- Model Evaluation Dashboard
- Model Diagnostics panel
- Confusion matrix table
- Regression residual table
- Feature importance table
- Best model evaluation auto-loading

## API Endpoint

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/ml/models/{model_id}/evaluation` | Get model evaluation artifacts |

## Evaluation Flow

```text
Train Model
→ Evaluate metrics
→ Build confusion matrix or residual samples
→ Extract feature importance
→ Save evaluation artifacts JSON
→ Register artifact path
→ Frontend loads evaluation by model ID
→ Render diagnostics dashboard
