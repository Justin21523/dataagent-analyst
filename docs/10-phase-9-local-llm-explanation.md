# Phase 9: Local LLM Explanation with llama.cpp

## Goal

Add optional local LLM explanations to DataAgent Analyst.

This phase integrates an OpenAI-compatible local LLM server through HTTP. The system works even when the LLM server is disabled or offline by using deterministic fallback explanations.

## Backend Deliverables

- Local LLM service
- AI Insight service
- AI Insight API routes
- LLM status endpoint
- EDA explanation endpoint
- Model explanation endpoint
- Report summary endpoint
- Fallback explanations when LLM is disabled

## Frontend Deliverables

- AI Insight Panel
- Local LLM status card
- Optional user goal input
- Generate EDA explanation button
- Generate model explanation button
- Generate report summary button
- AI explanation output viewer

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/ai-insights/status` | Check local LLM status |
| POST | `/api/ai-insights/{dataset_id}/eda` | Generate EDA explanation |
| POST | `/api/ai-insights/{dataset_id}/models/{model_id}` | Generate model explanation |
| POST | `/api/ai-insights/{dataset_id}/report-summary` | Generate report summary |

## Local LLM Runtime

Install optional LLM dependencies:

```bash
uv pip install -r requirements-llm.txt
