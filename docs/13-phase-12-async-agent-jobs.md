# Phase 12: Async Agent Jobs & SSE Progress

## Goal

Upgrade agent workflow execution from synchronous API calls to background jobs with event history and SSE progress streaming.

## Backend Deliverables

- Agent job schemas
- Agent job registry service
- Agent job runner service
- Background task route
- Job status API
- Job list API
- Job event API
- SSE event stream endpoint
- Agent workflow callback support
- Agent job tests

## Frontend Deliverables

- Background Agent Job panel
- Start background job button
- Job history list
- Live / replay event panel
- EventSource SSE integration
- Job refresh action

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/agent-jobs/{dataset_id}/start` | Start background agent job |
| GET | `/api/agent-jobs/{job_id}` | Get job detail |
| GET | `/api/agent-jobs/dataset/{dataset_id}` | List jobs for dataset |
| GET | `/api/agent-jobs/{job_id}/events` | List job events |
| GET | `/api/agent-jobs/{job_id}/events/stream` | Stream job events with SSE |

## Job Flow

```text
Client starts job
→ API creates queued job
→ API returns job_id with 202 Accepted
→ BackgroundTasks runs workflow
→ Workflow emits step events
→ Job registry saves events
→ Frontend streams events with EventSource
→ Job completes and saves result
