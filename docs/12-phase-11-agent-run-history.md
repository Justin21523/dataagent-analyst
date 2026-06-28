# Phase 11: Agent Workflow History & Timeline Replay

## Goal

Add persistent workflow history and timeline replay to the Agent Workflow Lab.

This phase stores completed agent workflow runs in a JSON registry, exposes history APIs, and allows the frontend to replay previous workflow timelines and outputs.

## Backend Deliverables

- Agent run registry service
- Agent run summary schema
- Agent run list API
- Agent run detail API
- Agent run steps API
- Agent workflow persistence
- Agent run history tests

## Frontend Deliverables

- Agent Run History panel
- Refresh Runs button
- Workflow replay by clicking a previous run
- Timeline replay rendering
- Final outputs replay rendering

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/agents/{dataset_id}/runs` | List agent workflow runs for dataset |
| GET | `/api/agents/runs/{workflow_id}` | Get full workflow run detail |
| GET | `/api/agents/runs/{workflow_id}/steps` | Get workflow timeline steps |

## Storage

Agent runs are stored in:

```text
data/processed/agent_runs_registry.json
