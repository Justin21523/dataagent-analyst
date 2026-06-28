# Phase 0: Project Foundation

## Goal

Build the initial project foundation for DataAgent Analyst.

This phase creates a working FastAPI backend, a simple vanilla JavaScript frontend shell, health check endpoints, basic testing, and development tooling.

## Deliverables

- FastAPI application factory
- Centralized settings
- Basic logging configuration
- Health check API
- Readiness check API
- pytest API tests
- Vanilla JavaScript dashboard shell
- Backend API status check in frontend
- requirements.txt
- requirements-dev.txt

## Backend Architecture

```text
backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes_health.py
│   └── core/
│       ├── config.py
│       └── logging.py
└── tests/
    └── test_health.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root API navigation |
| GET | `/api/health` | Basic health check |
| GET | `/api/health/ready` | Runtime readiness check |

## Frontend Architecture

```text
frontend/
├── index.html
├── css/
│   └── base.css
└── js/
    └── app.js
```

## Run Backend

```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

## Run Frontend

```bash
python -m http.server 5173 --directory frontend
```

Open:

```text
http://127.0.0.1:5173
```

## Test Commands

```bash
python -m pytest
python -m ruff check backend
python -m ruff format backend
python -m mypy backend
```

## Completion Criteria

- Backend starts successfully.
- `/api/health` returns `status: ok`.
- `/api/health/ready` returns `status: ready`.
- Frontend can connect to backend health API.
- pytest passes.
- ruff check passes.
