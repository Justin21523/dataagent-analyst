# Phase 14: Docker Compose and Production-like Local Deployment

## Goal

Containerize DataAgent Analyst and provide one-command local deployment.

## Services

| Service | Purpose | Host Port |
|---|---|---|
| frontend | Nginx static frontend and API proxy | Dynamic by default |
| backend | FastAPI application | Dynamic by default |
| postgres | Relational database infrastructure | Dynamic by default |
| qdrant | Vector database infrastructure | Dynamic by default |
| llama | Optional local LLM server | Dynamic by default |

Docker Compose publishes host ports with `published: 0` by default. To inspect
the assigned ports, run:

```bash
docker compose --env-file .env.docker port frontend 80
docker compose --env-file .env.docker port backend 8000
docker compose --env-file .env.docker port postgres 5432
docker compose --env-file .env.docker port qdrant 6333
```

Set `FRONTEND_PORT`, `BACKEND_PORT`, `POSTGRES_PORT`, `QDRANT_PORT`,
`QDRANT_GRPC_PORT`, or `LLM_PORT` in `.env.docker` only when a fixed host port is
explicitly required.

## Architecture

```text
Browser
→ Nginx Frontend
  → Static frontend files
  → /api reverse proxy
    → FastAPI Backend
      → PostgreSQL metadata store
      → Qdrant
      → Optional llama.cpp
