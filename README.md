# DataAgent Analyst

DataAgent Analyst is a local-first AI data analysis and machine learning lifecycle workbench for tabular data.

It supports CSV upload, profiling, EDA, visualization, feature preparation, baseline ML training, model evaluation, explainability, drift analysis, champion/challenger retraining, agent workflow automation, and portable bundle export/import.

## Current Status

This repository is a portfolio-grade MVP with a working FastAPI backend, modular vanilla JavaScript frontend, local artifact storage, switchable JSON/PostgreSQL metadata persistence, Docker Compose deployment, and automated tests.

Implemented capabilities:

- CSV upload, dataset registry, preview, schema profiling, and dataset versioning
- Transformation recipes for missing values, duplicate rows, column drops, type casts, datetime parts, and IQR clipping
- EDA: missing values, duplicates, statistics, outliers, correlations, recommendations
- Visualization recommendations and ECharts-based Visualization Lab
- Baseline ML training and ML Workbench experiments for classification, regression, clustering, and anomaly detection
- Model registry with lifecycle status: candidate, staging, production, archived
- Prediction APIs for JSON records and batch CSV
- Model evaluation, SHAP explainability, permutation importance, error samples, threshold analysis, segment metrics, and what-if prediction
- Drift Center for schema, feature, target, prediction, and performance drift
- Champion/challenger retraining from drifted dataset versions
- Markdown report generation and optional local LLM explanations through llama.cpp
- LangGraph agent workflow with async jobs, history, and SSE progress events
- Bundle export/import for dataset and model artifacts

Known limitations:

- Metadata persistence defaults to JSON registries for local development and tests; Docker can run the same metadata contracts on PostgreSQL.
- Qdrant is available in Docker for future RAG work, but dataset knowledge RAG is not implemented yet.
- The local LLM is optional and defaults to disabled; AI explanations fall back to deterministic text when offline.
- Runtime artifacts under `data/` are local state and should not be committed.

## Architecture

```text
Browser
  -> Vanilla JS frontend
     -> FastAPI /api
     -> Dataset, EDA, Visualization, ML, Explainability, Drift, Bundle, Agent services
     -> Metadata repository: JSON registries or PostgreSQL tables
     -> CSV/model/report artifacts in data/
     -> Optional llama.cpp local LLM
```

Main backend entrypoint:

- `backend/app/main.py`

Important service areas:

- Dataset lifecycle: `DatasetService`
- ML training and experiments: `MLTrainingService`, `MLWorkbenchService`
- Model lifecycle and diagnostics: `ModelRegistryService`, `ModelDiagnosticsService`, `ModelRetrainingService`
- Drift: `DriftService`
- Bundles: `BundleService`
- Agent workflow: `AgentWorkflowService`, `AgentJobRunnerService`

## Quick Start

### 1. Create and activate the Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you use `uv`:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

### 2. Run the backend

```bash
make run-backend
```

Backend defaults to:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### 3. Run the frontend

```bash
make run-frontend
```

Frontend defaults to:

```text
http://127.0.0.1:5173
```

### 4. Generate demo data

```bash
make demo-data
```

Sample CSV files live in:

```text
data/samples/
```

## Development Checks

Run the full project check:

```bash
make check
```

Individual checks:

```bash
make test
make lint
make typecheck
make frontend-check
make frontend-js-check
```

Smoke test against a running backend:

```bash
make smoke
```

## Metadata Persistence

Local development defaults to JSON registries under `data/processed`:

```bash
METADATA_BACKEND=json
```

Docker defaults to PostgreSQL metadata tables while keeping CSV, model, report,
and bundle artifacts in the existing `data/` volumes:

```bash
METADATA_BACKEND=postgres
DATABASE_URL=postgresql://dataagent:dataagent_dev_password@postgres:5432/dataagent
```

To migrate existing JSON registry metadata into PostgreSQL:

```bash
python scripts/migrate_metadata_to_postgres.py --dry-run
python scripts/migrate_metadata_to_postgres.py
```

The migration is idempotent and updates rows by metadata ID. It does not copy
artifact files; it validates artifact references and stores their relative paths.

## Docker Compose

Initialize Docker environment variables:

```bash
make docker-init
```

Start the production-like local stack:

```bash
make docker-up
```

Run Docker smoke tests:

```bash
make docker-smoke
```

Optional local LLM profile:

```bash
make docker-up-llm
```

CUDA variant:

```bash
make docker-up-cuda
```

Docker services include frontend/Nginx, backend/FastAPI, PostgreSQL, Qdrant, and optional llama.cpp.

## Lifecycle Workflow

Recommended demo flow:

1. Upload a CSV dataset.
2. Review preview, schema, column profiles, EDA, and visualizations.
3. Create a derived dataset version with a transformation recipe.
4. Train baseline or ML Workbench models.
5. Promote the best supervised model to production.
6. Create a new dataset version and run a drift report against the reference version.
7. If drift recommends action, retrain a challenger model on the current version.
8. Compare champion vs challenger.
9. Promote the challenger if it outperforms the champion.
10. Generate a Markdown report or bundle export for migration.

See `docs/17-lifecycle-workbench.md` for implementation details.

## Local LLM

The LLM integration uses an OpenAI-compatible llama.cpp server.

Default:

```text
LLM_ENABLED=false
```

To enable locally, set:

```bash
LLM_ENABLED=true
LLM_BASE_URL=http://127.0.0.1:8080
LLM_MODEL=local-model
```

When the LLM is disabled or unreachable, the API returns deterministic fallback explanations so the application remains usable.

## Runtime Data Policy

Committed files under `data/` should be limited to:

- `data/samples/*.csv`
- `.gitkeep` placeholders

Generated datasets, models, reports, registries, drift reports, bundles, imports, and vector-store files are local runtime artifacts and are ignored by Git.

<!-- portfolio-readme:begin -->

## Portfolio Documentation

### Project Overview

**DataAgent Analyst** is maintained as part of the Justin21523 GitHub portfolio. Demo: dataagent-analyst

### Features

- Demo: dataagent-analyst
- Python workflow for data processing, automation, AI, or backend tasks.
- API-oriented backend structure suitable for service integration.

### Tech Stack

- HTML
- Python
- FastAPI
- pandas
- Makefile

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

- Run `python -m venv .venv`
- Run `source .venv/bin/activate`
- Run `pip install -r requirements.txt`

### Project Structure

```text
dataagent-analyst/
  .cache/
  .dockerignore
  .editorconfig
  .env.docker
  .env.docker.example
  .env.example
  .gitignore
  .mypy_cache/
  .pytest_cache/
  .ruff_cache/
  Makefile
  README.md
```

### Environment Variables

- `APP_ENV`: TODO: Document expected value and whether it is required.
- `LOG_LEVEL`: TODO: Document expected value and whether it is required.
- `LLM_ENABLED`: TODO: Document expected value and whether it is required.
- `LLM_BASE_URL`: TODO: Document expected value and whether it is required.
- `LLM_MODEL`: TODO: Document expected value and whether it is required.
- `LLM_TIMEOUT_SECONDS`: TODO: Document expected value and whether it is required.
- `LLM_TEMPERATURE`: TODO: Document expected value and whether it is required.
- `LLM_MAX_TOKENS`: TODO: Document expected value and whether it is required.
- `INFRASTRUCTURE_CHECKS_ENABLED`: TODO: Document expected value and whether it is required.
- `INFRASTRUCTURE_TIMEOUT_SECONDS`: TODO: Document expected value and whether it is required.
- `METADATA_BACKEND`: TODO: Document expected value and whether it is required.
- `DATABASE_URL`: TODO: Document expected value and whether it is required.
- `QDRANT_URL`: TODO: Document expected value and whether it is required.
- `QDRANT_API_KEY`: TODO: Document expected value and whether it is required.
- `EXPLAINABILITY_MAX_SAMPLE_SIZE`: TODO: Document expected value and whether it is required.
- `EXPLAINABILITY_MAX_BACKGROUND_SIZE`: TODO: Document expected value and whether it is required.
- `EXPLAINABILITY_MAX_PERMUTATION_REPEATS`: TODO: Document expected value and whether it is required.
- `EXPLAINABILITY_MAX_BEESWARM_FEATURES`: TODO: Document expected value and whether it is required.
- `EXPLAINABILITY_MAX_LOCAL_FEATURES`: TODO: Document expected value and whether it is required.

### Deployment

- Demo / GitHub Pages: https://justin21523.github.io/dataagent-analyst/
- Static demo build: `make static-demo`
- Portfolio media capture: `make portfolio-media`

The GitHub Pages demo is a static interactive build of the real frontend. It uses deterministic fixture data and a browser-side mock API so the UI, route shell, guided helper, model lifecycle views, reports, and backtest viewer can be reviewed without running FastAPI on GitHub Pages.

### Demo

- Live demo: https://justin21523.github.io/dataagent-analyst/
- Source: https://github.com/Justin21523/dataagent-analyst
- README: https://github.com/Justin21523/dataagent-analyst#readme
- Demo video: linked from the portfolio media gallery.

### Screenshots

- Portfolio media includes Playwright-captured screenshots for upload, guided tour, dataset preview, schema, EDA, visualization lab, ML Workbench, model registry, prediction, explainability, drift, reports, backtests, agent jobs, AI insights, and mobile guide views.
- The default portfolio video demonstrates the guided helper overlay and the end-to-end DataAgent workflow shell.

### License

- TODO: Add or confirm the repository license.

### Maintainer

- Justin21523 - https://github.com/Justin21523

<!-- portfolio-readme:end -->
