.RECIPEPREFIX := >

PYTHON := .venv/bin/python
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8000
FRONTEND_PORT ?= 5173
BACKTEST_BACKEND_PORT ?= 8010
BACKTEST_FRONTEND_PORT ?= 5174

.PHONY: install test lint format typecheck frontend-check frontend-js-check check demo-data run-backend run-frontend smoke backtest scenario-backtest ui-backtest full-backtest playwright-install static-demo portfolio-media

install:
>uv pip install --python $(PYTHON) -r requirements.txt

test:
>$(PYTHON) -m pytest

lint:
>$(PYTHON) -m ruff check backend scripts

format:
>$(PYTHON) -m ruff format backend scripts

typecheck:
>$(PYTHON) -m mypy backend

frontend-check:
>$(PYTHON) scripts/check_frontend_contract.py

frontend-js-check:
>find frontend/js -type f -name '*.js' -print | sort | xargs -r -n 1 node --check

check: test lint typecheck frontend-check frontend-js-check

demo-data:
>$(PYTHON) scripts/generate_demo_dataset.py

run-backend:
>$(PYTHON) -m uvicorn backend.app.main:app --reload --host $(BACKEND_HOST) --port $(BACKEND_PORT)

run-frontend:
>$(PYTHON) -m http.server $(FRONTEND_PORT) --directory frontend

smoke:
>bash scripts/smoke_test.sh

backtest:
>$(PYTHON) scripts/backtest_full_workflow.py --base-url http://$(BACKEND_HOST):$(BACKTEST_BACKEND_PORT) --frontend-url http://$(BACKEND_HOST):$(BACKTEST_FRONTEND_PORT)

scenario-backtest:
>$(PYTHON) scripts/backtest_scenarios.py --base-url http://$(BACKEND_HOST):$(BACKTEST_BACKEND_PORT) --frontend-url http://$(BACKEND_HOST):$(BACKTEST_FRONTEND_PORT)

playwright-install:
>$(PYTHON) -m playwright install chromium

ui-backtest: playwright-install
>$(PYTHON) scripts/ui_backtest_workflow.py --base-url http://$(BACKEND_HOST):$(BACKTEST_BACKEND_PORT) --frontend-url http://$(BACKEND_HOST):$(BACKTEST_FRONTEND_PORT)

full-backtest: playwright-install
>$(PYTHON) scripts/backtest_full_workflow.py --base-url http://$(BACKEND_HOST):$(BACKTEST_BACKEND_PORT) --frontend-url http://$(BACKEND_HOST):$(BACKTEST_FRONTEND_PORT)
>$(PYTHON) scripts/backtest_scenarios.py --base-url http://$(BACKEND_HOST):$(BACKTEST_BACKEND_PORT) --frontend-url http://$(BACKEND_HOST):$(BACKTEST_FRONTEND_PORT)
>$(PYTHON) scripts/ui_backtest_workflow.py --base-url http://$(BACKEND_HOST):$(BACKTEST_BACKEND_PORT) --frontend-url http://$(BACKEND_HOST):$(BACKTEST_FRONTEND_PORT)

static-demo:
>$(PYTHON) scripts/build_static_demo.py

portfolio-media: playwright-install
>$(PYTHON) scripts/capture_portfolio_media.py --portfolio-dir ../justin-portfolio --base-url http://$(BACKEND_HOST):$(BACKTEST_BACKEND_PORT) --frontend-url http://$(BACKEND_HOST):$(BACKTEST_FRONTEND_PORT)

DOCKER_ENV_FILE ?= .env.docker
COMPOSE := docker compose --env-file $(DOCKER_ENV_FILE)

.PHONY: docker-init docker-config docker-build docker-pull docker-up docker-up-llm docker-up-cuda docker-down docker-reset docker-logs docker-ps docker-smoke

docker-init:
>test -f $(DOCKER_ENV_FILE) || cp .env.docker.example $(DOCKER_ENV_FILE)

docker-config: docker-init
>$(COMPOSE) config

docker-pull: docker-init
>$(COMPOSE) --profile llm pull

docker-build: docker-init frontend-vendor
>$(COMPOSE) build

docker-up: docker-init frontend-vendor
>$(COMPOSE) up -d --build

docker-up-llm: docker-init frontend-vendor
>LLM_ENABLED=true $(COMPOSE) --profile llm up -d --build

docker-up-cuda: docker-init frontend-vendor
>LLM_ENABLED=true docker compose --env-file $(DOCKER_ENV_FILE) -f compose.yaml -f compose.cuda.yaml --profile llm up -d --build

docker-down: docker-init
>$(COMPOSE) --profile llm down --remove-orphans

docker-reset: docker-init
>$(COMPOSE) --profile llm down --volumes --remove-orphans

docker-logs: docker-init
>$(COMPOSE) --profile llm logs --follow

docker-ps: docker-init
>$(COMPOSE) --profile llm ps

docker-smoke: docker-init
>bash scripts/docker_smoke_test.sh

.PHONY: frontend-vendor

frontend-vendor:
>bash scripts/install_frontend_vendor.sh
