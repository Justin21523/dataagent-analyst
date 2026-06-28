#!/usr/bin/env bash

set -euo pipefail

ENV_FILE="${ENV_FILE:-.env.docker}"
PYTHON="${PYTHON:-.venv/bin/python}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Run: cp .env.docker.example .env.docker"
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

COMPOSE_COMMAND=(
  docker
  compose
  --env-file
  "${ENV_FILE}"
)

resolve_service_url() {
  local service="$1"
  local private_port="$2"
  local published
  local host
  local port

  published="$("${COMPOSE_COMMAND[@]}" port "${service}" "${private_port}" | tail -n 1)"

  if [[ -z "${published}" ]]; then
    echo "Unable to resolve published port for ${service}:${private_port}" >&2
    exit 1
  fi

  host="${published%:*}"
  port="${published##*:}"

  if [[ "${host}" == "0.0.0.0" || "${host}" == "::" || "${host}" == "[::]" ]]; then
    host="127.0.0.1"
  fi

  printf 'http://%s:%s' "${host}" "${port}"
}

FRONTEND_URL="$(resolve_service_url frontend 80)"
BACKEND_URL="$(resolve_service_url backend 8000)"

echo "==> Resolved dynamic ports"
echo "    frontend=${FRONTEND_URL}"
echo "    backend=${BACKEND_URL}"

echo "==> Waiting for frontend"

frontend_ready=false

for _ in $(seq 1 90); do
  if curl -fsS "${FRONTEND_URL}/healthz" >/dev/null; then
    frontend_ready=true
    break
  fi

  sleep 1
done

if [[ "${frontend_ready}" != "true" ]]; then
  echo "Frontend did not become ready."
  "${COMPOSE_COMMAND[@]}" ps
  exit 1
fi

echo "==> Checking direct backend health"
curl -fsS "${BACKEND_URL}/api/health" >/dev/null

echo "==> Checking backend through Nginx"
curl -fsS "${FRONTEND_URL}/api/health" >/dev/null

echo "==> Checking infrastructure status"
infrastructure_response="$(
  curl -fsS "${FRONTEND_URL}/api/health/infrastructure"
)"

printf '%s' "${infrastructure_response}" |
  "${PYTHON}" -c '
import json
import sys

payload = json.load(sys.stdin)

if payload["status"] != "healthy":
    print(json.dumps(payload, indent=2))
    raise SystemExit("Infrastructure status is not healthy.")

print("Infrastructure services are healthy.")
'

echo "==> Running existing end-to-end smoke test through Nginx"
BASE_URL="${FRONTEND_URL}" bash scripts/smoke_test.sh

echo "==> Checking metadata persistence across backend restart"
dataset_count_before="$(
  curl -fsS "${FRONTEND_URL}/api/datasets" |
    "${PYTHON}" -c '
import json
import sys

payload = json.load(sys.stdin)
print(payload["total"])
'
)"

if [[ "${dataset_count_before}" -lt 1 ]]; then
  echo "Expected at least one dataset before backend restart."
  exit 1
fi

"${COMPOSE_COMMAND[@]}" restart backend >/dev/null

backend_ready=false

for _ in $(seq 1 90); do
  if curl -fsS "${FRONTEND_URL}/api/health" >/dev/null 2>&1; then
    backend_ready=true
    break
  fi

  sleep 1
done

if [[ "${backend_ready}" != "true" ]]; then
  echo "Backend did not become ready after restart."
  "${COMPOSE_COMMAND[@]}" ps
  exit 1
fi

backend_container="$("${COMPOSE_COMMAND[@]}" ps -q backend)"
backend_healthy=false

for _ in $(seq 1 90); do
  backend_health_status="$(
    docker inspect \
      --format '{{.State.Health.Status}}' \
      "${backend_container}" \
      2>/dev/null || true
  )"

  if [[ "${backend_health_status}" == "healthy" ]]; then
    backend_healthy=true
    break
  fi

  sleep 1
done

if [[ "${backend_healthy}" != "true" ]]; then
  echo "Backend Docker healthcheck did not become healthy after restart."
  "${COMPOSE_COMMAND[@]}" ps
  exit 1
fi

dataset_count_after="$(
  curl -fsS "${FRONTEND_URL}/api/datasets" |
    "${PYTHON}" -c '
import json
import sys

payload = json.load(sys.stdin)
print(payload["total"])
'
)"

if [[ "${dataset_count_after}" -lt "${dataset_count_before}" ]]; then
  echo "Dataset metadata count decreased after backend restart."
  echo "before=${dataset_count_before} after=${dataset_count_after}"
  exit 1
fi

echo "==> Docker service status"
"${COMPOSE_COMMAND[@]}" ps

echo "==> Docker smoke test passed"
