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

echo "==> Docker service status"
"${COMPOSE_COMMAND[@]}" ps

echo "==> Docker smoke test passed"
