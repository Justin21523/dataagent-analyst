#!/usr/bin/env bash

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

echo "==> Running pytest"
"${PYTHON}" -m pytest

echo "==> Running Ruff lint"
"${PYTHON}" -m ruff check backend scripts

echo "==> Checking Ruff formatting"
"${PYTHON}" -m ruff format --check backend scripts

echo "==> Running mypy"
"${PYTHON}" -m mypy backend

echo "==> Checking frontend DOM contract"
"${PYTHON}" scripts/check_frontend_contract.py

echo "==> Checking frontend JavaScript syntax"
while IFS= read -r javascript_file; do
  node --check "${javascript_file}" >/dev/null
done < <(find frontend/js -type f -name '*.js' | sort)

echo "==> All project checks passed"
