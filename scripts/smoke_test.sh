#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
PYTHON="${PYTHON:-.venv/bin/python}"
SAMPLE_FILE="${SAMPLE_FILE:-data/samples/customer_churn_demo.csv}"

extract_json_value() {
  local expression="$1"

  "${PYTHON}" -c "
import json
import sys

payload = json.load(sys.stdin)
print(${expression})
"
}

echo "==> Checking API health"
curl -fsS "${BASE_URL}/api/health" >/dev/null

if [[ ! -f "${SAMPLE_FILE}" ]]; then
  echo "==> Demo dataset missing; generating it"
  "${PYTHON}" scripts/generate_demo_dataset.py
fi

echo "==> Uploading demo dataset"
upload_response="$(
  curl -fsS \
    -X POST \
    -F "file=@${SAMPLE_FILE}" \
    "${BASE_URL}/api/datasets/upload"
)"

dataset_id="$(
  printf '%s' "${upload_response}" |
    extract_json_value 'payload["dataset"]["id"]'
)"

echo "    dataset_id=${dataset_id}"

echo "==> Checking schema"
curl -fsS \
  "${BASE_URL}/api/datasets/${dataset_id}/schema" \
  >/dev/null

echo "==> Checking EDA"
curl -fsS \
  "${BASE_URL}/api/eda/${dataset_id}/summary" \
  >/dev/null

echo "==> Checking visualization recommendations"
curl -fsS \
  "${BASE_URL}/api/visualizations/${dataset_id}/recommendations" \
  >/dev/null


echo "==> Checking Visualization Lab v2"
curl -fsS \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "target_column": "churn",
    "sample_rows": 80,
    "max_numeric_columns": 8,
    "max_categories": 12
  }' \
  "${BASE_URL}/api/visualizations/${dataset_id}/lab" \
  >/dev/null

echo "==> Checking ML Workbench v2 plan"
curl -fsS \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "target_column": "churn",
    "task_type": "auto",
    "include_datetime": true,
    "include_text": false
  }' \
  "${BASE_URL}/api/ml-workbench/${dataset_id}/plan" \
  >/dev/null

echo "==> Running ML Workbench v2 experiment"
curl -fsS \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "target_column": "churn",
    "task_type": "binary_classification",
    "selected_models": ["logistic_regression"],
    "include_datetime": true,
    "include_text": false,
    "cv_folds": 3,
    "test_size": 0.2
  }' \
  "${BASE_URL}/api/ml-workbench/${dataset_id}/experiments" \
  >/dev/null

echo "==> Training baseline ML models"
training_response="$(
  curl -fsS \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
      "target_column": "churn",
      "task_type": "auto",
      "test_size": 0.25,
      "random_state": 42
    }' \
    "${BASE_URL}/api/ml/${dataset_id}/train"
)"

model_id="$(
  printf '%s' "${training_response}" |
    extract_json_value 'payload["best_model_id"]'
)"

echo "    best_model_id=${model_id}"

echo "==> Checking model evaluation"
curl -fsS \
  "${BASE_URL}/api/ml/models/${model_id}/evaluation" \
  >/dev/null


echo "==> Checking Model Explainability"
curl -fsS \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "sample_size": 40,
    "background_size": 20,
    "permutation_repeats": 3,
    "local_row_position": 0,
    "include_permutation": true,
    "include_shap": true
  }' \
  "${BASE_URL}/api/explainability/models/${model_id}/analyze" \
  >/dev/null

echo "==> Running a prediction"
curl -fsS \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "age": 34,
        "gender": "Female",
        "region": "North",
        "monthly_charges": 980,
        "tenure_months": 14,
        "contract_type": "Monthly",
        "payment_method": "Credit Card",
        "is_active": true,
        "support_tickets": 1,
        "satisfaction_score": 4
      }
    ]
  }' \
  "${BASE_URL}/api/ml/models/${model_id}/predict" \
  >/dev/null

echo "==> Generating Markdown report"
report_response="$(
  curl -fsS \
    -X POST \
    "${BASE_URL}/api/reports/${dataset_id}/generate"
)"

report_id="$(
  printf '%s' "${report_response}" |
    extract_json_value 'payload["report"]["id"]'
)"

echo "    report_id=${report_id}"

echo "==> Starting lightweight background agent job"
job_response="$(
  curl -fsS \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
      "user_goal": "Run smoke test workflow.",
      "target_column": "churn",
      "run_ml": false,
      "generate_report": false,
      "generate_ai_insight": false
    }' \
    "${BASE_URL}/api/agent-jobs/${dataset_id}/start"
)"

job_id="$(
  printf '%s' "${job_response}" |
    extract_json_value 'payload["job"]["job_id"]'
)"

echo "    job_id=${job_id}"

terminal_status=""

for _ in $(seq 1 60); do
  job_detail="$(
    curl -fsS "${BASE_URL}/api/agent-jobs/${job_id}"
  )"

  terminal_status="$(
    printf '%s' "${job_detail}" |
      extract_json_value 'payload["status"]'
  )"

  case "${terminal_status}" in
    success|completed_with_warnings)
      break
      ;;
    failed)
      echo "Background agent job failed"
      exit 1
      ;;
  esac

  sleep 0.5
done

if [[ "${terminal_status}" != "success" && "${terminal_status}" != "completed_with_warnings" ]]; then
  echo "Background agent job did not complete in time"
  exit 1
fi

echo "==> Smoke test passed"
echo "    dataset_id=${dataset_id}"
echo "    model_id=${model_id}"
echo "    report_id=${report_id}"
echo "    job_id=${job_id}"
