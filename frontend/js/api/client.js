import { API_BASE_URL } from "../config.js";


async function apiFetch(path, options = {}) {
  // 集中處理 fetch、JSON 解析與錯誤訊息，避免每個頁面重複寫 try/catch。
  const response = await fetch(`${API_BASE_URL}${path}`, options);

  let payload = null;

  try {
    payload = await response.json();
  } catch {
    payload = {};
  }

  if (!response.ok) {
    const legacyMessage = typeof payload.detail === "string"
      ? payload.detail
      : null;

    const message =
      payload.error?.message ||
      legacyMessage ||
      `Unexpected status code: ${response.status}`;

    // 全域 toast 提醒使用者，同時保留 Error 讓呼叫端顯示區域錯誤。
    window.dispatchEvent(
      new CustomEvent("dataagent:toast", {
        detail: {
          type: "error",
          message,
        },
      }),
    );

    const error = new Error(message);
    error.payload = payload;
    throw error;
  }

  return payload;
}


export function getHealthApi() {
  return apiFetch("/api/health");
}


export function listDatasetsApi() {
  return apiFetch("/api/datasets");
}


export function uploadDatasetApi(file) {
  // FormData 讓瀏覽器自動建立 multipart/form-data request。
  const formData = new FormData();
  formData.append("file", file);

  return apiFetch("/api/datasets/upload", {
    method: "POST",
    body: formData,
  });
}


export function getDatasetPreviewApi(datasetId, maxRows = 20) {
  return apiFetch(`/api/datasets/${datasetId}/preview?max_rows=${maxRows}`);
}

export function listDatasetVersionsApi(datasetId) {
  return apiFetch(`/api/datasets/${datasetId}/versions`);
}

export function transformDatasetApi(datasetId, payload) {
  return apiFetch(`/api/datasets/${datasetId}/transform`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}


export function getDatasetSchemaApi(datasetId) {
  return apiFetch(`/api/datasets/${datasetId}/schema`);
}


export function getEdaSummaryApi(datasetId) {
  return apiFetch(`/api/eda/${datasetId}/summary`);
}


export function getVisualizationRecommendationsApi(datasetId) {
  return apiFetch(`/api/visualizations/${datasetId}/recommendations`);
}

export function trainMLModelsApi(datasetId, payload) {
  return apiFetch(`/api/ml/${datasetId}/train`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}


export function listDatasetModelsApi(datasetId) {
  return apiFetch(`/api/ml/${datasetId}/models`);
}

export function updateModelStatusApi(modelId, lifecycleStatus) {
  return apiFetch(`/api/ml/models/${modelId}/status`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ lifecycle_status: lifecycleStatus }),
  });
}

export function getModelEvaluationApi(modelId) {
  return apiFetch(`/api/ml/models/${modelId}/evaluation`);
}

export function getThresholdAnalysisApi(modelId, payload) {
  return apiFetch(`/api/ml/models/${modelId}/threshold-analysis`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function getSegmentMetricsApi(modelId, payload) {
  return apiFetch(`/api/ml/models/${modelId}/segment-metrics`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function runWhatIfApi(modelId, payload) {
  return apiFetch(`/api/ml/models/${modelId}/what-if`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function checkModelMigrationApi(modelId) {
  return apiFetch(`/api/ml/models/${modelId}/migration-check`, {
    method: "POST",
  });
}

export function getRetrainPlanApi(modelId, payload) {
  return apiFetch(`/api/ml/models/${modelId}/retrain-plan`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function retrainModelApi(modelId, payload) {
  return apiFetch(`/api/ml/models/${modelId}/retrain`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function promoteChallengerApi(modelId, challengerModelId) {
  return apiFetch(`/api/ml/models/${modelId}/promote-challenger`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ challenger_model_id: challengerModelId }),
  });
}

export function predictWithModelApi(modelId, records) {
  return apiFetch(`/api/ml/models/${modelId}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ records }),
  });
}


export function predictCsvWithModelApi(modelId, file) {
  const formData = new FormData();
  formData.append("file", file);

  return apiFetch(`/api/ml/models/${modelId}/predict-csv`, {
    method: "POST",
    body: formData,
  });
}

export function generateReportApi(datasetId) {
  return apiFetch(`/api/reports/${datasetId}/generate`, {
    method: "POST",
  });
}


export function listReportsApi(datasetId) {
  return apiFetch(`/api/reports/dataset/${datasetId}`);
}


export function getReportApi(reportId) {
  return apiFetch(`/api/reports/${reportId}`);
}

export function getLLMStatusApi() {
  return apiFetch("/api/ai-insights/status");
}


export function generateEdaInsightApi(datasetId, userGoal = null) {
  return apiFetch(`/api/ai-insights/${datasetId}/eda`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_goal: userGoal }),
  });
}


export function generateModelInsightApi(datasetId, modelId, userGoal = null) {
  return apiFetch(`/api/ai-insights/${datasetId}/models/${modelId}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_goal: userGoal }),
  });
}


export function generateReportSummaryInsightApi(datasetId, userGoal = null) {
  return apiFetch(`/api/ai-insights/${datasetId}/report-summary`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_goal: userGoal }),
  });
}

export function runAgentWorkflowApi(datasetId, payload) {
  return apiFetch(`/api/agents/${datasetId}/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function listAgentRunsApi(datasetId) {
  return apiFetch(`/api/agents/${datasetId}/runs`);
}


export function getAgentRunApi(workflowId) {
  return apiFetch(`/api/agents/runs/${workflowId}`);
}

export function startAgentJobApi(datasetId, payload) {
  return apiFetch(`/api/agent-jobs/${datasetId}/start`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}


export function getAgentJobApi(jobId) {
  return apiFetch(`/api/agent-jobs/${jobId}`);
}


export function listAgentJobsApi(datasetId) {
  return apiFetch(`/api/agent-jobs/dataset/${datasetId}`);
}


export function listAgentJobEventsApi(jobId) {
  return apiFetch(`/api/agent-jobs/${jobId}/events`);
}

export function createDriftReportApi(payload) {
  return apiFetch("/api/drift/reports", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function exportBundleApi(datasetId) {
  return apiFetch("/api/bundles/export", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ dataset_id: datasetId }),
  });
}



export function getVisualizationLabApi(
  datasetId,
  payload = {},
) {
  return apiFetch(
    `/api/visualizations/${datasetId}/lab`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
}


export function buildVisualizationChartApi(
  datasetId,
  payload,
) {
  return apiFetch(
    `/api/visualizations/${datasetId}/build`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
}


export function getMlWorkbenchPlanApi(
  datasetId,
  payload,
) {
  return apiFetch(
    `/api/ml-workbench/${datasetId}/plan`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
}


export function runMlWorkbenchExperimentApi(
  datasetId,
  payload,
) {
  return apiFetch(
    `/api/ml-workbench/${datasetId}/experiments`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
}


export function listMlWorkbenchExperimentsApi(
  datasetId,
) {
  return apiFetch(
    `/api/ml-workbench/${datasetId}/experiments`,
  );
}


export function getMlWorkbenchExperimentApi(
  experimentId,
) {
  return apiFetch(
    `/api/ml-workbench/experiments/${experimentId}`,
  );
}


export function analyzeModelExplainabilityApi(
  modelId,
  payload,
) {
  return apiFetch(
    `/api/explainability/models/${modelId}/analyze`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
}


export function generateExplainabilityInsightApi(
  datasetId,
  modelId,
  userGoal = null,
) {
  return apiFetch(
    (
      `/api/ai-insights/${datasetId}`
      + `/models/${modelId}/explainability`
    ),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_goal: userGoal,
      }),
    },
  );
}
