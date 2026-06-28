import {
  generateEdaInsightApi,
  generateModelInsightApi,
  generateReportSummaryInsightApi,
  checkModelMigrationApi,
  createDriftReportApi,
  exportBundleApi,
  getLLMStatusApi,
  getDatasetPreviewApi,
  getDatasetSchemaApi,
  getSegmentMetricsApi,
  getThresholdAnalysisApi,
  getRetrainPlanApi,
  generateReportApi,
  getEdaSummaryApi,
  getHealthApi,
  getModelEvaluationApi,
  getReportApi,
  listDatasetModelsApi,
  getAgentRunApi,
  listAgentRunsApi,
  listDatasetsApi,
  listReportsApi,
  predictCsvWithModelApi,
  predictWithModelApi,
  promoteChallengerApi,
  retrainModelApi,
  runWhatIfApi,
  runAgentWorkflowApi,
  getAgentJobApi,
  listAgentJobEventsApi,
  listAgentJobsApi,
  startAgentJobApi,
  transformDatasetApi,
  listDatasetVersionsApi,
  updateModelStatusApi,
  uploadDatasetApi,
} from "./api/client.js";
import {
  initModelExplainability,
} from "./features/modelExplainability.js";
import {
  initMlWorkbench,
  loadMlWorkbench,
} from "./features/mlWorkbench.js";
import {
  initVisualizationLab,
  loadVisualizationLab,
} from "./features/visualizationLab.js";
import {
  renderAgentJobEvents,
  renderAgentJobList,
  renderAgentJobStatus,
} from "./components/agentJobView.js";
import {
  renderAgentOutputs,
  renderAgentRunHistory,
  renderAgentResult,
  renderAgentStatus,
} from "./components/agentView.js";
import {
  renderAIInsight,
  renderAIInsightLoading,
  renderLLMStatus,
} from "./components/aiInsightView.js";
import { renderDatasetList, renderPreviewTable } from "./components/datasetView.js";
import { renderEdaSummary } from "./components/edaView.js";
import {
  renderFeatureImportanceTable,
  renderModelDiagnostics,
  renderModelLeaderboard,
  renderModelOptions,
  renderPredictionResults,
  renderPredictionTemplate,
  renderTargetOptions,
} from "./components/mlView.js";
import { renderColumnProfiles, renderSchemaSummary } from "./components/schemaView.js";
import { API_BASE_URL } from "./config.js";
import {
  renderReportList,
  renderReportStatus,
  renderReportViewer,
} from "./components/reportView.js";
import { renderJson } from "./utils/format.js";


const elements = {
  apiStatus: document.querySelector("#apiStatus"),
  healthOutput: document.querySelector("#healthOutput"),
  checkApiButton: document.querySelector("#checkApiButton"),
  uploadForm: document.querySelector("#uploadForm"),
  csvFileInput: document.querySelector("#csvFileInput"),
  uploadStatus: document.querySelector("#uploadStatus"),
  datasetList: document.querySelector("#datasetList"),
  refreshDatasetsButton: document.querySelector("#refreshDatasetsButton"),
  previewTitle: document.querySelector("#previewTitle"),
  previewMeta: document.querySelector("#previewMeta"),
  previewTable: document.querySelector("#previewTable"),
  refreshDatasetVersionsButton: document.querySelector("#refreshDatasetVersionsButton"),
  datasetVersionsPanel: document.querySelector("#datasetVersionsPanel"),
  datasetTransformRecipeInput: document.querySelector("#datasetTransformRecipeInput"),
  transformDropColumnsInput: document.querySelector("#transformDropColumnsInput"),
  transformFillColumnInput: document.querySelector("#transformFillColumnInput"),
  transformFillStrategySelect: document.querySelector("#transformFillStrategySelect"),
  transformFillValueInput: document.querySelector("#transformFillValueInput"),
  transformDatetimeColumnInput: document.querySelector("#transformDatetimeColumnInput"),
  transformDropDuplicatesCheckbox: document.querySelector("#transformDropDuplicatesCheckbox"),
  applyDatasetTransformButton: document.querySelector("#applyDatasetTransformButton"),
  datasetTransformOutput: document.querySelector("#datasetTransformOutput"),
  schemaSummary: document.querySelector("#schemaSummary"),
  columnProfiles: document.querySelector("#columnProfiles"),
  edaQualitySummary: document.querySelector("#edaQualitySummary"),
  missingAnalysisTable: document.querySelector("#missingAnalysisTable"),
  numericStatisticsTable: document.querySelector("#numericStatisticsTable"),
  outlierAnalysisTable: document.querySelector("#outlierAnalysisTable"),
  correlationAnalysisTable: document.querySelector("#correlationAnalysisTable"),
  edaRecommendations: document.querySelector("#edaRecommendations"),
  agentGoalInput: document.querySelector("#agentGoalInput"),
  agentRunMlCheckbox: document.querySelector("#agentRunMlCheckbox"),
  agentGenerateReportCheckbox: document.querySelector("#agentGenerateReportCheckbox"),
  agentGenerateInsightCheckbox: document.querySelector("#agentGenerateInsightCheckbox"),
  runAgentWorkflowButton: document.querySelector("#runAgentWorkflowButton"),
  agentStatusPanel: document.querySelector("#agentStatusPanel"),
  agentTimelinePanel: document.querySelector("#agentTimelinePanel"),
  agentOutputsPanel: document.querySelector("#agentOutputsPanel"),
  startAgentJobButton: document.querySelector("#startAgentJobButton"),
  refreshAgentJobsButton: document.querySelector("#refreshAgentJobsButton"),
  agentJobStatusPanel: document.querySelector("#agentJobStatusPanel"),
  agentJobListPanel: document.querySelector("#agentJobListPanel"),
  agentJobEventsPanel: document.querySelector("#agentJobEventsPanel"),
  refreshAgentRunsButton: document.querySelector("#refreshAgentRunsButton"),
  agentRunHistoryPanel: document.querySelector("#agentRunHistoryPanel"),
  refreshLLMStatusButton: document.querySelector("#refreshLLMStatusButton"),
  llmStatusPanel: document.querySelector("#llmStatusPanel"),
  aiInsightGoalInput: document.querySelector("#aiInsightGoalInput"),
  generateEdaInsightButton: document.querySelector("#generateEdaInsightButton"),
  generateModelInsightButton: document.querySelector("#generateModelInsightButton"),
  generateReportInsightButton: document.querySelector("#generateReportInsightButton"),
  aiInsightOutput: document.querySelector("#aiInsightOutput"),
  targetColumnSelect: document.querySelector("#targetColumnSelect"),
  modelLeaderboard: document.querySelector("#modelLeaderboard"),
  promoteModelButton: document.querySelector("#promoteModelButton"),
  checkModelMigrationButton: document.querySelector("#checkModelMigrationButton"),
  challengerModelIdInput: document.querySelector("#challengerModelIdInput"),
  promoteChallengerButton: document.querySelector("#promoteChallengerButton"),
  modelLifecycleOutput: document.querySelector("#modelLifecycleOutput"),
  segmentColumnInput: document.querySelector("#segmentColumnInput"),
  runThresholdAnalysisButton: document.querySelector("#runThresholdAnalysisButton"),
  runSegmentMetricsButton: document.querySelector("#runSegmentMetricsButton"),
  runWhatIfButton: document.querySelector("#runWhatIfButton"),
  modelAdvancedDiagnosticsOutput: document.querySelector("#modelAdvancedDiagnosticsOutput"),
  modelDiagnosticsPanel: document.querySelector("#modelDiagnosticsPanel"),
  featureImportancePanel: document.querySelector("#featureImportancePanel"),
  predictionModelSelect: document.querySelector("#predictionModelSelect"),
  predictionJsonInput: document.querySelector("#predictionJsonInput"),
  runPredictionButton: document.querySelector("#runPredictionButton"),
  predictionCsvInput: document.querySelector("#predictionCsvInput"),
  runBatchPredictionButton: document.querySelector("#runBatchPredictionButton"),
  predictionStatus: document.querySelector("#predictionStatus"),
  predictionResultsTable: document.querySelector("#predictionResultsTable"),
  generateReportButton: document.querySelector("#generateReportButton"),
  reportStatus: document.querySelector("#reportStatus"),
  reportList: document.querySelector("#reportList"),
  reportViewer: document.querySelector("#reportViewer"),
  downloadReportButton: document.querySelector("#downloadReportButton"),
  driftReferenceVersionInput: document.querySelector("#driftReferenceVersionInput"),
  driftCurrentVersionInput: document.querySelector("#driftCurrentVersionInput"),
  runDriftReportButton: document.querySelector("#runDriftReportButton"),
  buildRetrainPlanButton: document.querySelector("#buildRetrainPlanButton"),
  runRetrainChallengerButton: document.querySelector("#runRetrainChallengerButton"),
  driftReportOutput: document.querySelector("#driftReportOutput"),
  exportBundleButton: document.querySelector("#exportBundleButton"),
  bundleOutput: document.querySelector("#bundleOutput"),
};


let currentDatasetId = null;
let currentSchemaColumns = [];
let currentModels = [];
let currentReportMarkdown = "";
let currentAgentJobEventSource = null;
let currentDriftReport = null;


function updateApiStatus(status, message) {
  // 統一處理 API 狀態文字與顏色。
  elements.apiStatus.className = `status-pill ${status}`;
  elements.apiStatus.textContent = message;
}


async function checkApiHealth() {
  updateApiStatus("checking", "Checking API...");
  elements.healthOutput.textContent = "Requesting backend health endpoint...";

  try {
    const payload = await getHealthApi();

    updateApiStatus("online", "API Online");
    renderJson(elements.healthOutput, payload);
  } catch (error) {
    updateApiStatus("offline", "API Offline");

    renderJson(elements.healthOutput, {
      status: "error",
      message: "Unable to connect to the backend API.",
      detail: error.message,
      expected_backend_url: `${API_BASE_URL}/api/health`,
    });
  }
}


async function uploadDataset(event) {
  event.preventDefault();

  const selectedFile = elements.csvFileInput.files[0];

  if (!selectedFile) {
    renderJson(elements.uploadStatus, {
      status: "error",
      message: "Please select a CSV file before uploading.",
    });
    return;
  }

  renderJson(elements.uploadStatus, {
    status: "uploading",
    filename: selectedFile.name,
    size_bytes: selectedFile.size,
  });

  try {
    const payload = await uploadDatasetApi(selectedFile);

    renderJson(elements.uploadStatus, {
      status: "success",
      message: payload.message,
      dataset_id: payload.dataset.id,
      rows: payload.dataset.row_count,
      columns: payload.dataset.column_count,
    });

    elements.csvFileInput.value = "";
    await loadDatasets();
    await loadDatasetWorkspace(payload.dataset.id);
  } catch (error) {
    renderJson(elements.uploadStatus, {
      status: "error",
      message: error.message,
    });
  }
}


async function loadDatasets() {
  elements.datasetList.innerHTML = '<p class="empty-state">Loading datasets...</p>';

  try {
    const payload = await listDatasetsApi();

    renderDatasetList(elements.datasetList, payload.datasets, loadDatasetWorkspace);
  } catch (error) {
    elements.datasetList.innerHTML = `
      <p class="empty-state">Failed to load datasets: ${error.message}</p>
    `;
  }
}


async function loadDatasetWorkspace(datasetId) {
  // 點選資料集後，同步載入 preview、schema、EDA、視覺化建議與模型紀錄。
  currentDatasetId = datasetId;

  await Promise.all([
    loadDatasetPreview(datasetId),
    loadDatasetVersions(datasetId),
    loadDatasetSchema(datasetId),
    loadEdaSummary(datasetId),
    loadVisualizationLab(datasetId),
    loadModelLeaderboard(datasetId),
    loadMlWorkbench(datasetId),
    loadReports(datasetId),
    loadAgentRunHistory(datasetId),
    loadAgentJobs(datasetId),
  ]);
}


async function loadDatasetPreview(datasetId) {
  elements.previewTitle.textContent = "Loading Preview";
  elements.previewMeta.textContent = "Requesting dataset preview...";
  elements.previewTable.innerHTML = '<p class="empty-state">Loading preview rows...</p>';

  try {
    const payload = await getDatasetPreviewApi(datasetId, 20);

    elements.previewTitle.textContent = "Dataset Preview";
    elements.previewMeta.textContent =
      `${payload.row_count} total rows · showing ${payload.preview_row_count} preview rows`;

    renderPreviewTable(elements.previewTable, payload.columns, payload.rows);
  } catch (error) {
    elements.previewMeta.textContent = `Failed to load preview: ${error.message}`;
    elements.previewTable.innerHTML = '<p class="empty-state">No preview available.</p>';
  }
}

async function loadDatasetVersions(datasetId) {
  elements.datasetVersionsPanel.innerHTML =
    '<p class="empty-state">Loading dataset versions...</p>';

  try {
    const payload = await listDatasetVersionsApi(datasetId);

    elements.datasetVersionsPanel.innerHTML = `
      <table class="preview-table">
        <thead>
          <tr>
            <th>Version</th>
            <th>Source</th>
            <th>Kind</th>
            <th>Shape</th>
            <th>Delta</th>
            <th>Recipe</th>
          </tr>
        </thead>
        <tbody>
          ${payload.versions.map((version) => `
            <tr>
              <td>${version.version_id}</td>
              <td>${version.source_version_id || "-"}</td>
              <td>${version.kind}</td>
              <td>${version.row_count} x ${version.column_count}</td>
              <td>${formatVersionDelta(version.profile_diff)}</td>
              <td>${formatRecipeSummary(version.recipe)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;

    elements.driftCurrentVersionInput.value = payload.latest_version_id;
  } catch (error) {
    elements.datasetVersionsPanel.innerHTML = `
      <p class="empty-state">Failed to load versions: ${error.message}</p>
    `;
  }
}

async function applyDatasetTransform() {
  if (!currentDatasetId) {
    renderJson(elements.datasetTransformOutput, {
      status: "error",
      message: "Please select a dataset before applying a transform.",
    });
    return;
  }

  try {
    const recipe = buildTransformRecipe();

    renderJson(elements.datasetTransformOutput, {
      status: "running",
      message: "Applying transformation recipe.",
    });

    const payload = await transformDatasetApi(currentDatasetId, recipe);

    renderJson(elements.datasetTransformOutput, payload);

    await loadDatasetWorkspace(currentDatasetId);
  } catch (error) {
    renderJson(elements.datasetTransformOutput, {
      status: "error",
      message: error.message,
    });
  }
}

function buildTransformRecipe() {
  const advancedRecipe = JSON.parse(elements.datasetTransformRecipeInput.value || "{}");
  const dropColumns = elements.transformDropColumnsInput.value
    .split(",")
    .map((column) => column.trim())
    .filter(Boolean);
  const fillColumn = elements.transformFillColumnInput.value.trim();
  const datetimeColumn = elements.transformDatetimeColumnInput.value.trim();
  const fillMissing = [...(advancedRecipe.fill_missing || [])];
  const datetimeParts = [...(advancedRecipe.datetime_parts || [])];

  if (fillColumn) {
    fillMissing.push({
      column: fillColumn,
      strategy: elements.transformFillStrategySelect.value,
      value: elements.transformFillValueInput.value || null,
    });
  }

  if (datetimeColumn) {
    datetimeParts.push({
      column: datetimeColumn,
      parts: ["year", "month", "day", "dayofweek"],
    });
  }

  return {
    ...advancedRecipe,
    drop_columns: [
      ...(advancedRecipe.drop_columns || []),
      ...dropColumns,
    ],
    fill_missing: fillMissing,
    datetime_parts: datetimeParts,
    drop_duplicate_rows: (
      advancedRecipe.drop_duplicate_rows
      || elements.transformDropDuplicatesCheckbox.checked
    ),
  };
}

function formatVersionDelta(profileDiff) {
  if (!profileDiff || Object.keys(profileDiff).length === 0) {
    return "-";
  }

  return [
    `rows ${profileDiff.row_delta ?? 0}`,
    `cols ${profileDiff.column_delta ?? 0}`,
    `missing ${profileDiff.missing_cell_delta ?? 0}`,
  ].join("<br />");
}

function formatRecipeSummary(recipe) {
  if (!recipe) {
    return "original";
  }

  const parts = [];

  if (recipe.drop_columns?.length) {
    parts.push(`${recipe.drop_columns.length} drop`);
  }

  if (recipe.fill_missing?.length) {
    parts.push(`${recipe.fill_missing.length} fill`);
  }

  if (recipe.datetime_parts?.length) {
    parts.push(`${recipe.datetime_parts.length} datetime`);
  }

  if (recipe.drop_duplicate_rows) {
    parts.push("dedupe");
  }

  return parts.join(", ") || "custom";
}


async function loadDatasetSchema(datasetId) {
  elements.schemaSummary.innerHTML = '<p class="empty-state">Loading schema summary...</p>';
  elements.columnProfiles.innerHTML = '<p class="empty-state">Loading column profiles...</p>';

  try {
    const payload = await getDatasetSchemaApi(datasetId);

    currentSchemaColumns = payload.columns;

    renderSchemaSummary(elements.schemaSummary, payload.summary);
    renderColumnProfiles(elements.columnProfiles, payload.columns);
    renderTargetOptions(elements.targetColumnSelect, payload.columns);
  } catch (error) {
    elements.schemaSummary.innerHTML = `
      <p class="empty-state">Failed to load schema: ${error.message}</p>
    `;
    elements.columnProfiles.innerHTML = '<p class="empty-state">No column profiles available.</p>';
  }
}


async function loadEdaSummary(datasetId) {
  elements.edaQualitySummary.innerHTML = '<p class="empty-state">Running EDA summary...</p>';
  elements.missingAnalysisTable.innerHTML = '<p class="empty-state">Loading missing analysis...</p>';
  elements.numericStatisticsTable.innerHTML = '<p class="empty-state">Loading statistics...</p>';
  elements.outlierAnalysisTable.innerHTML = '<p class="empty-state">Loading outliers...</p>';
  elements.correlationAnalysisTable.innerHTML = '<p class="empty-state">Loading correlation...</p>';
  elements.edaRecommendations.innerHTML = '<p class="empty-state">Loading recommendations...</p>';

  try {
    const payload = await getEdaSummaryApi(datasetId);

    renderEdaSummary(
      {
        qualitySummary: elements.edaQualitySummary,
        missingTable: elements.missingAnalysisTable,
        statisticsTable: elements.numericStatisticsTable,
        outlierTable: elements.outlierAnalysisTable,
        correlationTable: elements.correlationAnalysisTable,
        recommendations: elements.edaRecommendations,
      },
      payload,
    );
  } catch (error) {
    elements.edaQualitySummary.innerHTML = `
      <p class="empty-state">Failed to load EDA summary: ${error.message}</p>
    `;
  }
}


async function trainBaselineModels() {
  if (!currentDatasetId) {
    renderTrainingStatus(elements.mlTrainingStatus, {
      status: "error",
      message: "Please select a dataset before training models.",
    });
    return;
  }

  const targetColumn = elements.targetColumnSelect.value;

  if (!targetColumn) {
    renderTrainingStatus(elements.mlTrainingStatus, {
      status: "error",
      message: "Please select a target column.",
    });
    return;
  }

  renderTrainingStatus(elements.mlTrainingStatus, {
    status: "training",
    message: "Training baseline models. This may take a few seconds.",
  });

  try {
    const result = await trainMLModelsApi(currentDatasetId, {
      target_column: targetColumn,
      task_type: "auto",
      test_size: 0.25,
      random_state: 42,
    });

    currentModels = result.models;

    renderTrainingResult(elements.mlTrainingStatus, result);
    renderModelLeaderboard(elements.modelLeaderboard, result.models);
    renderModelOptions(elements.predictionModelSelect, result.models);
    elements.predictionModelSelect.value = result.best_model_id;
    updatePredictionTemplate();
    await loadModelEvaluation(result.best_model_id);
  } catch (error) {
    renderTrainingStatus(elements.mlTrainingStatus, {
      status: "error",
      message: error.message,
    });
  }
}


async function loadModelLeaderboard(datasetId) {
  elements.modelLeaderboard.innerHTML = '<p class="empty-state">Loading trained models...</p>';

  try {
    const payload = await listDatasetModelsApi(datasetId);
    currentModels = payload.models;

    window.dispatchEvent(
      new CustomEvent(
        "dataagent:models-loaded",
        {
          detail: {
            datasetId,
            models: payload.models,
          },
        },
      ),
    );

    renderModelLeaderboard(elements.modelLeaderboard, payload.models);
    renderModelOptions(elements.predictionModelSelect, payload.models);

    if (payload.models.length) {
      elements.predictionModelSelect.value = payload.models[0].id;
      updatePredictionTemplate();
      await loadModelEvaluation(payload.models[0].id);
    } else {
      elements.modelDiagnosticsPanel.innerHTML =
        '<p class="empty-state">No trained model diagnostics yet.</p>';
      elements.featureImportancePanel.innerHTML =
        '<p class="empty-state">No feature importance available yet.</p>';
      renderPredictionTemplate(elements.predictionJsonInput, null);
    }
  } catch (error) {
    elements.modelLeaderboard.innerHTML = `
      <p class="empty-state">Failed to load models: ${error.message}</p>
    `;
  }
}



async function loadModelEvaluation(modelId) {
  elements.modelDiagnosticsPanel.innerHTML =
    '<p class="empty-state">Loading model diagnostics...</p>';
  elements.featureImportancePanel.innerHTML =
    '<p class="empty-state">Loading feature importance...</p>';

  try {
    const evaluation = await getModelEvaluationApi(modelId);

    renderModelDiagnostics(elements.modelDiagnosticsPanel, evaluation);
    renderFeatureImportanceTable(elements.featureImportancePanel, evaluation.feature_importance);
  } catch (error) {
    elements.modelDiagnosticsPanel.innerHTML = `
      <p class="empty-state">Failed to load model diagnostics: ${error.message}</p>
    `;
    elements.featureImportancePanel.innerHTML =
      '<p class="empty-state">No feature importance available.</p>';
  }
}

async function promoteSelectedModel() {
  const selectedModel = getSelectedPredictionModel();

  if (!selectedModel) {
    renderJson(elements.modelLifecycleOutput, {
      status: "error",
      message: "Please select a saved model first.",
    });
    return;
  }

  try {
    const payload = await updateModelStatusApi(selectedModel.id, "production");
    renderJson(elements.modelLifecycleOutput, payload);

    if (currentDatasetId) {
      await loadModelLeaderboard(currentDatasetId);
    }
  } catch (error) {
    renderJson(elements.modelLifecycleOutput, {
      status: "error",
      message: error.message,
    });
  }
}

async function checkSelectedModelMigration() {
  const selectedModel = getSelectedPredictionModel();

  if (!selectedModel) {
    renderJson(elements.modelLifecycleOutput, {
      status: "error",
      message: "Please select a saved model first.",
    });
    return;
  }

  try {
    const payload = await checkModelMigrationApi(selectedModel.id);
    renderJson(elements.modelLifecycleOutput, payload);
  } catch (error) {
    renderJson(elements.modelLifecycleOutput, {
      status: "error",
      message: error.message,
    });
  }
}

async function promoteChallengerModel() {
  const selectedModel = getSelectedPredictionModel();
  const challengerModelId = elements.challengerModelIdInput.value.trim();

  if (!selectedModel || !challengerModelId) {
    renderJson(elements.modelLifecycleOutput, {
      status: "error",
      message: "Select a champion model and enter a challenger model ID.",
    });
    return;
  }

  try {
    const payload = await promoteChallengerApi(selectedModel.id, challengerModelId);
    renderJson(elements.modelLifecycleOutput, payload);

    if (currentDatasetId) {
      await loadModelLeaderboard(currentDatasetId);
    }
  } catch (error) {
    renderJson(elements.modelLifecycleOutput, {
      status: "error",
      message: error.message,
    });
  }
}

async function runThresholdAnalysis() {
  const selectedModel = getSelectedPredictionModel();

  if (!selectedModel) {
    renderJson(elements.modelAdvancedDiagnosticsOutput, {
      status: "error",
      message: "Please select a saved model first.",
    });
    return;
  }

  try {
    const payload = await getThresholdAnalysisApi(selectedModel.id, {
      thresholds: [0.3, 0.4, 0.5, 0.6, 0.7],
    });
    renderJson(elements.modelAdvancedDiagnosticsOutput, payload);
  } catch (error) {
    renderJson(elements.modelAdvancedDiagnosticsOutput, {
      status: "error",
      message: error.message,
    });
  }
}

async function runSegmentMetrics() {
  const selectedModel = getSelectedPredictionModel();
  const segmentColumn = elements.segmentColumnInput.value.trim();

  if (!selectedModel || !segmentColumn) {
    renderJson(elements.modelAdvancedDiagnosticsOutput, {
      status: "error",
      message: "Please select a model and enter a segment column.",
    });
    return;
  }

  try {
    const payload = await getSegmentMetricsApi(selectedModel.id, {
      segment_column: segmentColumn,
    });
    renderJson(elements.modelAdvancedDiagnosticsOutput, payload);
  } catch (error) {
    renderJson(elements.modelAdvancedDiagnosticsOutput, {
      status: "error",
      message: error.message,
    });
  }
}

async function runWhatIfSample() {
  const selectedModel = getSelectedPredictionModel();

  if (!selectedModel) {
    renderJson(elements.modelAdvancedDiagnosticsOutput, {
      status: "error",
      message: "Please select a saved model first.",
    });
    return;
  }

  try {
    const baseRecord = JSON.parse(elements.predictionJsonInput.value || "{}");
    const firstFeature = selectedModel.feature_columns[0];

    if (!firstFeature) {
      throw new Error("Selected model has no feature columns.");
    }

    const payload = await runWhatIfApi(selectedModel.id, {
      base_record: baseRecord,
      scenarios: [
        {
          name: `baseline_${firstFeature}`,
          changes: {
            [firstFeature]: baseRecord[firstFeature],
          },
        },
      ],
    });
    renderJson(elements.modelAdvancedDiagnosticsOutput, payload);
  } catch (error) {
    renderJson(elements.modelAdvancedDiagnosticsOutput, {
      status: "error",
      message: error.message,
    });
  }
}



function getSelectedPredictionModel() {
  const selectedModelId = elements.predictionModelSelect.value;

  return currentModels.find((model) => model.id === selectedModelId) || null;
}


function updatePredictionTemplate() {
  const selectedModel = getSelectedPredictionModel();
  renderPredictionTemplate(elements.predictionJsonInput, selectedModel);
}


async function runSinglePrediction() {
  const selectedModel = getSelectedPredictionModel();

  if (!selectedModel) {
    renderJson(elements.predictionStatus, {
      status: "error",
      message: "Please select a trained model first.",
    });
    return;
  }

  try {
    const parsedInput = JSON.parse(elements.predictionJsonInput.value);
    const records = Array.isArray(parsedInput) ? parsedInput : [parsedInput];

    renderJson(elements.predictionStatus, {
      status: "predicting",
      model_id: selectedModel.id,
      record_count: records.length,
    });

    const response = await predictWithModelApi(selectedModel.id, records);

    renderJson(elements.predictionStatus, {
      status: "success",
      total: response.total,
      model_name: response.model_name,
    });

    renderPredictionResults(elements.predictionResultsTable, response);
  } catch (error) {
    renderJson(elements.predictionStatus, {
      status: "error",
      message: error.message,
    });
  }
}


async function runBatchPrediction() {
  const selectedModel = getSelectedPredictionModel();
  const selectedFile = elements.predictionCsvInput.files[0];

  if (!selectedModel) {
    renderJson(elements.predictionStatus, {
      status: "error",
      message: "Please select a trained model first.",
    });
    return;
  }

  if (!selectedFile) {
    renderJson(elements.predictionStatus, {
      status: "error",
      message: "Please select a CSV file for batch prediction.",
    });
    return;
  }

  try {
    renderJson(elements.predictionStatus, {
      status: "predicting",
      model_id: selectedModel.id,
      filename: selectedFile.name,
    });

    const response = await predictCsvWithModelApi(selectedModel.id, selectedFile);

    renderJson(elements.predictionStatus, {
      status: "success",
      total: response.total,
      filename: response.original_filename,
    });

    renderPredictionResults(elements.predictionResultsTable, response);
  } catch (error) {
    renderJson(elements.predictionStatus, {
      status: "error",
      message: error.message,
    });
  }
}



async function generateDatasetReport() {
  if (!currentDatasetId) {
    renderReportStatus(elements.reportStatus, {
      status: "error",
      title: "No Dataset Selected",
      message: "Please select a dataset before generating a report.",
    });
    return;
  }

  renderReportStatus(elements.reportStatus, {
    status: "generating",
    title: "Generating Report",
    message: "Building Markdown report from dataset, EDA, visualization, and ML results.",
  });

  try {
    const payload = await generateReportApi(currentDatasetId);

    currentReportMarkdown = payload.report.markdown_content;

    renderReportStatus(elements.reportStatus, {
      status: "success",
      title: "Report Generated",
      message: payload.message,
    });

    renderReportViewer(elements.reportViewer, currentReportMarkdown);
    await loadReports(currentDatasetId);
  } catch (error) {
    renderReportStatus(elements.reportStatus, {
      status: "error",
      title: "Report Generation Failed",
      message: error.message,
    });
  }
}


async function loadReports(datasetId) {
  elements.reportList.innerHTML = '<p class="empty-state">Loading reports...</p>';

  try {
    const payload = await listReportsApi(datasetId);

    renderReportList(elements.reportList, payload.reports, loadReportDetail);
  } catch (error) {
    elements.reportList.innerHTML = `
      <p class="empty-state">Failed to load reports: ${error.message}</p>
    `;
  }
}


async function loadReportDetail(reportId) {
  elements.reportViewer.innerHTML = '<p class="empty-state">Loading report...</p>';

  try {
    const report = await getReportApi(reportId);
    currentReportMarkdown = report.markdown_content;
    renderReportViewer(elements.reportViewer, currentReportMarkdown);
  } catch (error) {
    elements.reportViewer.innerHTML = `
      <p class="empty-state">Failed to load report: ${error.message}</p>
    `;
  }
}


function downloadCurrentReport() {
  if (!currentReportMarkdown) {
    renderReportStatus(elements.reportStatus, {
      status: "error",
      title: "No Report Available",
      message: "Generate or select a report before downloading.",
    });
    return;
  }

  const blob = new Blob([currentReportMarkdown], {
    type: "text/markdown;charset=utf-8",
  });

  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");

  anchor.href = url;
  anchor.download = "dataagent-analysis-report.md";
  anchor.click();

  URL.revokeObjectURL(url);
}

async function runDriftReport() {
  if (!currentDatasetId) {
    renderJson(elements.driftReportOutput, {
      status: "error",
      message: "Please select a dataset before running drift analysis.",
    });
    return;
  }

  const selectedModel = getSelectedPredictionModel();
  const referenceVersion = elements.driftReferenceVersionInput.value.trim() || "v1";
  const currentVersion = elements.driftCurrentVersionInput.value.trim();

  if (!currentVersion) {
    renderJson(elements.driftReportOutput, {
      status: "error",
      message: "Please enter the current dataset version.",
    });
    return;
  }

  try {
    const payload = await createDriftReportApi({
      dataset_id: currentDatasetId,
      reference_version_id: referenceVersion,
      current_version_id: currentVersion,
      model_id: selectedModel?.id || null,
      target_column: selectedModel?.target_column || null,
    });

    currentDriftReport = payload;
    renderDriftReportSummary(payload);
  } catch (error) {
    renderJson(elements.driftReportOutput, {
      status: "error",
      message: error.message,
    });
  }
}

async function buildRetrainPlan() {
  const selectedModel = getSelectedPredictionModel();
  const payload = buildRetrainPayload();

  if (!selectedModel || !payload.current_version_id) {
    renderJson(elements.driftReportOutput, {
      status: "error",
      message: "Select a model and provide the current version.",
    });
    return;
  }

  try {
    const plan = await getRetrainPlanApi(selectedModel.id, payload);
    renderJson(elements.driftReportOutput, plan);
  } catch (error) {
    renderJson(elements.driftReportOutput, {
      status: "error",
      message: error.message,
    });
  }
}

async function retrainChallenger() {
  const selectedModel = getSelectedPredictionModel();
  const payload = buildRetrainPayload();

  if (!selectedModel || !payload.current_version_id) {
    renderJson(elements.driftReportOutput, {
      status: "error",
      message: "Select a model and provide the current version.",
    });
    return;
  }

  try {
    const response = await retrainModelApi(selectedModel.id, payload);
    elements.challengerModelIdInput.value = response.challenger_model.id;
    renderJson(elements.driftReportOutput, response);

    if (currentDatasetId) {
      await loadModelLeaderboard(currentDatasetId);
    }
  } catch (error) {
    renderJson(elements.driftReportOutput, {
      status: "error",
      message: error.message,
    });
  }
}

function buildRetrainPayload() {
  return {
    current_version_id: elements.driftCurrentVersionInput.value.trim(),
    reference_version_id: elements.driftReferenceVersionInput.value.trim() || null,
    drift_report_id: currentDriftReport?.report_id || null,
    auto_promote: false,
  };
}

function renderDriftReportSummary(report) {
  const recommendation = report.retraining_recommendation || {};

  elements.driftReportOutput.innerHTML = `
    <div class="quality-summary-card">
      <span class="panel-tag">Status</span>
      <strong>${report.status}</strong>
      <p>Retraining score: ${recommendation.score ?? "N/A"}</p>
      <p>Action: ${recommendation.action || "N/A"}</p>
    </div>
    <div class="table-wrapper">
      <table class="preview-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Type</th>
            <th>Status</th>
            <th>PSI</th>
            <th>KS / JS</th>
          </tr>
        </thead>
        <tbody>
          ${report.feature_drift.map((metric) => `
            <tr>
              <td>${metric.column}</td>
              <td>${metric.drift_type}</td>
              <td>${metric.status}</td>
              <td>${metric.psi ?? "N/A"}</td>
              <td>${metric.ks_statistic ?? metric.js_distance ?? "N/A"}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
    <pre class="code-output small-output">${JSON.stringify({
      schema_drift: report.schema_drift,
      recommendations: report.recommendations,
      retraining_recommendation: recommendation,
    }, null, 2)}</pre>
  `;
}

async function exportCurrentBundle() {
  if (!currentDatasetId) {
    renderJson(elements.bundleOutput, {
      status: "error",
      message: "Please select a dataset before exporting a bundle.",
    });
    return;
  }

  try {
    const payload = await exportBundleApi(currentDatasetId);
    renderJson(elements.bundleOutput, payload);
  } catch (error) {
    renderJson(elements.bundleOutput, {
      status: "error",
      message: error.message,
    });
  }
}



async function refreshLLMStatus() {
  try {
    const status = await getLLMStatusApi();
    renderLLMStatus(elements.llmStatusPanel, status);
  } catch (error) {
    elements.llmStatusPanel.innerHTML = `
      <p class="empty-state">Failed to check LLM status: ${error.message}</p>
    `;
  }
}


function getInsightGoal() {
  return elements.aiInsightGoalInput.value.trim() || null;
}


async function generateEdaInsight() {
  if (!currentDatasetId) {
    renderAIInsightLoading(elements.aiInsightOutput, "Please select a dataset first.");
    return;
  }

  renderAIInsightLoading(elements.aiInsightOutput, "Generating EDA explanation...");

  try {
    const insight = await generateEdaInsightApi(currentDatasetId, getInsightGoal());
    renderAIInsight(elements.aiInsightOutput, insight);
  } catch (error) {
    renderAIInsightLoading(elements.aiInsightOutput, `Failed to generate EDA insight: ${error.message}`);
  }
}


async function generateModelInsight() {
  if (!currentDatasetId) {
    renderAIInsightLoading(elements.aiInsightOutput, "Please select a dataset first.");
    return;
  }

  const selectedModel = getSelectedPredictionModel();

  if (!selectedModel) {
    renderAIInsightLoading(elements.aiInsightOutput, "Please train or select a model first.");
    return;
  }

  renderAIInsightLoading(elements.aiInsightOutput, "Generating model explanation...");

  try {
    const insight = await generateModelInsightApi(
      currentDatasetId,
      selectedModel.id,
      getInsightGoal(),
    );
    renderAIInsight(elements.aiInsightOutput, insight);
  } catch (error) {
    renderAIInsightLoading(elements.aiInsightOutput, `Failed to generate model insight: ${error.message}`);
  }
}


async function generateReportInsight() {
  if (!currentDatasetId) {
    renderAIInsightLoading(elements.aiInsightOutput, "Please select a dataset first.");
    return;
  }

  renderAIInsightLoading(elements.aiInsightOutput, "Generating report summary...");

  try {
    const insight = await generateReportSummaryInsightApi(currentDatasetId, getInsightGoal());
    renderAIInsight(elements.aiInsightOutput, insight);
  } catch (error) {
    renderAIInsightLoading(elements.aiInsightOutput, `Failed to generate report insight: ${error.message}`);
  }
}



async function runAgentWorkflow() {
  if (!currentDatasetId) {
    renderAgentStatus(elements.agentStatusPanel, {
      status: "error",
      title: "No Dataset Selected",
      message: "Please select a dataset before running the agent workflow.",
    });
    return;
  }

  renderAgentStatus(elements.agentStatusPanel, {
    status: "running",
    title: "Agent Running",
    message: "The agent workflow is orchestrating profiling, EDA, visualization, ML, report, and insight tools.",
  });

  elements.agentTimelinePanel.innerHTML = '<p class="empty-state">Running workflow...</p>';
  elements.agentOutputsPanel.innerHTML = '<p class="empty-state">Waiting for workflow outputs...</p>';

  const targetColumn = elements.targetColumnSelect.value || null;

  try {
    const result = await runAgentWorkflowApi(currentDatasetId, {
      user_goal: elements.agentGoalInput.value.trim() || null,
      target_column: targetColumn,
      run_ml: elements.agentRunMlCheckbox.checked,
      generate_report: elements.agentGenerateReportCheckbox.checked,
      generate_ai_insight: elements.agentGenerateInsightCheckbox.checked,
    });

    renderAgentStatus(elements.agentStatusPanel, {
      status: "success",
      title: "Agent Workflow Complete",
      message: result.final_summary,
    });

    renderAgentResult(elements.agentTimelinePanel, result);
    renderAgentOutputs(elements.agentOutputsPanel, result);

    await Promise.all([
      loadModelLeaderboard(currentDatasetId),
      loadReports(currentDatasetId),
      loadAgentRunHistory(currentDatasetId),
    ]);
  } catch (error) {
    renderAgentStatus(elements.agentStatusPanel, {
      status: "error",
      title: "Agent Workflow Failed",
      message: error.message,
    });
  }
}



async function loadAgentRunHistory(datasetId) {
  elements.agentRunHistoryPanel.innerHTML =
    '<p class="empty-state">Loading agent run history...</p>';

  try {
    const payload = await listAgentRunsApi(datasetId);
    renderAgentRunHistory(elements.agentRunHistoryPanel, payload.runs, loadAgentRunDetail);
  } catch (error) {
    elements.agentRunHistoryPanel.innerHTML = `
      <p class="empty-state">Failed to load agent run history: ${error.message}</p>
    `;
  }
}


async function loadAgentRunDetail(workflowId) {
  elements.agentTimelinePanel.innerHTML = '<p class="empty-state">Loading workflow timeline...</p>';
  elements.agentOutputsPanel.innerHTML = '<p class="empty-state">Loading workflow outputs...</p>';

  try {
    const result = await getAgentRunApi(workflowId);
    renderAgentResult(elements.agentTimelinePanel, result);
    renderAgentOutputs(elements.agentOutputsPanel, result);
    renderAgentStatus(elements.agentStatusPanel, {
      status: result.status,
      title: "Workflow Replay Loaded",
      message: result.final_summary,
    });
  } catch (error) {
    renderAgentStatus(elements.agentStatusPanel, {
      status: "error",
      title: "Workflow Replay Failed",
      message: error.message,
    });
  }
}



async function startBackgroundAgentJob() {
  if (!currentDatasetId) {
    renderAgentJobStatus(elements.agentJobStatusPanel, {
      status: "error",
      title: "No Dataset Selected",
      message: "Please select a dataset before starting a background job.",
    });
    return;
  }

  closeCurrentAgentJobStream();

  renderAgentJobStatus(elements.agentJobStatusPanel, {
    status: "running",
    title: "Starting Background Job",
    message: "Submitting agent workflow to background runner.",
  });

  elements.agentJobEventsPanel.innerHTML =
    '<p class="empty-state">Waiting for job events...</p>';

  const targetColumn = elements.targetColumnSelect.value || null;

  try {
    const payload = await startAgentJobApi(currentDatasetId, {
      user_goal: elements.agentGoalInput.value.trim() || null,
      target_column: targetColumn,
      run_ml: elements.agentRunMlCheckbox.checked,
      generate_report: elements.agentGenerateReportCheckbox.checked,
      generate_ai_insight: elements.agentGenerateInsightCheckbox.checked,
    });

    renderAgentJobStatus(elements.agentJobStatusPanel, {
      status: payload.job.status,
      title: "Background Job Accepted",
      message: `Job ID: ${payload.job.job_id}`,
    });

    await loadAgentJobs(currentDatasetId);
    streamAgentJobEvents(payload.job.job_id);
  } catch (error) {
    renderAgentJobStatus(elements.agentJobStatusPanel, {
      status: "error",
      title: "Background Job Failed",
      message: error.message,
    });
  }
}


async function loadAgentJobs(datasetId) {
  elements.agentJobListPanel.innerHTML =
    '<p class="empty-state">Loading background agent jobs...</p>';

  try {
    const payload = await listAgentJobsApi(datasetId);
    renderAgentJobList(elements.agentJobListPanel, payload.jobs, loadAgentJobDetail);
  } catch (error) {
    elements.agentJobListPanel.innerHTML = `
      <p class="empty-state">Failed to load background jobs: ${error.message}</p>
    `;
  }
}


async function loadAgentJobDetail(jobId) {
  closeCurrentAgentJobStream();

  elements.agentJobEventsPanel.innerHTML =
    '<p class="empty-state">Loading job events...</p>';

  try {
    const job = await getAgentJobApi(jobId);
    renderAgentJobStatus(elements.agentJobStatusPanel, {
      status: job.status,
      title: "Background Job Loaded",
      message: `Job ID: ${job.job_id}`,
    });
    renderAgentJobEvents(elements.agentJobEventsPanel, job.events);
  } catch (error) {
    renderAgentJobStatus(elements.agentJobStatusPanel, {
      status: "error",
      title: "Failed to Load Job",
      message: error.message,
    });
  }
}


function streamAgentJobEvents(jobId) {
  closeCurrentAgentJobStream();

  const events = [];
  const streamUrl = `${API_BASE_URL}/api/agent-jobs/${jobId}/events/stream`;

  currentAgentJobEventSource = new EventSource(streamUrl);

  currentAgentJobEventSource.onmessage = async (event) => {
    const payload = JSON.parse(event.data);

    if (payload.event_type === "stream_closed") {
      closeCurrentAgentJobStream();

      renderAgentJobStatus(elements.agentJobStatusPanel, {
        status: payload.status,
        title: "Background Job Complete",
        message: `Workflow ID: ${payload.workflow_id || "N/A"}`,
      });

      if (currentDatasetId) {
        await Promise.all([
          loadAgentJobs(currentDatasetId),
          loadAgentRunHistory(currentDatasetId),
          loadModelLeaderboard(currentDatasetId),
          loadReports(currentDatasetId),
        ]);
      }

      return;
    }

    events.push(payload);
    renderAgentJobEvents(elements.agentJobEventsPanel, events);
  };

  currentAgentJobEventSource.onerror = () => {
    closeCurrentAgentJobStream();
  };
}


function closeCurrentAgentJobStream() {
  if (currentAgentJobEventSource) {
    currentAgentJobEventSource.close();
    currentAgentJobEventSource = null;
  }
}


let appInitialized = false;


function initializeApp() {
  // Dynamic partial loading may finish before or after DOMContentLoaded。
  if (appInitialized) {
    return;
  }

  appInitialized = true;
  // 頁面載入後先確認 API 狀態，再載入 dataset registry。
  initVisualizationLab();
  initMlWorkbench();
  initModelExplainability();
  checkApiHealth();
  refreshLLMStatus();
  loadDatasets();

  window.addEventListener(
    "dataagent:ml-experiment-completed",
    async (event) => {
      const datasetId = event.detail?.datasetId;
      const bestModelId = event.detail?.bestModelId;

      if (!datasetId) {
        return;
      }

      await loadModelLeaderboard(datasetId);

      if (bestModelId) {
        await loadModelEvaluation(bestModelId);
      }
    },
  );

  elements.checkApiButton.addEventListener("click", () => {
    checkApiHealth();
  });

  elements.uploadForm.addEventListener("submit", uploadDataset);

  elements.refreshDatasetsButton.addEventListener("click", () => {
    loadDatasets();
  });

  elements.refreshDatasetVersionsButton.addEventListener("click", () => {
    if (currentDatasetId) {
      loadDatasetVersions(currentDatasetId);
    }
  });

  elements.applyDatasetTransformButton.addEventListener("click", () => {
    applyDatasetTransform();
  });

  elements.predictionModelSelect.addEventListener("change", () => {
    updatePredictionTemplate();

    const selectedModel = getSelectedPredictionModel();

    if (selectedModel) {
      loadModelEvaluation(selectedModel.id);
    }
  });

  elements.runPredictionButton.addEventListener("click", () => {
    runSinglePrediction();
  });

  elements.runBatchPredictionButton.addEventListener("click", () => {
    runBatchPrediction();
  });

  elements.promoteModelButton.addEventListener("click", () => {
    promoteSelectedModel();
  });

  elements.checkModelMigrationButton.addEventListener("click", () => {
    checkSelectedModelMigration();
  });

  elements.promoteChallengerButton.addEventListener("click", () => {
    promoteChallengerModel();
  });

  elements.runThresholdAnalysisButton.addEventListener("click", () => {
    runThresholdAnalysis();
  });

  elements.runSegmentMetricsButton.addEventListener("click", () => {
    runSegmentMetrics();
  });

  elements.runWhatIfButton.addEventListener("click", () => {
    runWhatIfSample();
  });

  elements.generateReportButton.addEventListener("click", () => {
    generateDatasetReport();
  });

  elements.downloadReportButton.addEventListener("click", () => {
    downloadCurrentReport();
  });

  elements.runDriftReportButton.addEventListener("click", () => {
    runDriftReport();
  });

  elements.buildRetrainPlanButton.addEventListener("click", () => {
    buildRetrainPlan();
  });

  elements.runRetrainChallengerButton.addEventListener("click", () => {
    retrainChallenger();
  });

  elements.exportBundleButton.addEventListener("click", () => {
    exportCurrentBundle();
  });

  elements.refreshLLMStatusButton.addEventListener("click", () => {
    refreshLLMStatus();
  });

  elements.generateEdaInsightButton.addEventListener("click", () => {
    generateEdaInsight();
  });

  elements.generateModelInsightButton.addEventListener("click", () => {
    generateModelInsight();
  });

  elements.generateReportInsightButton.addEventListener("click", () => {
    generateReportInsight();
  });

  elements.runAgentWorkflowButton.addEventListener("click", () => {
    runAgentWorkflow();
  });

  elements.refreshAgentRunsButton.addEventListener("click", () => {
    if (currentDatasetId) {
      loadAgentRunHistory(currentDatasetId);
    }
  });

  elements.startAgentJobButton.addEventListener("click", () => {
    startBackgroundAgentJob();
  });

  elements.refreshAgentJobsButton.addEventListener("click", () => {
    if (currentDatasetId) {
      loadAgentJobs(currentDatasetId);
    }
  });

}


if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeApp, {
    once: true,
  });
} else {
  initializeApp();
}
