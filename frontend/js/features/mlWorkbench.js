import {
  getMlWorkbenchExperimentApi,
  getMlWorkbenchPlanApi,
  listMlWorkbenchExperimentsApi,
  runMlWorkbenchExperimentApi,
} from "../api/client.js";
import {
  populateMlWorkbenchFeatures,
  renderMlExperimentHistory,
  renderMlExperimentLeaderboard,
  renderMlModelOptions,
  renderMlPipelineSteps,
  renderMlPlanSummary,
  renderMlWarnings,
  renderMlWorkbenchStatus,
} from "../components/mlWorkbenchView.js";


let initialized = false;
let currentDatasetId = null;
let currentPlan = null;
let elements = null;
let metricChart = null;
let projectionChart = null;


export function initMlWorkbench() {
  if (initialized) {
    return;
  }

  elements = {
    taskType: document.querySelector(
      "#mlTaskTypeSelect",
    ),
    target: document.querySelector(
      "#targetColumnSelect",
    ),
    featureSelect: document.querySelector(
      "#mlFeatureSelect",
    ),
    includeDatetime: document.querySelector(
      "#mlIncludeDatetimeCheckbox",
    ),
    includeText: document.querySelector(
      "#mlIncludeTextCheckbox",
    ),
    scaler: document.querySelector(
      "#mlScalerSelect",
    ),
    cvFolds: document.querySelector(
      "#mlCvFoldsInput",
    ),
    classWeight: document.querySelector(
      "#mlClassWeightSelect",
    ),
    clusterCount: document.querySelector(
      "#mlClusterCountInput",
    ),
    contamination: document.querySelector(
      "#mlContaminationInput",
    ),
    generatePlanButton: document.querySelector(
      "#generateMlPlanButton",
    ),
    runExperimentButton: document.querySelector(
      "#runMlExperimentButton",
    ),
    refreshExperimentsButton: document.querySelector(
      "#refreshMlExperimentsButton",
    ),
    status: document.querySelector(
      "#mlWorkbenchStatus",
    ),
    planSummary: document.querySelector(
      "#mlPlanSummary",
    ),
    warnings: document.querySelector(
      "#mlPlanWarnings",
    ),
    pipelineSteps: document.querySelector(
      "#mlPipelineSteps",
    ),
    modelOptions: document.querySelector(
      "#mlModelOptions",
    ),
    leaderboard: document.querySelector(
      "#mlExperimentLeaderboard",
    ),
    metricChart: document.querySelector(
      "#mlMetricComparisonChart",
    ),
    projectionChart: document.querySelector(
      "#mlProjectionChart",
    ),
    history: document.querySelector(
      "#mlExperimentHistory",
    ),
  };

  const missingElement = Object.entries(
    elements,
  ).find(([, element]) => !element);

  if (missingElement) {
    throw new Error(
      `ML Workbench element not found: ${
        missingElement[0]
      }`,
    );
  }

  elements.generatePlanButton.addEventListener(
    "click",
    generatePlan,
  );

  elements.runExperimentButton.addEventListener(
    "click",
    runExperiment,
  );

  elements.refreshExperimentsButton.addEventListener(
    "click",
    () => {
      if (currentDatasetId) {
        loadExperimentHistory(
          currentDatasetId,
        );
      }
    },
  );

  elements.taskType.addEventListener(
    "change",
    updateTaskControls,
  );

  initialized = true;
  updateTaskControls();
}


export async function loadMlWorkbench(
  datasetId,
) {
  if (!initialized) {
    initMlWorkbench();
  }

  currentDatasetId = datasetId;

  await Promise.all([
    generatePlan(),
    loadExperimentHistory(datasetId),
  ]);
}


async function generatePlan() {
  if (!currentDatasetId) {
    renderMlWorkbenchStatus(
      elements.status,
      {
        status: "error",
        title: "No Dataset Selected",
        message: "Please select a dataset first.",
      },
    );
    return;
  }

  renderMlWorkbenchStatus(
    elements.status,
    {
      status: "running",
      title: "Planning Experiment",
      message: "Inspecting task, features, preprocessing, and models.",
    },
  );

  try {
    const plan = await getMlWorkbenchPlanApi(
      currentDatasetId,
      buildPlanPayload(),
    );

    currentPlan = plan;

    renderMlPlanSummary(
      elements.planSummary,
      plan,
    );

    renderMlPipelineSteps(
      elements.pipelineSteps,
      plan.preprocessing_steps,
    );

    renderMlWarnings(
      elements.warnings,
      plan.warnings,
    );

    renderMlModelOptions(
      elements.modelOptions,
      plan.available_models,
    );

    populateMlWorkbenchFeatures(
      elements.featureSelect,
      plan.feature_groups,
    );

    if (plan.target_column) {
      elements.target.value = plan.target_column;
    }

    renderMlWorkbenchStatus(
      elements.status,
      {
        status: "success",
        title: "Experiment Plan Ready",
        message: [
          plan.detected_task_type,
          `${plan.estimated_feature_count} estimated features`,
          `${plan.available_models.length} model option(s)`,
        ].join(" · "),
      },
    );

    updateTaskControls();
  } catch (error) {
    renderMlWorkbenchStatus(
      elements.status,
      {
        status: "error",
        title: "Planning Failed",
        message: error.message,
      },
    );
  }
}


async function runExperiment() {
  if (!currentDatasetId) {
    return;
  }

  const selectedModels = [
    ...elements.modelOptions.querySelectorAll(
      'input[type="checkbox"]:checked',
    ),
  ].map((input) => input.value);

  if (!selectedModels.length) {
    dispatchErrorToast(
      "Select at least one model.",
    );
    return;
  }

  renderMlWorkbenchStatus(
    elements.status,
    {
      status: "running",
      title: "Experiment Running",
      message: "Training and evaluating selected models.",
    },
  );

  try {
    const experiment = (
      await runMlWorkbenchExperimentApi(
        currentDatasetId,
        {
          ...buildPlanPayload(),
          selected_models: selectedModels,
          cv_folds: Number(
            elements.cvFolds.value,
          ),
          test_size: 0.2,
          random_state: 42,
          class_weight_mode: (
            elements.classWeight.value
          ),
          n_clusters: Number(
            elements.clusterCount.value,
          ),
          dbscan_eps: 0.7,
          dbscan_min_samples: 5,
          contamination: Number(
            elements.contamination.value,
          ),
        },
      )
    );

    renderExperiment(experiment);

    renderMlWorkbenchStatus(
      elements.status,
      {
        status: (
          experiment.status === "failed"
            ? "error"
            : "success"
        ),
        title: "Experiment Complete",
        message: [
          experiment.status,
          `Best model: ${
            experiment.best_model_name || "N/A"
          }`,
          `${experiment.primary_metric}: ${
            experiment.best_metric_value ?? "N/A"
          }`,
        ].join(" · "),
      },
    );

    await loadExperimentHistory(
      currentDatasetId,
    );

    window.dispatchEvent(
      new CustomEvent(
        "dataagent:ml-experiment-completed",
        {
          detail: {
            datasetId: currentDatasetId,
            bestModelId: (
              experiment.best_model_id
            ),
          },
        },
      ),
    );
  } catch (error) {
    renderMlWorkbenchStatus(
      elements.status,
      {
        status: "error",
        title: "Experiment Failed",
        message: error.message,
      },
    );
  }
}


async function loadExperimentHistory(
  datasetId,
) {
  try {
    const payload = (
      await listMlWorkbenchExperimentsApi(
        datasetId,
      )
    );

    renderMlExperimentHistory(
      elements.history,
      payload.experiments,
      loadExperimentDetail,
    );
  } catch (error) {
    elements.history.innerHTML = `
      <p class="empty-state">
        Failed to load experiment history:
        ${error.message}
      </p>
    `;
  }
}


async function loadExperimentDetail(
  experimentId,
) {
  try {
    const experiment = (
      await getMlWorkbenchExperimentApi(
        experimentId,
      )
    );

    renderExperiment(experiment);

    renderMlWorkbenchStatus(
      elements.status,
      {
        status: "success",
        title: "Experiment Replay Loaded",
        message: experiment.experiment_id,
      },
    );
  } catch (error) {
    dispatchErrorToast(error.message);
  }
}


function renderExperiment(experiment) {
  renderMlExperimentLeaderboard(
    elements.leaderboard,
    experiment,
  );

  renderMetricChart(experiment);
  renderProjectionChart(experiment);
}


function renderMetricChart(experiment) {
  disposeMetricChart();

  if (!window.echarts) {
    return;
  }

  const successfulResults = (
    experiment.model_results.filter(
      (result) => result.status === "success",
    )
  );

  if (!successfulResults.length) {
    elements.metricChart.innerHTML = `
      <p class="empty-state">
        No successful model metrics available.
      </p>
    `;
    return;
  }

  const labels = successfulResults.map(
    (result) => result.model_label,
  );

  const cvValues = successfulResults.map(
    (result) => {
      const metric = result.cv_metrics.find(
        (item) => {
          return (
            item.name
            === experiment.primary_metric
          );
        },
      );

      return metric?.mean ?? null;
    },
  );

  const holdoutValues = successfulResults.map(
    (result) => {
      return (
        result.holdout_metrics[
          experiment.primary_metric
        ] ?? null
      );
    },
  );

  metricChart = window.echarts.init(
    elements.metricChart,
  );

  metricChart.setOption({
    animationDuration: 500,
    aria: {
      enabled: true,
    },
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "shadow",
      },
    },
    legend: {
      data: [
        "CV Mean",
        "Holdout",
      ],
    },
    grid: {
      left: 60,
      right: 25,
      top: 45,
      bottom: 80,
      containLabel: true,
    },
    xAxis: {
      type: "category",
      data: labels,
      axisLabel: {
        interval: 0,
        rotate: labels.length > 3 ? 25 : 0,
      },
    },
    yAxis: {
      type: "value",
      name: experiment.primary_metric,
      scale: true,
    },
    series: [
      {
        name: "CV Mean",
        type: "bar",
        data: cvValues,
      },
      {
        name: "Holdout",
        type: "bar",
        data: holdoutValues,
      },
    ],
  });
}


function renderProjectionChart(experiment) {
  disposeProjectionChart();

  if (!window.echarts) {
    return;
  }

  const result = experiment.model_results.find(
    (item) => {
      return (
        item.status === "success"
        && item.artifact?.projection_points?.length
      );
    },
  );

  if (!result) {
    elements.projectionChart.innerHTML = `
      <p class="empty-state">
        Projection is available for clustering and anomaly experiments.
      </p>
    `;
    return;
  }

  const groupedPoints = new Map();

  result.artifact.projection_points.forEach(
    (point) => {
      if (!groupedPoints.has(point.label)) {
        groupedPoints.set(
          point.label,
          [],
        );
      }

      groupedPoints.get(point.label).push([
        point.x,
        point.y,
        point.score ?? null,
      ]);
    },
  );

  projectionChart = window.echarts.init(
    elements.projectionChart,
  );

  projectionChart.setOption({
    animationDuration: 500,
    aria: {
      enabled: true,
    },
    tooltip: {
      trigger: "item",
    },
    legend: {
      type: "scroll",
      top: 0,
    },
    grid: {
      left: 55,
      right: 25,
      top: 45,
      bottom: 55,
      containLabel: true,
    },
    xAxis: {
      type: "value",
      name: "Component 1",
      scale: true,
    },
    yAxis: {
      type: "value",
      name: "Component 2",
      scale: true,
    },
    dataZoom: [
      {
        type: "inside",
      },
    ],
    series: [
      ...groupedPoints.entries(),
    ].map(([label, points]) => {
      return {
        name: label,
        type: "scatter",
        data: points,
        symbolSize: 9,
        emphasis: {
          focus: "series",
        },
      };
    }),
  });
}


function buildPlanPayload() {
  return {
    target_column: getTargetColumn(),
    task_type: elements.taskType.value,
    feature_columns: getSelectedFeatures(),
    excluded_columns: [],
    include_datetime: (
      elements.includeDatetime.checked
    ),
    include_text: (
      elements.includeText.checked
    ),
    numeric_imputer: "median",
    scaler: elements.scaler.value,
    one_hot_min_frequency: 1,
    text_max_features: 300,
  };
}


function getTargetColumn() {
  const taskType = elements.taskType.value;

  if (
    taskType === "clustering"
    || taskType === "anomaly_detection"
  ) {
    return null;
  }

  return elements.target.value || null;
}


function getSelectedFeatures() {
  return [
    ...elements.featureSelect.selectedOptions,
  ].map((option) => option.value);
}


function updateTaskControls() {
  const unsupervised = [
    "clustering",
    "anomaly_detection",
  ].includes(elements.taskType.value);

  elements.target.disabled = unsupervised;
  elements.classWeight.disabled = unsupervised;
}


function disposeMetricChart() {
  if (metricChart) {
    metricChart.dispose();
    metricChart = null;
  }
}


function disposeProjectionChart() {
  if (projectionChart) {
    projectionChart.dispose();
    projectionChart = null;
  }
}


function dispatchErrorToast(message) {
  window.dispatchEvent(
    new CustomEvent("dataagent:toast", {
      detail: {
        type: "error",
        message,
      },
    }),
  );
}
