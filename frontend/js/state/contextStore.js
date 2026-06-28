const DEFAULT_CONTEXT = {
  dataset: null,
  version: null,
  targetColumn: null,
  models: [],
  selectedModelId: null,
  activeRoute: "data-upload",
  workflowFlags: {},
  profileLoaded: false,
  evaluationLoaded: false,
  driftStatus: "Not checked",
  driftReportId: null,
  retrainCandidateId: null,
};

let workspaceContext = { ...DEFAULT_CONTEXT };
let persistWorkspaceContext = null;


export function initWorkspaceContextUI(options = {}) {
  persistWorkspaceContext = options.persistWorkspaceContext || null;

  window.addEventListener("dataagent:context-changed", (event) => {
    renderWorkspaceContext(event.detail || workspaceContext);
  });

  renderWorkspaceContext(workspaceContext);
}


export function setWorkspaceContext(partialContext, options = {}) {
  workspaceContext = {
    ...workspaceContext,
    ...partialContext,
    workflowFlags: options.replaceWorkflowFlags
      ? { ...(partialContext.workflowFlags || {}) }
      : {
        ...workspaceContext.workflowFlags,
        ...(partialContext.workflowFlags || {}),
      },
  };

  window.dispatchEvent(
    new CustomEvent("dataagent:context-changed", {
      detail: workspaceContext,
    }),
  );

  if (options.persist !== false && persistWorkspaceContext) {
    persistWorkspaceContext(workspaceContext);
  }
}


export function getWorkspaceContext() {
  return workspaceContext;
}


function renderWorkspaceContext(context) {
  const dataset = context.dataset;
  const productionModel = findProductionModel(context.models);
  const selectedModel = findSelectedModel(context.models, context.selectedModelId);
  const displayModel = productionModel || selectedModel;
  const workflowState = deriveWorkflowState(context);
  const rowCount = dataset ? getDatasetValue(dataset, "row_count", "rows") : null;
  const columnCount = dataset ? getDatasetValue(dataset, "column_count", "columns") : null;

  setText("#contextDatasetName", dataset?.name || "No dataset selected");
  setText(
    "#contextDatasetMeta",
    dataset
      ? `${formatCount(rowCount)} rows · ${formatCount(columnCount)} columns · ${dataset.status || "ready"}`
      : "Upload or select a dataset to begin.",
  );
  setText("#contextVersion", context.version?.version_id || dataset?.latest_version_id || "-");
  setText("#contextModel", displayModel?.model_name || "-");
  setText("#contextDrift", context.driftStatus || "Not checked");
  setText("#inspectorDataset", dataset?.name || "None selected");
  setText(
    "#inspectorShape",
    dataset ? `${formatCount(rowCount)} x ${formatCount(columnCount)}` : "-",
  );
  setText("#inspectorTarget", context.targetColumn || displayModel?.target_column || "Auto / none");
  setText(
    "#inspectorProductionModel",
    productionModel ? `${productionModel.model_name} (${productionModel.task_type})` : "-",
  );

  renderWorkflowTimeline(workflowState);
  renderNextActions(workflowState);
}


function renderWorkflowTimeline(workflowState) {
  const container = document.querySelector("#workflowTimeline");

  if (!container) {
    return;
  }

  container.innerHTML = workflowState.steps.map((step) => `
    <div
      class="workflow-step ${step.status}"
      data-workflow-step-id="${step.id}"
      data-workflow-step-status="${step.status}"
    >
      <span>${step.label}</span>
      <strong>${step.statusLabel}</strong>
    </div>
  `).join("");
}


function renderNextActions(workflowState) {
  const container = document.querySelector("#nextActionList");

  if (!container) {
    return;
  }

  const actions = buildNextActions(workflowState);

  container.innerHTML = actions.slice(0, 5).map((action) => `
    <button
      class="next-action-button ${action.status} ${action.priority}"
      type="button"
      data-view-target="${action.route}"
      data-next-action-id="${action.id}"
      data-next-action-priority="${action.priority}"
      data-next-action-status="${action.status}"
    >
      <span>${action.label}</span>
      <small>${action.reason}</small>
    </button>
  `).join("");
}


export function deriveWorkflowState(context) {
  const flags = context.workflowFlags || {};
  const productionModel = findProductionModel(context.models);
  const selectedModel = findSelectedModel(context.models, context.selectedModelId);
  const hasDataset = Boolean(context.dataset);
  const hasSchema = Boolean(flags.schemaReviewed || context.profileLoaded || context.targetColumn);
  const hasEda = Boolean(flags.edaReviewed);
  const hasTarget = Boolean(context.targetColumn || selectedModel?.target_column);
  const hasModel = Boolean(selectedModel || context.models.length);
  const hasTraining = Boolean(flags.trainingCompleted || hasModel);
  const hasEvaluation = Boolean(flags.evaluationCompleted || context.evaluationLoaded);
  const hasExplainability = Boolean(flags.explainabilityCompleted);
  const hasPrediction = Boolean(flags.predictionCompleted);
  const hasReport = Boolean(flags.reportGenerated);
  const hasProductionModel = Boolean(productionModel);
  const hasDriftCheck = context.driftStatus && context.driftStatus !== "Not checked";
  const driftWarning = isDriftWarning(context.driftStatus);
  const hasRetrainCandidate = Boolean(context.retrainCandidateId);
  const hasMigrationCheck = Boolean(flags.migrationChecked);

  return {
    context,
    flags,
    productionModel,
    selectedModel,
    hasDataset,
    hasSchema,
    hasEda,
    hasTarget,
    hasModel,
    hasTraining,
    hasEvaluation,
    hasExplainability,
    hasPrediction,
    hasReport,
    hasProductionModel,
    hasDriftCheck,
    driftWarning,
    hasRetrainCandidate,
    hasMigrationCheck,
    steps: [
      workflowStep("upload", "Upload", hasDataset ? "complete" : "ready"),
      workflowStep("profile", "Profile", hasSchema ? "complete" : hasDataset ? "ready" : "blocked"),
      workflowStep("target", "Target", hasTarget ? "complete" : hasSchema ? "ready" : "blocked"),
      workflowStep("train", "Train", hasTraining ? "complete" : hasTarget ? "ready" : "blocked"),
      workflowStep(
        "evaluate",
        "Evaluate",
        hasEvaluation ? "complete" : hasTraining ? "ready" : "blocked",
      ),
      workflowStep(
        "explain",
        "Explain",
        hasExplainability ? "complete" : hasEvaluation ? "ready" : "blocked",
      ),
      workflowStep(
        "monitor",
        "Monitor",
        driftWarning ? "warning" : hasDriftCheck ? "complete" : hasProductionModel ? "ready" : "blocked",
      ),
    ],
  };
}


export function buildNextActions(workflowState) {
  const {
    hasDataset,
    hasSchema,
    hasEda,
    hasTarget,
    hasTraining,
    hasEvaluation,
    hasExplainability,
    hasProductionModel,
    hasDriftCheck,
    driftWarning,
    hasRetrainCandidate,
    hasMigrationCheck,
    hasPrediction,
    hasReport,
  } = workflowState;

  if (!hasDataset) {
    return [
      nextAction(
        "upload-dataset",
        "data-upload",
        "Upload a dataset",
        "Start the workspace",
        "required",
        "ready",
      ),
    ];
  }

  if (!hasSchema) {
    return [
      nextAction(
        "inspect-schema",
        "analyze-schema",
        "Inspect schema",
        "Profile columns and detect target candidates",
        "required",
        "ready",
      ),
      nextAction(
        "preview-rows",
        "data-preview",
        "Preview rows",
        "Confirm the uploaded data shape",
        "recommended",
        "ready",
      ),
    ];
  }

  if (!hasTarget) {
    return [
      nextAction(
        "choose-target",
        "model-workbench",
        "Choose target",
        "Select or confirm the prediction target",
        "required",
        "ready",
      ),
      nextAction(
        "review-eda",
        "analyze-eda",
        "Review EDA",
        "Check data quality before training",
        "recommended",
        "ready",
      ),
    ];
  }

  if (!hasTraining) {
    return [
      nextAction(
        "run-ml-experiment",
        "model-workbench",
        "Run ML experiment",
        "Train candidate models",
        "required",
        "ready",
      ),
      nextAction(
        "open-visualization-lab",
        "analyze-visualization",
        "Open Visualization Lab",
        "Inspect target relationships",
        "recommended",
        "ready",
      ),
    ];
  }

  if (!hasEvaluation) {
    return [
      nextAction(
        "review-evaluation",
        "model-registry",
        "Review evaluation",
        "Inspect metrics and diagnostics",
        "recommended",
        "ready",
      ),
    ];
  }

  if (!hasExplainability) {
    return [
      nextAction(
        "explain-model",
        "model-explainability",
        "Explain model",
        "Generate model explanations",
        "recommended",
        "ready",
      ),
    ];
  }

  if (driftWarning) {
    return [
      nextAction(
        "build-retrain-plan",
        "lifecycle-drift",
        "Build retrain plan",
        "Drift warning needs action",
        "required",
        "warning",
      ),
      nextAction(
        "retrain-challenger",
        "lifecycle-drift",
        "Retrain challenger",
        "Create a candidate model on current data",
        "recommended",
        "warning",
      ),
    ];
  }

  if (hasRetrainCandidate) {
    return [
      nextAction(
        "run-migration-check",
        "model-registry",
        "Run migration check",
        "Validate champion and challenger compatibility",
        "recommended",
        hasMigrationCheck ? "complete" : "ready",
      ),
      nextAction(
        "promote-challenger",
        "model-registry",
        "Promote challenger",
        "Use after reviewing comparison results",
        "optional",
        "ready",
      ),
    ];
  }

  if (hasProductionModel && !hasDriftCheck) {
    return [
      nextAction(
        "run-drift-check",
        "lifecycle-drift",
        "Run drift check",
        "Production model needs monitoring",
        "recommended",
        "ready",
      ),
      nextAction(
        "generate-report",
        "lifecycle-reports",
        "Generate report",
        "Create a lifecycle snapshot",
        "optional",
        hasReport ? "complete" : "ready",
      ),
    ];
  }

  const actions = [
    nextAction(
      "run-prediction",
      "model-prediction",
      "Run prediction",
      "Use the selected model",
      "optional",
      hasPrediction ? "complete" : "ready",
    ),
    nextAction(
      "generate-report",
      "lifecycle-reports",
      "Generate report",
      "Export current findings",
      "optional",
      hasReport ? "complete" : "ready",
    ),
    nextAction(
      "run-agent-job",
      "agent-workflow",
      "Run agent job",
      "Automate the workflow",
      "optional",
      "ready",
    ),
  ];

  if (!hasEda) {
    actions.unshift(
      nextAction(
        "review-eda",
        "analyze-eda",
        "Review EDA",
        "Check data quality before reporting",
        "recommended",
        "ready",
      ),
    );
  }

  return actions;
}


function workflowStep(id, label, status) {
  const statusLabels = {
    blocked: "Blocked",
    complete: "Done",
    ready: "Ready",
    warning: "Warning",
  };

  return {
    id,
    label,
    status,
    statusLabel: statusLabels[status] || status,
  };
}


function nextAction(id, route, label, reason, priority, status, blocker = null) {
  return {
    id,
    route,
    label,
    reason,
    priority,
    status,
    blocker,
  };
}


function isDriftWarning(status) {
  return ["warning", "fail", "failed", "drift", "drifted", "critical"].includes(
    String(status || "").toLowerCase(),
  );
}


function findProductionModel(models) {
  return models.find((model) => model.lifecycle_status === "production") || null;
}


function findSelectedModel(models, selectedModelId) {
  if (!selectedModelId) {
    return models[0] || null;
  }

  return models.find((model) => model.id === selectedModelId) || models[0] || null;
}


function setText(selector, value) {
  const element = document.querySelector(selector);

  if (element) {
    element.textContent = value;
  }
}


function getDatasetValue(dataset, primaryKey, fallbackKey) {
  return dataset[primaryKey] ?? dataset[fallbackKey] ?? "-";
}


function formatCount(value) {
  return value ?? "-";
}
