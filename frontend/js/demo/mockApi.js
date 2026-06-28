const DEMO_DATASET_ID = "demo-customer-churn";
const DEMO_VERSION_ID = "v1";
const DEMO_VERSION_DRIFT_ID = "v2";
const DEMO_MODEL_ID = "model-logistic-production";
const DEMO_CHALLENGER_ID = "model-logistic-challenger";
const DEMO_REPORT_ID = "report-demo-001";
const DEMO_RUN_ID = "playwright-demo-run";
const DEMO_JOB_ID = "backtest-job-demo";
const DEMO_AGENT_JOB_ID = "agent-job-demo";
const DEMO_WORKFLOW_ID = "agent-workflow-demo";

const now = () => new Date().toISOString();

const columns = [
  "customer_id",
  "tenure",
  "monthly_charges",
  "total_charges",
  "contract_type",
  "support_calls",
  "is_senior",
  "churn",
];

const previewRows = [
  {
    customer_id: "C-1001",
    tenure: 22,
    monthly_charges: 72.4,
    total_charges: 1592.8,
    contract_type: "month-to-month",
    support_calls: 3,
    is_senior: false,
    churn: "yes",
  },
  {
    customer_id: "C-1002",
    tenure: 54,
    monthly_charges: 64.1,
    total_charges: 3461.4,
    contract_type: "two-year",
    support_calls: 0,
    is_senior: false,
    churn: "no",
  },
  {
    customer_id: "C-1003",
    tenure: 8,
    monthly_charges: 91.2,
    total_charges: 729.6,
    contract_type: "month-to-month",
    support_calls: 5,
    is_senior: true,
    churn: "yes",
  },
  {
    customer_id: "C-1004",
    tenure: 37,
    monthly_charges: 58.9,
    total_charges: 2179.3,
    contract_type: "one-year",
    support_calls: 1,
    is_senior: false,
    churn: "no",
  },
  {
    customer_id: "C-1005",
    tenure: 15,
    monthly_charges: 83.5,
    total_charges: 1252.5,
    contract_type: "month-to-month",
    support_calls: 4,
    is_senior: true,
    churn: "yes",
  },
];

const dataset = {
  id: DEMO_DATASET_ID,
  name: "customer_churn_demo.csv",
  original_filename: "customer_churn_demo.csv",
  stored_filename: "customer_churn_demo.csv",
  file_size_bytes: 24218,
  row_count: 300,
  column_count: columns.length,
  encoding: "utf-8",
  status: "ready",
  created_at: now(),
  updated_at: now(),
  latest_version_id: DEMO_VERSION_DRIFT_ID,
  columns,
  preview_rows: previewRows,
};

let workspaceState = {
  workspace_id: "default",
  active_route: "data-upload",
  dataset_id: DEMO_DATASET_ID,
  dataset_version_id: DEMO_VERSION_ID,
  target_column: "churn",
  selected_model_id: DEMO_MODEL_ID,
  drift_report_id: "drift-demo-001",
  drift_status: "warning",
  workflow_flags: {
    schemaReviewed: true,
    edaReviewed: true,
    trainingCompleted: true,
    evaluationCompleted: true,
    explainabilityCompleted: true,
    driftChecked: true,
  },
  retrain_candidate_id: DEMO_CHALLENGER_ID,
  created_at: now(),
  updated_at: now(),
};

const columnProfiles = [
  profile("customer_id", "categorical", "identifier", 0, 300, ["C-1001", "C-1002"]),
  profile("tenure", "numeric", "feature", 0, 72, [22, 54, 8], {
    count: 300,
    mean: 31.4,
    std: 18.1,
    min: 1,
    q25: 12,
    median: 29,
    q75: 47,
    max: 72,
  }),
  profile("monthly_charges", "numeric", "feature", 2, 210, [72.4, 64.1, 91.2], {
    count: 298,
    mean: 74.8,
    std: 19.2,
    min: 28.9,
    q25: 58.4,
    median: 73.1,
    q75: 90.4,
    max: 119.6,
  }),
  profile("total_charges", "numeric", "feature", 4, 290, [1592.8, 3461.4, 729.6], {
    count: 296,
    mean: 2291.3,
    std: 1241.9,
    min: 58.9,
    q25: 1041.2,
    median: 2108.5,
    q75: 3370.7,
    max: 8271.4,
  }),
  profile("contract_type", "categorical", "feature", 0, 3, ["month-to-month", "one-year"], null, [
    { value: "month-to-month", count: 162 },
    { value: "one-year", count: 78 },
    { value: "two-year", count: 60 },
  ]),
  profile("support_calls", "numeric", "feature", 0, 8, [3, 0, 5], {
    count: 300,
    mean: 1.8,
    std: 1.5,
    min: 0,
    q25: 1,
    median: 2,
    q75: 3,
    max: 8,
  }),
  profile("is_senior", "boolean", "feature", 0, 2, [false, true]),
  profile("churn", "categorical", "target", 0, 2, ["yes", "no"], null, [
    { value: "no", count: 198 },
    { value: "yes", count: 102 },
  ]),
];

const model = {
  id: DEMO_MODEL_ID,
  dataset_id: DEMO_DATASET_ID,
  dataset_version_id: DEMO_VERSION_ID,
  model_name: "logistic_regression",
  task_type: "classification",
  target_column: "churn",
  feature_columns: ["tenure", "monthly_charges", "total_charges", "contract_type", "support_calls", "is_senior"],
  metrics: {
    accuracy: 0.873,
    precision: 0.821,
    recall: 0.784,
    f1: 0.802,
    roc_auc: 0.912,
  },
  feature_importance: [
    { feature: "contract_type", importance: 0.31 },
    { feature: "tenure", importance: 0.24 },
    { feature: "support_calls", importance: 0.18 },
    { feature: "monthly_charges", importance: 0.14 },
    { feature: "is_senior", importance: 0.08 },
  ],
  model_path: "data/models/demo-logistic-regression.joblib",
  evaluation_artifacts_path: "data/models/demo-logistic-regression-evaluation.json",
  status: "trained",
  lifecycle_status: "production",
  training_config: {
    test_size: 0.25,
    random_state: 42,
  },
  feature_schema: {
    required_columns: ["tenure", "monthly_charges", "total_charges", "contract_type", "support_calls", "is_senior"],
  },
  preprocessing_recipe: {
    numeric_scaler: "standard",
    categorical_encoder: "one_hot",
  },
  created_at: now(),
};

const challenger = {
  ...model,
  id: DEMO_CHALLENGER_ID,
  model_name: "logistic_regression_challenger",
  lifecycle_status: "candidate",
  metrics: {
    accuracy: 0.889,
    precision: 0.842,
    recall: 0.807,
    f1: 0.824,
    roc_auc: 0.928,
  },
};

function profile(name, inferredType, semanticRole, missingCount, uniqueCount, sampleValues, numericStats = null, topValues = []) {
  return {
    name,
    inferred_type: inferredType,
    semantic_role: semanticRole,
    missing_count: missingCount,
    missing_ratio: missingCount / dataset.row_count,
    unique_count: uniqueCount,
    unique_ratio: uniqueCount / dataset.row_count,
    sample_values: sampleValues,
    top_values: topValues,
    numeric_stats: numericStats,
    datetime_stats: null,
  };
}

function schemaResponse() {
  return {
    summary: {
      dataset_id: DEMO_DATASET_ID,
      row_count: dataset.row_count,
      column_count: dataset.column_count,
      type_counts: {
        numeric: 4,
        categorical: 3,
        boolean: 1,
      },
      missing_cell_count: 6,
      missing_cell_ratio: 0.0025,
      duplicate_row_count: 3,
      duplicate_row_ratio: 0.01,
      target_candidates: ["churn"],
    },
    columns: columnProfiles,
  };
}

function edaResponse() {
  return {
    dataset_id: DEMO_DATASET_ID,
    row_count: dataset.row_count,
    column_count: dataset.column_count,
    data_quality_score: 91.8,
    data_quality_grade: "A-",
    missing: {
      dataset_id: DEMO_DATASET_ID,
      total_missing_cells: 6,
      missing_cell_ratio: 0.0025,
      columns: [
        { name: "monthly_charges", inferred_type: "numeric", missing_count: 2, missing_ratio: 0.0067 },
        { name: "total_charges", inferred_type: "numeric", missing_count: 4, missing_ratio: 0.0133 },
      ],
    },
    duplicates: {
      dataset_id: DEMO_DATASET_ID,
      duplicate_row_count: 3,
      duplicate_row_ratio: 0.01,
    },
    numeric_statistics: {
      dataset_id: DEMO_DATASET_ID,
      columns: columnProfiles.filter((item) => item.numeric_stats).map((item) => ({
        name: item.name,
        ...item.numeric_stats,
        skewness: item.name === "total_charges" ? 0.72 : 0.18,
        kurtosis: item.name === "support_calls" ? 1.24 : 0.33,
      })),
    },
    outliers: {
      dataset_id: DEMO_DATASET_ID,
      columns: [
        { name: "monthly_charges", method: "iqr", lower_bound: 10.4, upper_bound: 138.4, outlier_count: 0, outlier_ratio: 0 },
        { name: "support_calls", method: "iqr", lower_bound: -2, upper_bound: 6, outlier_count: 7, outlier_ratio: 0.023 },
      ],
    },
    correlation: {
      dataset_id: DEMO_DATASET_ID,
      method: "pearson",
      columns: ["tenure", "monthly_charges", "total_charges", "support_calls"],
      matrix: {
        tenure: { tenure: 1, monthly_charges: -0.18, total_charges: 0.82, support_calls: -0.37 },
        monthly_charges: { tenure: -0.18, monthly_charges: 1, total_charges: 0.41, support_calls: 0.29 },
        total_charges: { tenure: 0.82, monthly_charges: 0.41, total_charges: 1, support_calls: -0.12 },
        support_calls: { tenure: -0.37, monthly_charges: 0.29, total_charges: -0.12, support_calls: 1 },
      },
      strongest_pairs: [
        { column_x: "tenure", column_y: "total_charges", correlation: 0.82 },
        { column_x: "tenure", column_y: "support_calls", correlation: -0.37 },
      ],
    },
    recommendations: [
      "Review support_calls outliers before retraining.",
      "Use churn as the supervised learning target.",
      "Monitor contract_type distribution because month-to-month customers dominate churn risk.",
    ],
    generated_at: now(),
  };
}

function visualizationRecommendations() {
  return {
    dataset_id: DEMO_DATASET_ID,
    recommendations: [
      chartSpec("histogram", "tenure", null, "Tenure Distribution"),
      chartSpec("bar", "contract_type", "churn", "Churn by Contract"),
      chartSpec("scatter", "monthly_charges", "total_charges", "Charges Relationship"),
    ],
    total: 3,
  };
}

function visualizationLab() {
  return {
    dataset_id: DEMO_DATASET_ID,
    summary: {
      row_count: dataset.row_count,
      column_count: dataset.column_count,
      sampled_row_count: 80,
      target_column: "churn",
      numeric_columns: ["tenure", "monthly_charges", "total_charges", "support_calls"],
      categorical_columns: ["contract_type", "churn"],
      datetime_columns: [],
      column_options: columnProfiles.map((item) => ({
        name: item.name,
        inferred_type: item.inferred_type,
        semantic_role: item.semantic_role,
      })),
    },
    warnings: ["Demo mode uses a deterministic fixture sample."],
    charts: [
      chartSpec("bar", "contract_type", "churn", "Churn Rate by Contract"),
      chartSpec("scatter", "monthly_charges", "tenure", "Charge vs Tenure"),
      chartSpec("box", "contract_type", "monthly_charges", "Charges by Contract"),
    ],
  };
}

function chartSpec(type, xColumn, yColumn, title) {
  return {
    chart_id: `${type}-${xColumn}-${yColumn || "count"}`,
    chart_type: type,
    title,
    description: `Demo ${type} chart for ${xColumn}${yColumn ? ` and ${yColumn}` : ""}.`,
    x_column: xColumn,
    y_column: yColumn,
    group_column: yColumn === "churn" ? "churn" : null,
    option: {
      title: { text: title, left: "center" },
      tooltip: { trigger: "axis" },
      grid: { left: 42, right: 20, bottom: 40, top: 56 },
      xAxis: { type: "category", data: ["low", "medium", "high"] },
      yAxis: { type: "value" },
      series: [
        {
          type: type === "scatter" ? "scatter" : type === "line" ? "line" : "bar",
          data: type === "scatter" ? [[24, 62], [41, 73], [13, 94], [55, 59]] : [42, 88, 51],
          itemStyle: { color: "#2563eb" },
        },
      ],
    },
    sample_size: 80,
    warnings: [],
  };
}

function trainResponse() {
  return {
    dataset_id: DEMO_DATASET_ID,
    task_type: "classification",
    target_column: "churn",
    feature_columns: model.feature_columns,
    model_count: 2,
    best_model_id: DEMO_MODEL_ID,
    best_metric_name: "roc_auc",
    best_metric_value: model.metrics.roc_auc,
    models: [model, challenger],
  };
}

function evaluationResponse(modelId = DEMO_MODEL_ID) {
  const selected = modelId === DEMO_CHALLENGER_ID ? challenger : model;
  return {
    model: selected,
    confusion_matrix: {
      labels: ["no", "yes"],
      matrix: [[47, 5], [7, 16]],
    },
    regression_residuals: [],
    feature_importance: selected.feature_importance,
  };
}

function mlPlan() {
  return {
    dataset_id: DEMO_DATASET_ID,
    detected_task_type: "classification",
    requested_task_type: "auto",
    target_column: "churn",
    estimated_feature_count: 12,
    feature_groups: {
      numeric: ["tenure", "monthly_charges", "total_charges", "support_calls"],
      categorical: ["contract_type"],
      boolean: ["is_senior"],
      datetime: [],
      text: [],
    },
    preprocessing_steps: [
      { name: "Impute missing values", description: "Median/mode imputation for sparse missing cells.", applies_to: ["monthly_charges", "total_charges"] },
      { name: "Scale numeric features", description: "Standard scaling for model stability.", applies_to: ["tenure", "monthly_charges", "total_charges"] },
      { name: "Encode categoricals", description: "One-hot encode contract_type.", applies_to: ["contract_type"] },
    ],
    available_models: [
      { key: "logistic_regression", label: "Logistic Regression", task_type: "classification", selected: true },
      { key: "random_forest", label: "Random Forest", task_type: "classification", selected: true },
    ],
    warnings: ["Demo mode returns deterministic planning output."],
  };
}

function mlExperiment() {
  return {
    experiment_id: "experiment-demo-001",
    dataset_id: DEMO_DATASET_ID,
    dataset_version_id: DEMO_VERSION_ID,
    task_type: "classification",
    target_column: "churn",
    status: "success",
    primary_metric: "roc_auc",
    best_model_id: DEMO_MODEL_ID,
    best_model_name: "logistic_regression",
    best_metric_value: 0.912,
    model_results: [
      experimentResult("logistic_regression", "Logistic Regression", 0.912, 0.873),
      experimentResult("random_forest", "Random Forest", 0.904, 0.861),
    ],
    projection: {
      points: previewRows.map((row, index) => ({
        x: index - 2,
        y: index % 2 ? 1.2 : -0.8,
        label: row.churn,
      })),
    },
    warnings: [],
    created_at: now(),
  };
}

function experimentResult(key, label, rocAuc, accuracy) {
  return {
    model_key: key,
    model_label: label,
    model_id: key === "logistic_regression" ? DEMO_MODEL_ID : "model-random-forest-demo",
    status: "success",
    metrics: { roc_auc: rocAuc, accuracy },
    cv_metrics: [
      { name: "roc_auc", mean: rocAuc - 0.01, std: 0.025 },
      { name: "accuracy", mean: accuracy - 0.008, std: 0.019 },
    ],
    test_metrics: [
      { name: "roc_auc", value: rocAuc },
      { name: "accuracy", value: accuracy },
    ],
  };
}

function explainabilityResponse() {
  return {
    model_id: DEMO_MODEL_ID,
    dataset_id: DEMO_DATASET_ID,
    task_type: "classification",
    target_column: "churn",
    cache_hit: false,
    holdout: {
      row_count: 75,
      positive_class: "yes",
      metrics: model.metrics,
    },
    permutation_importance: {
      available: true,
      items: model.feature_importance.map((item) => ({
        feature: item.feature,
        importance_mean: item.importance,
        importance_std: item.importance / 8,
      })),
    },
    shap: {
      available: true,
      explainer_type: "kernel-demo",
      feature_importance: model.feature_importance.map((item) => ({
        feature: item.feature,
        mean_abs_shap: item.importance,
      })),
      beeswarm: model.feature_importance.slice(0, 4).map((item, index) => ({
        feature: item.feature,
        values: [-0.18, -0.06, 0.04, 0.13].map((value) => value + index * 0.02),
      })),
      local_explanation: {
        row_index: 0,
        base_value: 0.34,
        prediction_value: 0.71,
        contributions: model.feature_importance.slice(0, 5).map((item) => ({
          feature: item.feature,
          value: item.importance > 0.15 ? "high" : "medium",
          contribution: item.importance / 2,
        })),
      },
    },
    curves: {
      roc: {
        points: [
          { x: 0, y: 0 },
          { x: 0.08, y: 0.48 },
          { x: 0.18, y: 0.71 },
          { x: 0.42, y: 0.9 },
          { x: 1, y: 1 },
        ],
        auc: 0.912,
      },
      precision_recall: {
        points: [
          { x: 0.2, y: 0.94 },
          { x: 0.45, y: 0.88 },
          { x: 0.7, y: 0.79 },
          { x: 0.9, y: 0.62 },
        ],
        auc: 0.84,
      },
    },
    error_samples: [
      { row_index: 4, actual: "yes", predicted: "no", probability: 0.42, raw_record: previewRows[4] },
      { row_index: 1, actual: "no", predicted: "yes", probability: 0.58, raw_record: previewRows[1] },
    ],
    warnings: ["SHAP values are fixture values for the static demo."],
    generated_at: now(),
  };
}

function driftReport() {
  return {
    report_id: "drift-demo-001",
    dataset_id: DEMO_DATASET_ID,
    reference_version_id: DEMO_VERSION_ID,
    current_version_id: DEMO_VERSION_DRIFT_ID,
    model_id: DEMO_MODEL_ID,
    target_column: "churn",
    status: "warning",
    schema_drift: {
      added_columns: [],
      removed_columns: [],
      type_changes: [],
      status: "stable",
    },
    feature_drift: [
      { column: "contract_type", drift_type: "categorical", status: "warning", psi: 0.22, js_distance: 0.11 },
      { column: "monthly_charges", drift_type: "numeric", status: "stable", psi: 0.08, ks_statistic: 0.09 },
      { column: "support_calls", drift_type: "numeric", status: "warning", psi: 0.19, ks_statistic: 0.16 },
    ],
    recommendations: [
      "Review month-to-month contract mix before promotion.",
      "Retrain a challenger model on v2 if warning persists.",
    ],
    retraining_recommendation: {
      action: "retrain",
      score: 0.68,
      reasons: ["Two monitored features moved beyond warning thresholds."],
    },
    created_at: now(),
  };
}

function reportMarkdown() {
  return [
    "# DataAgent Analyst Demo Report",
    "",
    "The customer churn fixture has 300 rows and 8 columns.",
    "",
    "## EDA",
    "- Data quality score: 91.8",
    "- Strongest correlation: tenure and total_charges",
    "- Support calls show mild outlier pressure.",
    "",
    "## Model",
    "- Production model: logistic_regression",
    "- ROC AUC: 0.912",
    "- Most important drivers: contract_type, tenure, support_calls.",
    "",
    "## Lifecycle",
    "Drift status is warning. Retraining a challenger is recommended before migration.",
  ].join("\n");
}

function agentRun() {
  const steps = [
    agentStep("planner", "success", "Selected churn as target and planned the workflow."),
    agentStep("profiler", "success", "Profiled schema, missingness, and data quality."),
    agentStep("visualizer", "success", "Rendered recommended charts."),
    agentStep("trainer", "success", "Trained baseline classifier."),
    agentStep("explainer", "success", "Computed feature importance and SHAP-style diagnostics."),
    agentStep("reporter", "success", "Generated the Markdown analysis report."),
  ];

  return {
    workflow_id: DEMO_WORKFLOW_ID,
    dataset_id: DEMO_DATASET_ID,
    status: "success",
    user_goal: "Run the complete portfolio demo workflow.",
    steps,
    outputs: {
      report_id: DEMO_REPORT_ID,
      best_model_id: DEMO_MODEL_ID,
      drift_report_id: "drift-demo-001",
    },
    errors: [],
    final_summary: "Completed the guided demo analysis, training, explanation, and reporting workflow.",
  };
}

function agentStep(name, status, message) {
  return {
    name,
    status,
    message,
    payload: {},
    created_at: now(),
  };
}

function backtestRunDetail() {
  return {
    summary: {
      run_id: DEMO_RUN_ID,
      status: "success",
      created_at: now(),
      metadata: { suite: "ui", description: "Static demo verified workflow" },
      step_count: 6,
      failed_step_count: 0,
      assertion_count: 8,
      failed_assertion_count: 0,
      screenshot_count: 4,
      payload_count: 1,
      has_summary: true,
      error: null,
    },
    steps: [
      { name: "open_demo", status: "success", duration_seconds: 0.42, payload_path: null, error: null, metadata: { route: "data-upload" } },
      { name: "guide_tour", status: "success", duration_seconds: 2.8, payload_path: null, error: null, metadata: { steps: 18 } },
      { name: "upload_fixture", status: "success", duration_seconds: 0.55, payload_path: "payloads/upload.json", error: null, metadata: {} },
      { name: "train_model", status: "success", duration_seconds: 1.2, payload_path: null, error: null, metadata: {} },
      { name: "check_drift", status: "success", duration_seconds: 0.66, payload_path: null, error: null, metadata: {} },
      { name: "render_report", status: "success", duration_seconds: 0.35, payload_path: null, error: null, metadata: {} },
    ],
    assertions: [
      { name: "no_console_errors", status: "success", expected: true, actual: true, message: "No blocking browser errors." },
      { name: "guide_visible", status: "success", expected: "visible", actual: "visible", message: "Guide tour auto-started." },
      { name: "workflow_complete", status: "success", expected: "success", actual: "success", message: "Fixture flow completed." },
    ],
    payloads: [
      { name: "demo_payload.json", path: "payloads/demo_payload.json", size_bytes: 412 },
    ],
    screenshots: [
      { name: "upload.png", path: "screenshots/upload.png", size_bytes: 182340 },
      { name: "schema.png", path: "screenshots/schema.png", size_bytes: 192104 },
      { name: "ml-workbench.png", path: "screenshots/ml-workbench.png", size_bytes: 201102 },
      { name: "guide-tour.png", path: "screenshots/guide-tour.png", size_bytes: 221943 },
    ],
    summary_markdown: "## Static Demo Backtest\n\nAll fixture checks passed for the GitHub Pages demo.",
  };
}

function apiPayload(method, path) {
  if (path === "/api/health") {
    return { status: "ok", service: "DataAgent Analyst static demo", demo_mode: true };
  }

  if (path === "/api/workspaces/default/state" && method === "GET") {
    return workspaceState;
  }

  if (path === "/api/workspaces/default/state" && method === "PATCH") {
    workspaceState = { ...workspaceState, updated_at: now() };
    return workspaceState;
  }

  if (path === "/api/datasets" && method === "GET") {
    return { datasets: [dataset], total: 1 };
  }

  if (path === "/api/datasets/upload" && method === "POST") {
    return { message: "Demo dataset loaded from static fixture.", dataset };
  }

  if (path === `/api/datasets/${DEMO_DATASET_ID}/preview`) {
    return { dataset_id: DEMO_DATASET_ID, version_id: DEMO_VERSION_ID, columns, rows: previewRows, row_count: 300, preview_row_count: previewRows.length };
  }

  if (path === `/api/datasets/${DEMO_DATASET_ID}/versions`) {
    return {
      dataset_id: DEMO_DATASET_ID,
      versions: [
        version(DEMO_VERSION_ID, null, "original", 300, columns.length, null),
        version(DEMO_VERSION_DRIFT_ID, DEMO_VERSION_ID, "derived", 294, columns.length, {
          drop_duplicate_rows: true,
          fill_missing: [{ column: "total_charges", strategy: "median" }],
        }),
      ],
      total: 2,
      latest_version_id: DEMO_VERSION_DRIFT_ID,
    };
  }

  if (path === `/api/datasets/${DEMO_DATASET_ID}/transform` && method === "POST") {
    return {
      dataset_id: DEMO_DATASET_ID,
      source_version_id: DEMO_VERSION_ID,
      row_count_before: 300,
      row_count_after: 294,
      column_count_before: columns.length,
      column_count_after: columns.length,
      columns,
      preview_rows: previewRows,
      profile_diff: { row_delta: -6, column_delta: 0, missing_cell_delta: -6 },
      warnings: [],
      version: version(DEMO_VERSION_DRIFT_ID, DEMO_VERSION_ID, "derived", 294, columns.length, {}),
    };
  }

  if (path === `/api/datasets/${DEMO_DATASET_ID}/schema`) return schemaResponse();
  if (path === `/api/eda/${DEMO_DATASET_ID}/summary`) return edaResponse();
  if (path === `/api/visualizations/${DEMO_DATASET_ID}/recommendations`) return visualizationRecommendations();
  if (path === `/api/visualizations/${DEMO_DATASET_ID}/lab` && method === "POST") return visualizationLab();
  if (path === `/api/visualizations/${DEMO_DATASET_ID}/build` && method === "POST") return chartSpec("bar", "contract_type", "churn", "Custom Demo Chart");
  if (path === `/api/ml-workbench/${DEMO_DATASET_ID}/plan`) return mlPlan();
  if (path === `/api/ml-workbench/${DEMO_DATASET_ID}/experiments` && method === "GET") return { dataset_id: DEMO_DATASET_ID, experiments: [mlExperiment()], total: 1 };
  if (path === `/api/ml-workbench/${DEMO_DATASET_ID}/experiments` && method === "POST") return mlExperiment();
  if (path === "/api/ml-workbench/experiments/experiment-demo-001") return mlExperiment();
  if (path === `/api/ml/${DEMO_DATASET_ID}/train` && method === "POST") return trainResponse();
  if (path === `/api/ml/${DEMO_DATASET_ID}/models`) return { dataset_id: DEMO_DATASET_ID, models: [model, challenger], total: 2 };
  if (path === `/api/ml/models/${DEMO_MODEL_ID}/evaluation` || path === `/api/ml/models/${DEMO_CHALLENGER_ID}/evaluation`) return evaluationResponse(path.includes(DEMO_CHALLENGER_ID) ? DEMO_CHALLENGER_ID : DEMO_MODEL_ID);
  if (path === `/api/ml/models/${DEMO_MODEL_ID}/status` && method === "PATCH") return { ...model, lifecycle_status: "production" };
  if (path === `/api/ml/models/${DEMO_MODEL_ID}/migration-check` && method === "POST") return { model_id: DEMO_MODEL_ID, compatible: true, checks: [{ name: "feature_schema", status: "passed" }, { name: "preprocessing", status: "passed" }], warnings: [] };
  if (path === `/api/ml/models/${DEMO_MODEL_ID}/threshold-analysis` && method === "POST") return { model_id: DEMO_MODEL_ID, dataset_id: DEMO_DATASET_ID, dataset_version_id: DEMO_VERSION_ID, positive_class: "yes", points: [0.3, 0.4, 0.5, 0.6, 0.7].map((threshold) => ({ threshold, precision: 0.78 + threshold / 10, recall: 0.91 - threshold / 5, f1: 0.82, confusion_matrix: { tp: 16, fp: 5, tn: 47, fn: 7 } })) };
  if (path === `/api/ml/models/${DEMO_MODEL_ID}/segment-metrics` && method === "POST") return { model_id: DEMO_MODEL_ID, dataset_id: DEMO_DATASET_ID, dataset_version_id: DEMO_VERSION_ID, segment_column: "contract_type", segments: [{ segment: "month-to-month", row_count: 162, metrics: { f1: 0.79 } }, { segment: "two-year", row_count: 60, metrics: { f1: 0.88 } }] };
  if (path === `/api/ml/models/${DEMO_MODEL_ID}/what-if` && method === "POST") return { model_id: DEMO_MODEL_ID, dataset_id: DEMO_DATASET_ID, results: [{ scenario: "baseline_tenure", record: previewRows[0], prediction: "yes", probabilities: { no: 0.29, yes: 0.71 } }] };
  if (path === `/api/ml/models/${DEMO_MODEL_ID}/predict` && method === "POST") return predictionResponse();
  if (path === `/api/ml/models/${DEMO_MODEL_ID}/predict-csv` && method === "POST") return { ...predictionResponse(), original_filename: "batch_demo.csv" };
  if (path === `/api/ml/models/${DEMO_MODEL_ID}/retrain-plan` && method === "POST") return { champion_model_id: DEMO_MODEL_ID, dataset_id: DEMO_DATASET_ID, current_version_id: DEMO_VERSION_DRIFT_ID, target_column: "churn", task_type: "classification", selected_model: "logistic_regression", feature_columns: model.feature_columns, primary_metric: "roc_auc", warnings: ["Drift warning detected in contract_type."] };
  if (path === `/api/ml/models/${DEMO_MODEL_ID}/retrain` && method === "POST") return { champion_model: model, challenger_model: challenger, comparison: { dataset_id: DEMO_DATASET_ID, primary_metric: "roc_auc", models: [{ model_id: DEMO_CHALLENGER_ID, model_name: challenger.model_name, lifecycle_status: "candidate", task_type: "classification", target_column: "churn", metrics: challenger.metrics, primary_metric_value: 0.928, rank: 1 }, { model_id: DEMO_MODEL_ID, model_name: model.model_name, lifecycle_status: "production", task_type: "classification", target_column: "churn", metrics: model.metrics, primary_metric_value: 0.912, rank: 2 }] }, recommendation: "promote", reasons: ["Challenger improves ROC AUC."], promoted: false };
  if (path === `/api/ml/models/${DEMO_MODEL_ID}/promote-challenger` && method === "POST") return { promoted_model: { ...challenger, lifecycle_status: "production" }, archived_model_id: DEMO_MODEL_ID };
  if (path === "/api/drift/reports" && method === "POST") return driftReport();
  if (path === `/api/reports/${DEMO_DATASET_ID}/generate` && method === "POST") return { message: "Demo report generated.", report: reportRecord() };
  if (path === `/api/reports/dataset/${DEMO_DATASET_ID}`) return { dataset_id: DEMO_DATASET_ID, reports: [reportRecord()], total: 1 };
  if (path === `/api/reports/${DEMO_REPORT_ID}`) return reportRecord();
  if (path === "/api/bundles/export" && method === "POST") return { dataset_id: DEMO_DATASET_ID, bundle_path: "data/bundles/demo.zip", included_artifacts: ["dataset", "models", "reports"] };
  if (path === "/api/ai-insights/status") return { enabled: true, provider: "static-demo", model: "fixture-explainer", available: true, detail: "Static demo insight generator is enabled." };
  if (path.includes("/api/ai-insights/") && method === "POST") return insightResponse();
  if (path === `/api/explainability/models/${DEMO_MODEL_ID}/analyze` && method === "POST") return explainabilityResponse();
  if (path === `/api/explainability/models/${DEMO_CHALLENGER_ID}/analyze` && method === "POST") return explainabilityResponse();
  if (path === `/api/agents/${DEMO_DATASET_ID}/run` && method === "POST") return agentRun();
  if (path === `/api/agents/${DEMO_DATASET_ID}/runs`) return { dataset_id: DEMO_DATASET_ID, runs: [agentRunSummary()], total: 1 };
  if (path === `/api/agents/runs/${DEMO_WORKFLOW_ID}`) return agentRun();
  if (path === `/api/agent-jobs/${DEMO_DATASET_ID}` && method === "POST") return { message: "Demo background job accepted.", job: agentJobSummary("queued") };
  if (path === `/api/agent-jobs/${DEMO_DATASET_ID}` && method === "GET") return { dataset_id: DEMO_DATASET_ID, jobs: [agentJobSummary("success")], total: 1 };
  if (path === `/api/agent-jobs/${DEMO_AGENT_JOB_ID}`) return { ...agentJobSummary("success"), request: {}, events: agentEvents(), result: agentRun() };
  if (path === `/api/agent-jobs/${DEMO_AGENT_JOB_ID}/events`) return { job_id: DEMO_AGENT_JOB_ID, events: agentEvents(), total: agentEvents().length };
  if (path === "/api/backtests/runs") return { runs: [backtestRunDetail().summary], total: 1 };
  if (path === `/api/backtests/runs/${DEMO_RUN_ID}`) return backtestRunDetail();
  if (path === `/api/backtests/runs/${DEMO_RUN_ID}/payloads/demo_payload.json`) return { run_id: DEMO_RUN_ID, name: "demo_payload.json", payload: { fixture_payload: true, route: "lifecycle-backtests" } };
  if (path === "/api/backtests/suites") return { suites: [{ suite_id: "suite-demo-ui", suite: "ui", status: "success", created_at: now(), runs: [{ run_id: DEMO_RUN_ID }], error: null }], total: 1 };
  if (path === "/api/backtests/jobs" && method === "GET") return { jobs: [backtestJobSummary("success")], total: 1 };
  if (path === "/api/backtests/jobs" && method === "POST") return { message: "Demo backtest job started.", job: backtestJobSummary("queued") };
  if (path === `/api/backtests/jobs/${DEMO_JOB_ID}`) return { ...backtestJobSummary("success"), request: { suite_type: "ui", backend_port: 8010, frontend_port: 5174, keep_servers: false }, events: backtestEvents(), result: { run_ids: [DEMO_RUN_ID] } };
  if (path === `/api/backtests/jobs/${DEMO_JOB_ID}/events`) return { job_id: DEMO_JOB_ID, events: backtestEvents(), total: backtestEvents().length };

  return null;
}

function version(versionId, sourceVersionId, kind, rowCount, columnCount, recipe) {
  return {
    dataset_id: DEMO_DATASET_ID,
    version_id: versionId,
    source_version_id: sourceVersionId,
    version_index: versionId === DEMO_VERSION_ID ? 1 : 2,
    kind,
    file_path: `data/processed/${versionId}.csv`,
    row_count: rowCount,
    column_count: columnCount,
    columns,
    content_hash: `${versionId}-fixture-hash`,
    recipe,
    profile_diff: versionId === DEMO_VERSION_ID ? {} : { row_delta: -6, column_delta: 0, missing_cell_delta: -6 },
    warnings: [],
    created_at: now(),
  };
}

function predictionResponse() {
  return {
    model_id: DEMO_MODEL_ID,
    dataset_id: DEMO_DATASET_ID,
    model_name: model.model_name,
    task_type: "classification",
    target_column: "churn",
    feature_columns: model.feature_columns,
    predictions: [
      { row_index: 0, prediction: "yes", probabilities: { no: 0.29, yes: 0.71 } },
      { row_index: 1, prediction: "no", probabilities: { no: 0.82, yes: 0.18 } },
    ],
    total: 2,
  };
}

function reportRecord() {
  return {
    id: DEMO_REPORT_ID,
    dataset_id: DEMO_DATASET_ID,
    title: "Customer Churn Demo Analysis",
    markdown_content: reportMarkdown(),
    created_at: now(),
    file_path: "data/reports/customer_churn_demo.md",
  };
}

function insightResponse() {
  return {
    dataset_id: DEMO_DATASET_ID,
    model_id: DEMO_MODEL_ID,
    status: "success",
    insight_type: "demo",
    content: "The fixture indicates churn risk is concentrated among month-to-month customers with short tenure and elevated support calls. Review drift before promoting retrained models.",
    warnings: [],
    generated_at: now(),
  };
}

function agentRunSummary() {
  return {
    workflow_id: DEMO_WORKFLOW_ID,
    dataset_id: DEMO_DATASET_ID,
    status: "success",
    user_goal: "Run the complete portfolio demo workflow.",
    step_count: 6,
    error_count: 0,
    started_at: now(),
    finished_at: now(),
    final_summary: "Completed the guided demo analysis workflow.",
  };
}

function agentJobSummary(status) {
  return {
    job_id: DEMO_AGENT_JOB_ID,
    dataset_id: DEMO_DATASET_ID,
    workflow_id: status === "success" ? DEMO_WORKFLOW_ID : null,
    status,
    user_goal: "Run the complete portfolio demo workflow.",
    created_at: now(),
    started_at: now(),
    finished_at: status === "success" ? now() : null,
    event_count: agentEvents().length,
    error_message: null,
  };
}

function agentEvents() {
  return [
    { event_id: "agent-event-1", job_id: DEMO_AGENT_JOB_ID, workflow_id: DEMO_WORKFLOW_ID, event_type: "running", status: "running", message: "Agent workflow started.", payload: {}, created_at: now() },
    { event_id: "agent-event-2", job_id: DEMO_AGENT_JOB_ID, workflow_id: DEMO_WORKFLOW_ID, event_type: "completed", status: "success", message: "Agent workflow completed.", payload: {}, created_at: now() },
  ];
}

function backtestJobSummary(status) {
  return {
    job_id: DEMO_JOB_ID,
    suite_type: "ui",
    status,
    created_at: now(),
    started_at: now(),
    finished_at: status === "success" ? now() : null,
    event_count: backtestEvents().length,
    run_ids: status === "success" ? [DEMO_RUN_ID] : [],
    suite_ids: status === "success" ? ["suite-demo-ui"] : [],
    error_message: null,
  };
}

function backtestEvents() {
  return [
    { event_id: "backtest-event-1", job_id: DEMO_JOB_ID, event_type: "running", status: "running", message: "Launching static demo quality gate.", payload: {}, created_at: now() },
    { event_id: "backtest-event-2", job_id: DEMO_JOB_ID, event_type: "log", status: "running", message: "Guide tour and route screenshots verified.", payload: {}, created_at: now() },
    { event_id: "backtest-event-3", job_id: DEMO_JOB_ID, event_type: "completed", status: "success", message: "Backtest completed.", payload: { run_ids: [DEMO_RUN_ID] }, created_at: now(), run_ids: [DEMO_RUN_ID] },
  ];
}

function normalizeApiPath(url) {
  const parsed = new URL(url, window.location.href);
  const apiIndex = parsed.pathname.indexOf("/api/");

  if (apiIndex === -1) {
    return null;
  }

  return `${parsed.pathname.slice(apiIndex)}${parsed.search ? "" : ""}`;
}

function pngResponse() {
  const bytes = Uint8Array.from(atob("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFeAJ5rgXJYQAAAABJRU5ErkJggg=="), (char) => char.charCodeAt(0));
  return new Response(bytes, { status: 200, headers: { "Content-Type": "image/png" } });
}

function jsonResponse(payload) {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "X-DataAgent-Demo": "true",
    },
  });
}

function installMockFetch() {
  const realFetch = window.fetch.bind(window);

  window.fetch = async (resource, options = {}) => {
    const url = typeof resource === "string" ? resource : resource.url;
    const path = normalizeApiPath(url);

    if (!path) {
      return realFetch(resource, options);
    }

    if (path.includes("/screenshots/")) {
      return pngResponse();
    }

    const method = (options.method || "GET").toUpperCase();
    const pathWithoutQuery = path.split("?")[0];
    const payload = apiPayload(method, pathWithoutQuery);

    if (payload) {
      return jsonResponse(payload);
    }

    return jsonResponse({
      status: "demo_unimplemented",
      path: pathWithoutQuery,
      method,
      message: "The static demo returned a generic fixture response.",
    });
  };
}

class DemoEventSource {
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.onopen = null;
    this.onmessage = null;
    this.onerror = null;
    window.setTimeout(() => this.emit(), 250);
  }

  emit() {
    this.readyState = 1;
    this.onopen?.({ type: "open" });
    const events = this.url.includes("/backtests/jobs/") ? backtestEvents() : agentEvents();
    events.forEach((event, index) => {
      window.setTimeout(() => {
        this.onmessage?.({ data: JSON.stringify(event) });
        if (index === events.length - 1) {
          this.onmessage?.({
            data: JSON.stringify({
              ...event,
              event_type: "stream_closed",
              status: "success",
              run_ids: [DEMO_RUN_ID],
              workflow_id: DEMO_WORKFLOW_ID,
            }),
          });
        }
      }, 200 + index * 220);
    });
  }

  close() {
    this.readyState = 2;
  }
}

function addDemoBadge() {
  const badge = document.createElement("div");
  badge.className = "demo-mode-badge";
  badge.textContent = "Static GitHub Pages demo";
  document.body.appendChild(badge);
}

function installDemoStyles() {
  const style = document.createElement("style");
  style.textContent = `
    .demo-mode-badge {
      position: fixed;
      left: 1rem;
      bottom: 1rem;
      z-index: 80;
      border: 1px solid #bfdbfe;
      border-radius: 999px;
      background: rgba(239, 246, 255, 0.94);
      color: #1d4ed8;
      padding: 0.45rem 0.75rem;
      font-size: 0.75rem;
      font-weight: 700;
      box-shadow: 0 10px 24px rgba(37, 99, 235, 0.14);
      pointer-events: none;
    }
  `;
  document.head.appendChild(style);
}

window.DATAAGENT_CONFIG = {
  ...(window.DATAAGENT_CONFIG || {}),
  apiBaseUrl: ".",
  demoMode: true,
};
window.EventSource = DemoEventSource;
installMockFetch();
installDemoStyles();
window.addEventListener("DOMContentLoaded", addDemoBadge);
