import { escapeHtml, formatCellValue } from "../utils/format.js";


export function renderTargetOptions(selectElement, columns) {
  // target 候選欄位排前面，降低使用者選錯 target 的機率。
  const sortedColumns = [...columns].sort((a, b) => {
    if (a.semantic_role === "target_candidate") return -1;
    if (b.semantic_role === "target_candidate") return 1;
    return a.name.localeCompare(b.name);
  });

  const options = sortedColumns
    .filter((column) => column.semantic_role !== "identifier")
    .map((column) => `
      <option value="${escapeHtml(column.name)}">
        ${escapeHtml(column.name)} · ${escapeHtml(column.inferred_type)} · ${escapeHtml(column.semantic_role)}
      </option>
    `)
    .join("");

  selectElement.innerHTML = options;
}


export function renderTrainingStatus(container, payload) {
  container.innerHTML = `
    <div class="training-status-card">
      <span>Training Status</span>
      <strong>${escapeHtml(payload.status)}</strong>
      <p>${escapeHtml(payload.message)}</p>
    </div>
  `;
}


export function renderTrainingResult(container, result) {
  container.innerHTML = `
    <div class="training-status-card success">
      <span>Training Complete</span>
      <strong>${result.model_count} models trained</strong>
      <p>
        Task: ${escapeHtml(result.task_type)} ·
        Target: ${escapeHtml(result.target_column)} ·
        Best metric: ${escapeHtml(result.best_metric_name)} =
        ${formatCellValue(result.best_metric_value)}
      </p>
    </div>
  `;
}


export function renderModelLeaderboard(container, models) {
  if (!models.length) {
    container.innerHTML = '<p class="empty-state">No trained models yet.</p>';
    return;
  }

  const rows = models
    .map((model) => `
      <tr>
        <td>${escapeHtml(model.model_name)}</td>
        <td><span class="badge">${escapeHtml(model.lifecycle_status || "candidate")}</span></td>
        <td>${escapeHtml(model.task_type)}</td>
        <td>${escapeHtml(model.target_column)}</td>
        <td>${escapeHtml(model.dataset_version_id || "latest")}</td>
        <td>${renderMetricSummary(model.metrics)}</td>
        <td>${escapeHtml(model.created_at)}</td>
      </tr>
    `)
    .join("");

  container.innerHTML = `
    <table class="preview-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Status</th>
          <th>Task</th>
          <th>Target</th>
          <th>Version</th>
          <th>Metrics</th>
          <th>Created At</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}


function renderMetricSummary(metrics) {
  return Object.entries(metrics)
    .map(([name, value]) => `${escapeHtml(name)}: ${formatCellValue(value)}`)
    .join("<br />");
}


export function renderFeatureImportanceTable(container, featureImportance) {
  if (!featureImportance.length) {
    container.innerHTML = '<p class="empty-state">No feature importance available.</p>';
    return;
  }

  const rows = featureImportance
    .slice(0, 12)
    .map((item) => `
      <tr>
        <td>${escapeHtml(item.feature)}</td>
        <td>
          <div class="importance-bar">
            <span style="width: ${Math.min(Number(item.importance) * 100, 100)}%"></span>
          </div>
        </td>
        <td>${formatCellValue(item.importance)}</td>
      </tr>
    `)
    .join("");

  container.innerHTML = `
    <table class="preview-table">
      <thead>
        <tr>
          <th>Feature</th>
          <th>Importance</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}


export function renderModelDiagnostics(container, evaluation) {
  if (evaluation.confusion_matrix) {
    renderConfusionMatrix(container, evaluation.confusion_matrix);
    return;
  }

  if (evaluation.regression_residuals.length) {
    renderRegressionResiduals(container, evaluation.regression_residuals);
    return;
  }

  container.innerHTML = '<p class="empty-state">No diagnostics available.</p>';
}


function renderConfusionMatrix(container, confusionMatrix) {
  const labels = confusionMatrix.labels;

  const headerHtml = labels
    .map((label) => `<th>Pred ${escapeHtml(label)}</th>`)
    .join("");

  const bodyHtml = confusionMatrix.matrix
    .map((row, rowIndex) => `
      <tr>
        <th>Actual ${escapeHtml(labels[rowIndex])}</th>
        ${row.map((value) => `<td>${value}</td>`).join("")}
      </tr>
    `)
    .join("");

  container.innerHTML = `
    <table class="preview-table confusion-table">
      <thead>
        <tr>
          <th></th>
          ${headerHtml}
        </tr>
      </thead>
      <tbody>${bodyHtml}</tbody>
    </table>
  `;
}


function renderRegressionResiduals(container, residuals) {
  const rows = residuals
    .slice(0, 20)
    .map((item) => `
      <tr>
        <td>${formatCellValue(item.actual)}</td>
        <td>${formatCellValue(item.predicted)}</td>
        <td>${formatCellValue(item.residual)}</td>
      </tr>
    `)
    .join("");

  container.innerHTML = `
    <table class="preview-table">
      <thead>
        <tr>
          <th>Actual</th>
          <th>Predicted</th>
          <th>Residual</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}


export function renderModelOptions(selectElement, models) {
  if (!models.length) {
    selectElement.innerHTML = '<option value="">No trained models available</option>';
    return;
  }

  selectElement.innerHTML = models
    .map((model) => `
      <option value="${escapeHtml(model.id)}">
        ${escapeHtml(model.model_name)} · ${escapeHtml(model.task_type)} · ${escapeHtml(model.target_column)}
      </option>
    `)
    .join("");
}


export function renderPredictionTemplate(textareaElement, model) {
  if (!model) {
    textareaElement.value = "";
    textareaElement.placeholder = "Train a model first.";
    return;
  }

  const template = {};

  model.feature_columns.forEach((column) => {
    template[column] = null;
  });

  textareaElement.value = JSON.stringify(template, null, 2);
}


export function renderPredictionResults(container, response) {
  if (!response.predictions.length) {
    container.innerHTML = '<p class="empty-state">No prediction results available.</p>';
    return;
  }

  const probabilityHeader = response.task_type === "classification"
    ? "<th>Probabilities</th>"
    : "";

  const rows = response.predictions
    .map((item) => `
      <tr>
        <td>${item.row_index}</td>
        <td>${escapeHtml(formatCellValue(item.prediction))}</td>
        ${
          response.task_type === "classification"
            ? `<td>${formatProbabilities(item.probabilities)}</td>`
            : ""
        }
      </tr>
    `)
    .join("");

  container.innerHTML = `
    <table class="preview-table">
      <thead>
        <tr>
          <th>Row</th>
          <th>Prediction</th>
          ${probabilityHeader}
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}


function formatProbabilities(probabilities) {
  if (!probabilities) {
    return "";
  }

  return Object.entries(probabilities)
    .map(([label, value]) => `${escapeHtml(label)}: ${formatCellValue(value)}`)
    .join("<br />");
}
