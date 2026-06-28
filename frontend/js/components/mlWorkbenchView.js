import {
  escapeHtml,
  formatCellValue,
} from "../utils/format.js";


export function renderMlWorkbenchStatus(
  container,
  payload,
) {
  container.innerHTML = `
    <div class="training-status-card ${
      payload.status === "success"
        ? "success"
        : ""
    }">
      <span>ML Workbench</span>
      <strong>${escapeHtml(payload.title)}</strong>
      <p>${escapeHtml(payload.message)}</p>
    </div>
  `;
}


export function populateMlWorkbenchFeatures(
  selectElement,
  featureGroups,
) {
  const groupEntries = [
    [
      "Numeric",
      featureGroups.numeric,
    ],
    [
      "Categorical",
      featureGroups.categorical,
    ],
    [
      "Datetime",
      featureGroups.datetime,
    ],
    [
      "Text",
      featureGroups.text,
    ],
  ];

  selectElement.innerHTML = groupEntries
    .filter(([, columns]) => columns.length)
    .map(([label, columns]) => {
      const options = columns
        .map((column) => {
          return `
            <option value="${escapeHtml(column)}">
              ${escapeHtml(column)}
            </option>
          `;
        })
        .join("");

      return `
        <optgroup label="${escapeHtml(label)}">
          ${options}
        </optgroup>
      `;
    })
    .join("");
}


export function renderMlPlanSummary(
  container,
  plan,
) {
  container.innerHTML = `
    <div class="summary-grid">
      <article class="summary-card">
        <span>Detected Task</span>
        <strong>
          ${escapeHtml(plan.detected_task_type)}
        </strong>
      </article>

      <article class="summary-card">
        <span>Target</span>
        <strong>
          ${escapeHtml(plan.target_column || "None")}
        </strong>
      </article>

      <article class="summary-card">
        <span>Estimated Features</span>
        <strong>
          ${plan.estimated_feature_count}
        </strong>
      </article>

      <article class="summary-card">
        <span>Primary Metric</span>
        <strong>
          ${escapeHtml(plan.primary_metric)}
        </strong>
      </article>
    </div>
  `;
}


export function renderMlPipelineSteps(
  container,
  steps,
) {
  if (!steps.length) {
    container.innerHTML = `
      <p class="empty-state">
        No preprocessing steps available.
      </p>
    `;
    return;
  }

  container.innerHTML = steps
    .map((step, index) => {
      return `
        <article class="ml-pipeline-step-card">
          <span>${index + 1}</span>

          <div>
            <h4>${escapeHtml(step.label)}</h4>
            <p>
              Columns:
              ${escapeHtml(
                step.columns.join(", "),
              )}
            </p>
            <p>
              Operations:
              ${escapeHtml(
                step.operations.join(" → "),
              )}
            </p>
          </div>
        </article>
      `;
    })
    .join("");
}


export function renderMlWarnings(
  container,
  warnings,
) {
  if (!warnings.length) {
    container.innerHTML = "";
    return;
  }

  container.innerHTML = warnings
    .map((warning) => {
      return `
        <div class="ml-warning-card">
          ${escapeHtml(warning)}
        </div>
      `;
    })
    .join("");
}


export function renderMlModelOptions(
  container,
  models,
) {
  if (!models.length) {
    container.innerHTML = `
      <p class="empty-state">
        No models are available.
      </p>
    `;
    return;
  }

  container.innerHTML = models
    .map((model) => {
      return `
        <label class="ml-model-option-card">
          <input
            type="checkbox"
            value="${escapeHtml(model.id)}"
            ${model.recommended ? "checked" : ""}
          />

          <span>
            <strong>${escapeHtml(model.label)}</strong>
            <small>
              ${escapeHtml(model.description)}
            </small>
          </span>
        </label>
      `;
    })
    .join("");
}


export function renderMlExperimentLeaderboard(
  container,
  experiment,
) {
  if (!experiment.model_results.length) {
    container.innerHTML = `
      <p class="empty-state">
        No model result is available.
      </p>
    `;
    return;
  }

  const rows = experiment.model_results
    .map((result) => {
      return `
        <tr>
          <td>${escapeHtml(result.model_label)}</td>
          <td>${escapeHtml(result.status)}</td>
          <td>
            ${renderMetricValue(
              result,
              experiment.primary_metric,
            )}
          </td>
          <td>${result.feature_count}</td>
          <td>${result.training_seconds}</td>
          <td>
            ${escapeHtml(
              result.error_message || "",
            )}
          </td>
        </tr>
      `;
    })
    .join("");

  container.innerHTML = `
    <table class="preview-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Status</th>
          <th>
            ${escapeHtml(experiment.primary_metric)}
          </th>
          <th>Features</th>
          <th>Seconds</th>
          <th>Error</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}


export function renderMlExperimentHistory(
  container,
  experiments,
  onSelectExperiment,
) {
  if (!experiments.length) {
    container.innerHTML = `
      <p class="empty-state">
        No Workbench experiments yet.
      </p>
    `;
    return;
  }

  container.innerHTML = "";

  experiments.forEach((experiment) => {
    const button = document.createElement(
      "button",
    );

    button.type = "button";
    button.className = "dataset-card";

    button.innerHTML = `
      <h4>${escapeHtml(experiment.task_type)}</h4>
      <p>
        Target:
        ${escapeHtml(
          experiment.target_column || "None",
        )}
      </p>
      <p>
        ${experiment.model_count} model result(s)
      </p>
      <p>
        Best:
        ${escapeHtml(
          experiment.best_model_name || "N/A",
        )}
        ·
        ${formatCellValue(
          experiment.best_metric_value,
        )}
      </p>
      <p>${escapeHtml(experiment.created_at)}</p>
    `;

    button.addEventListener(
      "click",
      () => {
        onSelectExperiment(
          experiment.experiment_id,
        );
      },
    );

    container.appendChild(button);
  });
}


function renderMetricValue(
  result,
  metricName,
) {
  const cvMetric = result.cv_metrics.find(
    (metric) => metric.name === metricName,
  );

  if (cvMetric) {
    return [
      `CV: ${formatCellValue(cvMetric.mean)}`,
      `± ${formatCellValue(cvMetric.std)}`,
    ].join(" ");
  }

  return formatCellValue(
    result.holdout_metrics[metricName],
  );
}
