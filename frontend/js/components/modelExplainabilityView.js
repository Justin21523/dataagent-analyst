import {
  escapeHtml,
  formatCellValue,
} from "../utils/format.js";


export function populateExplainabilityModels(
  selectElement,
  models,
) {
  if (!models.length) {
    selectElement.innerHTML = `
      <option value="">
        No saved supervised models available
      </option>
    `;
    return;
  }

  selectElement.innerHTML = models
    .map((model) => {
      return `
        <option value="${escapeHtml(model.id)}">
          ${escapeHtml(model.model_name)}
          · ${escapeHtml(model.task_type)}
          · ${escapeHtml(model.target_column)}
        </option>
      `;
    })
    .join("");
}


export function renderExplainabilityStatus(
  container,
  payload,
) {
  container.innerHTML = `
    <div class="training-status-card ${
      payload.status === "success"
        ? "success"
        : ""
    }">
      <span>Model Explainability</span>
      <strong>${escapeHtml(payload.title)}</strong>
      <p>${escapeHtml(payload.message)}</p>
    </div>
  `;
}


export function renderExplainabilityOverview(
  container,
  result,
) {
  container.innerHTML = `
    <div class="summary-grid">
      <article class="summary-card">
        <span>Model</span>
        <strong>${escapeHtml(result.model_name)}</strong>
      </article>

      <article class="summary-card">
        <span>Task</span>
        <strong>${escapeHtml(result.task_type)}</strong>
      </article>

      <article class="summary-card">
        <span>Holdout Rows</span>
        <strong>${result.holdout.row_count}</strong>
      </article>

      <article class="summary-card">
        <span>Holdout Source</span>
        <strong>
          ${escapeHtml(result.holdout.source)}
        </strong>
      </article>

      <article class="summary-card">
        <span>SHAP</span>
        <strong>
          ${
            result.shap.available
              ? escapeHtml(
                  result.shap.explainer_type,
                )
              : "Unavailable"
          }
        </strong>
      </article>

      <article class="summary-card">
        <span>Cache</span>
        <strong>
          ${result.cache_hit ? "Hit" : "Generated"}
        </strong>
      </article>
    </div>
  `;
}


export function renderExplainabilityWarnings(
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


export function renderExplainabilityErrorSamples(
  container,
  result,
) {
  if (!result.error_samples.length) {
    container.innerHTML = `
      <p class="empty-state">
        No model errors were found in the current holdout.
      </p>
    `;
    return;
  }

  const isClassification = (
    result.task_type === "classification"
  );

  const rows = result.error_samples
    .map((sample) => {
      return `
        <tr>
          <td>
            ${escapeHtml(sample.dataset_index)}
          </td>
          <td>
            ${escapeHtml(
              formatCellValue(sample.actual),
            )}
          </td>
          <td>
            ${escapeHtml(
              formatCellValue(sample.predicted),
            )}
          </td>
          <td>
            ${
              isClassification
                ? escapeHtml(
                    formatCellValue(
                      sample.confidence,
                    ),
                  )
                : escapeHtml(
                    formatCellValue(
                      sample.absolute_error,
                    ),
                  )
            }
          </td>
          <td>
            <code>
              ${escapeHtml(
                JSON.stringify(sample.record),
              )}
            </code>
          </td>
        </tr>
      `;
    })
    .join("");

  container.innerHTML = `
    <table class="preview-table">
      <thead>
        <tr>
          <th>Dataset Index</th>
          <th>Actual</th>
          <th>Predicted</th>
          <th>
            ${
              isClassification
                ? "Confidence"
                : "Absolute Error"
            }
          </th>
          <th>Input Record</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}


export function renderExplainabilityInsight(
  container,
  insight,
) {
  container.innerHTML = `
    <article class="ai-insight-card">
      <div class="ai-insight-header">
        <div>
          <span class="chart-family">
            ${escapeHtml(insight.source)}
          </span>
          <h4>${escapeHtml(insight.title)}</h4>
        </div>

        <span class="badge">
          ${escapeHtml(insight.model)}
        </span>
      </div>

      <pre class="ai-insight-content">${
        escapeHtml(insight.content)
      }</pre>
    </article>
  `;
}
