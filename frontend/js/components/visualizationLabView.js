import { escapeHtml } from "../utils/format.js";


export function renderVisualizationLabSummary(
  container,
  summary,
) {
  container.innerHTML = `
    <article class="visualization-summary-card">
      <span>Charts</span>
      <strong>${summary.chart_count}</strong>
    </article>

    <article class="visualization-summary-card">
      <span>Sampled Rows</span>
      <strong>${summary.sampled_rows}</strong>
    </article>

    <article class="visualization-summary-card">
      <span>Numeric Features</span>
      <strong>${summary.numeric_columns.length}</strong>
    </article>

    <article class="visualization-summary-card">
      <span>Categorical Features</span>
      <strong>${summary.categorical_columns.length}</strong>
    </article>

    <article class="visualization-summary-card">
      <span>Datetime Features</span>
      <strong>${summary.datetime_columns.length}</strong>
    </article>

    <article class="visualization-summary-card">
      <span>Target</span>
      <strong>
        ${escapeHtml(summary.target_column || "Not Selected")}
      </strong>
    </article>
  `;
}


export function renderVisualizationWarnings(
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
        <div class="visualization-warning-card">
          ${escapeHtml(warning)}
        </div>
      `;
    })
    .join("");
}


export function populateVisualizationControls(
  elements,
  summary,
) {
  const options = summary.column_options
    .map((column) => {
      return `
        <option value="${escapeHtml(column.name)}">
          ${escapeHtml(column.name)}
          · ${escapeHtml(column.inferred_type)}
          · ${escapeHtml(column.semantic_role)}
        </option>
      `;
    })
    .join("");

  elements.xColumn.innerHTML = `
    <option value="">Select Column</option>
    ${options}
  `;

  elements.yColumn.innerHTML = `
    <option value="">Optional</option>
    ${options}
  `;

  elements.groupColumn.innerHTML = `
    <option value="">Optional</option>
    ${options}
  `;

  const targetOptions = summary.column_options
    .filter((column) => {
      return column.semantic_role !== "identifier";
    })
    .map((column) => {
      return `
        <option value="${escapeHtml(column.name)}">
          ${escapeHtml(column.name)}
          · ${escapeHtml(column.inferred_type)}
        </option>
      `;
    })
    .join("");

  elements.target.innerHTML = `
    <option value="">Auto Detect</option>
    ${targetOptions}
  `;

  if (summary.target_column) {
    elements.target.value = summary.target_column;
  }

  const firstNumeric = summary.numeric_columns[0] || "";
  const secondNumeric = summary.numeric_columns[1] || "";

  elements.xColumn.value = firstNumeric;
  elements.yColumn.value = secondNumeric;
}


export function renderCustomVisualizationMeta(
  container,
  specification,
) {
  container.innerHTML = `
    <article class="custom-visualization-info">
      <div>
        <span class="chart-family">
          ${escapeHtml(specification.chart_family)}
        </span>
        <h4>${escapeHtml(specification.title)}</h4>
      </div>

      <p>${escapeHtml(specification.description)}</p>

      <div class="column-meta">
        ${specification.columns
          .map((column) => {
            return `
              <span class="badge">
                ${escapeHtml(column)}
              </span>
            `;
          })
          .join("")}
      </div>

      <div class="echart-insight">
        <strong>Insight</strong>
        <p>${escapeHtml(specification.insight)}</p>
      </div>
    </article>
  `;
}


export function renderVisualizationLoading(
  container,
  message,
) {
  container.innerHTML = `
    <div class="visualization-skeleton">
      <span></span>
      <span></span>
      <span></span>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}
