import { escapeHtml, formatCellValue, formatPercent } from "../utils/format.js";


export function renderEdaSummary(elements, edaSummary) {
  renderQualitySummary(elements.qualitySummary, edaSummary);
  renderMissingTable(elements.missingTable, edaSummary.missing.columns);
  renderNumericStatisticsTable(elements.statisticsTable, edaSummary.numeric_statistics.columns);
  renderOutlierTable(elements.outlierTable, edaSummary.outliers.columns);
  renderCorrelationTable(elements.correlationTable, edaSummary.correlation.strongest_pairs);
  renderRecommendations(elements.recommendations, edaSummary.recommendations);
}


function renderQualitySummary(container, edaSummary) {
  container.innerHTML = `
    <div class="summary-grid quality-grid">
      <div class="summary-card quality-score-card">
        <span>Data Quality Score</span>
        <strong>${edaSummary.data_quality_score}</strong>
        <p>${escapeHtml(edaSummary.data_quality_grade)}</p>
      </div>
      <div class="summary-card">
        <span>Rows</span>
        <strong>${edaSummary.row_count}</strong>
      </div>
      <div class="summary-card">
        <span>Columns</span>
        <strong>${edaSummary.column_count}</strong>
      </div>
      <div class="summary-card">
        <span>Generated At</span>
        <strong class="timestamp-text">${escapeHtml(edaSummary.generated_at)}</strong>
      </div>
    </div>
  `;
}


function renderMissingTable(container, columns) {
  const rows = columns
    .filter((column) => column.missing_count > 0)
    .map((column) => [
      column.name,
      column.inferred_type,
      column.missing_count,
      formatPercent(column.missing_ratio),
    ]);

  renderSimpleTable(container, ["Column", "Type", "Missing Count", "Missing Ratio"], rows);
}


function renderNumericStatisticsTable(container, columns) {
  const rows = columns.map((column) => [
    column.name,
    column.count,
    column.mean,
    column.std,
    column.min,
    column.median,
    column.max,
  ]);

  renderSimpleTable(
    container,
    ["Column", "Count", "Mean", "Std", "Min", "Median", "Max"],
    rows,
  );
}


function renderOutlierTable(container, columns) {
  const rows = columns.map((column) => [
    column.name,
    column.method,
    column.outlier_count,
    formatPercent(column.outlier_ratio),
    column.lower_bound,
    column.upper_bound,
  ]);

  renderSimpleTable(
    container,
    ["Column", "Method", "Outliers", "Ratio", "Lower Bound", "Upper Bound"],
    rows,
  );
}


function renderCorrelationTable(container, pairs) {
  const rows = pairs.map((pair) => [
    pair.column_x,
    pair.column_y,
    pair.correlation,
  ]);

  renderSimpleTable(container, ["Column X", "Column Y", "Correlation"], rows);
}


function renderRecommendations(container, recommendations) {
  if (!recommendations.length) {
    container.innerHTML = '<p class="empty-state">No recommendations available.</p>';
    return;
  }

  container.innerHTML = recommendations
    .map((recommendation) => `
      <div class="recommendation-card">
        ${escapeHtml(recommendation)}
      </div>
    `)
    .join("");
}


function renderSimpleTable(container, headers, rows) {
  if (!rows.length) {
    container.innerHTML = '<p class="empty-state">No data available.</p>';
    return;
  }

  const headerHtml = headers
    .map((header) => `<th>${escapeHtml(header)}</th>`)
    .join("");

  const bodyHtml = rows
    .map((row) => `
      <tr>
        ${row.map((cell) => `<td>${escapeHtml(formatCellValue(cell))}</td>`).join("")}
      </tr>
    `)
    .join("");

  container.innerHTML = `
    <table class="preview-table">
      <thead>
        <tr>${headerHtml}</tr>
      </thead>
      <tbody>${bodyHtml}</tbody>
    </table>
  `;
}
