import { escapeHtml, formatCellValue, formatPercent } from "../utils/format.js";


export function renderSchemaSummary(container, summary) {
  const typePills = Object.entries(summary.type_counts)
    .map(([type, count]) => `<span class="type-pill">${escapeHtml(type)} · ${count}</span>`)
    .join("");

  const targets = summary.target_candidates.length
    ? summary.target_candidates
      .map((target) => `<span class="type-pill">${escapeHtml(target)}</span>`)
      .join("")
    : '<span class="type-pill">No target candidate detected</span>';

  container.innerHTML = `
    <div class="summary-grid">
      <div class="summary-card">
        <span>Rows</span>
        <strong>${summary.row_count}</strong>
      </div>
      <div class="summary-card">
        <span>Columns</span>
        <strong>${summary.column_count}</strong>
      </div>
      <div class="summary-card">
        <span>Missing Cells</span>
        <strong>${summary.missing_cell_count}</strong>
      </div>
      <div class="summary-card">
        <span>Duplicate Rows</span>
        <strong>${summary.duplicate_row_count}</strong>
      </div>
    </div>

    <div>
      <p class="helper-text"><strong>Column Type Distribution</strong></p>
      <div class="type-counts">${typePills}</div>
    </div>

    <div>
      <p class="helper-text"><strong>Target Candidates</strong></p>
      <div class="target-list">${targets}</div>
    </div>

    <p class="helper-text">
      Missing cell ratio: ${formatPercent(summary.missing_cell_ratio)} ·
      Duplicate row ratio: ${formatPercent(summary.duplicate_row_ratio)}
    </p>
  `;
}


export function renderColumnProfiles(container, columns) {
  if (!columns.length) {
    container.innerHTML = '<p class="empty-state">No columns available.</p>';
    return;
  }

  container.innerHTML = "";

  columns.forEach((column) => {
    const card = document.createElement("article");
    card.className = "column-card";

    card.innerHTML = `
      <div class="column-card-header">
        <div>
          <h4>${escapeHtml(column.name)}</h4>
          <div class="column-meta">
            <span class="badge ${escapeHtml(column.inferred_type)}">
              ${escapeHtml(column.inferred_type)}
            </span>
            <span class="badge">${escapeHtml(column.semantic_role)}</span>
            <span class="badge">${column.unique_count} unique</span>
            <span class="badge">${formatPercent(column.missing_ratio)} missing</span>
          </div>
        </div>
      </div>

      <div class="column-section">
        <strong>Samples:</strong>
        ${formatValueList(column.sample_values)}
      </div>

      <div class="column-section">
        <strong>Top Values:</strong>
        ${formatTopValues(column.top_values)}
      </div>

      ${formatNumericStats(column.numeric_stats)}
      ${formatDatetimeStats(column.datetime_stats)}
    `;

    container.appendChild(card);
  });
}


function formatNumericStats(stats) {
  if (!stats) {
    return "";
  }

  return `
    <div class="column-section">
      <strong>Numeric Stats:</strong>
      mean ${formatCellValue(stats.mean)}, std ${formatCellValue(stats.std)},
      min ${formatCellValue(stats.min)}, median ${formatCellValue(stats.median)},
      max ${formatCellValue(stats.max)}
    </div>
  `;
}


function formatDatetimeStats(stats) {
  if (!stats) {
    return "";
  }

  return `
    <div class="column-section">
      <strong>Date Range:</strong>
      ${escapeHtml(stats.min)} → ${escapeHtml(stats.max)}
    </div>
  `;
}


function formatTopValues(topValues) {
  if (!topValues.length) {
    return "No values";
  }

  return topValues
    .map((item) => `${escapeHtml(formatCellValue(item.value))} (${item.count})`)
    .join(", ");
}


function formatValueList(values) {
  if (!values.length) {
    return "No samples";
  }

  return values.map((value) => escapeHtml(formatCellValue(value))).join(", ");
}
