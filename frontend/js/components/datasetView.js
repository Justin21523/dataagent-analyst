import { escapeHtml, formatCellValue } from "../utils/format.js";


export function renderDatasetList(container, datasets, onSelectDataset) {
  if (!datasets.length) {
    container.innerHTML = '<p class="empty-state">No datasets uploaded yet.</p>';
    return;
  }

  container.innerHTML = "";

  datasets.forEach((dataset) => {
    const button = document.createElement("button");
    button.className = "dataset-card";
    button.type = "button";

    button.innerHTML = `
      <h4>${escapeHtml(dataset.name)}</h4>
      <p>${escapeHtml(dataset.original_filename)}</p>
      <p>${dataset.row_count} rows · ${dataset.column_count} columns</p>
      <p>Encoding: ${escapeHtml(dataset.encoding)}</p>
    `;

    button.addEventListener("click", () => {
      onSelectDataset(dataset.id);
    });

    container.appendChild(button);
  });
}


export function renderPreviewTable(container, columns, rows) {
  if (!rows.length) {
    container.innerHTML = '<p class="empty-state">No rows available.</p>';
    return;
  }

  const table = document.createElement("table");
  table.className = "preview-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");

  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column;
    headerRow.appendChild(th);
  });

  thead.appendChild(headerRow);

  const tbody = document.createElement("tbody");

  rows.forEach((row) => {
    const tr = document.createElement("tr");

    columns.forEach((column) => {
      const td = document.createElement("td");
      td.textContent = formatCellValue(row[column]);
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);

  container.innerHTML = "";
  container.appendChild(table);
}
