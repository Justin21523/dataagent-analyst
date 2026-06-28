import { escapeHtml } from "../utils/format.js";


export function renderReportStatus(container, payload) {
  container.innerHTML = `
    <div class="training-status-card ${payload.status === "success" ? "success" : ""}">
      <span>Report Status</span>
      <strong>${escapeHtml(payload.title)}</strong>
      <p>${escapeHtml(payload.message)}</p>
    </div>
  `;
}


export function renderReportList(container, reports, onSelectReport) {
  if (!reports.length) {
    container.innerHTML = '<p class="empty-state">No reports generated yet.</p>';
    return;
  }

  container.innerHTML = "";

  reports.forEach((report) => {
    const button = document.createElement("button");
    button.className = "dataset-card";
    button.type = "button";

    button.innerHTML = `
      <h4>${escapeHtml(report.title)}</h4>
      <p>${escapeHtml(report.created_at)}</p>
      <p>Status: ${escapeHtml(report.status)}</p>
    `;

    button.addEventListener("click", () => {
      onSelectReport(report.id);
    });

    container.appendChild(button);
  });
}


export function renderReportViewer(container, markdownContent) {
  if (!markdownContent) {
    container.innerHTML = '<p class="empty-state">No report content available.</p>';
    return;
  }

  // Phase 8 先用 pre 顯示 Markdown，避免自己寫不完整的 Markdown parser。
  container.innerHTML = `
    <pre class="markdown-preview">${escapeHtml(markdownContent)}</pre>
  `;
}
