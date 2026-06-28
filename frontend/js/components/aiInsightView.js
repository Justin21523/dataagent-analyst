import { escapeHtml } from "../utils/format.js";


export function renderLLMStatus(container, status) {
  const statusLabel = status.enabled && status.online
    ? "Online"
    : status.enabled
      ? "Enabled but Offline"
      : "Disabled";

  container.innerHTML = `
    <div class="training-status-card ${status.online ? "success" : ""}">
      <span>Local LLM Status</span>
      <strong>${escapeHtml(statusLabel)}</strong>
      <p>
        Model: ${escapeHtml(status.model)}<br />
        Base URL: ${escapeHtml(status.base_url)}<br />
        ${escapeHtml(status.message)}
      </p>
    </div>
  `;
}


export function renderAIInsight(container, insight) {
  container.innerHTML = `
    <article class="ai-insight-card">
      <div class="ai-insight-header">
        <div>
          <span class="chart-family">${escapeHtml(insight.source)}</span>
          <h4>${escapeHtml(insight.title)}</h4>
        </div>
        <span class="badge">${escapeHtml(insight.model)}</span>
      </div>

      <pre class="ai-insight-content">${escapeHtml(insight.content)}</pre>
    </article>
  `;
}


export function renderAIInsightLoading(container, message) {
  container.innerHTML = `
    <p class="empty-state">${escapeHtml(message)}</p>
  `;
}
