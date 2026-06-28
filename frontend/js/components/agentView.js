import { escapeHtml, formatCellValue } from "../utils/format.js";


export function renderAgentStatus(container, payload) {
  container.innerHTML = `
    <div class="training-status-card ${payload.status === "success" ? "success" : ""}">
      <span>Agent Workflow</span>
      <strong>${escapeHtml(payload.title)}</strong>
      <p>${escapeHtml(payload.message)}</p>
    </div>
  `;
}


export function renderAgentResult(container, result) {
  const stepItems = result.steps
    .map((step) => `
      <article class="agent-step-card ${escapeHtml(step.status)}">
        <div class="agent-step-header">
          <div>
            <span>${escapeHtml(step.status)}</span>
            <h4>${escapeHtml(step.name)}</h4>
          </div>
          <small>${escapeHtml(step.created_at)}</small>
        </div>
        <p>${escapeHtml(step.message)}</p>
        ${renderStepPayload(step.payload)}
      </article>
    `)
    .join("");

  container.innerHTML = `
    <div class="agent-result-summary">
      <span>Status</span>
      <strong>${escapeHtml(result.status)}</strong>
      <p>${escapeHtml(result.final_summary)}</p>
    </div>

    <div class="agent-step-list">
      ${stepItems}
    </div>
  `;
}


export function renderAgentOutputs(container, result) {
  container.innerHTML = `
    <pre class="code-output agent-output-json">${escapeHtml(
      JSON.stringify(result.outputs, null, 2),
    )}</pre>
  `;
}


function renderStepPayload(payload) {
  if (!payload || !Object.keys(payload).length) {
    return "";
  }

  const rows = Object.entries(payload)
    .map(([key, value]) => `
      <tr>
        <td>${escapeHtml(key)}</td>
        <td>${escapeHtml(formatCellValue(formatPayloadValue(value)))}</td>
      </tr>
    `)
    .join("");

  return `
    <table class="agent-payload-table">
      <tbody>${rows}</tbody>
    </table>
  `;
}


function formatPayloadValue(value) {
  if (Array.isArray(value)) {
    return value.join(", ");
  }

  if (typeof value === "object" && value !== null) {
    return JSON.stringify(value);
  }

  return value;
}


export function renderAgentRunHistory(container, runs, onSelectRun) {
  if (!runs.length) {
    container.innerHTML = '<p class="empty-state">No agent workflow history yet.</p>';
    return;
  }

  container.innerHTML = "";

  runs.forEach((run) => {
    const button = document.createElement("button");
    button.className = "dataset-card agent-run-card";
    button.type = "button";

    button.innerHTML = `
      <h4>${escapeHtml(run.status)}</h4>
      <p>${escapeHtml(run.final_summary)}</p>
      <p>${run.step_count} steps · ${run.error_count} warning(s)</p>
      <p>Finished: ${escapeHtml(run.finished_at)}</p>
    `;

    button.addEventListener("click", () => {
      onSelectRun(run.workflow_id);
    });

    container.appendChild(button);
  });
}
