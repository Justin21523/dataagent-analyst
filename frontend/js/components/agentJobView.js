import { escapeHtml, formatCellValue } from "../utils/format.js";


export function renderAgentJobStatus(container, payload) {
  container.innerHTML = `
    <div class="training-status-card ${payload.status === "success" ? "success" : ""}">
      <span>Background Agent Job</span>
      <strong>${escapeHtml(payload.title)}</strong>
      <p>${escapeHtml(payload.message)}</p>
    </div>
  `;
}


export function renderAgentJobList(container, jobs, onSelectJob) {
  if (!jobs.length) {
    container.innerHTML = '<p class="empty-state">No background agent jobs yet.</p>';
    return;
  }

  container.innerHTML = "";

  jobs.forEach((job) => {
    const button = document.createElement("button");
    button.className = "dataset-card agent-run-card";
    button.type = "button";

    button.innerHTML = `
      <h4>${escapeHtml(job.status)}</h4>
      <p>Job ID: ${escapeHtml(job.job_id)}</p>
      <p>${job.event_count} event(s)</p>
      <p>Created: ${escapeHtml(job.created_at)}</p>
    `;

    button.addEventListener("click", () => {
      onSelectJob(job.job_id);
    });

    container.appendChild(button);
  });
}


export function renderAgentJobEvents(container, events) {
  if (!events.length) {
    container.innerHTML = '<p class="empty-state">No job events yet.</p>';
    return;
  }

  container.innerHTML = events
    .map((event) => `
      <article class="agent-step-card ${escapeHtml(event.status)}">
        <div class="agent-step-header">
          <div>
            <span>${escapeHtml(event.event_type)}</span>
            <h4>${escapeHtml(event.step_name || event.status)}</h4>
          </div>
          <small>${escapeHtml(event.created_at)}</small>
        </div>
        <p>${escapeHtml(event.message)}</p>
        ${renderEventPayload(event.payload)}
      </article>
    `)
    .join("");
}


function renderEventPayload(payload) {
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
