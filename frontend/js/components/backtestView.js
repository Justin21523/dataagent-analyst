import { escapeHtml } from "../utils/format.js";


const STATUS_LABELS = {
  queued: "Queued",
  running: "Running",
  success: "Success",
  failed: "Failed",
  skipped: "Skipped",
  invalid: "Invalid",
  unknown: "Unknown",
};


export function renderBacktestJobList(container, jobs, selectedJobId) {
  if (!jobs.length) {
    container.innerHTML = '<p class="empty-state">No backtest jobs started yet.</p>';
    return;
  }

  container.innerHTML = jobs.map((job) => `
    <button
      class="dataset-card backtest-job-card ${job.job_id === selectedJobId ? "active" : ""}"
      type="button"
      data-backtest-job-id="${escapeHtml(job.job_id)}"
    >
      <div class="backtest-card-header">
        <h4>${escapeHtml(job.suite_type.toUpperCase())}</h4>
        ${statusBadge(job.status)}
      </div>
      <p>${escapeHtml(formatTimestamp(job.created_at))}</p>
      <div class="backtest-metric-row">
        <span>${job.event_count} events</span>
        <span>${job.run_ids.length} runs</span>
        <span>${job.suite_ids.length} suites</span>
      </div>
      ${job.error_message ? `<p class="backtest-failure-text">${escapeHtml(job.error_message)}</p>` : ""}
    </button>
  `).join("");
}


export function renderBacktestJobStatus(container, job) {
  if (!job) {
    container.innerHTML = '<p class="empty-state">Start a backtest job to stream progress here.</p>';
    return;
  }

  container.innerHTML = `
    <div class="backtest-job-status-card">
      <div>
        <span class="panel-tag">Job</span>
        <h4>${escapeHtml(job.suite_type.toUpperCase())}</h4>
        <p>${escapeHtml(job.job_id)}</p>
      </div>
      ${statusBadge(job.status)}
    </div>
    <div class="backtest-metric-row">
      <span>${job.event_count} events</span>
      <span>${job.run_ids.length} runs</span>
      <span>${job.suite_ids.length} suites</span>
    </div>
  `;
}


export function renderBacktestJobLog(container, events) {
  if (!events.length) {
    container.textContent = "No job events yet.";
    return;
  }

  container.textContent = events.map((event) => {
    const time = formatTimestamp(event.created_at);
    return `[${time}] ${event.event_type} ${event.status}: ${event.message}`;
  }).join("\n");
}


export function renderBacktestRunList(container, runs, selectedRunId, onSelectRun) {
  if (!runs.length) {
    container.innerHTML = '<p class="empty-state">No backtest runs found.</p>';
    return;
  }

  container.innerHTML = "";

  runs.forEach((run) => {
    const button = document.createElement("button");
    button.className = `dataset-card backtest-run-card ${
      run.run_id === selectedRunId ? "active" : ""
    }`;
    button.type = "button";
    button.dataset.backtestRunId = run.run_id;
    button.innerHTML = `
      <div class="backtest-card-header">
        <h4>${escapeHtml(run.run_id)}</h4>
        ${statusBadge(run.status)}
      </div>
      <p>${escapeHtml(formatTimestamp(run.created_at))}</p>
      <div class="backtest-metric-row">
        <span>${run.step_count} steps</span>
        <span>${run.assertion_count} assertions</span>
        <span>${run.screenshot_count} screenshots</span>
      </div>
      ${renderFailureLine(run)}
    `;

    button.addEventListener("click", () => onSelectRun(run.run_id));
    container.appendChild(button);
  });
}


export function renderBacktestSuites(container, suites) {
  if (!suites.length) {
    container.innerHTML = '<p class="empty-state">No suite summaries found.</p>';
    return;
  }

  container.innerHTML = suites.map((suite) => `
    <div class="backtest-suite-row">
      <div>
        <strong>${escapeHtml(suite.suite)}</strong>
        <span>${escapeHtml(formatTimestamp(suite.created_at))}</span>
      </div>
      ${statusBadge(suite.status)}
      <small>${suite.runs.length} runs</small>
    </div>
  `).join("");
}


export function renderBacktestRunDetail(
  container,
  detail,
  {
    buildScreenshotUrl,
    onSelectPayload,
  },
) {
  const { summary, steps, assertions, payloads, screenshots, summary_markdown: markdown } = detail;
  const failedSteps = steps.filter((step) => step.status === "failed");
  const failedAssertions = assertions.filter((assertion) => assertion.status === "failed");

  container.innerHTML = `
    <div class="backtest-detail-heading">
      <div>
        <span class="panel-tag">Run Detail</span>
        <h3>${escapeHtml(summary.run_id)}</h3>
        <p>${escapeHtml(formatTimestamp(summary.created_at))}</p>
      </div>
      ${statusBadge(summary.status)}
    </div>

    <div class="summary-grid backtest-summary-grid">
      ${summaryTile("Steps", summary.step_count, `${summary.failed_step_count} failed`)}
      ${summaryTile("Assertions", summary.assertion_count, `${summary.failed_assertion_count} failed`)}
      ${summaryTile("Payloads", summary.payload_count, "JSON files")}
      ${summaryTile("Screenshots", summary.screenshot_count, "PNG captures")}
    </div>

    ${renderMetadata(summary)}
    ${renderFailures(failedSteps, failedAssertions, summary.error)}
    ${renderMarkdown(markdown)}
    ${renderStepsTable(steps)}
    ${renderAssertionsTable(assertions)}
    ${renderPayloadList(payloads)}
    ${renderScreenshotGallery(summary.run_id, screenshots, buildScreenshotUrl)}
  `;

  container.querySelectorAll("[data-backtest-payload-name]").forEach((button) => {
    button.addEventListener("click", () => {
      onSelectPayload(summary.run_id, button.dataset.backtestPayloadName);
    });
  });
}


export function renderBacktestPayload(container, payloadDetail) {
  if (!payloadDetail) {
    container.innerHTML = '<p class="empty-state">Select a payload to preview JSON.</p>';
    return;
  }

  container.innerHTML = `
    <div class="panel-header compact-header">
      <div>
        <span class="panel-tag">Payload</span>
        <h3>${escapeHtml(payloadDetail.name)}</h3>
      </div>
    </div>
    <pre class="code-output backtest-payload-output">${escapeHtml(
      JSON.stringify(payloadDetail.payload, null, 2),
    )}</pre>
  `;
}


export function renderBacktestLoading(container, message) {
  container.innerHTML = `<p class="empty-state">${escapeHtml(message)}</p>`;
}


function renderFailureLine(run) {
  const failures = Number(run.failed_step_count || 0) + Number(run.failed_assertion_count || 0);

  if (run.status === "invalid") {
    return `<p class="backtest-failure-text">${escapeHtml(run.error || "Invalid artifact")}</p>`;
  }

  if (!failures) {
    return '<p class="backtest-success-text">No failed checks recorded.</p>';
  }

  return `<p class="backtest-failure-text">${failures} failed checks need review.</p>`;
}


function renderMetadata(summary) {
  const entries = Object.entries(summary.metadata || {});

  if (!entries.length) {
    return "";
  }

  return `
    <section class="backtest-detail-section">
      <h4>Metadata</h4>
      <dl class="backtest-metadata-list">
        ${entries.map(([key, value]) => `
          <div>
            <dt>${escapeHtml(key)}</dt>
            <dd>${escapeHtml(formatValue(value))}</dd>
          </div>
        `).join("")}
      </dl>
    </section>
  `;
}


function renderFailures(failedSteps, failedAssertions, artifactError) {
  if (!failedSteps.length && !failedAssertions.length && !artifactError) {
    return "";
  }

  return `
    <section class="backtest-detail-section backtest-warning-section">
      <h4>Needs Review</h4>
      ${artifactError ? `<p>${escapeHtml(artifactError)}</p>` : ""}
      ${failedSteps.map((step) => `
        <p><strong>Step:</strong> ${escapeHtml(step.name)} ${escapeHtml(step.error || "")}</p>
      `).join("")}
      ${failedAssertions.map((assertion) => `
        <p><strong>Assertion:</strong> ${escapeHtml(assertion.name)} ${escapeHtml(assertion.message || "")}</p>
      `).join("")}
    </section>
  `;
}


function renderMarkdown(markdown) {
  if (!markdown) {
    return "";
  }

  return `
    <section class="backtest-detail-section">
      <h4>Summary</h4>
      <pre class="markdown-preview backtest-markdown-preview">${escapeHtml(markdown)}</pre>
    </section>
  `;
}


function renderStepsTable(steps) {
  if (!steps.length) {
    return '<section class="backtest-detail-section"><p class="empty-state">No steps recorded.</p></section>';
  }

  return `
    <section class="backtest-detail-section">
      <h4>Steps</h4>
      <div class="table-wrapper">
        <table class="preview-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Status</th>
              <th>Duration</th>
              <th>Payload</th>
              <th>Error</th>
            </tr>
          </thead>
          <tbody>
            ${steps.map((step) => `
              <tr>
                <td>${escapeHtml(step.name)}</td>
                <td>${statusBadge(step.status)}</td>
                <td>${Number(step.duration_seconds || 0).toFixed(2)}s</td>
                <td>${escapeHtml(step.payload_path || "-")}</td>
                <td>${escapeHtml(step.error || "-")}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </section>
  `;
}


function renderAssertionsTable(assertions) {
  if (!assertions.length) {
    return '<section class="backtest-detail-section"><p class="empty-state">No assertions recorded.</p></section>';
  }

  return `
    <section class="backtest-detail-section">
      <h4>Assertions</h4>
      <div class="table-wrapper">
        <table class="preview-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Status</th>
              <th>Expected</th>
              <th>Actual</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody>
            ${assertions.map((assertion) => `
              <tr>
                <td>${escapeHtml(assertion.name)}</td>
                <td>${statusBadge(assertion.status)}</td>
                <td>${escapeHtml(formatValue(assertion.expected))}</td>
                <td>${escapeHtml(formatValue(assertion.actual))}</td>
                <td>${escapeHtml(assertion.message || "-")}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </section>
  `;
}


function renderPayloadList(payloads) {
  if (!payloads.length) {
    return "";
  }

  return `
    <section class="backtest-detail-section">
      <h4>Payloads</h4>
      <div class="backtest-payload-list">
        ${payloads.map((payload) => `
          <button
            class="secondary-button backtest-payload-button"
            type="button"
            data-backtest-payload-name="${escapeHtml(payload.name)}"
          >
            ${escapeHtml(payload.name)}
          </button>
        `).join("")}
      </div>
    </section>
  `;
}


function renderScreenshotGallery(runId, screenshots, buildScreenshotUrl) {
  if (!screenshots.length) {
    return "";
  }

  return `
    <section class="backtest-detail-section">
      <h4>Screenshots</h4>
      <div class="backtest-screenshot-grid">
        ${screenshots.map((screenshot) => `
          <figure class="backtest-screenshot">
            <img
              src="${escapeHtml(buildScreenshotUrl(runId, screenshot.name))}"
              alt="${escapeHtml(screenshot.name)}"
              loading="lazy"
            />
            <figcaption>${escapeHtml(screenshot.name)}</figcaption>
          </figure>
        `).join("")}
      </div>
    </section>
  `;
}


function summaryTile(label, value, helper) {
  return `
    <div class="summary-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <p>${escapeHtml(helper)}</p>
    </div>
  `;
}


function statusBadge(status) {
  const normalized = String(status || "unknown").toLowerCase();
  const label = STATUS_LABELS[normalized] || normalized;

  return `<span class="badge backtest-status ${escapeHtml(normalized)}">${escapeHtml(label)}</span>`;
}


function formatTimestamp(value) {
  if (!value) {
    return "No timestamp";
  }

  return String(value).replace("T", " ").replace("+00:00", " UTC");
}


function formatValue(value) {
  if (value === null || value === undefined) {
    return "-";
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value);
}
