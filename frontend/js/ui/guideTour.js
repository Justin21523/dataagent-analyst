const TOUR_STEPS = [
  {
    route: "data-upload",
    selector: ".workspace-sidebar",
    label: "Navigation",
    title: "Start with the left sidebar",
    body: "Use the sidebar to move through Data, Analyze, Model, Lifecycle, and Agent work areas without scrolling through one large page.",
  },
  {
    route: "data-upload",
    selector: ".workspace-topbar",
    label: "Context",
    title: "Watch the active workspace context",
    body: "The top bar keeps the selected dataset, version, model, and drift state visible while you move across tools.",
  },
  {
    route: "data-upload",
    selector: ".workspace-inspector",
    label: "Guidance",
    title: "Use the right inspector for next actions",
    body: "The inspector summarizes current state and suggests the next workflow action based on what has already been completed.",
  },
  {
    route: "data-upload",
    selector: "#uploadForm",
    label: "Step 1",
    title: "Upload a CSV dataset",
    body: "Begin by uploading a CSV. This anchors profiling, EDA, ML, reporting, and lifecycle checks to the same dataset context.",
  },
  {
    route: "data-preview",
    selector: "#previewTable",
    label: "Step 2",
    title: "Preview rows and columns",
    body: "Open Dataset Preview to confirm shape, sample rows, and whether the uploaded data looks usable before deeper analysis.",
  },
  {
    route: "data-versions",
    selector: "#datasetVersionsPanel",
    label: "Step 3",
    title: "Track versions and transforms",
    body: "Versions & Transform records derived data versions so drift, retraining, and migration checks can compare reproducible snapshots.",
  },
  {
    route: "analyze-schema",
    selector: "#schemaSummary",
    label: "Step 4",
    title: "Inspect schema and target candidates",
    body: "Schema profiling helps detect data types, missing values, duplicate rows, and likely target columns for prediction.",
  },
  {
    route: "analyze-eda",
    selector: "#edaQualitySummary",
    label: "Step 5",
    title: "Review EDA quality signals",
    body: "EDA summarizes quality, missingness, outliers, correlations, and recommendations before you commit to modeling.",
  },
  {
    route: "analyze-visualization",
    selector: "#visualizationLabSummary",
    label: "Step 6",
    title: "Explore the Visualization Lab",
    body: "Visualization Lab gives chart recommendations and a custom chart builder to inspect relationships in the data.",
  },
  {
    route: "model-workbench",
    selector: "#mlTaskTypeSelect",
    label: "Step 7",
    title: "Configure an ML experiment",
    body: "ML Workbench prepares task detection, feature choices, model options, cross-validation, and experiment runs.",
  },
  {
    route: "model-registry",
    selector: "#modelLeaderboard",
    label: "Step 8",
    title: "Compare saved models",
    body: "Registry & Lifecycle shows saved models, metrics, promotion controls, challenger checks, and migration diagnostics.",
  },
  {
    route: "model-prediction",
    selector: "#predictionJsonInput",
    label: "Step 9",
    title: "Run predictions",
    body: "Prediction supports single JSON records and batch CSV scoring once a model has been trained or selected.",
  },
  {
    route: "model-explainability",
    selector: "#explainabilityOverview",
    label: "Step 10",
    title: "Explain model behavior",
    body: "Explainability provides diagnostics, feature importance, SHAP-style views, and local explanations for model review.",
  },
  {
    route: "lifecycle-drift",
    selector: "#driftReportOutput",
    label: "Step 11",
    title: "Monitor drift and retraining needs",
    body: "Drift Center compares dataset versions and model behavior, then recommends retraining or migration actions when needed.",
  },
  {
    route: "lifecycle-reports",
    selector: "#reportViewer",
    label: "Step 12",
    title: "Generate analysis reports",
    body: "Reports collect analysis, EDA, visualization, model results, and lifecycle information into Markdown output.",
  },
  {
    route: "lifecycle-backtests",
    selector: "#backtestRunList",
    label: "Step 13",
    title: "Run and inspect quality gates",
    body: "Backtest Runs lets you launch automated checks, inspect artifacts, view screenshots, and review payloads from regression workflows.",
  },
  {
    route: "agent-workflow",
    selector: "#agentGoalInput",
    label: "Step 14",
    title: "Automate with Agent Jobs",
    body: "Agent Jobs can orchestrate profiling, analysis, ML, reports, and AI summaries while preserving run history and events.",
  },
  {
    route: "agent-insights",
    selector: "#aiInsightOutput",
    label: "Step 15",
    title: "Ask for AI insight summaries",
    body: "AI Insights turns EDA, model, and report context into natural-language explanations when local LLM support is enabled.",
  },
  {
    route: "data-upload",
    selector: "#nextActionList",
    label: "Finish",
    title: "Follow Suggested Actions when unsure",
    body: "At any point, return to the right inspector. Suggested Actions will point to the next useful workflow step for the current state.",
  },
];

const PARTICLE_POINTS = [
  ["-0.7rem", "-0.55rem"],
  ["18%", "-0.8rem"],
  ["50%", "-0.7rem"],
  ["82%", "-0.8rem"],
  ["calc(100% + 0.45rem)", "-0.4rem"],
  ["calc(100% + 0.55rem)", "28%"],
  ["calc(100% + 0.45rem)", "68%"],
  ["calc(100% + 0.25rem)", "calc(100% + 0.35rem)"],
  ["65%", "calc(100% + 0.55rem)"],
  ["32%", "calc(100% + 0.55rem)"],
  ["-0.65rem", "calc(100% + 0.25rem)"],
  ["-0.75rem", "42%"],
];

let root = null;
let launcher = null;
let active = false;
let skippedThisSession = false;
let currentIndex = 0;


export function initGuideTour() {
  root = document.querySelector("#guideTourRoot");

  if (!root) {
    return;
  }

  root.className = "guide-tour-root";
  root.hidden = true;
  root.innerHTML = buildTourMarkup();
  launcher = document.createElement("button");
  launcher.className = "guide-tour-launcher";
  launcher.type = "button";
  launcher.textContent = "Guide";
  launcher.hidden = true;
  launcher.dataset.guideTourLauncher = "true";
  document.body.appendChild(launcher);

  root.querySelector('[data-guide-action="next"]').addEventListener("click", nextStep);
  root.querySelector('[data-guide-action="back"]').addEventListener("click", previousStep);
  root.querySelector('[data-guide-action="skip"]').addEventListener("click", skipTour);
  launcher.addEventListener("click", () => startGuideTour({ force: true }));

  window.addEventListener("resize", updateTargetFrame);
  window.addEventListener("scroll", updateTargetFrame, true);
  document.addEventListener("keydown", handleKeydown);

  window.setTimeout(() => {
    if (!skippedThisSession) {
      startGuideTour();
    }
  }, 650);
}


export async function startGuideTour(options = {}) {
  if (active || (!options.force && skippedThisSession)) {
    return;
  }

  active = true;
  currentIndex = 0;
  root.hidden = false;
  launcher.hidden = true;
  await renderStep();
}


async function renderStep() {
  const step = TOUR_STEPS[currentIndex];

  await navigateToRoute(step.route);
  const target = await waitForTarget(step.selector);
  target?.scrollIntoView({
    block: "center",
    inline: "center",
    behavior: "smooth",
  });

  await wait(220);
  updatePanel(step);
  updateTargetFrame();
}


async function navigateToRoute(route) {
  if (window.location.hash === `#${route}`) {
    return;
  }

  const routeChanged = new Promise((resolve) => {
    const timeout = window.setTimeout(resolve, 3500);
    window.addEventListener(
      "dataagent:route-changed",
      () => {
        window.clearTimeout(timeout);
        resolve();
      },
      { once: true },
    );
  });

  window.dispatchEvent(
    new CustomEvent("dataagent:navigate-route", {
      detail: {
        route,
      },
    }),
  );

  await routeChanged;
}


async function waitForTarget(selector) {
  for (let attempt = 0; attempt < 35; attempt += 1) {
    const target = document.querySelector(selector);

    if (target && target.getClientRects().length) {
      return target;
    }

    await wait(100);
  }

  return document.querySelector(".app-shell") || document.body;
}


function updatePanel(step) {
  root.querySelector("[data-guide-step-label]").textContent = step.label;
  root.querySelector("[data-guide-title]").textContent = step.title;
  root.querySelector("[data-guide-body]").textContent = step.body;
  root.querySelector("[data-guide-step-count]").textContent =
    `${currentIndex + 1} / ${TOUR_STEPS.length}`;
  root.querySelector('[data-guide-action="back"]').disabled = currentIndex === 0;
  root.querySelector('[data-guide-action="next"]').textContent =
    currentIndex === TOUR_STEPS.length - 1 ? "Finish" : "Next";
}


function updateTargetFrame() {
  if (!active || !root) {
    return;
  }

  const step = TOUR_STEPS[currentIndex];
  const target = document.querySelector(step.selector) || document.querySelector(".app-shell");
  const rect = target.getBoundingClientRect();
  const padding = 10;
  const x = Math.max(8, rect.left - padding);
  const y = Math.max(8, rect.top - padding);
  const width = Math.min(window.innerWidth - x - 8, rect.width + padding * 2);
  const height = Math.min(window.innerHeight - y - 8, rect.height + padding * 2);

  root.style.setProperty("--tour-x", `${x}px`);
  root.style.setProperty("--tour-y", `${y}px`);
  root.style.setProperty("--tour-w", `${width}px`);
  root.style.setProperty("--tour-h", `${height}px`);
  root.style.setProperty("--tour-center-x", `${x + width / 2}px`);
  root.style.setProperty("--tour-center-y", `${y + height / 2}px`);
}


async function nextStep() {
  if (currentIndex >= TOUR_STEPS.length - 1) {
    finishTour();
    return;
  }

  currentIndex += 1;
  await renderStep();
}


async function previousStep() {
  if (currentIndex === 0) {
    return;
  }

  currentIndex -= 1;
  await renderStep();
}


function skipTour() {
  skippedThisSession = true;
  closeTour();
}


function finishTour() {
  closeTour();
}


function closeTour() {
  active = false;
  root.hidden = true;
  launcher.hidden = false;
}


function handleKeydown(event) {
  if (!active) {
    return;
  }

  if (event.key === "Escape") {
    skipTour();
  } else if (event.key === "ArrowRight") {
    nextStep();
  } else if (event.key === "ArrowLeft") {
    previousStep();
  }
}


function buildTourMarkup() {
  return `
    <div class="guide-tour-scrim" data-guide-tour-scrim></div>
    <div class="guide-tour-frame" data-guide-tour-frame></div>
    <div class="guide-tour-particles" aria-hidden="true">
      ${PARTICLE_POINTS.map(([x, y], index) => `
        <span
          class="guide-tour-particle"
          style="--particle-x: ${x}; --particle-y: ${y}; --particle-delay: ${index * 0.08}s"
        ></span>
      `).join("")}
    </div>
    <section
      class="guide-tour-panel"
      role="dialog"
      aria-live="polite"
      aria-label="Product guide"
      data-guide-tour-panel
    >
      <div class="guide-tour-panel-header">
        <div>
          <span class="guide-tour-step-label" data-guide-step-label>Guide</span>
          <h2 data-guide-title>DataAgent Analyst guide</h2>
        </div>
        <span class="guide-tour-step-count" data-guide-step-count>1 / ${TOUR_STEPS.length}</span>
      </div>
      <p data-guide-body></p>
      <div class="guide-tour-actions">
        <button class="guide-tour-action ghost" type="button" data-guide-action="skip">
          Skip
        </button>
        <button class="guide-tour-action" type="button" data-guide-action="back">
          Back
        </button>
        <button class="guide-tour-action primary" type="button" data-guide-action="next">
          Next
        </button>
      </div>
    </section>
  `;
}


function wait(milliseconds) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}
