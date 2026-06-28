import { initNavigation } from "./ui/navigation.js";
import { bindGlobalToasts, showToast } from "./ui/toast.js";


const PARTIALS = [
  {
    mount: "#overviewView",
    path: "./partials/overview.html",
  },
  {
    mount: "#analysisView",
    path: "./partials/analysis.html",
  },
  {
    mount: "#machineLearningView",
    path: "./partials/machine-learning.html",
  },
  {
    mount: "#intelligenceView",
    path: "./partials/intelligence.html",
  },
];


async function loadPartial({ mount, path }) {
  const container = document.querySelector(mount);

  if (!container) {
    throw new Error(`Partial mount not found: ${mount}`);
  }

  const response = await fetch(path);

  if (!response.ok) {
    throw new Error(`Failed to load partial ${path}: ${response.status}`);
  }

  container.innerHTML = await response.text();
}


bindGlobalToasts();

try {
  // 先載入所有 HTML partials，再 import app.js，確保 DOM elements 已存在。
  await Promise.all(PARTIALS.map(loadPartial));

  initNavigation();

  await import("./app.js");
} catch (error) {
  console.error(error);

  showToast(
    `Application bootstrap failed: ${error.message}`,
    "error",
  );

  const overviewMount = document.querySelector("#overviewView");

  if (overviewMount) {
    overviewMount.innerHTML = `
      <section class="panel bootstrap-error-panel">
        <h2>Application Bootstrap Failed</h2>
        <p>${error.message}</p>
      </section>
    `;
  }
}
