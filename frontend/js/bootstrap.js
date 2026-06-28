import { initNavigation } from "./ui/navigation.js";
import { initGuideTour } from "./ui/guideTour.js";
import { bindGlobalToasts, showToast } from "./ui/toast.js";
import {
  ensureRouteInitialized,
  initializeAppShell,
} from "./app.js";


const PARTIALS = {
  data: {
    mount: "#overviewView",
    path: "./partials/overview.html",
  },
  analyze: {
    mount: "#analysisView",
    path: "./partials/analysis.html",
  },
  model: {
    mount: "#machineLearningView",
    path: "./partials/machine-learning.html",
  },
  intelligence: {
    mount: "#intelligenceView",
    path: "./partials/intelligence.html",
  },
};

const loadedGroups = new Set();


async function loadPartial(group) {
  if (loadedGroups.has(group)) {
    return;
  }

  const partial = PARTIALS[group];
  const { mount, path } = partial;
  const container = document.querySelector(mount);

  if (!container) {
    throw new Error(`Partial mount not found: ${mount}`);
  }

  const response = await fetch(path);

  if (!response.ok) {
    throw new Error(`Failed to load partial ${path}: ${response.status}`);
  }

  container.innerHTML = await response.text();
  loadedGroups.add(group);
}


bindGlobalToasts();

try {
  await initializeAppShell();

  initNavigation({
    beforeActivate: async (routeDetail) => {
      await loadPartial(routeDetail.group);
      await ensureRouteInitialized(routeDetail);
    },
  });
  initGuideTour();
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
