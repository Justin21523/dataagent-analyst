const DEFAULT_ROUTE = "data-upload";

export const ROUTES = {
  "data-upload": {
    group: "data",
    section: "data-upload",
    eyebrow: "Data",
    title: "Upload",
  },
  "data-preview": {
    group: "data",
    section: "data-preview",
    eyebrow: "Data",
    title: "Dataset Preview",
  },
  "data-versions": {
    group: "data",
    section: "data-versions",
    eyebrow: "Data",
    title: "Versions & Transform",
  },
  "analyze-schema": {
    group: "analyze",
    section: "analyze-schema",
    eyebrow: "Analyze",
    title: "Schema",
  },
  "analyze-eda": {
    group: "analyze",
    section: "analyze-eda",
    eyebrow: "Analyze",
    title: "Exploratory Data Analysis",
  },
  "analyze-visualization": {
    group: "analyze",
    section: "analyze-visualization",
    eyebrow: "Analyze",
    title: "Visualization Lab",
  },
  "model-workbench": {
    group: "model",
    section: "model-workbench",
    eyebrow: "Model",
    title: "ML Workbench",
  },
  "model-registry": {
    group: "model",
    section: "model-registry",
    eyebrow: "Model",
    title: "Registry & Lifecycle",
  },
  "model-prediction": {
    group: "model",
    section: "model-prediction",
    eyebrow: "Model",
    title: "Prediction",
  },
  "model-explainability": {
    group: "model",
    section: "model-explainability",
    eyebrow: "Model",
    title: "Explainability",
  },
  "lifecycle-drift": {
    group: "intelligence",
    section: "lifecycle-drift",
    eyebrow: "Lifecycle",
    title: "Drift Center",
  },
  "lifecycle-reports": {
    group: "intelligence",
    section: "lifecycle-reports",
    eyebrow: "Lifecycle",
    title: "Reports",
  },
  "lifecycle-backtests": {
    group: "intelligence",
    section: "lifecycle-backtests",
    eyebrow: "Lifecycle",
    title: "Backtest Runs",
  },
  "agent-workflow": {
    group: "intelligence",
    section: "agent-workflow",
    eyebrow: "Agent",
    title: "Agent Jobs",
  },
  "agent-insights": {
    group: "intelligence",
    section: "agent-insights",
    eyebrow: "Agent",
    title: "AI Insights",
  },
};


export function getRoute(routeName) {
  const selectedRoute = ROUTES[routeName] ? routeName : DEFAULT_ROUTE;
  return {
    name: selectedRoute,
    ...ROUTES[selectedRoute],
  };
}


export function getDefaultRouteName() {
  return DEFAULT_ROUTE;
}


export function getCurrentRouteName() {
  return normalizeHashRoute();
}


export function initNavigation(options = {}) {
  const mounts = [...document.querySelectorAll("[data-view-mount]")];
  const routeEyebrow = document.querySelector("#routeEyebrow");
  const routeTitle = document.querySelector("#routeTitle");
  const sectionSubnav = document.querySelector("#sectionSubnav");
  let pendingRouteRequest = Promise.resolve();

  async function activateRoute(routeName, updateHash = true) {
    const route = ROUTES[routeName] || ROUTES[DEFAULT_ROUTE];
    const selectedRoute = ROUTES[routeName] ? routeName : DEFAULT_ROUTE;
    const routeDetail = {
      route: selectedRoute,
      group: route.group,
      section: route.section,
    };

    pendingRouteRequest = pendingRouteRequest.then(async () => {
      if (options.beforeActivate) {
        await options.beforeActivate(routeDetail);
      }

      renderRoute(route, selectedRoute, routeDetail, updateHash);
    });

    await pendingRouteRequest;
  }

  function renderRoute(route, selectedRoute, routeDetail, updateHash) {

    mounts.forEach((mount) => {
      const isActiveMount = mount.dataset.viewMount === route.group;

      mount.hidden = !isActiveMount;

      if (isActiveMount) {
        activateSection(mount, route.section);
      }
    });

    document.querySelectorAll("[data-view-target]").forEach((button) => {
      const isActive = button.dataset.viewTarget === selectedRoute;

      button.classList.toggle("active", isActive);
      button.setAttribute("aria-current", isActive ? "page" : "false");
    });

    if (routeEyebrow) {
      routeEyebrow.textContent = route.eyebrow;
    }

    if (routeTitle) {
      routeTitle.textContent = route.title;
    }

    renderSubnav(route.group, selectedRoute, sectionSubnav);

    if (updateHash) {
      history.replaceState(null, "", `#${selectedRoute}`);
    }

    window.dispatchEvent(
      new CustomEvent("dataagent:route-changed", {
        detail: routeDetail,
      }),
    );

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  }

  document.addEventListener("click", (event) => {
    const trigger = event.target.closest("[data-view-target]");

    if (!trigger) {
      return;
    }

    activateRoute(trigger.dataset.viewTarget || DEFAULT_ROUTE);
  });

  window.addEventListener("hashchange", () => {
    activateRoute(normalizeHashRoute(), false);
  });

  window.addEventListener("dataagent:navigate-route", (event) => {
    const routeName = event.detail?.route || DEFAULT_ROUTE;
    activateRoute(routeName);
  });

  activateRoute(normalizeHashRoute(), false);
}


function activateSection(mount, sectionName) {
  const sections = [...mount.querySelectorAll("[data-route-section]")];
  const hasMatchingSection = sections.some(
    (section) => section.dataset.routeSection === sectionName,
  );

  sections.forEach((section) => {
    section.hidden = hasMatchingSection && section.dataset.routeSection !== sectionName;
  });
}


function renderSubnav(group, activeRoute, container) {
  if (!container) {
    return;
  }

  const routes = Object.entries(ROUTES).filter(([, route]) => route.group === group);

  container.innerHTML = routes.map(([routeName, route]) => `
    <button
      class="section-subnav-button ${routeName === activeRoute ? "active" : ""}"
      type="button"
      data-view-target="${routeName}"
    >
      ${route.title}
    </button>
  `).join("");
}


function normalizeHashRoute() {
  const routeName = window.location.hash.slice(1);
  return ROUTES[routeName] ? routeName : DEFAULT_ROUTE;
}
