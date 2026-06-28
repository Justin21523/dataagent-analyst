const DEFAULT_VIEW = "overview";


export function initNavigation() {
  const buttons = [...document.querySelectorAll("[data-view-target]")];
  const mounts = [...document.querySelectorAll("[data-view-mount]")];

  function activateView(viewName, updateHash = true) {
    const validView = mounts.some(
      (mount) => mount.dataset.viewMount === viewName,
    );

    const selectedView = validView ? viewName : DEFAULT_VIEW;

    mounts.forEach((mount) => {
      mount.hidden = mount.dataset.viewMount !== selectedView;
    });

    buttons.forEach((button) => {
      const isActive = button.dataset.viewTarget === selectedView;

      button.classList.toggle("active", isActive);
      button.setAttribute("aria-current", isActive ? "page" : "false");
    });

    if (updateHash) {
      history.replaceState(null, "", `#${selectedView}`);
    }

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      activateView(button.dataset.viewTarget || DEFAULT_VIEW);
    });
  });

  window.addEventListener("hashchange", () => {
    const requestedView = window.location.hash.slice(1) || DEFAULT_VIEW;
    activateView(requestedView, false);
  });

  activateView(window.location.hash.slice(1) || DEFAULT_VIEW, false);
}
