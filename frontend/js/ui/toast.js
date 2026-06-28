const TOAST_DURATION_MS = 4500;


export function bindGlobalToasts() {
  window.addEventListener("dataagent:toast", (event) => {
    const message = event.detail?.message || "Unexpected application message.";
    const type = event.detail?.type || "info";

    showToast(message, type);
  });
}


export function showToast(message, type = "info") {
  const region = document.querySelector("#toastRegion");

  if (!region) {
    return;
  }

  const toast = document.createElement("div");

  toast.className = `toast-message ${type}`;
  toast.setAttribute("role", type === "error" ? "alert" : "status");
  toast.textContent = message;

  region.appendChild(toast);

  requestAnimationFrame(() => {
    toast.classList.add("visible");
  });

  window.setTimeout(() => {
    toast.classList.remove("visible");

    window.setTimeout(() => {
      toast.remove();
    }, 200);
  }, TOAST_DURATION_MS);
}
