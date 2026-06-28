const DEVELOPMENT_FRONTEND_PORTS = new Set([
  "5173",
  "5500",
]);

const API_BASE_URL_STORAGE_KEY = "dataagent.apiBaseUrl";


function readConfiguredApiBaseUrl() {
  const queryApiBaseUrl = new URLSearchParams(window.location.search).get(
    "apiBaseUrl",
  );

  if (queryApiBaseUrl) {
    localStorage.setItem(API_BASE_URL_STORAGE_KEY, queryApiBaseUrl);
    return queryApiBaseUrl;
  }

  return localStorage.getItem(API_BASE_URL_STORAGE_KEY);
}


function resolveApiBaseUrl() {
  const configuredApiBaseUrl = readConfiguredApiBaseUrl();

  if (configuredApiBaseUrl) {
    return configuredApiBaseUrl;
  }

  // 本機 static server 開發時直接連 FastAPI port。
  if (DEVELOPMENT_FRONTEND_PORTS.has(window.location.port)) {
    return "http://127.0.0.1:8000";
  }

  // Docker / Nginx 模式使用同源 /api reverse proxy。
  return "";
}


export const API_BASE_URL =
  window.DATAAGENT_CONFIG?.apiBaseUrl ??
  resolveApiBaseUrl();
