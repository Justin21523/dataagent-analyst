export function renderJson(element, payload) {
  // 開發階段直接顯示 JSON，方便確認前後端資料格式。
  element.textContent = JSON.stringify(payload, null, 2);
}


export function formatCellValue(value) {
  // null / undefined 統一顯示為空字串，避免前端表格出現不友善文字。
  if (value === null || value === undefined) {
    return "";
  }

  return String(value);
}


export function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}


export function escapeHtml(value) {
  // 避免檔名、欄位名稱或資料內容被當成 HTML 插入頁面。
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
