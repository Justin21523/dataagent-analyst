import {
  buildVisualizationChartApi,
  getVisualizationLabApi,
} from "../api/client.js";
import {
  renderEChartsGrid,
  renderSingleEChart,
} from "../components/echartsRenderer.js";
import {
  populateVisualizationControls,
  renderCustomVisualizationMeta,
  renderVisualizationLabSummary,
  renderVisualizationLoading,
  renderVisualizationWarnings,
} from "../components/visualizationLabView.js";


let initialized = false;
let currentDatasetId = null;
let currentResponse = null;
let elements = null;


export function initVisualizationLab() {
  if (initialized) {
    return;
  }

  elements = {
    summary: document.querySelector(
      "#visualizationLabSummary",
    ),
    warnings: document.querySelector(
      "#visualizationLabWarnings",
    ),
    refreshButton: document.querySelector(
      "#refreshVisualizationLabButton",
    ),
    target: document.querySelector(
      "#visualizationTargetSelect",
    ),
    chartType: document.querySelector(
      "#visualizationChartTypeSelect",
    ),
    xColumn: document.querySelector(
      "#visualizationXColumnSelect",
    ),
    yColumn: document.querySelector(
      "#visualizationYColumnSelect",
    ),
    groupColumn: document.querySelector(
      "#visualizationGroupColumnSelect",
    ),
    buildButton: document.querySelector(
      "#buildVisualizationChartButton",
    ),
    chartGrid: document.querySelector(
      "#visualizationChartGrid",
    ),
    customMeta: document.querySelector(
      "#customVisualizationMeta",
    ),
    customChart: document.querySelector(
      "#customVisualizationChart",
    ),
  };

  const missingElement = Object.entries(elements).find(
    ([, element]) => !element,
  );

  if (missingElement) {
    throw new Error(
      `Visualization Lab element not found: ${missingElement[0]}`,
    );
  }

  elements.refreshButton.addEventListener(
    "click",
    () => {
      if (currentDatasetId) {
        loadVisualizationLab(
          currentDatasetId,
          getOptionalValue(elements.target),
        );
      }
    },
  );

  elements.target.addEventListener(
    "change",
    () => {
      if (currentDatasetId) {
        loadVisualizationLab(
          currentDatasetId,
          getOptionalValue(elements.target),
        );
      }
    },
  );

  elements.buildButton.addEventListener(
    "click",
    buildCustomVisualization,
  );

  initialized = true;
}


export async function loadVisualizationLab(
  datasetId,
  targetColumn = null,
) {
  if (!initialized) {
    initVisualizationLab();
  }

  currentDatasetId = datasetId;

  renderVisualizationLoading(
    elements.chartGrid,
    "Generating analytical visualizations...",
  );

  elements.summary.innerHTML = `
    <p class="empty-state">
      Loading Visualization Lab summary...
    </p>
  `;

  try {
    const response = await getVisualizationLabApi(
      datasetId,
      {
        target_column: targetColumn,
        selected_columns: [],
        sample_rows: 80,
        max_numeric_columns: 8,
        max_categories: 12,
      },
    );

    currentResponse = response;

    renderVisualizationLabSummary(
      elements.summary,
      response.summary,
    );

    renderVisualizationWarnings(
      elements.warnings,
      response.warnings,
    );

    populateVisualizationControls(
      elements,
      response.summary,
    );

    if (
      targetColumn
      && response.summary.column_options.some(
        (column) => column.name === targetColumn,
      )
    ) {
      elements.target.value = targetColumn;
    }

    renderEChartsGrid(
      elements.chartGrid,
      response.charts,
    );
  } catch (error) {
    elements.summary.innerHTML = `
      <p class="empty-state">
        Failed to load Visualization Lab:
        ${error.message}
      </p>
    `;

    elements.chartGrid.innerHTML = `
      <p class="empty-state">
        No analytical charts were rendered.
      </p>
    `;
  }
}


async function buildCustomVisualization() {
  if (!currentDatasetId) {
    dispatchErrorToast(
      "Please select a dataset before building a chart.",
    );
    return;
  }

  renderVisualizationLoading(
    elements.customMeta,
    "Building custom visualization...",
  );

  elements.customChart.innerHTML = "";

  try {
    const specification = await buildVisualizationChartApi(
      currentDatasetId,
      {
        chart_type: elements.chartType.value,
        x_column: getOptionalValue(elements.xColumn),
        y_column: getOptionalValue(elements.yColumn),
        group_column: getOptionalValue(
          elements.groupColumn,
        ),
        target_column: getOptionalValue(elements.target),
        sample_rows: 300,
        max_categories: 15,
      },
    );

    renderCustomVisualizationMeta(
      elements.customMeta,
      specification,
    );

    renderSingleEChart(
      elements.customChart,
      specification,
    );
  } catch (error) {
    elements.customMeta.innerHTML = `
      <p class="empty-state">
        Failed to build custom chart:
        ${error.message}
      </p>
    `;
  }
}


function getOptionalValue(selectElement) {
  return selectElement.value || null;
}


function dispatchErrorToast(message) {
  window.dispatchEvent(
    new CustomEvent("dataagent:toast", {
      detail: {
        type: "error",
        message,
      },
    }),
  );
}
