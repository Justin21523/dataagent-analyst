import {
  analyzeModelExplainabilityApi,
  generateExplainabilityInsightApi,
} from "../api/client.js";
import {
  renderExplainabilityCharts,
} from "../components/modelExplainabilityCharts.js";
import {
  populateExplainabilityModels,
  renderExplainabilityErrorSamples,
  renderExplainabilityInsight,
  renderExplainabilityOverview,
  renderExplainabilityStatus,
  renderExplainabilityWarnings,
} from "../components/modelExplainabilityView.js";


let initialized = false;
let currentDatasetId = null;
let currentModels = [];
let currentResult = null;
let elements = null;


export function initModelExplainability() {
  if (initialized) {
    return;
  }

  elements = {
    modelSelect: document.querySelector(
      "#explainabilityModelSelect",
    ),
    sampleSize: document.querySelector(
      "#explainabilitySampleSizeInput",
    ),
    backgroundSize: document.querySelector(
      "#explainabilityBackgroundSizeInput",
    ),
    permutationRepeats: document.querySelector(
      "#explainabilityPermutationRepeatsInput",
    ),
    localRow: document.querySelector(
      "#explainabilityLocalRowInput",
    ),
    positiveClass: document.querySelector(
      "#explainabilityPositiveClassInput",
    ),
    includePermutation: document.querySelector(
      "#includePermutationCheckbox",
    ),
    includeShap: document.querySelector(
      "#includeShapCheckbox",
    ),
    runButton: document.querySelector(
      "#runExplainabilityButton",
    ),
    insightButton: document.querySelector(
      "#generateExplainabilityInsightButton",
    ),
    status: document.querySelector(
      "#explainabilityStatus",
    ),
    overview: document.querySelector(
      "#explainabilityOverview",
    ),
    warnings: document.querySelector(
      "#explainabilityWarnings",
    ),
    insightOutput: document.querySelector(
      "#explainabilityInsightOutput",
    ),
    primaryCurve: document.querySelector(
      "#explainabilityPrimaryCurveChart",
    ),
    secondaryCurve: document.querySelector(
      "#explainabilitySecondaryCurveChart",
    ),
    permutation: document.querySelector(
      "#explainabilityPermutationChart",
    ),
    shapImportance: document.querySelector(
      "#explainabilityShapImportanceChart",
    ),
    shapBeeswarm: document.querySelector(
      "#explainabilityShapBeeswarmChart",
    ),
    local: document.querySelector(
      "#explainabilityLocalChart",
    ),
    errorSamples: document.querySelector(
      "#explainabilityErrorSamples",
    ),
  };

  const missingElement = Object.entries(
    elements,
  ).find(([, element]) => !element);

  if (missingElement) {
    throw new Error(
      `Explainability element not found: ${
        missingElement[0]
      }`,
    );
  }

  elements.runButton.addEventListener(
    "click",
    runExplainability,
  );

  elements.insightButton.addEventListener(
    "click",
    generateInsight,
  );

  window.addEventListener(
    "dataagent:models-loaded",
    (event) => {
      currentDatasetId = (
        event.detail?.datasetId || null
      );

      currentModels = (
        event.detail?.models || []
      );

      populateExplainabilityModels(
        elements.modelSelect,
        currentModels,
      );
    },
  );

  window.addEventListener(
    "dataagent:ml-experiment-completed",
    (event) => {
      if (event.detail?.datasetId) {
        currentDatasetId = (
          event.detail.datasetId
        );
      }
    },
  );

  initialized = true;
}


async function runExplainability() {
  const modelId = elements.modelSelect.value;

  if (!modelId) {
    renderExplainabilityStatus(
      elements.status,
      {
        status: "error",
        title: "No Model Selected",
        message: "Train or select a saved model first.",
      },
    );
    return;
  }

  renderExplainabilityStatus(
    elements.status,
    {
      status: "running",
      title: "Analyzing Model",
      message: (
        "Computing diagnostics, permutation importance, "
        + "SHAP, and local explanation."
      ),
    },
  );

  try {
    const result = (
      await analyzeModelExplainabilityApi(
        modelId,
        {
          sample_size: Number(
            elements.sampleSize.value,
          ),
          background_size: Number(
            elements.backgroundSize.value,
          ),
          permutation_repeats: Number(
            elements.permutationRepeats.value,
          ),
          local_row_position: Number(
            elements.localRow.value,
          ),
          positive_class: (
            elements.positiveClass.value.trim()
            || null
          ),
          include_permutation: (
            elements.includePermutation.checked
          ),
          include_shap: (
            elements.includeShap.checked
          ),
          force_recompute: false,
        },
      )
    );

    currentResult = result;

    renderExplainabilityOverview(
      elements.overview,
      result,
    );

    renderExplainabilityWarnings(
      elements.warnings,
      result.warnings,
    );

    renderExplainabilityErrorSamples(
      elements.errorSamples,
      result,
    );

    renderExplainabilityCharts(
      {
        primaryCurve: elements.primaryCurve,
        secondaryCurve: (
          elements.secondaryCurve
        ),
        permutation: elements.permutation,
        shapImportance: (
          elements.shapImportance
        ),
        shapBeeswarm: elements.shapBeeswarm,
        local: elements.local,
      },
      result,
    );

    renderExplainabilityStatus(
      elements.status,
      {
        status: "success",
        title: "Explainability Ready",
        message: [
          `${result.holdout.row_count} holdout rows`,
          result.shap.available
            ? result.shap.explainer_type
            : "SHAP unavailable",
          result.cache_hit
            ? "cache hit"
            : "new analysis",
        ].join(" · "),
      },
    );

    window.dispatchEvent(
      new CustomEvent(
        "dataagent:model-explainability-completed",
        {
          detail: {
            modelId,
          },
        },
      ),
    );
  } catch (error) {
    renderExplainabilityStatus(
      elements.status,
      {
        status: "error",
        title: "Explainability Failed",
        message: error.message,
      },
    );
  }
}


async function generateInsight() {
  if (
    !currentDatasetId
    || !elements.modelSelect.value
  ) {
    return;
  }

  elements.insightOutput.innerHTML = `
    <p class="empty-state">
      Generating model interpretation...
    </p>
  `;

  try {
    const insight = (
      await generateExplainabilityInsightApi(
        currentDatasetId,
        elements.modelSelect.value,
        (
          "Explain model reliability, important drivers, "
          + "error risks, limitations, and next experiments."
        ),
      )
    );

    renderExplainabilityInsight(
      elements.insightOutput,
      insight,
    );
  } catch (error) {
    elements.insightOutput.innerHTML = `
      <p class="empty-state">
        Failed to generate interpretation:
        ${error.message}
      </p>
    `;
  }
}
