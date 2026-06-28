const charts = new Map();


export function renderExplainabilityCharts(
  elements,
  result,
) {
  disposeExplainabilityCharts();

  if (!window.echarts) {
    return;
  }

  if (result.classification) {
    renderRocChart(
      elements.primaryCurve,
      result.classification,
    );

    renderPrecisionRecallChart(
      elements.secondaryCurve,
      result.classification,
    );
  } else if (result.regression) {
    renderPredictedActualChart(
      elements.primaryCurve,
      result.regression,
    );

    renderResidualChart(
      elements.secondaryCurve,
      result.regression,
    );
  }

  renderImportanceChart(
    elements.permutation,
    result.permutation_importance,
    "Permutation Importance",
    "permutation",
  );

  renderImportanceChart(
    elements.shapImportance,
    result.shap.source_feature_importance,
    "Mean |SHAP Value|",
    "shap-importance",
  );

  renderShapBeeswarm(
    elements.shapBeeswarm,
    result.shap,
  );

  renderLocalWaterfall(
    elements.local,
    result.shap,
  );
}


export function disposeExplainabilityCharts() {
  charts.forEach((chart) => {
    chart.dispose();
  });

  charts.clear();
}


function renderRocChart(
  element,
  classification,
) {
  const chart = createChart(
    element,
    "roc",
  );

  const series = classification.curves.map(
    (curve) => {
      return {
        name: [
          curve.class_label,
          `AUC ${formatNumber(curve.roc_auc)}`,
        ].join(" · "),
        type: "line",
        showSymbol: false,
        smooth: true,
        data: curve.roc_points.map((point) => {
          return [
            point.x,
            point.y,
          ];
        }),
      };
    },
  );

  series.push({
    name: "Random",
    type: "line",
    showSymbol: false,
    lineStyle: {
      type: "dashed",
    },
    data: [
      [0, 0],
      [1, 1],
    ],
  });

  chart.setOption({
    ...commonOption(),
    legend: {
      type: "scroll",
      top: 0,
    },
    xAxis: {
      type: "value",
      name: "False Positive Rate",
      min: 0,
      max: 1,
    },
    yAxis: {
      type: "value",
      name: "True Positive Rate",
      min: 0,
      max: 1,
    },
    series,
  });
}


function renderPrecisionRecallChart(
  element,
  classification,
) {
  const chart = createChart(
    element,
    "precision-recall",
  );

  chart.setOption({
    ...commonOption(),
    legend: {
      type: "scroll",
      top: 0,
    },
    xAxis: {
      type: "value",
      name: "Recall",
      min: 0,
      max: 1,
    },
    yAxis: {
      type: "value",
      name: "Precision",
      min: 0,
      max: 1,
    },
    series: classification.curves.map(
      (curve) => {
        return {
          name: [
            curve.class_label,
            `AP ${formatNumber(
              curve.average_precision,
            )}`,
          ].join(" · "),
          type: "line",
          showSymbol: false,
          smooth: true,
          data: (
            curve.precision_recall_points.map(
              (point) => {
                return [
                  point.x,
                  point.y,
                ];
              },
            )
          ),
        };
      },
    ),
  });
}


function renderPredictedActualChart(
  element,
  regression,
) {
  const chart = createChart(
    element,
    "predicted-actual",
  );

  const values = regression.points.flatMap(
    (point) => [
      point.actual,
      point.predicted,
    ],
  );

  const minimum = Math.min(...values);
  const maximum = Math.max(...values);

  chart.setOption({
    ...commonOption(),
    xAxis: {
      type: "value",
      name: "Actual",
      scale: true,
    },
    yAxis: {
      type: "value",
      name: "Predicted",
      scale: true,
    },
    series: [
      {
        name: "Prediction",
        type: "scatter",
        symbolSize: 9,
        data: regression.points.map(
          (point) => {
            return [
              point.actual,
              point.predicted,
            ];
          },
        ),
      },
      {
        name: "Ideal",
        type: "line",
        showSymbol: false,
        lineStyle: {
          type: "dashed",
        },
        data: [
          [minimum, minimum],
          [maximum, maximum],
        ],
      },
    ],
  });
}


function renderResidualChart(
  element,
  regression,
) {
  const chart = createChart(
    element,
    "residual",
  );

  chart.setOption({
    ...commonOption(),
    xAxis: {
      type: "value",
      name: "Predicted",
      scale: true,
    },
    yAxis: {
      type: "value",
      name: "Residual",
      scale: true,
    },
    series: [
      {
        name: "Residual",
        type: "scatter",
        symbolSize: 9,
        data: regression.points.map(
          (point) => {
            return [
              point.predicted,
              point.residual,
            ];
          },
        ),
      },
      {
        name: "Zero Residual",
        type: "line",
        showSymbol: false,
        lineStyle: {
          type: "dashed",
        },
        data: [
          [
            Math.min(
              ...regression.points.map(
                (point) => point.predicted,
              ),
            ),
            0,
          ],
          [
            Math.max(
              ...regression.points.map(
                (point) => point.predicted,
              ),
            ),
            0,
          ],
        ],
      },
    ],
  });
}


function renderImportanceChart(
  element,
  items,
  valueLabel,
  chartKey,
) {
  if (!items.length) {
    element.innerHTML = `
      <p class="empty-state">
        No importance result is available.
      </p>
    `;
    return;
  }

  const visibleItems = items
    .slice(0, 15)
    .reverse();

  const chart = createChart(
    element,
    chartKey,
  );

  chart.setOption({
    ...commonOption(),
    grid: {
      left: 150,
      right: 35,
      top: 25,
      bottom: 45,
      containLabel: true,
    },
    xAxis: {
      type: "value",
      name: valueLabel,
      scale: true,
    },
    yAxis: {
      type: "category",
      data: visibleItems.map(
        (item) => item.feature,
      ),
    },
    series: [
      {
        type: "bar",
        data: visibleItems.map(
          (item) => {
            return {
              value: item.importance_mean,
              itemStyle: {
                opacity: (
                  item.importance_mean < 0
                    ? 0.55
                    : 1
                ),
              },
            };
          },
        ),
        barMaxWidth: 24,
      },
    ],
  });
}


function renderShapBeeswarm(
  element,
  shap,
) {
  if (
    !shap.available
    || !shap.beeswarm_points.length
  ) {
    element.innerHTML = `
      <p class="empty-state">
        No SHAP beeswarm result is available.
      </p>
    `;
    return;
  }

  const topFeatures = (
    shap.transformed_feature_importance
    .slice(0, 15)
    .map((item) => item.feature)
  );

  const featureIndex = new Map(
    topFeatures.map(
      (feature, index) => [
        feature,
        index,
      ],
    ),
  );

  const values = shap.beeswarm_points
    .filter((point) => {
      return featureIndex.has(point.feature);
    })
    .map((point) => {
      const jitter = (
        (point.row_position % 9 - 4) * 0.035
      );

      return [
        point.shap_value,
        featureIndex.get(point.feature) + jitter,
        point.feature_value,
        point.feature,
      ];
    });

  const chart = createChart(
    element,
    "shap-beeswarm",
  );

  chart.setOption({
    ...commonOption(),
    tooltip: {
      formatter: (params) => {
        const value = params.value;

        return [
          value[3],
          `SHAP: ${formatNumber(value[0])}`,
          `Feature value: ${
            formatNumber(value[2])
          }`,
        ].join("<br />");
      },
    },
    grid: {
      left: 170,
      right: 35,
      top: 30,
      bottom: 65,
      containLabel: true,
    },
    xAxis: {
      type: "value",
      name: "SHAP Value",
      scale: true,
    },
    yAxis: {
      type: "value",
      min: -0.6,
      max: topFeatures.length - 0.4,
      interval: 1,
      axisLabel: {
        formatter: (value) => {
          return topFeatures[
            Math.round(value)
          ] || "";
        },
      },
    },
    visualMap: {
      dimension: 2,
      min: Math.min(
        ...values
          .map((value) => value[2])
          .filter(Number.isFinite),
        0,
      ),
      max: Math.max(
        ...values
          .map((value) => value[2])
          .filter(Number.isFinite),
        1,
      ),
      orient: "horizontal",
      left: "center",
      bottom: 5,
      text: [
        "High Value",
        "Low Value",
      ],
      calculable: true,
    },
    series: [
      {
        type: "scatter",
        data: values,
        symbolSize: 7,
      },
    ],
  });
}


function renderLocalWaterfall(
  element,
  shap,
) {
  const local = shap.local_explanation;

  if (
    !shap.available
    || !local
    || !local.contributions.length
  ) {
    element.innerHTML = `
      <p class="empty-state">
        No local SHAP explanation is available.
      </p>
    `;
    return;
  }

  const contributions = [
    ...local.contributions,
  ]
    .slice(0, 15)
    .reverse();

  const chart = createChart(
    element,
    "local-waterfall",
  );

  chart.setOption({
    ...commonOption(),
    tooltip: {
      formatter: (params) => {
        const item = contributions[
          params.dataIndex
        ];

        return [
          item.feature,
          `Source: ${item.source_feature}`,
          `Value: ${formatNumber(
            item.feature_value,
          )}`,
          `SHAP: ${formatNumber(
            item.shap_value,
          )}`,
        ].join("<br />");
      },
    },
    grid: {
      left: 170,
      right: 40,
      top: 30,
      bottom: 50,
      containLabel: true,
    },
    xAxis: {
      type: "value",
      name: "SHAP Contribution",
      scale: true,
    },
    yAxis: {
      type: "category",
      data: contributions.map(
        (item) => item.feature,
      ),
    },
    series: [
      {
        type: "bar",
        data: contributions.map(
          (item) => {
            return {
              value: item.shap_value,
              itemStyle: {
                color: (
                  item.shap_value >= 0
                    ? "#ef4444"
                    : "#2563eb"
                ),
              },
            };
          },
        ),
      },
    ],
  });
}


function commonOption() {
  return {
    animationDuration: 500,
    aria: {
      enabled: true,
    },
    tooltip: {
      trigger: "item",
      confine: true,
    },
    toolbox: {
      right: 8,
      feature: {
        restore: {},
        saveAsImage: {
          pixelRatio: 2,
        },
      },
    },
    grid: {
      left: 65,
      right: 30,
      top: 45,
      bottom: 60,
      containLabel: true,
    },
  };
}


function createChart(
  element,
  chartKey,
) {
  const oldChart = charts.get(chartKey);

  if (oldChart) {
    oldChart.dispose();
  }

  element.innerHTML = "";

  const chart = window.echarts.init(element);

  charts.set(chartKey, chart);

  return chart;
}


function formatNumber(value) {
  if (
    value === null
    || value === undefined
    || Number.isNaN(Number(value))
  ) {
    return "N/A";
  }

  return Number(value).toFixed(4);
}
