import { escapeHtml } from "../utils/format.js";


const chartInstances = new Map();

const resizeObserver = new ResizeObserver((entries) => {
  entries.forEach((entry) => {
    const chartKey = entry.target.dataset.echartKey;
    const chart = chartInstances.get(chartKey);

    if (chart) {
      chart.resize();
    }
  });
});


export function renderEChartsGrid(container, specifications) {
  disposeChartsByPrefix("auto:");

  if (!window.echarts) {
    container.innerHTML = `
      <p class="empty-state">
        Apache ECharts is not available.
        Run scripts/install_frontend_vendor.sh.
      </p>
    `;
    return;
  }

  if (!specifications.length) {
    container.innerHTML = `
      <p class="empty-state">
        No analytical charts are available.
      </p>
    `;
    return;
  }

  container.innerHTML = "";

  specifications.forEach((specification) => {
    const card = createChartCard(specification);
    const canvas = card.querySelector(".echart-canvas");

    container.appendChild(card);

    initializeChart(
      canvas,
      specification,
      `auto:${specification.id}`,
    );
  });
}


export function renderSingleEChart(
  container,
  specification,
) {
  disposeChart("custom");

  if (!window.echarts) {
    container.innerHTML = `
      <p class="empty-state">
        Apache ECharts is not available.
      </p>
    `;
    return;
  }

  container.innerHTML = "";

  initializeChart(
    container,
    specification,
    "custom",
  );
}


export function disposeAllECharts() {
  chartInstances.forEach((chart, chartKey) => {
    resizeObserver.unobserve(chart.getDom());
    chart.dispose();
    chartInstances.delete(chartKey);
  });
}


function createChartCard(specification) {
  const card = document.createElement("article");

  card.className = "echart-card";

  card.innerHTML = `
    <div class="echart-card-header">
      <div>
        <span class="chart-family">
          ${escapeHtml(specification.chart_family)}
        </span>
        <h4>${escapeHtml(specification.title)}</h4>
      </div>

      <span class="badge">
        ${escapeHtml(specification.chart_type)}
      </span>
    </div>

    <p class="echart-description">
      ${escapeHtml(specification.description)}
    </p>

    <div class="echart-canvas"></div>

    <div class="echart-insight">
      <strong>Insight</strong>
      <p>${escapeHtml(specification.insight)}</p>
    </div>
  `;

  return card;
}


function initializeChart(
  element,
  specification,
  chartKey,
) {
  element.dataset.echartKey = chartKey;

  const chart = window.echarts.init(
    element,
    null,
    {
      renderer: "canvas",
    },
  );

  chart.setOption(
    buildOption(specification),
    {
      notMerge: true,
      lazyUpdate: false,
    },
  );

  chartInstances.set(chartKey, chart);
  resizeObserver.observe(element);
}


function buildOption(specification) {
  switch (specification.chart_type) {
    case "missing_heatmap":
      return buildMissingHeatmapOption(specification);

    case "correlation_heatmap":
      return buildCorrelationHeatmapOption(specification);

    case "boxplot":
    case "target_boxplot":
      return buildBoxplotOption(specification);

    case "scatter":
    case "feature_target_scatter":
      return buildScatterOption(specification);

    case "line":
      return buildLineOption(specification);

    case "target_stacked_bar":
      return buildStackedBarOption(specification);

    case "missing_bar":
    case "histogram":
    case "category_bar":
    case "target_distribution":
      return buildDatasetBarOption(specification);

    default:
      return buildUnsupportedOption(specification);
  }
}


function buildCommonOption(specification) {
  return {
    animationDuration: 550,
    animationEasing: "cubicOut",
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
          name: specification.id,
          pixelRatio: 2,
        },
      },
    },
    grid: {
      left: 60,
      right: 30,
      top: 45,
      bottom: 70,
      containLabel: true,
    },
  };
}


function buildDatasetBarOption(specification) {
  const source = specification.data.source || [];
  const rowCount = Math.max(source.length - 1, 0);

  return {
    ...buildCommonOption(specification),
    dataset: {
      source,
    },
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "shadow",
      },
      confine: true,
    },
    xAxis: {
      type: "category",
      axisLabel: {
        interval: 0,
        rotate: rowCount > 8 ? 30 : 0,
      },
    },
    yAxis: {
      type: "value",
      minInterval: 1,
    },
    dataZoom: rowCount > 12
      ? [
          {
            type: "inside",
          },
          {
            type: "slider",
            bottom: 10,
            height: 18,
          },
        ]
      : [],
    series: [
      {
        type: "bar",
        encode: {
          x: specification.config.x_dimension,
          y: specification.config.y_dimension,
          tooltip: specification.data.dimensions,
        },
        barMaxWidth: 46,
        emphasis: {
          focus: "series",
        },
      },
    ],
  };
}


function buildMissingHeatmapOption(specification) {
  const {
    x_labels: xLabels,
    y_labels: yLabels,
    values,
  } = specification.data;

  return {
    ...buildCommonOption(specification),
    tooltip: {
      position: "top",
      formatter: (params) => {
        const value = params.value;

        return [
          `Column: ${xLabels[value[0]]}`,
          `Sample row: ${yLabels[value[1]]}`,
          `Missing: ${value[2] === 1 ? "Yes" : "No"}`,
        ].join("<br />");
      },
    },
    grid: {
      left: 75,
      right: 30,
      top: 35,
      bottom: 85,
      containLabel: true,
    },
    xAxis: {
      type: "category",
      data: xLabels,
      splitArea: {
        show: true,
      },
      axisLabel: {
        interval: 0,
        rotate: 35,
      },
    },
    yAxis: {
      type: "category",
      data: yLabels,
      splitArea: {
        show: true,
      },
    },
    visualMap: {
      min: 0,
      max: 1,
      calculable: false,
      orient: "horizontal",
      left: "center",
      bottom: 10,
      text: [
        "Missing",
        "Present",
      ],
    },
    dataZoom: yLabels.length > 40
      ? [
          {
            type: "inside",
            yAxisIndex: 0,
          },
          {
            type: "slider",
            yAxisIndex: 0,
            right: 5,
            width: 16,
          },
        ]
      : [],
    series: [
      {
        type: "heatmap",
        data: values,
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: "rgba(15, 23, 42, 0.35)",
          },
        },
      },
    ],
  };
}


function buildCorrelationHeatmapOption(specification) {
  const {
    x_labels: xLabels,
    y_labels: yLabels,
    values,
  } = specification.data;

  return {
    ...buildCommonOption(specification),
    tooltip: {
      position: "top",
      formatter: (params) => {
        const value = params.value;
        const correlation = value[2];

        return [
          `${yLabels[value[1]]} × ${xLabels[value[0]]}`,
          `Correlation: ${
            correlation === null
              ? "N/A"
              : Number(correlation).toFixed(4)
          }`,
        ].join("<br />");
      },
    },
    grid: {
      left: 80,
      right: 45,
      top: 30,
      bottom: 90,
      containLabel: true,
    },
    xAxis: {
      type: "category",
      data: xLabels,
      splitArea: {
        show: true,
      },
      axisLabel: {
        interval: 0,
        rotate: 35,
      },
    },
    yAxis: {
      type: "category",
      data: yLabels,
      splitArea: {
        show: true,
      },
    },
    visualMap: {
      min: specification.config.minimum ?? -1,
      max: specification.config.maximum ?? 1,
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: 10,
      precision: 2,
    },
    series: [
      {
        type: "heatmap",
        data: values,
        label: {
          show: xLabels.length <= 8,
          formatter: (params) => {
            const value = params.value[2];

            return value === null
              ? ""
              : Number(value).toFixed(2);
          },
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: "rgba(15, 23, 42, 0.35)",
          },
        },
      },
    ],
  };
}


function buildBoxplotOption(specification) {
  const {
    categories,
    boxes,
    outliers,
  } = specification.data;

  return {
    ...buildCommonOption(specification),
    tooltip: {
      trigger: "item",
      confine: true,
    },
    xAxis: {
      type: "category",
      data: categories,
      boundaryGap: true,
      axisLabel: {
        interval: 0,
        rotate: categories.length > 5 ? 25 : 0,
      },
    },
    yAxis: {
      type: "value",
      splitArea: {
        show: true,
      },
    },
    series: [
      {
        name: "Boxplot",
        type: "boxplot",
        data: boxes,
        tooltip: {
          formatter: (params) => {
            const values = params.value.slice(1);

            return [
              params.name,
              `Lower whisker: ${values[0]}`,
              `Q1: ${values[1]}`,
              `Median: ${values[2]}`,
              `Q3: ${values[3]}`,
              `Upper whisker: ${values[4]}`,
            ].join("<br />");
          },
        },
      },
      {
        name: "Outlier",
        type: "scatter",
        data: outliers,
        symbolSize: 7,
      },
    ],
  };
}


function buildScatterOption(specification) {
  const {
    dimensions,
    source,
  } = specification.data;

  return {
    ...buildCommonOption(specification),
    dataset: {
      source,
    },
    tooltip: {
      trigger: "item",
      confine: true,
    },
    xAxis: {
      type: "value",
      name: specification.config.x_dimension,
      nameLocation: "middle",
      nameGap: 42,
      scale: true,
    },
    yAxis: {
      type: "value",
      name: specification.config.y_dimension,
      nameLocation: "middle",
      nameGap: 52,
      scale: true,
    },
    dataZoom: [
      {
        type: "inside",
      },
      {
        type: "slider",
        bottom: 10,
        height: 18,
      },
    ],
    series: [
      {
        type: "scatter",
        encode: {
          x: specification.config.x_dimension,
          y: specification.config.y_dimension,
          tooltip: dimensions,
        },
        symbolSize: 9,
        large: source.length > 1000,
        emphasis: {
          focus: "series",
        },
      },
    ],
  };
}


function buildLineOption(specification) {
  return {
    ...buildCommonOption(specification),
    dataset: {
      source: specification.data.source,
    },
    tooltip: {
      trigger: "axis",
      confine: true,
    },
    xAxis: {
      type: "time",
      name: specification.config.x_dimension,
    },
    yAxis: {
      type: "value",
      name: specification.config.y_dimension,
      scale: true,
    },
    dataZoom: [
      {
        type: "inside",
      },
      {
        type: "slider",
        bottom: 10,
        height: 18,
      },
    ],
    series: [
      {
        type: "line",
        encode: {
          x: specification.config.x_dimension,
          y: specification.config.y_dimension,
        },
        showSymbol: false,
        smooth: true,
        sampling: "lttb",
      },
    ],
  };
}


function buildStackedBarOption(specification) {
  const {
    categories,
    series,
  } = specification.data;

  return {
    ...buildCommonOption(specification),
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "shadow",
      },
      valueFormatter: (value) => {
        return `${(Number(value) * 100).toFixed(1)}%`;
      },
    },
    legend: {
      type: "scroll",
      top: 0,
    },
    xAxis: {
      type: "category",
      data: categories,
    },
    yAxis: {
      type: "value",
      max: 1,
      axisLabel: {
        formatter: (value) => `${value * 100}%`,
      },
    },
    series: series.map((item) => {
      return {
        name: item.name,
        type: "bar",
        stack: "total",
        data: item.data,
        emphasis: {
          focus: "series",
        },
      };
    }),
  };
}


function buildUnsupportedOption(specification) {
  return {
    ...buildCommonOption(specification),
    title: {
      text: "Unsupported visualization",
      subtext: specification.chart_type,
      left: "center",
      top: "middle",
    },
  };
}


function disposeChartsByPrefix(prefix) {
  [...chartInstances.entries()].forEach(
    ([chartKey, chart]) => {
      if (!chartKey.startsWith(prefix)) {
        return;
      }

      resizeObserver.unobserve(chart.getDom());
      chart.dispose();
      chartInstances.delete(chartKey);
    },
  );
}


function disposeChart(chartKey) {
  const chart = chartInstances.get(chartKey);

  if (!chart) {
    return;
  }

  resizeObserver.unobserve(chart.getDom());
  chart.dispose();
  chartInstances.delete(chartKey);
}
