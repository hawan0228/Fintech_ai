const state = {
  health: null,
  models: [],
  overview: null,
  upload: null,
  run: null,
};

const elements = {
  heroMetrics: document.querySelector("#heroMetrics"),
  requirementCards: document.querySelector("#requirementCards"),
  projectDatasetSnapshot: document.querySelector("#projectDatasetSnapshot"),
  projectYearBars: document.querySelector("#projectYearBars"),
  timelineSteps: document.querySelector("#timelineSteps"),
  validationChecks: document.querySelector("#validationChecks"),
  leaderboardBars: document.querySelector("#leaderboardBars"),
  classificationOverviewTable: document.querySelector("#classificationOverviewTable"),
  bestModelTable: document.querySelector("#bestModelTable"),
  externalTable: document.querySelector("#externalTable"),
  figureGallery: document.querySelector("#figureGallery"),
  downloadLinks: document.querySelector("#downloadLinks"),
  latestDemoSummary: document.querySelector("#latestDemoSummary"),
  latestDemoTable: document.querySelector("#latestDemoTable"),
  uploadForm: document.querySelector("#uploadForm"),
  runForm: document.querySelector("#runForm"),
  datasetFile: document.querySelector("#datasetFile"),
  uploadButton: document.querySelector("#uploadButton"),
  runButton: document.querySelector("#runButton"),
  modelSelect: document.querySelector("#modelSelect"),
  topKRange: document.querySelector("#topKRange"),
  topKInput: document.querySelector("#topKInput"),
  statusCard: document.querySelector("#statusCard"),
  statusMessage: document.querySelector("#statusMessage"),
  healthSummary: document.querySelector("#healthSummary"),
  summaryCards: document.querySelector("#summaryCards"),
  uploadDatasetSnapshot: document.querySelector("#uploadDatasetSnapshot"),
  uploadYearBars: document.querySelector("#uploadYearBars"),
  featureCoverage: document.querySelector("#featureCoverage"),
  performanceSummary: document.querySelector("#performanceSummary"),
  curveChart: document.querySelector("#curveChart"),
  selectedTable: document.querySelector("#selectedTable"),
  predictionTable: document.querySelector("#predictionTable"),
  metricTable: document.querySelector("#metricTable"),
  classificationDetailTable: document.querySelector("#classificationDetailTable"),
  artifactPaths: document.querySelector("#artifactPaths"),
};

function setStatus(message, tone = "info") {
  elements.statusMessage.textContent = message;
  elements.statusCard.classList.remove("is-error", "is-success");

  if (tone === "error") {
    elements.statusCard.classList.add("is-error");
  } else if (tone === "success") {
    elements.statusCard.classList.add("is-success");
  }
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }

  return Number(value).toLocaleString("zh-TW", {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  });
}

function formatPercent(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }

  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatBoolean(value) {
  return value ? "是" : "否";
}

function formatCheckStatus(status) {
  const labels = {
    pass: "通過",
    fail: "失敗",
    warn: "警示",
  };

  return labels[status] || status;
}

function formatWeightMethod(value) {
  const labels = {
    equal: "等權",
    score: "分數加權",
  };

  return labels[value] || value;
}

function getModelDisplayName(modelName) {
  const matched = state.models.find((model) => model.model_name === modelName);
  return matched?.label || modelName;
}

function clampTopK(value) {
  const parsed = Number(value);
  if (Number.isNaN(parsed)) return 10;
  return Math.max(1, Math.min(200, Math.round(parsed)));
}

async function fileToBase64(file) {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";

  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }

  return btoa(binary);
}

function syncTopK(value) {
  const clamped = clampTopK(value);
  elements.topKInput.value = clamped;
  elements.topKRange.value = Math.max(1, Math.min(30, clamped));
}

function titleizeColumn(column) {
  const labels = {
    model_name: "模型",
    model_label: "模型",
    task_group: "任務",
    top_k: "Top-K",
    weight_method: "權重方式",
    weight: "權重",
    net_annualized_return: "淨年化報酬",
    net_cumulative_return: "淨累積報酬",
    net_sharpe_ratio: "淨 Sharpe Ratio",
    net_win_rate: "淨勝率",
    net_maximum_drawdown: "淨最大回撤",
    accuracy: "準確率",
    precision_label_1: "Precision(1)",
    recall_label_1: "Recall(1)",
    f1_label_1: "F1(1)",
    year: "年度",
    stock_id: "股票代號",
    stock_name: "股票名稱",
    rank: "名次",
    score_label_1: "分數 P(1)",
    actual_label: "實際 Label",
    predicted_label: "預測 Label",
    actual_return: "實際報酬",
    predicted_return: "預測報酬",
    split_id: "切分編號",
    train_years: "訓練年度",
    test_years: "測試年度",
    gross_return: "毛報酬",
    net_return: "淨報酬",
    cumulative_net_return: "累積淨報酬",
    n_years: "年數",
    n_selected: "入選數",
  };

  return labels[column] || column;
}

function formatCell(value, key = "") {
  if (value === null || value === undefined || value === "") {
    return "-";
  }

  if (key === "weight_method") {
    return formatWeightMethod(String(value));
  }

  if (typeof value === "number") {
    if (
      key.includes("return") ||
      key.includes("accuracy") ||
      key.includes("precision") ||
      key.includes("recall") ||
      key.includes("f1") ||
      key.includes("score") ||
      key.includes("weight") ||
      key.includes("ratio")
    ) {
      return formatNumber(value, 6);
    }
    return formatNumber(value, 4);
  }

  return String(value);
}

function renderTable(container, records, options = {}) {
  const { limit = 50, emptyText = "沒有資料。", preferredColumns = null } = options;

  if (!records || records.length === 0) {
    container.classList.add("empty-state");
    container.innerHTML = emptyText;
    return;
  }

  container.classList.remove("empty-state");
  const rows = records.slice(0, limit);
  const columns = preferredColumns || Object.keys(rows[0]);

  const thead = `
    <thead>
      <tr>${columns.map((column) => `<th>${titleizeColumn(column)}</th>`).join("")}</tr>
    </thead>
  `;

  const tbody = `
    <tbody>
      ${rows
        .map(
          (row) => `
            <tr>
              ${columns.map((column) => `<td>${formatCell(row[column], column)}</td>`).join("")}
            </tr>
          `
        )
        .join("")}
    </tbody>
  `;

  const note = rows.length < records.length
    ? `<div class="table-note">顯示前 ${rows.length} / ${records.length} 筆</div>`
    : `<div class="table-note">共 ${records.length} 筆</div>`;

  container.innerHTML = `${note}<table>${thead}${tbody}</table>`;
}

function renderBars(container, rows, config = {}) {
  const {
    labelKey = "label",
    valueKey = "value",
    metaBuilder = (row) => formatNumber(row[valueKey], 2),
    scale = null,
  } = config;

  if (!rows || rows.length === 0) {
    container.classList.add("empty-state");
    container.textContent = "沒有資料可視化。";
    return;
  }

  container.classList.remove("empty-state");
  const maxValue = scale || Math.max(...rows.map((row) => Number(row[valueKey]) || 0), 1);

  container.innerHTML = rows
    .map((row) => {
      const value = Number(row[valueKey]) || 0;
      const width = Math.max(3, (value / maxValue) * 100);

      return `
        <div class="bar-row">
          <div class="bar-label">
            <span>${row[labelKey]}</span>
            <span>${metaBuilder(row)}</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${width}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderKeyGrid(container, rows, emptyText = "沒有摘要資料。") {
  if (!rows || rows.length === 0) {
    container.classList.add("empty-state");
    container.textContent = emptyText;
    return;
  }

  container.classList.remove("empty-state");
  container.innerHTML = rows
    .map(
      (row) => `
        <div class="kv">
          <strong>${row.label}</strong>
          <span>${row.value}</span>
        </div>
      `
    )
    .join("");
}

function renderRequirements() {
  const cards = state.overview?.requirement_cards || [];

  elements.requirementCards.innerHTML = cards
    .map(
      (card) => `
        <article class="requirement-card">
          <span class="requirement-status">${card.status}</span>
          <div>
            <h3>${card.title}</h3>
            <p>${card.subtitle}</p>
          </div>
          <div class="metric-chip">
            <span>${card.metric_label}</span>
            <strong>${formatPercent(card.metric_value)}</strong>
          </div>
          <ul class="evidence-list">
            ${card.evidence.map((item) => `<li>${item}</li>`).join("")}
          </ul>
        </article>
      `
    )
    .join("");
}

function renderProjectDataset() {
  const dataset = state.overview?.dataset;
  if (!dataset) return;

  renderKeyGrid(elements.projectDatasetSnapshot, [
    { label: "資料筆數", value: formatNumber(dataset.rows, 0) },
    { label: "欄位數", value: formatNumber(dataset.columns, 0) },
    { label: "年度範圍", value: `${dataset.year_start} - ${dataset.year_end}` },
    { label: "特徵數", value: formatNumber(dataset.n_features, 0) },
    { label: "唯一股票代號數", value: formatNumber(dataset.n_unique_stock_ids, 0) },
    {
      label: "時序切分設計",
      value: `${state.overview.splits.next_year_split_count} + ${state.overview.splits.remaining_years_split_count}`,
    },
  ]);

  renderBars(elements.projectYearBars, dataset.rows_by_year, {
    labelKey: "year",
    valueKey: "n_rows",
    metaBuilder: (row) => `${formatNumber(row.n_rows, 0)} 筆`,
  });
}

function renderTimeline() {
  const steps = state.overview?.timeline || [];
  elements.timelineSteps.innerHTML = steps
    .map(
      (step) => `
        <div class="timeline-step">
          <div class="timeline-badge">${step.step}</div>
          <div class="timeline-card">
            <h4>${step.title}</h4>
            <p>${step.detail}</p>
          </div>
        </div>
      `
    )
    .join("");
}

function renderValidationChecks() {
  const checks = state.overview?.validation_checks || [];
  elements.validationChecks.innerHTML = checks
    .map(
      (check) => `
        <div class="validation-card ${check.status}">
          <span class="validation-pill">${formatCheckStatus(check.status)}</span>
          <strong>${check.name}</strong>
          <span>預期：${check.expected}</span>
          <span>實際：${check.actual}</span>
        </div>
      `
    )
    .join("");
}

function renderHeroMetrics() {
  const dataset = state.overview?.dataset;
  const bestModels = state.overview?.leaderboards?.best_models || [];
  const external = state.overview?.leaderboards?.best_external || [];
  const latestDemo = state.overview?.latest_demo;
  const bestInternal = bestModels.reduce((best, row) => {
    if (!best) return row;
    const bestValue = Number(best.net_annualized_return) || Number.NEGATIVE_INFINITY;
    const currentValue = Number(row.net_annualized_return) || Number.NEGATIVE_INFINITY;
    return currentValue > bestValue ? row : best;
  }, null);
  const bestExternal = external[0];
  const latestDemoPortfolio = latestDemo?.evaluation_summary?.portfolio;

  const items = [
    {
      label: "資料筆數",
      value: dataset ? `${formatNumber(dataset.rows, 0)} 筆` : "-",
    },
    {
      label: "內部最強模型",
      value: bestInternal
        ? `${bestInternal.model_label} | ${formatPercent(bestInternal.net_annualized_return)}`
        : "-",
    },
    {
      label: "外部資料重跑",
      value: bestExternal
        ? `${formatPercent(bestExternal.net_annualized_return)} | Top-${bestExternal.top_k}`
        : "-",
    },
    {
      label: "最新展示",
      value: latestDemoPortfolio
        ? formatPercent(latestDemoPortfolio.net_annualized_return)
        : "尚未產生",
    },
  ];

  elements.heroMetrics.innerHTML = items
    .map(
      (item) => `
        <div class="hero-metric">
          <span>${item.label}</span>
          <strong>${item.value}</strong>
        </div>
      `
    )
    .join("");
}

function renderHealth() {
  if (!state.health) return;

  const cards = [
    {
      label: "已清理資料",
      value: formatBoolean(state.health.cleaned_data_exists),
    },
    {
      label: "模型目錄",
      value: formatBoolean(state.health.saved_model_dir_exists),
    },
  ];

  Object.entries(state.health.saved_models || {}).forEach(([modelName, count]) => {
    cards.push({
      label: modelName,
      value: `${count} 個模型`,
    });
  });

  elements.healthSummary.innerHTML = cards
    .map(
      (card) => `
        <div class="mini-card">
          <strong>${card.label}</strong>
          <span>${card.value}</span>
        </div>
      `
    )
    .join("");
}

function renderLeaderboard() {
  const bestModels = state.overview?.leaderboards?.best_models || [];

  renderBars(elements.leaderboardBars, bestModels, {
    labelKey: "model_label",
    valueKey: "net_annualized_return",
    scale: Math.max(...bestModels.map((row) => Number(row.net_annualized_return) || 0), 0.01),
    metaBuilder: (row) => `${formatPercent(row.net_annualized_return)} | Top-${row.top_k} ${formatWeightMethod(row.weight_method)}`,
  });

  renderTable(elements.bestModelTable, bestModels, {
    limit: 10,
    preferredColumns: [
      "model_label",
      "task_group",
      "top_k",
      "weight_method",
      "net_annualized_return",
      "net_sharpe_ratio",
      "net_win_rate",
      "net_maximum_drawdown",
    ],
  });

  renderTable(elements.externalTable, state.overview?.leaderboards?.best_external || [], {
    limit: 5,
    preferredColumns: [
      "model_label",
      "top_k",
      "weight_method",
      "net_annualized_return",
      "net_sharpe_ratio",
      "net_win_rate",
      "net_maximum_drawdown",
    ],
  });

  renderTable(elements.classificationOverviewTable, state.overview?.leaderboards?.classification_overall || [], {
    limit: 10,
    preferredColumns: [
      "model_label",
      "accuracy",
      "precision_label_1",
      "recall_label_1",
      "f1_label_1",
    ],
  });
}

function renderGallery() {
  const figures = state.overview?.gallery || [];

  elements.figureGallery.innerHTML = figures
    .map(
      (figure) => `
        <article class="figure-card">
          ${figure.exists
            ? `<img src="${figure.url}" alt="${figure.title}" loading="lazy">`
            : `<div class="chart-surface empty-state">圖檔不存在</div>`}
          <div class="figure-body">
            <span class="figure-tag">${figure.section}</span>
            <h3>${figure.title}</h3>
            <p>${figure.caption}</p>
          </div>
        </article>
      `
    )
    .join("");
}

function renderDownloads() {
  const downloads = state.overview?.downloads || [];

  elements.downloadLinks.innerHTML = downloads
    .map(
      (item) => `
        <div class="download-item">
          <span>${item.label}</span>
          <a href="${item.url}" target="_blank" rel="noreferrer">開啟</a>
        </div>
      `
    )
    .join("");
}

function getActiveDemoResult() {
  return state.run || state.overview?.latest_demo || null;
}

function renderLatestDemoSnapshot() {
  const latestDemo = state.overview?.latest_demo;

  if (!latestDemo) {
    renderKeyGrid(elements.latestDemoSummary, [], "尚未找到展示輸出。");
    renderTable(elements.latestDemoTable, [], { emptyText: "尚未找到展示輸出。" });
    return;
  }

  const evaluation = latestDemo.evaluation_summary || {};
  const portfolio = evaluation.portfolio;

  renderKeyGrid(elements.latestDemoSummary, [
    { label: "模型", value: getModelDisplayName(latestDemo.request.model_name) },
    { label: "Top-K", value: latestDemo.request.top_k },
    { label: "預測筆數", value: formatNumber(evaluation.prediction_rows, 0) },
    { label: "入選筆數", value: formatNumber(evaluation.selected_rows, 0) },
    { label: "預測正類比例", value: formatPercent(evaluation.predicted_positive_rate) },
    {
      label: "淨年化報酬",
      value: portfolio ? formatPercent(portfolio.net_annualized_return) : "-",
    },
  ]);

  renderTable(elements.latestDemoTable, latestDemo.tables.selected_stocks || [], {
    limit: 8,
    preferredColumns: [
      "year",
      "rank",
      "stock_id",
      "stock_name",
      "score_label_1",
      "actual_return",
    ],
  });
}

function renderModelOptions() {
  if (!state.models.length) {
    elements.modelSelect.innerHTML = `<option value="random_forest">random_forest</option>`;
    return;
  }

  elements.modelSelect.innerHTML = state.models
    .map(
      (model) => `
        <option value="${model.model_name}">
          ${model.label} | ${model.task}
        </option>
      `
    )
    .join("");
}

function renderSummaryCards() {
  const uploadSummary = state.upload?.summary;
  const activeDemo = getActiveDemoResult();
  const evaluation = activeDemo?.evaluation_summary;
  const portfolio = evaluation?.portfolio;

  const cards = [
    {
      label: "可用筆數",
      value: uploadSummary ? formatNumber(uploadSummary.rows_prepared, 0) : "-",
      foot: uploadSummary ? `原始 ${formatNumber(uploadSummary.rows_raw, 0)} 筆` : "上傳後顯示",
    },
    {
      label: "年度跨度",
      value: uploadSummary?.years?.length
        ? `${uploadSummary.years[0]}-${uploadSummary.years[uploadSummary.years.length - 1]}`
        : "-",
      foot: uploadSummary?.years?.length ? `${uploadSummary.years.length} 個年度切片` : "上傳後顯示",
    },
    {
      label: "淨年化報酬",
      value: portfolio ? formatPercent(portfolio.net_annualized_return) : "-",
      foot: portfolio ? `淨 MDD ${formatPercent(portfolio.net_maximum_drawdown)}` : "展示結果顯示",
    },
    {
      label: "預測正類比例",
      value: evaluation ? formatPercent(evaluation.predicted_positive_rate) : "-",
      foot: evaluation ? `平均分數 ${formatNumber(evaluation.mean_score_label_1, 4)}` : "展示結果顯示",
    },
  ];

  elements.summaryCards.innerHTML = cards
    .map(
      (card) => `
        <article class="summary-card">
          <p class="summary-label">${card.label}</p>
          <p class="summary-value">${card.value}</p>
          <p class="summary-foot">${card.foot}</p>
        </article>
      `
    )
    .join("");
}

function renderUploadState() {
  if (!state.upload) {
    renderKeyGrid(elements.uploadDatasetSnapshot, [], "尚未上傳資料。");
    elements.uploadYearBars.classList.add("empty-state");
    elements.uploadYearBars.textContent = "尚未上傳資料。";
    elements.featureCoverage.classList.add("empty-state");
    elements.featureCoverage.textContent = "尚未上傳資料。";
    return;
  }

  const { filename, summary } = state.upload;
  const yearRange = summary.years.length
    ? `${summary.years[0]} - ${summary.years[summary.years.length - 1]}`
    : "-";

  renderKeyGrid(elements.uploadDatasetSnapshot, [
    { label: "檔名", value: filename },
    { label: "可用筆數", value: formatNumber(summary.rows_prepared, 0) },
    { label: "年度範圍", value: yearRange },
    { label: "缺少特徵欄位數", value: formatNumber(summary.missing_feature_columns.length, 0) },
    { label: "含 Return", value: formatBoolean(summary.has_return) },
    { label: "含 Label", value: formatBoolean(summary.has_label) },
  ]);

  renderBars(elements.uploadYearBars, summary.rows_by_year || [], {
    labelKey: "year",
    valueKey: "n_rows",
    metaBuilder: (row) => `${formatNumber(row.n_rows, 0)} 筆`,
  });

  const coverageRows = [...(summary.feature_coverage || [])]
    .sort((a, b) => a.non_null_ratio - b.non_null_ratio)
    .slice(0, 10);

  renderBars(elements.featureCoverage, coverageRows, {
    labelKey: "feature",
    valueKey: "non_null_ratio",
    scale: 1,
    metaBuilder: (row) => formatPercent(row.non_null_ratio),
  });
}

function renderCurve(curve) {
  if (!curve || curve.length === 0) {
    elements.curveChart.classList.add("empty-state");
    elements.curveChart.textContent = "若上傳資料缺少 Return，這裡不會繪製投組績效曲線。";
    return;
  }

  elements.curveChart.classList.remove("empty-state");

  const width = 760;
  const height = 300;
  const paddingX = 52;
  const paddingY = 28;

  const values = curve.map((item) => Number(item.cumulative_net_return));
  const years = curve.map((item) => Number(item.year));
  const minValue = Math.min(...values, 0);
  const maxValue = Math.max(...values, 0);
  const ySpan = Math.max(maxValue - minValue, 0.01);
  const xSpan = Math.max(years.length - 1, 1);

  const points = curve.map((item, index) => {
    const x = paddingX + (index / xSpan) * (width - paddingX * 2);
    const y = height - paddingY - ((Number(item.cumulative_net_return) - minValue) / ySpan) * (height - paddingY * 2);
    return { x, y, year: item.year, value: item.cumulative_net_return };
  });

  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");

  const areaPath = `${linePath} L ${points[points.length - 1].x} ${height - paddingY} L ${points[0].x} ${height - paddingY} Z`;

  const yTicks = 4;
  const tickLabels = Array.from({ length: yTicks + 1 }, (_, index) => {
    const ratio = index / yTicks;
    const value = maxValue - ratio * ySpan;
    const y = paddingY + ratio * (height - paddingY * 2);
    return { value, y };
  });

  elements.curveChart.innerHTML = `
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="累積淨報酬曲線">
      ${tickLabels
        .map(
          (tick) => `
            <line x1="${paddingX}" x2="${width - paddingX}" y1="${tick.y}" y2="${tick.y}" stroke="rgba(22,32,41,0.08)" />
            <text class="chart-label" x="8" y="${tick.y + 4}">${formatPercent(tick.value)}</text>
          `
        )
        .join("")}
      <path class="chart-area" d="${areaPath}"></path>
      <path class="chart-line" d="${linePath}"></path>
      ${points
        .map(
          (point) => `
            <circle cx="${point.x}" cy="${point.y}" r="4" fill="#0d6d73"></circle>
            <text class="chart-axis" x="${point.x - 16}" y="${height - 6}">${point.year}</text>
          `
        )
        .join("")}
    </svg>
  `;
}

function renderDemoResult() {
  const activeDemo = getActiveDemoResult();

  if (!activeDemo) {
    renderKeyGrid(elements.performanceSummary, [], "尚未執行展示。");
    renderCurve([]);
    renderTable(elements.selectedTable, [], { emptyText: "尚未執行展示。" });
    renderTable(elements.predictionTable, [], { emptyText: "尚未執行展示。" });
    renderTable(elements.metricTable, [], { emptyText: "尚未執行展示。" });
    renderTable(elements.classificationDetailTable, [], { emptyText: "若輸入含有 Label，這裡會顯示分類驗證。" });
    elements.artifactPaths.textContent = "尚未執行展示。";
    return;
  }

  const evaluation = activeDemo.evaluation_summary || {};
  const portfolio = evaluation.portfolio;
  const classification = evaluation.classification;

  const summaryRows = [
    { label: "模型", value: getModelDisplayName(activeDemo.request.model_name) },
    { label: "Top-K", value: activeDemo.request.top_k },
    { label: "預測筆數", value: formatNumber(evaluation.prediction_rows, 0) },
    { label: "入選筆數", value: formatNumber(evaluation.selected_rows, 0) },
    { label: "預測正類比例", value: formatPercent(evaluation.predicted_positive_rate) },
    { label: "平均分數", value: formatNumber(evaluation.mean_score_label_1, 4) },
  ];

  if (portfolio) {
    summaryRows.push(
      { label: "淨年化報酬", value: formatPercent(portfolio.net_annualized_return) },
      { label: "淨累積報酬", value: formatPercent(portfolio.net_cumulative_return) },
      { label: "淨 Sharpe Ratio", value: formatNumber(portfolio.net_sharpe_ratio, 4) },
      { label: "淨勝率", value: formatPercent(portfolio.net_win_rate) }
    );
  }

  if (classification) {
    summaryRows.push(
      { label: "準確率", value: formatPercent(classification.accuracy) },
      { label: "F1(label=1)", value: formatPercent(classification.f1_label_1) }
    );
  }

  renderKeyGrid(elements.performanceSummary, summaryRows);
  renderCurve(evaluation.portfolio_curve);

  renderTable(elements.selectedTable, activeDemo.tables.selected_stocks || [], {
    limit: 40,
    preferredColumns: [
      "year",
      "top_k",
      "rank",
      "stock_id",
      "stock_name",
      "score_label_1",
      "weight_method",
      "weight",
      "actual_return",
    ],
  });

  renderTable(elements.predictionTable, activeDemo.tables.predictions || [], {
    limit: 120,
    preferredColumns: [
      "year",
      "stock_id",
      "stock_name",
      "predicted_label",
      "score_label_1",
      "actual_label",
      "actual_return",
    ],
  });

  renderTable(elements.metricTable, activeDemo.tables.portfolio_metrics || [], {
    limit: 20,
    emptyText: "本次結果沒有投組指標。",
  });

  renderTable(elements.classificationDetailTable, activeDemo.tables.classification_metrics || [], {
    limit: 20,
    emptyText: "本次結果沒有分類驗證指標。",
  });

  const artifactText = Object.entries(activeDemo.artifacts || {})
    .filter(([, value]) => value)
    .map(([key, value]) => `${key}: ${value}`)
    .join("\n");

  elements.artifactPaths.textContent = artifactText || "沒有可顯示的輸出檔。";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();

  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || "請求失敗。");
  }

  return payload;
}

async function initialize() {
  try {
    setStatus("正在載入專題輸出與展示資訊...");

    const [healthPayload, modelPayload, overviewPayload] = await Promise.all([
      fetchJson("/api/health"),
      fetchJson("/api/models"),
      fetchJson("/api/project/overview"),
    ]);

    state.health = healthPayload;
    state.models = modelPayload.models || [];
    state.overview = overviewPayload;

    renderHeroMetrics();
    renderRequirements();
    renderProjectDataset();
    renderTimeline();
    renderValidationChecks();
    renderLeaderboard();
    renderGallery();
    renderDownloads();
    renderLatestDemoSnapshot();
    renderModelOptions();
    renderHealth();
    renderSummaryCards();
    renderUploadState();
    renderDemoResult();

    if (state.overview.latest_demo) {
      setStatus("頁面已載入既有成果與最新展示，你也可以直接上傳新資料重跑。", "success");
    } else {
      setStatus("頁面已載入既有成果，請上傳資料開始展示。", "success");
    }
  } catch (error) {
    setStatus(`初始化失敗: ${error.message}`, "error");
  }
}

elements.topKRange.addEventListener("input", (event) => {
  syncTopK(event.target.value);
});

elements.topKInput.addEventListener("input", (event) => {
  syncTopK(event.target.value);
});

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = elements.datasetFile.files[0];
  if (!file) {
    setStatus("請先選擇資料檔案。", "error");
    return;
  }

  try {
    elements.uploadButton.disabled = true;
    setStatus("正在檢查資料並套用正式 preprocessing...");

    const payload = await fetchJson("/api/demo/upload", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        filename: file.name,
        content_base64: await fileToBase64(file),
      }),
    });

    state.upload = payload.upload;
    state.run = null;
    elements.runButton.disabled = false;

    renderSummaryCards();
    renderUploadState();
    renderDemoResult();

    const hasReturn = state.upload.summary.has_return ? "含 Return" : "不含 Return";
    const hasLabel = state.upload.summary.has_label ? "含 Label" : "不含 Label";
    setStatus(`資料檢查完成: ${file.name} | ${hasReturn} | ${hasLabel}`, "success");
  } catch (error) {
    setStatus(`資料檢查失敗: ${error.message}`, "error");
  } finally {
    elements.uploadButton.disabled = false;
  }
});

elements.runForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!state.upload) {
    setStatus("請先上傳並檢查資料。", "error");
    return;
  }

  const topK = clampTopK(elements.topKInput.value);
  const modelName = elements.modelSelect.value;

  try {
    elements.runButton.disabled = true;
    setStatus("模型推論中，正在產生選股與績效摘要...");

    const payload = await fetchJson("/api/demo/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        upload_id: state.upload.upload_id,
        model_name: modelName,
        top_k: topK,
      }),
    });

    state.run = payload;
    renderSummaryCards();
    renderDemoResult();
    setStatus(`展示完成: ${modelName} | Top-${topK}`, "success");
  } catch (error) {
    setStatus(`展示執行失敗: ${error.message}`, "error");
  } finally {
    elements.runButton.disabled = false;
  }
});

syncTopK(10);
initialize();
