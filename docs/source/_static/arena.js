// UniKT Leaderboard — Artificial Analysis Bar Chart + Modern Table UI.
// Framework-free client reading the versioned data files under _static/arena/
// with verified dataset cohort metadata and model taxonomy.
(function () {
  "use strict";

  var TABULATOR_JS = "https://unpkg.com/tabulator-tables@6.5.0/dist/js/tabulator.min.js";
  var TABULATOR_CSS = "https://unpkg.com/tabulator-tables@6.5.0/dist/css/tabulator.min.css";

  function loadScript(src, onload, onerror) {
    var s = document.createElement("script");
    s.src = src;
    s.onload = onload;
    if (onerror) s.onerror = onerror;
    document.head.appendChild(s);
  }

  function loadStyle(href) {
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    document.head.appendChild(link);
  }

  var isEn = (document.documentElement.lang || "").toLowerCase().startsWith("en");

  // Data versions are labeled by the commit that produced them. The paper
  // board is frozen; the latest board follows the commit the docs build from,
  // injected into the page as <meta name="unikt-commit">.
  var PAPER_COMMIT = "f45f9a7";

  function latestCommit() {
    var meta = document.querySelector('meta[name="unikt-commit"]');
    var value = (meta && meta.getAttribute("content")) || window.UNIKT_COMMIT || "";
    return String(value).trim() || "HEAD";
  }

  var VERSIONS = [
    {
      key: "latest",
      label: latestCommit(),
      tag: isEn ? "Latest" : "最新",
      src: "leaderboard-live.json"
    },
    {
      key: "paper",
      label: PAPER_COMMIT,
      src: "leaderboard.json"
    }
  ];

  // Retained / raw student counts per dataset. `sampled` marks datasets that
  // keep a random student-level subsample before splitting.
  var DATASET_META = {
    "ASSISTments09":     { students:   3852, rawStudents:    4217, sampled:    0 },
    "ASSISTments12":     { students:  27485, rawStudents:   46674, sampled:    0 },
    "ASSISTments15":     { students:  19172, rawStudents:   19917, sampled:    0 },
    "ASSISTments17":     { students:   1708, rawStudents:    1709, sampled:    0 },
    "ASSISTments17-PQ":  { students:   1707, rawStudents:    1709, sampled:    0 },
    "Algebra2005":       { students:    574, rawStudents:     574, sampled:    0 },
    "Algebra2006":       { students:   1326, rawStudents:    1338, sampled:    0 },
    "Bridge2006":        { students:   1145, rawStudents:    1146, sampled:    0 },
    "NIPS2020_t34":      { students:   4918, rawStudents:    4918, sampled:    0 },
    "Slepemapy":         { students:  84837, rawStudents:   91331, sampled:    0 },
    "EdNet-KT1":         { students:   5000, rawStudents:  784309, sampled: 5000 },
    "XES3G5M":           { students:   2500, rawStudents:   18066, sampled: 2500 },
    "MOOC-Radar":        { students:  13597, rawStudents:   14224, sampled:    0 },
    "Junyi2015":         { students:   5000, rawStudents:  247606, sampled: 5000 },
    "Anatomy":           { students:  14239, rawStudents:   18563, sampled:    0 }
  };

  // Paper models (43) plus extended models synced in live leaderboard
  var MODEL_META = {
    "ABKT":         { gran: "Q",  year: 2022, family: "hybrid", doc: "abkt" },
    "AKT":          { gran: "KC", year: 2020, family: "attn",   doc: "akt" },
    "ATDKT":        { gran: "KC", year: 2023, family: "seq",    doc: "atdkt" },
    "ATKT":         { gran: "KC", year: 2021, family: "seq",    doc: "atkt" },
    "BDGKT":        { gran: "Q",  year: 2026, family: "graph",  doc: "bdgkt" },
    "CIKT":         { gran: "Q",  year: 2026, family: "hybrid", doc: "cikt" },
    "ClusterKT":    { gran: "Q",  year: 2025, family: "hybrid", doc: "clusterkt" },
    "csKT":         { gran: "KC", year: 2025, family: "attn",   doc: "cskt" },
    "DAGKT":        { gran: "Q",  year: 2023, family: "graph",  doc: "dagkt" },
    "Deep-IRT":     { gran: "KC", year: 2019, family: "seq",    doc: "deepirt" },
    "DenoiseKT":    { gran: "Q",  year: 2025, family: "attn",   doc: "denoisekt" },
    "DGEKT":        { gran: "Q",  year: 2024, family: "graph",  doc: "dgekt" },
    "DGMKT":        { gran: "KC", year: 2025, family: "hybrid", doc: "dgmkt" },
    "DIMKT":        { gran: "KC", year: 2022, family: "seq",    doc: "dimkt" },
    "DisKT":        { gran: "Q",  year: 2025, family: "attn",   doc: "diskt" },
    "DKT":          { gran: "KC", year: 2015, family: "seq",    doc: "dkt" },
    "DKT+":         { gran: "KC", year: 2018, family: "seq",    doc: "dktplus" },
    "DKT2":         { gran: "Q",  year: 2025, family: "seq",    doc: "dkt2" },
    "DKVMN":        { gran: "KC", year: 2017, family: "seq",    doc: "dkvmn" },
    "DTransformer": { gran: "KC", year: 2023, family: "attn",   doc: "dtransformer" },
    "DyGKT":        { gran: "Q",  year: 2024, family: "graph",  doc: "dygkt" },
    "extraKT":      { gran: "KC", year: 2024, family: "attn",   doc: "extrakt" },
    "FA-KT":        { gran: "KC", year: 2026, family: "hybrid", doc: "fakt" },
    "FlucKT":       { gran: "KC", year: 2025, family: "attn",   doc: "fluckt" },
    "FoLibiKT":     { gran: "KC", year: 2023, family: "attn",   doc: "folibikt" },
    "GIKT":         { gran: "Q",  year: 2020, family: "graph",  doc: "gikt" },
    "GKT":          { gran: "KC", year: 2019, family: "graph",  doc: "gkt" },
    "GRKT":         { gran: "Q",  year: 2024, family: "graph",  doc: "grkt" },
    "HawkesKT":     { gran: "Q",  year: 2021, family: "hybrid", doc: "hawkeskt" },
    "HGKT":         { gran: "Q",  year: 2025, family: "graph",  doc: "hgkt" },
    "HiTSKT":       { gran: "Q",  year: 2024, family: "attn",   doc: "hitskt" },
    "IEKT":         { gran: "Q",  year: 2021, family: "seq",    doc: "iekt" },
    "KQN":          { gran: "KC", year: 2019, family: "seq",    doc: "kqn" },
    "LBKT":         { gran: "Q",  year: 2023, family: "seq",    doc: "lbkt" },
    "LPKT":         { gran: "Q",  year: 2021, family: "seq",    doc: "lpkt" },
    "LPKT-S":       { gran: "Q",  year: 2023, family: "seq",    doc: "lpkt" },
    "Mamba4KT":     { gran: "KC", year: 2024, family: "hybrid", doc: "mamba4kt" },
    "MCKT":         { gran: "Q",  year: 2025, family: "attn",   doc: "mckt" },
    "MCSKT":        { gran: "KC", year: 2026, family: "hybrid", doc: "mcskt" },
    "MIKT":         { gran: "Q",  year: 2024, family: "seq",    doc: "mikt" },
    "MTKT":         { gran: "KC", year: 2025, family: "attn",   doc: "mtkt" },
    "PSKT":         { gran: "Q",  year: 2024, family: "seq",    doc: "pskt" },
    "QIKT":         { gran: "Q",  year: 2023, family: "seq",    doc: "qikt" },
    "ReKT":         { gran: "Q",  year: 2024, family: "seq",    doc: "rekt" },
    "RobustKT":     { gran: "KC", year: 2025, family: "attn",   doc: "robustkt" },
    "SAKT":         { gran: "KC", year: 2019, family: "attn",   doc: "sakt" },
    "SGKT":         { gran: "Q",  year: 2022, family: "graph",  doc: "sgkt" },
    "simpleKT":     { gran: "KC", year: 2023, family: "attn",   doc: "simplekt" },
    "SKVMN":        { gran: "KC", year: 2019, family: "seq",    doc: "skvmn" },
    "sparseKT":     { gran: "KC", year: 2023, family: "attn",   doc: "sparsekt" },
    "SQGKT":        { gran: "Q",  year: 2026, family: "graph",  doc: "sqgkt" },
    "stableKT":     { gran: "KC", year: 2024, family: "attn",   doc: "stablekt" },
    "TCKT":         { gran: "Q",  year: 2024, family: "hybrid", doc: "tckt" },
    "UKT":          { gran: "KC", year: 2025, family: "attn",   doc: "ukt" }
  };

  // Paper Section 3.2 four architectural families
  var FAMILY_LABELS = {
    seq:    isEn ? "Sequential state-update" : "序列状态更新",
    attn:   isEn ? "Attention-based"         : "基于注意力",
    graph:  isEn ? "Graph-based"             : "基于图结构",
    hybrid: isEn ? "Hybrid"                  : "混合架构"
  };

  function sortMetricVal(row, metric) {
    var raw = row && row[metric + "_raw"];
    if (raw != null) return raw;
    return (row && row[metric]) || 0;
  }

  function sortRowsDesc(a, b) {
    var aSpec = a.protocol === "model_specific";
    var bSpec = b.protocol === "model_specific";
    if (aSpec !== bSpec) return aSpec ? 1 : -1;
    var diff = sortMetricVal(b, selectedMetric) - sortMetricVal(a, selectedMetric);
    if (diff !== 0) return diff;
    return String(a.model || "").localeCompare(String(b.model || ""));
  }

  var allData = [];
  var versionCache = {};            // version key -> decorated rows
  var currentVersionKey = VERSIONS[0].key;
  var table = null;
  var applySeq = 0;
  var rankMap = new WeakMap();
  var currentRange = { min: 0.6, max: 0.95 };
  var rootPrefix = "";

  // UI state: default to the latest board and the first curated dataset
  var selectedDataset = "ASSISTments09";
  var searchQuery = "";
  var selectedMetric = "auc";       // "auc" | "acc"
  var selectedGran = "ALL";         // "ALL" | "Q" | "KC"
  // DOM references
  var shellEl = null;
  var versionWrapEl = null;
  var versionTriggerEl = null;
  var versionMenuEl = null;
  var versionOptionEls = {};
  var versionFocusPending = false;
  var chartCardEl = null;
  var tableCardEl = null;
  var titleEl = null;
  var subtitleEl = null;
  var legendEl = null;
  var stageEl = null;
  var tooltipEl = null;

  function fmtInt(n) {
    return Number(n).toLocaleString("en-US");
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function modelUrl(modelName) {
    var m = MODEL_META[modelName];
    return m ? (rootPrefix + "model-zoo/models/" + m.doc + ".html") : "#";
  }

  function decorateRows(data) {
    return data.map(function (r) {
      var meta = MODEL_META[r.model] || { gran: "Q", year: "", family: "seq", doc: "" };
      return Object.assign({}, r, {
        gran: meta.gran,
        family: meta.family,
        year: meta.year
      });
    });
  }

  function collectDatasets(rows) {
    var datasets = [];
    var seen = {};
    rows.forEach(function (r) {
      if (!seen[r.dataset]) {
        seen[r.dataset] = true;
        datasets.push(r.dataset);
      }
    });
    // Keep the DATASET_META key order (curated list), not alphabetical.
    var metaOrder = Object.keys(DATASET_META);
    datasets.sort(function (a, b) {
      var ia = metaOrder.indexOf(a);
      var ib = metaOrder.indexOf(b);
      return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
    });
    return datasets;
  }

  function metricSorter(_a, _b, aRow, bRow, column, dir) {
    var da = aRow.getData();
    var db = bRow.getData();
    var aSpec = da.protocol === "model_specific";
    var bSpec = db.protocol === "model_specific";
    if (aSpec !== bSpec) {
      var bottom = aSpec ? 1 : -1;
      return dir === "desc" ? -bottom : bottom;
    }
    var field = (column && column.getField()) || selectedMetric;
    var diff = sortMetricVal(da, field) - sortMetricVal(db, field);
    if (diff !== 0) return diff;
    var cmp = String(da.model || "").localeCompare(String(db.model || ""));
    return dir === "desc" ? -cmp : cmp;
  }

  function renderRankHtml(d, rankNum) {
    if (!d || d.protocol === "model_specific" || rankNum == null) {
      return '<span class="arena-rank-pill">—</span>';
    }
    var cls = rankNum <= 3 ? ("arena-rank-pill is-" + rankNum) : "arena-rank-pill";
    return '<span class="' + cls + '">' + rankNum + "</span>";
  }

  // Ranks always follow the selected metric, so a "#" stays comparable with
  // the chart tooltip even when the table is sorted by another column.
  function syncRanks(rows) {
    rankMap = new WeakMap();
    if (!rows || !rows.length) return;
    var data = rows.map(function (r) {
      return r && r.getData ? r.getData() : r;
    });
    var k = 0;
    data.slice().sort(sortRowsDesc).forEach(function (d) {
      if (!d || d.protocol === "model_specific") return;
      k += 1;
      rankMap.set(d, k);
    });
    for (var j = 0; j < rows.length; j++) {
      var r = rows[j];
      if (!r || !r.getElement) continue;
      var el = r.getElement();
      if (!el) continue;
      var firstCell = el.querySelector(".tabulator-cell");
      if (firstCell) {
        var rd = r.getData();
        firstCell.innerHTML = renderRankHtml(rd, rankMap.get(rd));
      }
    }
  }

  function rankFormatter(cell) {
    var d = cell.getRow().getData();
    if (!d || d.protocol === "model_specific") {
      return '<span class="arena-rank-pill">—</span>';
    }
    var r = rankMap.get(d);
    var n = r != null ? r : cell.getRow().getPosition(true);
    return renderRankHtml(d, n);
  }

  function modelFormatter(cell) {
    var m = cell.getValue();
    var meta = MODEL_META[m];
    var yr = meta && meta.year ? '<span class="arena-model-year">' + meta.year + "</span>" : "";
    return '<span class="arena-model-cell"><a class="arena-model-link" href="' + esc(modelUrl(m)) + '">' + esc(m) + "</a>" + yr + "</span>";
  }

  function familyFormatter(cell) {
    var f = cell.getValue() || "seq";
    return '<span class="arena-fam-tag"><span class="arena-fam-dot is-' + esc(f) + '"></span>' + esc(FAMILY_LABELS[f] || f) + "</span>";
  }

  function granFormatter(cell) {
    var g = cell.getValue();
    var isQ = g === "Q";
    return '<span class="arena-gran ' + (isQ ? "is-q" : "is-kc") + '">' + (isQ ? "Q" : "KC") + "</span>";
  }

  function makeScoreFormatter(stdField, showBar) {
    return function (cell) {
      var v = cell.getValue();
      if (v == null) return "—";
      var std = cell.getRow().getData()[stdField];
      var stdHtml = std != null ? ('<span class="arena-metric-std">±' + std.toFixed(3) + "</span>") : "";
      var barHtml = "";
      if (showBar) {
        var span = Math.max(0.03, currentRange.max - currentRange.min);
        var pct = Math.max(6, Math.min(100, ((v - currentRange.min) / span) * 100));
        barHtml = '<span class="arena-metric-bar"><span class="arena-metric-bar-fill" style="width:' + pct.toFixed(1) + '%"></span></span>';
      }
      return '<span class="arena-metric-wrap"><span class="arena-metric-nums"><span class="arena-metric-main">' + v.toFixed(3) + "</span>" + stdHtml + "</span>" + barHtml + "</span>";
    };
  }

  var COHORT_LABELS = {
    full:    { zh: "全量",   en: "Full" },
    sampled: { zh: "子采样", en: "Subsampled" },
    spec:    { zh: "专用",   en: "Model-specific" }
  };

  function cohortFormatter(cell) {
    var tone = cohortTone(cell.getRow().getData());
    var label = isEn ? COHORT_LABELS[tone].en : COHORT_LABELS[tone].zh;
    return '<span class="arena-tag is-' + tone + '">' + esc(label) + "</span>";
  }

  function cohortTooltip(_e, cell) {
    var d = cell.getRow().getData();
    var dsMeta = DATASET_META[d.dataset];
    if (!dsMeta) return esc(d.notes || "");
    var prefix = d.protocol === "model_specific"
      ? (isEn ? "Not evaluated under the shared protocol · " : "未在共享协议下评测 · ")
      : "";
    if (dsMeta.sampled) {
      return prefix + (isEn
        ? ("Random student-level subsampling prior to splitting: " + fmtInt(dsMeta.sampled) + " retained students (" + fmtInt(dsMeta.rawStudents) + " in raw release)")
        : ("切分前执行学生级随机子采样：保留 " + fmtInt(dsMeta.sampled) + " 名学生（原始发布 " + fmtInt(dsMeta.rawStudents) + " 名）"));
    }
    return prefix + (isEn
      ? ("Unified preprocessing on complete public release: " + fmtInt(dsMeta.students) + " retained students (" + fmtInt(dsMeta.rawStudents) + " in raw release)")
      : ("完整公开数据集经统一预处理后保留 " + fmtInt(dsMeta.students) + " 名学生（原始发布 " + fmtInt(dsMeta.rawStudents) + " 名）"));
  }

  function cohortTone(d) {
    if (d.protocol === "model_specific") return "spec";
    var m = DATASET_META[d.dataset];
    return (m && m.sampled) ? "sampled" : "full";
  }

  function cohortSorter(_a, _b, aRow, bRow) {
    var order = { full: 0, sampled: 1, spec: 2 };
    return order[cohortTone(aRow.getData())] - order[cohortTone(bRow.getData())];
  }

  function getColumns() {
    return [
      { title: "#", width: 48, minWidth: 48, hozAlign: "center", headerHozAlign: "center", headerSort: false, formatter: rankFormatter },
      { title: isEn ? "Model" : "模型", field: "model", sorter: "string", minWidth: 148, widthGrow: 1.15, formatter: modelFormatter },
      { title: isEn ? "Granularity" : "交互粒度", field: "gran", sorter: "string", width: 108, minWidth: 108, hozAlign: "center", headerHozAlign: "center", formatter: granFormatter },
      { title: isEn ? "Model Family" : "模型家族", field: "family", sorter: "string", minWidth: 156, widthGrow: 1.15, formatter: familyFormatter },
      {
        title: "AUC",
        field: "auc",
        sorter: metricSorter,
        hozAlign: "right",
        headerHozAlign: "right",
        minWidth: 132,
        widthGrow: 1,
        formatter: makeScoreFormatter("auc_std", true)
      },
      {
        title: "ACC",
        field: "acc",
        sorter: metricSorter,
        hozAlign: "right",
        headerHozAlign: "right",
        minWidth: 120,
        widthGrow: 0.9,
        formatter: makeScoreFormatter("acc_std", false)
      },
      {
        title: isEn ? "Protocol" : "评测设置",
        field: "notes",
        sorter: cohortSorter,
        minWidth: 128,
        widthGrow: 0.85,
        hozAlign: "left",
        tooltip: cohortTooltip,
        formatter: cohortFormatter
      }
    ];
  }

  function hideTooltip() {
    if (tooltipEl) tooltipEl.classList.remove("is-visible");
    if (!stageEl) return;
    var activeEls = stageEl.querySelectorAll(".is-active");
    for (var i = 0; i < activeEls.length; i++) activeEls[i].classList.remove("is-active");
  }

  function showTooltip(colEl, axisEl, item, rankNum) {
    if (!tooltipEl || !chartCardEl) return;
    hideTooltip();
    colEl.classList.add("is-active");
    if (axisEl) axisEl.classList.add("is-active");

    var mLabel = selectedMetric.toUpperCase();
    var mainVal = item[selectedMetric];
    var stdVal = item[selectedMetric + "_std"];
    var otherMetric = selectedMetric === "auc" ? "acc" : "auc";
    var otherVal = item[otherMetric];
    var otherStd = item[otherMetric + "_std"];

    var rankTxt = item.protocol === "model_specific"
      ? (isEn ? "Model-specific" : "模型专用协议")
      : ("#" + rankNum);

    var metaLine1 = otherMetric.toUpperCase() + ": " + otherVal.toFixed(3) +
      (otherStd != null ? (" ± " + otherStd.toFixed(3)) : "") +
      " · " + (FAMILY_LABELS[item.family] || "") +
      (item.year ? (" (" + item.year + ")") : "");

    // Cohort facts live in the title pill; only the ranking exclusion needs a note here.
    var metaLine2 = item.protocol === "model_specific"
      ? (isEn ? "Model-specific protocol (excluded from shared ranking)" : "模型专用协议（不纳入共享协议直接排名）")
      : "";

    var granCls = item.gran === "Q" ? "is-q" : "is-kc";
    tooltipEl.innerHTML =
      '<div class="aa-tip-head">' +
        '<span class="aa-tip-model"><span class="aa-gran-chip ' + granCls + '">' + esc(item.gran) + "</span>" + esc(item.model) + "</span>" +
        '<span class="aa-tip-rank">' + esc(rankTxt) + "</span>" +
      "</div>" +
      '<div class="aa-tip-score">' + mainVal.toFixed(3) +
        (stdVal != null ? ('<small>± ' + stdVal.toFixed(3) + " " + mLabel + "</small>") : ('<small>' + mLabel + "</small>")) +
      "</div>" +
      '<div class="aa-tip-meta">' + esc(metaLine1) +
        (metaLine2 ? ("<br>" + esc(metaLine2)) : "") +
      "</div>" +
      '<div class="aa-tip-cta">' + (isEn ? "Click for model details →" : "点击查看模型介绍 →") + "</div>";

    tooltipEl.classList.add("is-visible");

    var cardRect = chartCardEl.getBoundingClientRect();
    var colRect = colEl.getBoundingClientRect();
    var tipW = tooltipEl.offsetWidth || 225;
    var left = (colRect.right - cardRect.left) + 8;
    if (left + tipW > cardRect.width - 12) {
      left = Math.max(12, (colRect.left - cardRect.left) - tipW - 8);
    }
    var rawTop = (colRect.top - cardRect.top) + 18;
    var tipH = tooltipEl.offsetHeight || 130;
    var top = Math.min(Math.max(58, rawTop), Math.max(58, cardRect.height - tipH - 12));
    tooltipEl.style.left = left + "px";
    tooltipEl.style.top = top + "px";
  }

  function cohortPillHtml(dsMeta) {
    if (!dsMeta) return "";
    return dsMeta.sampled
      ? '<span class="arena-pill is-sampled">' + (isEn ? "Subsampled" : "学生级子采样") + "</span>"
      : '<span class="arena-pill is-full">' + (isEn ? "Full release" : "全量数据集") + "</span>";
  }

  function renderChart(chartItems, chartDs) {
    if (!stageEl || !titleEl || !legendEl || !subtitleEl) return;
    hideTooltip();

    var dsMeta = DATASET_META[chartDs];
    titleEl.textContent = chartDs;
    subtitleEl.innerHTML = cohortPillHtml(dsMeta);

    // Check if any model_specific item is present
    var hasSpec = chartItems.some(function (x) { return x.protocol === "model_specific"; });
    var legendHtml =
      '<span class="aa-legend-item"><span class="aa-swatch is-seq"></span>' + esc(FAMILY_LABELS.seq) + "</span>" +
      '<span class="aa-legend-item"><span class="aa-swatch is-attn"></span>' + esc(FAMILY_LABELS.attn) + "</span>" +
      '<span class="aa-legend-item"><span class="aa-swatch is-graph"></span>' + esc(FAMILY_LABELS.graph) + "</span>" +
      '<span class="aa-legend-item"><span class="aa-swatch is-hybrid"></span>' + esc(FAMILY_LABELS.hybrid) + "</span>";
    if (hasSpec) {
      legendHtml += '<span class="aa-legend-item"><span class="aa-swatch is-patterned"></span>' +
        (isEn ? "Model-specific protocol" : "模型专用协议") + "</span>";
    }
    legendEl.innerHTML = legendHtml;

    stageEl.innerHTML = "";
    if (!chartItems.length) {
      stageEl.innerHTML = '<div class="arena-error" style="padding:28px 0;color:var(--aa-muted)">' +
        (isEn ? "No matching models." : "当前筛选条件下暂无匹配模型。") + "</div>";
      return;
    }

    // Compute clean dynamic Y-axis bounds so bar heights show clear differentiation
    var vals = chartItems.map(function (x) { return x[selectedMetric]; });
    var minV = Math.min.apply(null, vals);
    var maxV = Math.max.apply(null, vals);
    var rawSpan = Math.max(0.04, maxV - minV);
    var yMin = Math.max(0.50, Math.floor((minV - Math.max(0.022, rawSpan * 0.22)) * 50) / 50);
    var yMax = Math.min(1.00, Math.ceil((maxV + Math.max(0.008, rawSpan * 0.08)) * 50) / 50);
    if (yMax - yMin < 0.06) yMin = Math.max(0.45, yMax - 0.08);

    var plotWrap = document.createElement("div");
    plotWrap.className = "aa-plot-wrap";

    // 4 horizontal gridlines + baseline
    var steps = 4;
    for (var g = 0; g <= steps; g++) {
      var frac = g / steps; // 0 = top, 1 = bottom axis
      var tickVal = yMax - frac * (yMax - yMin);
      var line = document.createElement("div");
      line.className = g === steps ? "aa-grid-line is-axis" : "aa-grid-line";
      line.style.top = (frac * 100).toFixed(2) + "%";
      var lbl = document.createElement("span");
      lbl.className = "aa-grid-label";
      lbl.textContent = tickVal.toFixed(2);
      line.appendChild(lbl);
      plotWrap.appendChild(line);
    }

    var barsRow = document.createElement("div");
    barsRow.className = "aa-bars-row";
    var axisRow = document.createElement("div");
    axisRow.className = "aa-axis-row";

    var rankCounter = 0;
    var insertedDivider = false;

    chartItems.forEach(function (item) {
      if (item.protocol === "model_specific" && !insertedDivider && rankCounter > 0) {
        insertedDivider = true;
        var divLine = document.createElement("div");
        divLine.className = "aa-spec-divider";
        barsRow.appendChild(divLine);
        var spacer = document.createElement("div");
        spacer.className = "aa-axis-spacer";
        axisRow.appendChild(spacer);
      }

      var thisRank = null;
      if (item.protocol !== "model_specific") {
        rankCounter += 1;
        thisRank = rankCounter;
      }

      var v = item[selectedMetric];
      var hPct = Math.max(12, Math.min(98, ((v - yMin) / (yMax - yMin)) * 100));

      var col = document.createElement("a");
      col.className = "aa-col";
      col.href = modelUrl(item.model);
      col.setAttribute("aria-label", item.model + ": " + v.toFixed(3) + " " + selectedMetric.toUpperCase());

      var bar = document.createElement("div");
      var famCls = "is-" + (item.family || "seq");
      bar.className = "aa-bar " + famCls +
        (item.protocol === "model_specific" ? " is-patterned" : "");
      bar.style.height = hPct.toFixed(1) + "%";

      var valSpan = document.createElement("span");
      valSpan.className = "aa-bar-val";
      valSpan.textContent = v.toFixed(3);
      bar.appendChild(valSpan);
      col.appendChild(bar);

      var axisCell = document.createElement("div");
      axisCell.className = "aa-axis-cell" + (thisRank && thisRank <= 3 ? " is-top" : "");

      var mLbl = document.createElement("span");
      mLbl.className = "aa-model-label";
      mLbl.textContent = item.model;
      axisCell.appendChild(mLbl);

      col.addEventListener("mouseenter", function () {
        showTooltip(col, axisCell, item, thisRank);
      });
      col.addEventListener("mouseleave", hideTooltip);
      col.addEventListener("focus", function () {
        showTooltip(col, axisCell, item, thisRank);
      });
      col.addEventListener("blur", hideTooltip);

      barsRow.appendChild(col);
      axisRow.appendChild(axisCell);
    });

    plotWrap.appendChild(barsRow);
    stageEl.appendChild(plotWrap);
    stageEl.appendChild(axisRow);
  }

  function updateCount(n) {
    var el = document.getElementById("arena-count");
    if (!el) return;
    el.textContent = isEn ? (n + " records") : ("共 " + n + " 条记录");
  }

  function downloadCsv() {
    var rows = table ? table.getRows("active") : [];
    if (!rows.length) return;
    var head = ["model", "dataset", "granularity", "family", "auc", "auc_std", "acc", "acc_std", "protocol"];
    var lines = [head.join(",")];
    rows.forEach(function (row) {
      var d = row.getData();
      var tone = cohortTone(d);
      lines.push([
        d.model,
        d.dataset,
        d.gran || "",
        d.family || "",
        d.auc != null ? d.auc.toFixed(4) : "",
        d.auc_std != null ? d.auc_std.toFixed(4) : "",
        d.acc != null ? d.acc.toFixed(4) : "",
        d.acc_std != null ? d.acc_std.toFixed(4) : "",
        isEn ? COHORT_LABELS[tone].en : COHORT_LABELS[tone].zh
      ].map(function (v) {
        return '"' + String(v).replace(/"/g, '""') + '"';
      }).join(","));
    });
    var blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "unikt-leaderboard-" + selectedDataset + "-" + selectedMetric + ".csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function applyFilters() {
    var q = searchQuery.trim().toLowerCase();

    // 1. Prepare Chart & Table items
    var filteredTable = allData.filter(function (r) {
      if (r.dataset !== selectedDataset) return false;
      if (selectedGran !== "ALL" && r.gran !== selectedGran) return false;
      if (q && r.model.toLowerCase().indexOf(q) === -1) return false;
      return true;
    });
    var filteredChart = filteredTable.slice().sort(sortRowsDesc);

    renderChart(filteredChart, selectedDataset);

    if (filteredTable.length) {
      var aucVals = filteredTable.map(function (r) { return r.auc; });
      currentRange = {
        min: Math.min.apply(null, aucVals),
        max: Math.max.apply(null, aucVals)
      };
    }

    syncRanks(filteredTable);

    if (!table) {
      table = new Tabulator("#arena-table", {
        data: filteredTable,
        layout: "fitColumns",
        height: "520px",
        placeholder: isEn ? "No matching records" : "暂无匹配数据",
        initialSort: [{ column: selectedMetric, dir: "desc" }],
        columns: getColumns()
      });
      table.on("dataSorted", function (_sorters, rows) {
        syncRanks(rows && rows.length ? rows : (table ? table.getRows("active") : []));
      });
      table.on("renderComplete", function () {
        if (table) syncRanks(table.getRows("active"));
      });
    } else {
      var mySeq = ++applySeq;
      Promise.resolve(table.setData(filteredTable)).then(function () {
        if (mySeq !== applySeq || !table) return;
        table.setSort(selectedMetric, "desc");
        syncRanks(table.getRows("active"));
      });
    }

    updateCount(filteredTable.length);
  }

  function setDataset(key, btn) {
    selectedDataset = key;
    var btns = document.querySelectorAll(".arena-ds-btn");
    for (var i = 0; i < btns.length; i++) {
      btns[i].classList.toggle("active", btns[i] === btn);
      btns[i].setAttribute("aria-pressed", btns[i] === btn ? "true" : "false");
    }
    applyFilters();
  }

  function buildDatasetTabs(datasets) {
    var el = document.getElementById("arena-tabs");
    if (!el) return;
    el.innerHTML = "";
    el.setAttribute("role", "group");
    el.setAttribute("aria-label", isEn ? "Dataset" : "数据集");

    datasets.forEach(function (ds) {
      var m = DATASET_META[ds];
      var b = document.createElement("button");
      b.type = "button";
      b.className = "arena-ds-btn" + (selectedDataset === ds ? " active" : "");
      b.setAttribute("aria-pressed", selectedDataset === ds ? "true" : "false");
      b.textContent = ds;
      if (m) {
        b.title = isEn
          ? (fmtInt(m.students) + " retained students (" + fmtInt(m.rawStudents) + " in raw release)")
          : ("保留 " + fmtInt(m.students) + " 名学生（原始发布 " + fmtInt(m.rawStudents) + " 名）");
      }
      b.addEventListener("click", function () {
        setDataset(ds, b);
      });
      el.appendChild(b);
    });
  }

  function syncVersionControl() {
    if (!versionTriggerEl) return;
    var current = null;
    VERSIONS.forEach(function (v) {
      if (v.key === currentVersionKey) current = v;
    });
    versionTriggerEl.querySelector(".arena-version-value").textContent =
      (current && current.label) || "HEAD";
    VERSIONS.forEach(function (v) {
      var optionEl = versionOptionEls[v.key];
      if (!optionEl) return;
      optionEl.disabled = !versionCache[v.key];
      optionEl.setAttribute("aria-selected", v.key === currentVersionKey ? "true" : "false");
    });
  }

  function closeVersionMenu() {
    if (!versionMenuEl) return;
    versionMenuEl.classList.remove("open");
    if (versionTriggerEl) versionTriggerEl.setAttribute("aria-expanded", "false");
  }

  function openVersionMenu() {
    if (!versionMenuEl || !versionTriggerEl || versionTriggerEl.disabled) return;
    syncVersionControl();
    versionMenuEl.classList.add("open");
    versionTriggerEl.setAttribute("aria-expanded", "true");
    var selected = versionMenuEl.querySelector('[aria-selected="true"]');
    (selected || versionMenuEl.querySelector(".arena-version-option")).focus();
  }

  function buildVersionSwitcher() {
    var wrap = document.createElement("div");
    wrap.className = "arena-version";
    wrap.title = isEn ? "Leaderboard code version" : "排行榜代码版本";
    versionWrapEl = wrap;

    var trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "arena-version-trigger";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    trigger.innerHTML =
      '<svg class="arena-version-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">' +
        '<circle cx="12" cy="12" r="3"/><path d="M12 3v6M12 15v6"/>' +
      "</svg>" +
      '<span class="arena-version-label">' + (isEn ? "Version" : "版本") + "</span>" +
      '<span class="arena-version-value"></span>' +
      '<svg class="arena-version-caret" viewBox="0 0 12 12" width="10" height="10" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">' +
        '<path d="m2.5 4.5 3.5 3.5 3.5-3.5"/>' +
      "</svg>";
    trigger.addEventListener("click", function (e) {
      e.stopPropagation();
      if (versionMenuEl.classList.contains("open")) closeVersionMenu();
      else openVersionMenu();
    });
    wrap.appendChild(trigger);

    var menu = document.createElement("div");
    menu.className = "arena-version-menu";
    menu.setAttribute("role", "listbox");
    menu.setAttribute("aria-label", isEn ? "Leaderboard code version" : "排行榜代码版本");
    VERSIONS.forEach(function (v) {
      var optionEl = document.createElement("button");
      optionEl.type = "button";
      optionEl.className = "arena-version-option";
      optionEl.setAttribute("role", "option");
      optionEl.setAttribute("aria-selected", "false");
      optionEl.innerHTML =
        '<span class="arena-version-opt-label">' + esc(v.label) + "</span>" +
        (v.tag ? '<span class="arena-version-opt-tag">' + esc(v.tag) + "</span>" : "") +
        '<svg class="arena-version-check" viewBox="0 0 12 12" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">' +
          '<path d="m2.5 6.2 2.4 2.4 4.6-4.8"/>' +
        "</svg>";
      optionEl.addEventListener("click", function (e) {
        e.stopPropagation();
        // Closing the menu hides the focused option; hand focus back to the
        // trigger once the swap finishes so keyboard flow survives.
        versionFocusPending = document.activeElement === optionEl;
        closeVersionMenu();
        setVersion(v.key);
      });
      menu.appendChild(optionEl);
      versionOptionEls[v.key] = optionEl;
    });
    wrap.appendChild(menu);

    versionTriggerEl = trigger;
    versionMenuEl = menu;
    syncVersionControl();

    // Dismiss the menu on outside click or Escape.
    document.addEventListener("click", function (e) {
      if (versionWrapEl && !versionWrapEl.contains(e.target)) closeVersionMenu();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closeVersionMenu();
    });
    return wrap;
  }

  function restoreVersionFocus() {
    if (!versionFocusPending) return;
    versionFocusPending = false;
    if (versionTriggerEl) versionTriggerEl.focus({ preventScroll: true });
  }

  function setVersion(key) {
    var next = key ? versionCache[key] : null;
    if (!next || key === currentVersionKey) {
      restoreVersionFocus();
      return;
    }

    currentVersionKey = key;
    if (versionTriggerEl) versionTriggerEl.disabled = true;
    syncVersionControl();
    if (shellEl) shellEl.classList.add("is-switching");

    // Fade out, swap the rendered rows, then fade back in so the version
    // change reads as one continuous transition.
    window.setTimeout(function () {
      allData = next;
      var datasets = collectDatasets(allData);
      if (datasets.length && datasets.indexOf(selectedDataset) === -1) {
        selectedDataset = datasets[0];
      }
      buildDatasetTabs(datasets);
      applyFilters();
      window.requestAnimationFrame(function () {
        if (shellEl) shellEl.classList.remove("is-switching");
        if (versionTriggerEl) versionTriggerEl.disabled = false;
        restoreVersionFocus();
      });
    }, 170);
  }

  function mountShell() {
    var tabsEl = document.getElementById("arena-tabs");
    var tableEl = document.getElementById("arena-table");
    var searchWrap = document.querySelector(".arena-search");
    var countEl = document.getElementById("arena-count");
    if (!tabsEl || !tableEl) return;

    var shell = document.createElement("div");
    shell.className = "arena-shell";
    shellEl = shell;
    tabsEl.parentNode.insertBefore(shell, tabsEl);

    var tabsRow = document.createElement("div");
    tabsRow.className = "arena-tabs-row";
    shell.appendChild(tabsRow);
    tabsRow.appendChild(tabsEl);
    tabsRow.appendChild(buildVersionSwitcher());

    // 1. Bar chart card
    chartCardEl = document.createElement("section");
    chartCardEl.className = "aa-card";
    chartCardEl.innerHTML =
      '<div class="aa-metric-tabs" data-role="metric">' +
        '<button type="button" data-val="auc" class="active" aria-pressed="true">AUC</button>' +
        '<button type="button" data-val="acc" aria-pressed="false">ACC</button>' +
      "</div>" +
      '<div class="aa-header">' +
        '<div class="aa-title-block">' +
          '<div class="aa-title-row">' +
            '<h3 class="aa-title"></h3>' +
            '<div class="aa-subtitle"></div>' +
          "</div>" +
        "</div>" +
        '<div class="aa-icon-group">' +
          '<button type="button" class="aa-icon-btn" data-role="download" aria-label="' + (isEn ? "Download CSV" : "下载 CSV") + '" title="' + (isEn ? "Download CSV" : "下载 CSV") + '">' +
            '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M12 3v12m0 0 4-4m-4 4-4-4M4 20h16"/></svg>' +
          "</button>" +
        "</div>" +
      "</div>" +
      '<div class="aa-legend-row">' +
        '<div class="aa-gran-switch" data-role="gran" role="group" aria-label="' + (isEn ? "Interaction granularity" : "交互粒度") + '">' +
          '<button type="button" data-val="ALL" class="active" aria-pressed="true">' + (isEn ? "All" : "全部") + "</button>" +
          '<button type="button" data-val="Q" aria-pressed="false">' + (isEn ? "Q-level" : "题目级") + "</button>" +
          '<button type="button" data-val="KC" aria-pressed="false">' + (isEn ? "KC-level" : "知识组件级") + "</button>" +
        "</div>" +
      "</div>" +
      '<div class="aa-chart-scroll"><div class="aa-chart-stage"></div></div>' +
      '<div class="aa-legend-row is-bottom"><div class="aa-legend"></div></div>' +
      '<div class="aa-tooltip"></div>';

    shell.appendChild(chartCardEl);

    titleEl = chartCardEl.querySelector(".aa-title");
    subtitleEl = chartCardEl.querySelector(".aa-subtitle");
    legendEl = chartCardEl.querySelector(".aa-legend");
    stageEl = chartCardEl.querySelector(".aa-chart-stage");
    tooltipEl = chartCardEl.querySelector(".aa-tooltip");

    chartCardEl.querySelector('[data-role="download"]').addEventListener("click", downloadCsv);

    // 2. Modern Table Card
    tableCardEl = document.createElement("section");
    tableCardEl.className = "arena-table-card";

    var toolbar = document.createElement("div");
    toolbar.className = "arena-table-toolbar";
    if (searchWrap) toolbar.appendChild(searchWrap);

    var rightBox = document.createElement("div");
    rightBox.className = "arena-toolbar-right";

    if (countEl) rightBox.appendChild(countEl);
    toolbar.appendChild(rightBox);

    tableCardEl.appendChild(toolbar);
    tableCardEl.appendChild(tableEl);
    shell.appendChild(tableCardEl);

    // Wire controls
    chartCardEl.querySelector('[data-role="metric"]').addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-val]");
      if (!btn) return;
      selectedMetric = btn.getAttribute("data-val");
      var btns = chartCardEl.querySelectorAll('[data-role="metric"] button');
      for (var i = 0; i < btns.length; i++) {
        btns[i].classList.toggle("active", btns[i] === btn);
        btns[i].setAttribute("aria-pressed", btns[i] === btn ? "true" : "false");
      }
      applyFilters();
    });

    chartCardEl.querySelector('[data-role="gran"]').addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-val]");
      if (!btn) return;
      selectedGran = btn.getAttribute("data-val");
      var btns = chartCardEl.querySelectorAll('[data-role="gran"] button');
      for (var i = 0; i < btns.length; i++) {
        btns[i].classList.toggle("active", btns[i] === btn);
        btns[i].setAttribute("aria-pressed", btns[i] === btn ? "true" : "false");
      }
      applyFilters();
    });

    var searchInput = document.getElementById("arena-filter");
    if (searchInput) {
      searchInput.placeholder = isEn ? "Filter by model name..." : "按模型名称实时筛选...";
      searchInput.addEventListener("input", function () {
        searchQuery = searchInput.value;
        applyFilters();
      });
    }
  }

  function initArena() {
    var el = document.getElementById("arena-table");
    if (!el || el.dataset.initialized) return;
    el.dataset.initialized = "true";

    loadStyle(TABULATOR_CSS);

    var script = document.querySelector('script[src*="arena.js"]');
    if (script) {
      var src = script.getAttribute("src") || "";
      var idx = src.indexOf("_static/");
      if (idx !== -1) {
        rootPrefix = src.substring(0, idx);
      }
    }
    mountShell();

    function fetchJson(url) {
      return fetch(url).then(function (r) {
        if (!r.ok) throw new Error("HTTP error " + r.status);
        return r.json();
      });
    }

    function fetchVersion(file) {
      var primaryUrl = rootPrefix + "_static/arena/" + file;
      return fetchJson(primaryUrl).catch(function () {
        var fallbackUrl = (rootPrefix === "" ? "../_static/arena/" : "_static/arena/") + file;
        return fetchJson(fallbackUrl);
      });
    }

    // Preload every version so switching only swaps rendered rows.
    Promise.all(
      VERSIONS.map(function (v) {
        return fetchVersion(v.src)
          .then(function (rows) {
            versionCache[v.key] = decorateRows(rows);
          })
          .catch(function () {
            versionCache[v.key] = null;
          });
      })
    )
      .then(function () {
        var initial = VERSIONS.filter(function (v) {
          return versionCache[v.key];
        })[0];
        if (!initial) throw new Error("no leaderboard data");
        currentVersionKey = initial.key;
        allData = versionCache[initial.key];

        var datasets = collectDatasets(allData);
        if (datasets.length && datasets.indexOf(selectedDataset) === -1) {
          selectedDataset = datasets[0];
        }
        buildDatasetTabs(datasets);
        syncVersionControl();
        applyFilters();
      })
      .catch(function () {
        el.innerHTML = isEn
          ? '<p class="arena-error">Failed to load leaderboard data.</p>'
          : '<p class="arena-error">加载排行榜数据失败。</p>';
      });
  }

  function boot() {
    if (window.Tabulator) { initArena(); return; }
    loadScript(TABULATOR_JS, initArena, function () {
      var el = document.getElementById("arena-table");
      if (el) {
        el.innerHTML = isEn
          ? '<p class="arena-error">Failed to load leaderboard table component.</p>'
          : '<p class="arena-error">加载排行榜表格组件失败。</p>';
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
