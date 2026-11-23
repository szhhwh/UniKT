// UniKT Benchmark Leaderboard — Artificial Analysis Bar Chart + Modern Table UI.
// Framework-free client reading _static/arena/leaderboard.json with verified
// dataset cohort metadata (full vs. student-subsampled) and model taxonomy.
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

  // Verified dataset statistics from paper Table 2 (raw) & Table 5 (preprocessed/sampled)
  var DATASET_META = {
    "ASSIST09":   { students: 3852,  rawStudents: 4217,   questions: 17737,  kcs: 123,  sampled: 0,    doc: "assistments09",   canonical: true },
    "ASSIST12":   { students: 27485, rawStudents: 46667,  questions: 53065,  kcs: 265,  sampled: 0,    doc: "assistments12",   canonical: true },
    "ASSIST15":   { students: 19172, rawStudents: 19917,  questions: 100,    kcs: 100,  sampled: 0,    doc: "assistments15",   canonical: true },
    "A17PQ":      { students: 1707,  rawStudents: 1709,   questions: 2997,   kcs: 102,  sampled: 0,    doc: "assistments17",   canonical: true,  variant: "pq" },
    "A17":        { students: 1708,  rawStudents: 1709,   questions: 3162,   kcs: 102,  sampled: 0,    doc: "assistments17",   canonical: false, variant: "step" },
    "Algebra05":  { students: 574,   rawStudents: 574,    questions: 176630, kcs: 112,  sampled: 0,    doc: "algebra2005",     canonical: true },
    "Algebra06":  { students: 1326,  rawStudents: 1338,   questions: 559909, kcs: 491,  sampled: 0,    doc: "algebra2006",     canonical: true },
    "Anatomy":    { students: 14239, rawStudents: 18563,  questions: 5776,   kcs: 2894, sampled: 0,    doc: "practiceanatomy", canonical: true },
    "Bridge06":   { students: 1145,  rawStudents: 1146,   questions: 130836, kcs: 493,  sampled: 0,    doc: "bridge2006",      canonical: true },
    "MOOC-Radar": { students: 13597, rawStudents: 14224,  questions: 2511,   kcs: 2792, sampled: 0,    doc: "moocradar",       canonical: true },
    "NIPS34":     { students: 4918,  rawStudents: 4918,   questions: 948,    kcs: 57,   sampled: 0,    doc: "nips2020",        canonical: true },
    "Slepemapy":  { students: 84837, rawStudents: 90767,  questions: 2911,   kcs: 1458, sampled: 0,    doc: "slepemapy",       canonical: true },
    "EdNet-KT1":  { students: 5000,  rawStudents: 784304, questions: 11949,  kcs: 189,  sampled: 5000, doc: "ednet",           canonical: true },
    "Junyi15":    { students: 5000,  rawStudents: 247606, questions: 670,    kcs: 39,   sampled: 5000, doc: "junyi2015",       canonical: true },
    "XES3G5M":    { students: 2500,  rawStudents: 18066,  questions: 7066,   kcs: 835,  sampled: 2500, doc: "xes3g5m",         canonical: true }
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

  var allData = [];
  var totalModelCount = 43;
  var boardKind = "paper";
  var table = null;
  var tableMode = "";
  var applySeq = 0;
  var rankMap = new WeakMap();
  var currentRange = { min: 0.6, max: 0.95 };
  var rootPrefix = "";

  // UI state: default to per-dataset paper leaderboard (ASSIST09)
  var selectedDataset = "ASSIST09"; // "ALL" | dataset name
  var lastChartDataset = "ASSIST09";
  var searchQuery = "";
  var selectedMetric = "auc";       // "auc" | "acc"
  var selectedGran = "ALL";         // "ALL" | "Q" | "KC"
  var viewMode = "both";            // "both" | "chart" | "table"
  var scopeMode = "default";        // "default" (all evaluated on dataset) | "alt" (top 20)
  // DOM references
  var chartCardEl = null;
  var tableCardEl = null;
  var titleEl = null;
  var subtitleEl = null;
  var scopeSelectEl = null;
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

  function datasetUrl(dsName) {
    var d = DATASET_META[dsName];
    return d ? (rootPrefix + "datasets/catalog/" + d.doc + ".html") : "#";
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

  function syncRanks(rows) {
    rankMap = new WeakMap();
    if (!rows || !rows.length) return;
    var k = 0;
    for (var i = 0; i < rows.length; i++) {
      var rowComp = rows[i];
      var d = rowComp && rowComp.getData ? rowComp.getData() : rowComp;
      if (!d || d.protocol === "model_specific") continue;
      k += 1;
      rankMap.set(d, k);
    }
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

  function granFormatter(cell) {
    var g = cell.getValue();
    var cls = g === "Q" ? "is-q" : "is-kc";
    var txt = g === "Q" ? "Q-level" : "KC-level";
    return '<span class="arena-pill ' + cls + '">' + txt + "</span>";
  }

  function familyFormatter(cell) {
    var f = cell.getValue() || "seq";
    return '<span class="arena-fam-tag"><span class="arena-fam-dot is-' + esc(f) + '"></span>' + esc(FAMILY_LABELS[f] || f) + "</span>";
  }

  function datasetFormatter(cell) {
    var ds = cell.getValue();
    var meta = DATASET_META[ds];
    var tag = "";
    if (meta && meta.sampled) {
      tag = ' <span class="arena-tab-tag">' + fmtInt(meta.sampled) + "</span>";
    }
    return '<span><a class="arena-ds-link" href="' + esc(datasetUrl(ds)) + '">' + esc(ds) + "</a>" + tag + "</span>";
  }

  function makeScoreFormatter(stdField, showBar) {
    return function (cell) {
      var v = cell.getValue();
      if (v == null) return "—";
      var d = cell.getRow().getData();
      var std = stdField ? d[stdField] : null;
      var stdHtml = std != null ? ('<span class="arena-metric-std">±' + std.toFixed(3) + "</span>") : "";
      var barHtml = "";
      if (showBar) {
        var span = Math.max(0.03, currentRange.max - currentRange.min);
        var pct = Math.max(8, Math.min(100, ((v - currentRange.min) / span) * 100));
        barHtml = '<span class="arena-metric-bar-track"><span class="arena-metric-bar-fill" style="width:' + pct.toFixed(1) + '%"></span></span>';
      }
      return '<div class="arena-metric-wrap"><span class="arena-metric-nums"><span class="arena-metric-main">' + v.toFixed(3) + "</span>" + stdHtml + "</span>" + barHtml + "</div>";
    };
  }

  function cohortFormatter(cell) {
    var d = cell.getRow().getData();
    var dsMeta = DATASET_META[d.dataset];
    if (d.protocol === "model_specific") {
      var extra = (dsMeta && dsMeta.sampled) ? (" · " + fmtInt(dsMeta.sampled)) : "";
      return '<span class="arena-pill is-spec">' + (isEn ? "Model-specific protocol" : "模型专用协议") + extra + "</span>";
    }
    if (dsMeta && dsMeta.sampled) {
      return '<span class="arena-pill is-sampled">' +
        (isEn ? ("Subsampled · " + fmtInt(dsMeta.sampled)) : ("学生级子采样 · " + fmtInt(dsMeta.sampled))) +
        "</span>";
    }
    var stu = dsMeta ? fmtInt(dsMeta.students) : "";
    return '<span class="arena-pill is-full">' +
      (isEn ? ("Full · " + stu + " students") : ("全量 · 保留 " + stu + " 人")) +
      "</span>";
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

  function getColumnsForMode(mode) {
    var cols = [
      { title: "#", width: 48, minWidth: 48, hozAlign: "center", headerHozAlign: "center", headerSort: false, formatter: rankFormatter },
      { title: isEn ? "Model" : "模型", field: "model", sorter: "string", minWidth: 128, widthGrow: 1.2, formatter: modelFormatter },
      { title: isEn ? "Granularity" : "交互粒度", field: "gran", sorter: "string", width: 128, minWidth: 128, hozAlign: "center", headerHozAlign: "center", formatter: granFormatter },
      { title: isEn ? "Model Family" : "模型家族", field: "family", sorter: "string", minWidth: 162, widthGrow: 1.25, formatter: familyFormatter }
    ];
    if (mode === "ALL") {
      cols.push({
        title: isEn ? "Dataset" : "数据集",
        field: "dataset",
        sorter: "string",
        minWidth: 146,
        widthGrow: 1.2,
        formatter: datasetFormatter
      });
    }
    cols.push(
      {
        title: "AUC",
        field: "auc",
        sorter: metricSorter,
        hozAlign: "right",
        headerHozAlign: "right",
        minWidth: 126,
        widthGrow: 1.05,
        formatter: makeScoreFormatter("auc_std", true)
      },
      {
        title: "ACC",
        field: "acc",
        sorter: metricSorter,
        hozAlign: "right",
        headerHozAlign: "right",
        minWidth: 118,
        widthGrow: 0.95,
        formatter: makeScoreFormatter("acc_std", false)
      },
      {
        title: isEn ? "Evaluation Setting" : "评测设置",
        field: "notes",
        sorter: "string",
        minWidth: 184,
        widthGrow: 1.3,
        hozAlign: "left",
        tooltip: cohortTooltip,
        formatter: cohortFormatter
      }
    );
    return cols;
  }

  function updateScopeOptions(chartPoolCount) {
    if (!scopeSelectEl) return;
    var prev = scopeMode;
    scopeSelectEl.innerHTML = "";
    var oAll = document.createElement("option");
    oAll.value = "default";
    oAll.textContent = isEn
      ? (chartPoolCount + " of " + totalModelCount + " models")
      : (chartPoolCount + " / " + totalModelCount + " 个评测模型");
    scopeSelectEl.appendChild(oAll);
    if (chartPoolCount > 20) {
      var oTop = document.createElement("option");
      oTop.value = "alt";
      oTop.textContent = isEn ? "Top 20 models" : "前 20 个模型 (Top 20)";
      scopeSelectEl.appendChild(oTop);
    } else {
      scopeMode = "default";
    }
    scopeSelectEl.value = scopeMode === prev ? scopeMode : "default";
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

    var metaLine2 = "";
    var dsMeta = DATASET_META[item.dataset];
    if (item.protocol === "model_specific") {
      metaLine2 = isEn ? "Model-specific protocol (excluded from shared ranking)" : "模型专用协议（不纳入共享协议直接排名）";
    } else if (dsMeta && dsMeta.sampled) {
      metaLine2 = isEn
        ? ("Student-level subsampling: " + fmtInt(dsMeta.sampled) + " retained (" + fmtInt(dsMeta.rawStudents) + " raw)")
        : ("学生级随机子采样：保留 " + fmtInt(dsMeta.sampled) + " 名学生（原始 " + fmtInt(dsMeta.rawStudents) + "）");
    } else if (dsMeta) {
      metaLine2 = isEn
        ? ("Complete release: " + fmtInt(dsMeta.students) + " retained students")
        : ("全量数据集：保留 " + fmtInt(dsMeta.students) + " 名学生");
    }

    var granCls = item.gran === "Q" ? "is-q" : "is-kc";
    tooltipEl.innerHTML =
      '<div class="aa-tip-head">' +
        '<span class="aa-tip-model"><span class="aa-gran-chip ' + granCls + '">' + esc(item.gran) + "</span>" + esc(item.model) + "</span>" +
        '<span class="aa-tip-rank">' + esc(rankTxt) + "</span>" +
      "</div>" +
      '<div class="aa-tip-score">' + mainVal.toFixed(3) +
        (stdVal != null ? ('<small>± ' + stdVal.toFixed(3) + " " + mLabel + "</small>") : ('<small>' + mLabel + "</small>")) +
      "</div>" +
      '<div class="aa-tip-meta">' + esc(metaLine1) + "<br>" + esc(metaLine2) + "</div>" +
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

  function renderChart(chartItems, chartDs, totalPoolCount, filteredTableCount) {
    if (!stageEl || !titleEl || !subtitleEl || !legendEl) return;
    hideTooltip();

    var mUpper = selectedMetric.toUpperCase();
    var dsMeta = DATASET_META[chartDs];
    if (boardKind === "live") {
      titleEl.textContent = isEn
        ? ("UniKT Live Leaderboard: " + chartDs + " — " + mUpper)
        : ("UniKT 最新排行榜：" + chartDs + " — " + mUpper);
    } else {
      titleEl.textContent = isEn
        ? ("UniKT Benchmark (KDD 2027): " + chartDs + " — " + mUpper)
        : ("UniKT 论文基准（KDD 2027）：" + chartDs + " — " + mUpper);
    }

    var pillHtml = "";
    var detailTxt = "";
    if (dsMeta && dsMeta.sampled) {
      pillHtml = '<span class="arena-cohort-pill is-sampled">' +
        (isEn
          ? ("Student-level Subsampling · " + fmtInt(dsMeta.sampled) + " retained students (raw: " + fmtInt(dsMeta.rawStudents) + ")")
          : ("学生级随机子采样 · 保留 " + fmtInt(dsMeta.sampled) + " 名学生（原始 " + fmtInt(dsMeta.rawStudents) + "）")) +
        "</span>";
    } else if (dsMeta) {
      pillHtml = '<span class="arena-cohort-pill is-full">' +
        (isEn
          ? ("Complete Release · " + fmtInt(dsMeta.students) + " retained students")
          : ("全量数据集 · 保留 " + fmtInt(dsMeta.students) + " 名学生")) +
        "</span>";
    }
    if (dsMeta) {
      var varTxt = "";
      if (dsMeta.variant === "pq") {
        varTxt = isEn ? " · Question-level reconstruction" : " · 题目级重构";
      } else if (dsMeta.variant === "step") {
        varTxt = isEn ? " · Original step-level records" : " · 原始步级记录";
      }
      var allExtra = selectedDataset === "ALL"
        ? (filteredTableCount == null || filteredTableCount === allData.length
            ? (isEn ? (" · Table below shows all " + allData.length + " records") : (" · 下方表格当前展示全部 " + allData.length + " 条记录"))
            : (isEn ? (" · Table below shows " + filteredTableCount + " of " + allData.length + " records") : (" · 下方表格当前展示 " + filteredTableCount + " / " + allData.length + " 条记录")))
        : "";
      detailTxt = (isEn
        ? (fmtInt(dsMeta.questions) + " questions · " + fmtInt(dsMeta.kcs) + " KCs · 5-fold cross-validation")
        : (fmtInt(dsMeta.questions) + " 道题目 · " + fmtInt(dsMeta.kcs) + " 个知识组件 (KC) · 5 折交叉验证")) + varTxt + allExtra;
    }
    subtitleEl.innerHTML = pillHtml + "<span>" + esc(detailTxt) + "</span>";
    updateScopeOptions(totalPoolCount);

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

      var granChip = document.createElement("span");
      granChip.className = "aa-gran-chip " + (item.gran === "Q" ? "is-q" : "is-kc");
      granChip.textContent = item.gran;
      axisCell.appendChild(granChip);

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

  function applyFilters() {
    var q = searchQuery.trim().toLowerCase();

    function sortRowsDesc(a, b) {
      var aSpec = a.protocol === "model_specific";
      var bSpec = b.protocol === "model_specific";
      if (aSpec !== bSpec) return aSpec ? 1 : -1;
      var diff = sortMetricVal(b, selectedMetric) - sortMetricVal(a, selectedMetric);
      if (diff !== 0) return diff;
      return String(a.model || "").localeCompare(String(b.model || ""));
    }

    // 1. Prepare Chart & Table items
    var chartDs = selectedDataset === "ALL" ? lastChartDataset : selectedDataset;
    var rawChartPool = allData.filter(function (r) { return r.dataset === chartDs; });

    var filteredChart = rawChartPool.filter(function (r) {
      if (selectedGran !== "ALL" && r.gran !== selectedGran) return false;
      if (q && r.model.toLowerCase().indexOf(q) === -1) return false;
      return true;
    });
    filteredChart.sort(sortRowsDesc);

    var nextTableMode = selectedDataset === "ALL" ? "ALL" : "SINGLE";
    var tableSource = allData.filter(function (r) {
      if (nextTableMode === "SINGLE" && r.dataset !== selectedDataset) return false;
      return true;
    });
    var filteredTable = tableSource.filter(function (r) {
      if (selectedGran !== "ALL" && r.gran !== selectedGran) return false;
      if (q && r.model.toLowerCase().indexOf(q) === -1) return false;
      return true;
    });

    var totalPoolCount = filteredChart.length;
    var chartDisplay = scopeMode === "alt" ? filteredChart.slice(0, 20) : filteredChart;

    renderChart(chartDisplay, chartDs, totalPoolCount, filteredTable.length);

    if (filteredTable.length) {
      var aucVals = filteredTable.map(function (r) { return r.auc; });
      currentRange = {
        min: Math.min.apply(null, aucVals),
        max: Math.max.apply(null, aucVals)
      };
    }

    syncRanks(filteredTable.slice().sort(sortRowsDesc));

    if (!table) {
      tableMode = nextTableMode;
      table = new Tabulator("#arena-table", {
        data: filteredTable,
        layout: "fitColumns",
        height: "520px",
        placeholder: isEn ? "No matching records" : "暂无匹配数据",
        initialSort: [{ column: selectedMetric, dir: "desc" }],
        columns: getColumnsForMode(nextTableMode)
      });
      table.on("dataSorted", function (_sorters, rows) {
        syncRanks(rows && rows.length ? rows : (table ? table.getRows("active") : []));
      });
      table.on("renderComplete", function () {
        if (table) syncRanks(table.getRows("active"));
      });
    } else {
      var mySeq = ++applySeq;
      if (tableMode !== nextTableMode) {
        tableMode = nextTableMode;
        table.setColumns(getColumnsForMode(nextTableMode));
      }
      Promise.resolve(table.setData(filteredTable)).then(function () {
        if (mySeq !== applySeq || !table) return;
        table.setSort(selectedMetric, "desc");
        syncRanks(table.getRows("active"));
      });
    }

    updateCount(filteredTable.length);
  }

  function setActiveTab(btn) {
    var tabs = document.querySelectorAll(".arena-tab");
    for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove("active");
    btn.classList.add("active");
  }

  function makeTabBtn(labelHtml, key, active, titleTxt) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = active ? "arena-tab active" : "arena-tab";
    b.innerHTML = labelHtml;
    if (titleTxt) b.title = titleTxt;
    b.onclick = function () {
      selectedDataset = key;
      if (key !== "ALL") lastChartDataset = key;
      scopeMode = "default";
      setActiveTab(b);
      applyFilters();
    };
    return b;
  }

  function buildTabs(datasets) {
    var el = document.getElementById("arena-tabs");
    if (!el) return;
    el.innerHTML = "";

    el.appendChild(makeTabBtn(
      esc(isEn ? ("All Records (" + allData.length + ")") : ("全部记录 (" + allData.length + ")")),
      "ALL",
      selectedDataset === "ALL",
      isEn
        ? ("All " + allData.length + " benchmark records across the 15 dataset settings")
        : ("全部 15 个评测设置下的 " + allData.length + " 条基准结果")
    ));

    var sep1 = document.createElement("span");
    sep1.className = "arena-tab-sep";
    el.appendChild(sep1);

    var fullList = [];
    var sampledList = [];
    datasets.forEach(function (ds) {
      var m = DATASET_META[ds];
      if (m && m.sampled) sampledList.push(ds);
      else fullList.push(ds);
    });

    var lblFull = document.createElement("span");
    lblFull.className = "arena-tab-group-label";
    lblFull.textContent = isEn ? "Full Release:" : "全量数据集:";
    el.appendChild(lblFull);

    fullList.forEach(function (ds) {
      var m = DATASET_META[ds];
      var badge = "";
      if (ds === "A17PQ") {
        badge = '<span class="arena-tab-tag is-pq">' + (isEn ? "Q-recon" : "题目重构") + "</span>";
      } else if (ds === "A17") {
        badge = '<span class="arena-tab-tag">' + (isEn ? "Step" : "步级") + "</span>";
      }
      var tip = m
        ? (isEn
            ? ("Complete release: " + fmtInt(m.students) + " retained students (" + fmtInt(m.rawStudents) + " raw)")
            : ("全量数据集：保留 " + fmtInt(m.students) + " 名学生（原始发布 " + fmtInt(m.rawStudents) + " 名）"))
        : ds;
      el.appendChild(makeTabBtn(esc(ds) + badge, ds, selectedDataset === ds, tip));
    });

    var sep2 = document.createElement("span");
    sep2.className = "arena-tab-sep";
    el.appendChild(sep2);

    var lblSub = document.createElement("span");
    lblSub.className = "arena-tab-group-label";
    lblSub.textContent = isEn ? "Subsampled:" : "学生级子采样:";
    el.appendChild(lblSub);

    sampledList.forEach(function (ds) {
      var m = DATASET_META[ds];
      var badge = '<span class="arena-tab-tag">' + fmtInt(m.sampled) + "</span>";
      var tip = isEn
        ? ("Random student-level subsampling: " + fmtInt(m.sampled) + " retained students (" + fmtInt(m.rawStudents) + " raw)")
        : ("学生级随机子采样：保留 " + fmtInt(m.sampled) + " 名学生（原始发布 " + fmtInt(m.rawStudents) + " 名）");
      el.appendChild(makeTabBtn(esc(ds) + badge, ds, selectedDataset === ds, tip));
    });
  }

  function mountShell() {
    var tabsEl = document.getElementById("arena-tabs");
    var tableEl = document.getElementById("arena-table");
    var searchWrap = document.querySelector(".arena-search");
    var countEl = document.getElementById("arena-count");
    if (!tabsEl || !tableEl) return;

    var shell = document.createElement("div");
    shell.className = "arena-shell";
    tabsEl.parentNode.insertBefore(shell, tabsEl);
    shell.appendChild(tabsEl);

    // 1. Artificial Analysis Bar Chart Card
    chartCardEl = document.createElement("section");
    chartCardEl.className = "aa-card";
    chartCardEl.innerHTML =
      '<div class="aa-header">' +
        '<div class="aa-title-block">' +
          '<h3 class="aa-title"></h3>' +
          '<div class="aa-subtitle"></div>' +
        "</div>" +
        '<div class="aa-controls">' +
          '<div class="aa-controls-row">' +
            '<div class="aa-seg" data-role="metric">' +
              '<button type="button" data-val="auc" class="active">AUC</button>' +
              '<button type="button" data-val="acc">ACC</button>' +
            "</div>" +
            '<div class="aa-icon-group" data-role="view">' +
              '<button type="button" class="aa-icon-btn active" data-val="both" title="' + (isEn ? "Chart + Table" : "柱状图 + 数据表") + '">' +
                '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="8" rx="1.5"/><rect x="3" y="14" width="18" height="7" rx="1.5"/></svg>' +
              "</button>" +
              '<button type="button" class="aa-icon-btn" data-val="chart" title="' + (isEn ? "Bar Chart Only" : "仅看柱状图") + '">' +
                '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>' +
              "</button>" +
              '<button type="button" class="aa-icon-btn" data-val="table" title="' + (isEn ? "Table Only" : "仅看数据表") + '">' +
                '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/></svg>' +
              "</button>" +
            "</div>" +
          "</div>" +
          '<div class="aa-controls-row">' +
            '<select class="aa-select" aria-label="Model scope"></select>' +
            '<div class="aa-seg" data-role="gran">' +
              '<button type="button" data-val="ALL" class="active">' + (isEn ? "All" : "全部") + "</button>" +
              '<button type="button" data-val="Q">Q-level</button>' +
              '<button type="button" data-val="KC">KC-level</button>' +
            "</div>" +
          "</div>" +
        "</div>" +
      "</div>" +
      '<div class="aa-legend-row">' +
        '<div class="aa-legend"></div>' +
        '<div class="aa-watermark"><span class="aa-watermark-dot"></span>' +
          (boardKind === "live" ? "UniKT Live Leaderboard" : "UniKT Benchmark (KDD 2027)") +
        '</div>' +
      "</div>" +
      '<div class="aa-chart-scroll"><div class="aa-chart-stage"></div></div>' +
      '<div class="aa-tooltip"></div>';

    shell.appendChild(chartCardEl);

    titleEl = chartCardEl.querySelector(".aa-title");
    subtitleEl = chartCardEl.querySelector(".aa-subtitle");
    scopeSelectEl = chartCardEl.querySelector(".aa-select");
    legendEl = chartCardEl.querySelector(".aa-legend");
    stageEl = chartCardEl.querySelector(".aa-chart-stage");
    tooltipEl = chartCardEl.querySelector(".aa-tooltip");

    // 2. Modern Table Card
    tableCardEl = document.createElement("section");
    tableCardEl.className = "arena-table-card";

    var toolbar = document.createElement("div");
    toolbar.className = "arena-table-toolbar";
    if (searchWrap) toolbar.appendChild(searchWrap);

    var rightBox = document.createElement("div");
    rightBox.className = "arena-toolbar-right";

    // Keep a secondary view restore button inside table toolbar when chart is hidden
    var showChartBtn = document.createElement("button");
    showChartBtn.type = "button";
    showChartBtn.className = "arena-tab";
    showChartBtn.style.display = "none";
    showChartBtn.textContent = isEn ? "Show Bar Chart" : "显示柱状图";
    rightBox.appendChild(showChartBtn);

    if (countEl) rightBox.appendChild(countEl);
    toolbar.appendChild(rightBox);

    tableCardEl.appendChild(toolbar);
    tableCardEl.appendChild(tableEl);
    shell.appendChild(tableCardEl);

    // Wire controls
    function syncViewVisibility() {
      var chartBtns = chartCardEl.querySelectorAll('[data-role="view"] button');
      for (var i = 0; i < chartBtns.length; i++) {
        chartBtns[i].classList.toggle("active", chartBtns[i].getAttribute("data-val") === viewMode);
      }
      chartCardEl.classList.toggle("is-hidden", viewMode === "table");
      tableCardEl.classList.toggle("is-hidden", viewMode === "chart");
      showChartBtn.style.display = viewMode === "table" ? "inline-flex" : "none";
      if (table && viewMode !== "chart") {
        setTimeout(function () { table.redraw(); }, 20);
      }
    }

    showChartBtn.onclick = function () {
      viewMode = "both";
      syncViewVisibility();
    };

    chartCardEl.querySelector('[data-role="view"]').addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-val]");
      if (!btn) return;
      viewMode = btn.getAttribute("data-val");
      syncViewVisibility();
    });

    chartCardEl.querySelector('[data-role="metric"]').addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-val]");
      if (!btn) return;
      selectedMetric = btn.getAttribute("data-val");
      var btns = chartCardEl.querySelectorAll('[data-role="metric"] button');
      for (var i = 0; i < btns.length; i++) {
        btns[i].classList.toggle("active", btns[i] === btn);
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
      }
      applyFilters();
    });

    scopeSelectEl.addEventListener("change", function () {
      scopeMode = scopeSelectEl.value;
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
    var dataFile = el.getAttribute("data-src") || "leaderboard.json";
    boardKind = el.getAttribute("data-board") || "paper";
    var primaryUrl = rootPrefix + "_static/arena/" + dataFile;

    mountShell();

    function fetchJson(url) {
      return fetch(url).then(function (r) {
        if (!r.ok) throw new Error("HTTP error " + r.status);
        return r.json();
      });
    }

    fetchJson(primaryUrl)
      .catch(function () {
        var fallbackUrl = (rootPrefix === "" ? "../_static/arena/" : "_static/arena/") + dataFile;
        return fetchJson(fallbackUrl);
      })
      .then(function (data) {
        var modelSeen = {};
        totalModelCount = 0;
        allData = data.map(function (r) {
          if (!modelSeen[r.model]) {
            modelSeen[r.model] = true;
            totalModelCount += 1;
          }
          var meta = MODEL_META[r.model] || { gran: "Q", year: "", family: "seq", doc: "" };
          return Object.assign({}, r, {
            gran: meta.gran,
            family: meta.family,
            year: meta.year
          });
        });

        var datasets = [];
        var seen = {};
        allData.forEach(function (r) {
          if (!seen[r.dataset]) {
            seen[r.dataset] = true;
            datasets.push(r.dataset);
          }
        });
        datasets.sort();
        buildTabs(datasets);
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
