// UniKT model-zoo leaderboard.
// Framework-free rewrite of the original VuePress client (docs/src/.vuepress/client.ts).
// Only Tabulator's structural base CSS is loaded; arena.css colours it from Furo's
// variables, so light/dark follows the page theme without JS. Data is fetched from
// _static/arena/leaderboard.json.
(function () {
  "use strict";

  var TABULATOR_JS = "https://unpkg.com/tabulator-tables@6.5.0/dist/js/tabulator.min.js";
  var TABULATOR_CSS = "https://unpkg.com/tabulator-tables@6.5.0/dist/css/tabulator.min.css";

  function loadScript(src, onload) {
    var s = document.createElement("script");
    s.src = src;
    s.onload = onload;
    document.head.appendChild(s);
  }

  function loadStyle(href) {
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    document.head.appendChild(link);
  }

  var allData = [];
  var table = null;
  var isEn = (document.documentElement.lang || "").toLowerCase().startsWith("en");

  function foldsTitle(folds) {
    return (isEn ? "Folds: " : "各折：") + (folds || []).map(function (v) { return v.toFixed(4); }).join(", ");
  }

  function stdFormatter(cell) {
    var v = cell.getValue();
    return v != null ? v.toFixed(6) : "—";
  }

  var notesDict = {
    "未调参数据": "Default parameters",
    "随机采样5000个学生": "Randomly sampled 5,000 students"
  };

  function noteFormatter(cell) {
    var v = cell.getValue();
    if (!v) return "—";
    if (isEn && notesDict[v]) {
      v = notesDict[v];
    }
    return '<span class="arena-note">' + v + "</span>";
  }

  function noteTooltip(e, cell) {
    var v = cell.getValue();
    if (!v) return "";
    return (isEn && notesDict[v]) ? notesDict[v] : v;
  }

  var numberCol = { sorter: "number", hozAlign: "right", headerHozAlign: "right", minWidth: 76 };

  function col(extra) {
    return Object.assign({}, numberCol, extra);
  }

  var columns = [
    { title: "#", width: 40, hozAlign: "right", headerHozAlign: "right", headerSort: false, formatter: "rownum", cssClass: "arena-rank" },
    { title: isEn ? "Model" : "模型", field: "model", sorter: "string", minWidth: 118, widthGrow: 2, cssClass: "arena-model" },
    { title: isEn ? "Dataset" : "数据集", field: "dataset", sorter: "string", minWidth: 88, widthGrow: 2 },
    col({
      title: "AUC", field: "auc",
      formatter: function (cell) {
        return '<span title="' + foldsTitle(cell.getRow().getData().auc_folds) + '">' + cell.getValue().toFixed(4) + "</span>";
      },
    }),
    col({ title: isEn ? "AUC Std" : "AUC 标准差", field: "auc_std", minWidth: 122, formatter: stdFormatter, cssClass: "arena-std" }),
    col({
      title: "ACC", field: "acc",
      formatter: function (cell) {
        return '<span title="' + foldsTitle(cell.getRow().getData().acc_folds) + '">' + cell.getValue().toFixed(4) + "</span>";
      },
    }),
    col({ title: isEn ? "ACC Std" : "ACC 标准差", field: "acc_std", minWidth: 122, formatter: stdFormatter, cssClass: "arena-std" }),
    {
      title: isEn ? "Notes" : "备注", field: "notes", sorter: "string", minWidth: 76, widthGrow: 1,
      tooltip: noteTooltip,
      formatter: noteFormatter,
    },
  ];

  function updateCount(n) {
    var el = document.getElementById("arena-count");
    if (el) el.textContent = isEn ? (n + " records in total") : ("共 " + n + " 条记录");
  }

  function buildTable(data) {
    if (table) {
      table.setData(data);
    } else {
      table = new Tabulator("#arena-table", {
        data: data,
        layout: "fitColumns",
        height: "500px",
        placeholder: isEn ? "No data available" : "暂无数据",
        initialSort: [{ column: "auc", dir: "desc" }],
        columns: columns,
      });
      var search = document.getElementById("arena-filter");
      if (search) {
        search.placeholder = isEn ? "Filter by model name..." : "按模型名称筛选...";
        search.addEventListener("input", function () { table.setFilter("model", "like", search.value); });
      }
    }
    updateCount(data.length);
  }

  function createTab(text, active) {
    var b = document.createElement("button");
    b.textContent = text;
    b.className = active ? "arena-tab active" : "arena-tab";
    return b;
  }

  function setActiveTab(active) {
    var tabs = document.querySelectorAll(".arena-tab");
    for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove("active");
    active.classList.add("active");
  }

  function buildTabs(datasets) {
    var el = document.getElementById("arena-tabs");
    if (!el) return;
    el.innerHTML = "";
    var all = createTab(isEn ? "All" : "全部", true);
    all.onclick = function () { setActiveTab(all); buildTable(allData); };
    el.appendChild(all);
    datasets.forEach(function (ds) {
      var b = createTab(ds, false);
      b.onclick = function () { setActiveTab(b); buildTable(allData.filter(function (r) { return r.dataset === ds; })); };
      el.appendChild(b);
    });
  }

  function initArena() {
    var el = document.getElementById("arena-table");
    if (!el || el.dataset.initialized) return;
    el.dataset.initialized = "true";

    loadStyle(TABULATOR_CSS);

    var script = document.querySelector('script[src*="arena.js"]');
    var prefix = "";
    if (script) {
      var src = script.getAttribute("src") || "";
      var idx = src.indexOf("_static/");
      if (idx !== -1) {
        prefix = src.substring(0, idx);
      }
    }
    var primaryUrl = prefix + "_static/arena/leaderboard.json";

    fetch(primaryUrl)
      .catch(function () {
        var fallbackUrl = prefix === "" ? "../_static/arena/leaderboard.json" : "_static/arena/leaderboard.json";
        return fetch(fallbackUrl);
      })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP error " + r.status);
        return r.json();
      })
      .then(function (data) {
        allData = data;
        var datasets = [];
        var seen = {};
        data.forEach(function (r) { if (!seen[r.dataset]) { seen[r.dataset] = true; datasets.push(r.dataset); } });
        datasets.sort();
        buildTabs(datasets);
        buildTable(data);
      })
      .catch(function () {
        el.innerHTML = isEn ? '<p class="arena-error">Failed to load leaderboard data.</p>' : '<p class="arena-error">加载排行榜数据失败。</p>';
      });
  }

  function boot() {
    var search = document.getElementById("arena-filter");
    if (search && isEn) {
      search.placeholder = "Filter by model name...";
    }
    if (window.Tabulator) { initArena(); return; }
    loadScript(TABULATOR_JS, initArena);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
