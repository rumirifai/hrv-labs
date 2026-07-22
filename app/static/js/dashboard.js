/* =========================================================
   dashboard.js — Dashboard HRV Research Visualisation
   All init code is safely inside DOMContentLoaded.
   ========================================================= */

"use strict";

// Init Theme immediately to prevent flash of content
(function() {
  var savedTheme = localStorage.getItem("theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
})();

// ── State ────────────────────────────────────────────────
let leaderboardData   = [];
let labelDistChart    = null;
let allAccuracyChart  = null;
let healthDistChart   = null;
let signalChart       = null;
let featContribChart  = null;
let currentSensor     = "ecg";
let currentSigType    = "ppg";
let extractedFeatures = null;
let sortState         = { col: "rank", asc: true };

// Preset demo data (real CLAS dataset values)
const PRESETS = {
  rendah1: { MeanNN_ms:636.929, SDNN_ms:58.956, RMSSD_ms:52.378, LF:0.050, HF:0.038, LF_HF_Ratio:1.292, label:"Part10 — Baseline (Beban Rendah)" },
  rendah2: { MeanNN_ms:699.770, SDNN_ms:84.269, RMSSD_ms:53.383, LF:0.038, HF:0.027, LF_HF_Ratio:1.443, label:"Part14 — Baseline (Beban Rendah)" },
  rendah3: { MeanNN_ms:793.229, SDNN_ms:66.911, RMSSD_ms:83.105, LF:0.009, HF:0.035, LF_HF_Ratio:0.270, label:"Part12 — Baseline (Beban Rendah)" },
  tinggi1: { MeanNN_ms:652.344, SDNN_ms:42.791, RMSSD_ms:45.930, LF:0.035, HF:0.060, LF_HF_Ratio:1.100, label:"Part10 — Math Test (Beban Tinggi)" },
  tinggi2: { MeanNN_ms:603.125, SDNN_ms:17.161, RMSSD_ms:15.502, LF:0.040, HF:0.020, LF_HF_Ratio:2.100, label:"Part10 — IQ Test (Beban Tinggi)" },
  mean:    { MeanNN_ms:700.412, SDNN_ms:53.907, RMSSD_ms:69.428, LF:0.035, HF:0.060, LF_HF_Ratio:1.109, label:"Nilai rata-rata dataset PPG" },
};

// ── Helper: show error on screen ─────────────────────────
function showError(msg) {
  let box = document.getElementById("js-error-box");
  if (!box) {
    box = document.createElement("div");
    box.id = "js-error-box";
    box.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:9999;background:#4d1e1e;color:#f85149;padding:10px 20px;font-size:13px;font-family:monospace;border-bottom:2px solid #f85149;";
    document.body.prepend(box);
  }
  box.textContent = "⚠️ JS Error: " + msg;
  console.error(msg);
}

// ── Main Init — semua kode berjalan setelah DOM siap ─────
document.addEventListener("DOMContentLoaded", function() {

  // Safety check for Chart.js
  if (typeof Chart === "undefined") {
    showError("Chart.js gagal dimuat. Periksa apakah file /static/js/chart.umd.min.js tersedia.");
    return;
  }

  // Theme setup helper
  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
    var icon = document.getElementById("theme-toggle-icon");
    if (icon) {
      icon.textContent = theme === "light" ? "☀️" : "🌙";
    }

    var textColor = theme === "light" ? "#57606a" : "#7d8590";
    var gridColor = theme === "light" ? "#d0d7de" : "#21262d";

    Chart.defaults.color = textColor;
    Chart.defaults.borderColor = gridColor;

    // Update dynamically existing charts
    if (labelDistChart) {
      labelDistChart.options.scales.y.grid.color = gridColor;
      labelDistChart.update();
    }
    if (allAccuracyChart) {
      allAccuracyChart.options.scales.x.grid.color = gridColor;
      allAccuracyChart.update();
    }
    if (healthDistChart) {
      healthDistChart.options.plugins.legend.labels.color = textColor;
      healthDistChart.update();
    }
    if (signalChart) {
      signalChart.options.scales.x.grid.color = gridColor;
      signalChart.options.scales.y.grid.color = gridColor;
      if (signalChart.options.scales.x.title) signalChart.options.scales.x.title.color = textColor;
      if (signalChart.options.scales.y.title) signalChart.options.scales.y.title.color = textColor;
      signalChart.update();
    }
    if (featContribChart) {
      featContribChart.options.scales.x.grid.color = gridColor;
      featContribChart.update();
    }
  }

  // Initial theme load
  var currentTheme = localStorage.getItem("theme") || "dark";
  applyTheme(currentTheme);

  // Setup toggle listener
  var toggleBtn = document.getElementById("theme-toggle");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", function() {
      var nextTheme = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
      applyTheme(nextTheme);
    });
  }

  Chart.defaults.font.family  = "'Inter', sans-serif";

  // ── Tab Navigation ──────────────────────────────────────
  document.querySelectorAll(".nav-tab").forEach(function(btn) {
    btn.addEventListener("click", function() {
      var tabId = btn.dataset.tab;
      document.querySelectorAll(".nav-tab").forEach(function(b) { b.classList.remove("active"); });
      document.querySelectorAll(".tab-panel").forEach(function(p) { p.classList.remove("active"); });
      btn.classList.add("active");
      var panel = document.getElementById("tab-" + tabId);
      if (panel) panel.classList.add("active");

      if (tabId === "dataset" && !labelDistChart) loadDataset("ecg");
      if (tabId === "models"  && leaderboardData.length === 0) loadLeaderboard();
    });
  });

  // ── Upload zone drag & drop ─────────────────────────────
  var uploadZone = document.getElementById("upload-zone");
  if (uploadZone) {
    uploadZone.addEventListener("dragover", function(e) {
      e.preventDefault();
      uploadZone.classList.add("drag-over");
    });
    uploadZone.addEventListener("dragleave", function() {
      uploadZone.classList.remove("drag-over");
    });
    uploadZone.addEventListener("drop", function(e) {
      e.preventDefault();
      uploadZone.classList.remove("drag-over");
      var file = e.dataTransfer.files[0];
      if (file) processSignalFile(file);
    });
  }

  // ── Lightbox ESC key ────────────────────────────────────
  document.addEventListener("keydown", function(e) {
    if (e.key === "Escape") {
      var lb = document.getElementById("lightbox");
      if (lb) lb.classList.remove("open");
    }
  });

  // ── Auto-load Dataset tab on start ─────────────────────
  loadDataset("ecg");

}); // end DOMContentLoaded


// ══════════════════════════════════════════════════════════
//  DATASET TAB
// ══════════════════════════════════════════════════════════

function loadDataset(sensor) {
  currentSensor = sensor;

  var btnEcg = document.getElementById("btn-ecg-ds");
  var btnPpg = document.getElementById("btn-ppg-ds");
  if (btnEcg) btnEcg.classList.toggle("active", sensor === "ecg");
  if (btnPpg) btnPpg.classList.toggle("active", sensor === "ppg");

  var elTotal   = document.getElementById("ds-total");
  var elRendah  = document.getElementById("ds-rendah");
  var elTinggi  = document.getElementById("ds-tinggi");
  if (elTotal)  elTotal.textContent  = "…";
  if (elRendah) elRendah.textContent = "…";
  if (elTinggi) elTinggi.textContent = "…";

  var featBox = document.getElementById("ds-feat-stats");
  if (featBox) featBox.innerHTML = "<p style='color:var(--text-muted);text-align:center;padding:2rem'>Memuat statistik...</p>";

  var sampleBody = document.getElementById("ds-sample-body");
  if (sampleBody) sampleBody.innerHTML = "<tr><td colspan='9' style='text-align:center;color:var(--text-muted);padding:2rem'>Memuat data...</td></tr>";

  fetch("/api/dataset/" + sensor)
    .then(function(resp) {
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      return resp.json();
    })
    .then(function(d) {
      if (elTotal)  elTotal.textContent  = d.total_rows || "—";
      if (elRendah) elRendah.textContent = (d.label_counts && d.label_counts["Rendah"]) || "—";
      if (elTinggi) elTinggi.textContent = (d.label_counts && d.label_counts["Tinggi"]) || "—";

      renderLabelDistChart(d.label_counts);
      renderFeatStats(d.stats);
      renderSampleTable(d.sample);
    })
    .catch(function(err) {
      showError("Gagal memuat dataset: " + err.message);
      if (elTotal)  elTotal.textContent  = "Error";
      if (elRendah) elRendah.textContent = "Error";
      if (elTinggi) elTinggi.textContent = "Error";
    });
}

function renderLabelDistChart(counts) {
  var canvas = document.getElementById("chart-label-dist");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  if (labelDistChart) { labelDistChart.destroy(); labelDistChart = null; }

  labelDistChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Beban Rendah", "Beban Tinggi"],
      datasets: [{
        label: "Jumlah Sampel",
        data: [(counts && counts["Rendah"]) || 0, (counts && counts["Tinggi"]) || 0],
        backgroundColor: ["rgba(63,185,80,.7)", "rgba(248,81,73,.7)"],
        borderColor:     ["#3fb950", "#f85149"],
        borderWidth: 1, borderRadius: 6,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: document.documentElement.getAttribute("data-theme") === "light" ? "#d0d7de" : "#21262d" } },
        x: { grid: { display: false } }
      }
    }
  });
}

function renderFeatStats(stats) {
  var box = document.getElementById("ds-feat-stats");
  if (!box) return;
  var LABELS = {
    MeanNN_ms:"MeanNN (ms)", SDNN_ms:"SDNN (ms)", RMSSD_ms:"RMSSD (ms)",
    LF:"LF Power (nu)", HF:"HF Power (nu)", LF_HF_Ratio:"LF/HF Ratio"
  };
  var html = "<table><thead><tr><th>Fitur</th><th>Rata-rata</th><th>Std Dev</th><th>Min</th><th>Maks</th></tr></thead><tbody>";
  for (var k in stats) {
    var v = stats[k];
    html += "<tr><td><strong>" + (LABELS[k] || k) + "</strong></td><td>" + v.mean + "</td><td>" + v.std + "</td><td>" + v.min + "</td><td>" + v.max + "</td></tr>";
  }
  html += "</tbody></table>";
  box.innerHTML = html;
}

function renderSampleTable(rows) {
  var tbody = document.getElementById("ds-sample-body");
  if (!tbody) return;
  if (!rows || rows.length === 0) {
    tbody.innerHTML = "<tr><td colspan='9' style='text-align:center;color:var(--text-muted);padding:2rem'>Tidak ada data</td></tr>";
    return;
  }
  var FIELDS = ["MeanNN_ms","SDNN_ms","RMSSD_ms","LF","HF","LF_HF_Ratio"];
  var html = "";
  for (var i = 0; i < rows.length; i++) {
    var r = rows[i];
    html += "<tr><td>" + (r.Subjek || "—") + "</td><td>" + (r.Tugas || "—") + "</td>";
    html += "<td><span class='badge " + (r.Label === "Rendah" ? "badge-green" : "badge-red") + "'>" + (r.Label || "—") + "</span></td>";
    for (var j = 0; j < FIELDS.length; j++) {
      var val = r[FIELDS[j]];
      html += "<td>" + (val != null ? Number(val).toFixed(3) : "<span style='color:var(--text-dim)'>—</span>") + "</td>";
    }
    html += "</tr>";
  }
  tbody.innerHTML = html;
}


// ══════════════════════════════════════════════════════════
//  LEADERBOARD TAB
// ══════════════════════════════════════════════════════════

function loadLeaderboard() {
  fetch("/api/leaderboard")
    .then(function(resp) {
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      return resp.json();
    })
    .then(function(data) {
      leaderboardData = data;
      updateLeaderboard(data);
      renderAllAccuracyChart(data);
      renderHealthDistChart(data);
    })
    .catch(function(err) {
      showError("Gagal memuat leaderboard: " + err.message);
    });
}

function updateLeaderboard(data) {
  var sehat   = data.filter(function(m) { return m.status === "SEHAT"; }).length;
  var anomali = data.filter(function(m) { return m.status === "ANOMALI"; }).length;
  var best    = Math.max.apply(null, data.map(function(m) { return m.accuracy_pct; }));

  var elSehat   = document.getElementById("cnt-sehat");
  var elAnomali = document.getElementById("cnt-anomali");
  var elBest    = document.getElementById("best-acc");
  if (elSehat)   elSehat.textContent   = sehat;
  if (elAnomali) elAnomali.textContent = anomali;
  if (elBest)    elBest.textContent    = isFinite(best) ? best.toFixed(2) + "%" : "—";

  var tbody = document.getElementById("leaderboard-body");
  if (!tbody) return;
  if (!data || data.length === 0) {
    tbody.innerHTML = "<tr><td colspan='8' style='text-align:center;padding:2rem'>Tidak ada data</td></tr>";
    return;
  }

  var html = "";
  for (var i = 0; i < data.length; i++) {
    var m = data[i];
    var rankClass = m.rank === 1 ? "gold" : m.rank === 2 ? "silver" : m.rank === 3 ? "bronze" : "";
    var statusBadge = m.status === "SEHAT"
      ? "<span class='badge badge-green'>✅ SEHAT</span>"
      : m.status === "ANOMALI"
      ? "<span class='badge badge-red'>⚠️ ANOMALI</span>"
      : "<span class='badge badge-yellow'>ℹ️ CUKUP</span>";
    var sensorBadge = (m.sensor || "").includes("ECG+PPG")
      ? "<span class='badge badge-purple'>ECG+PPG</span>"
      : (m.sensor || "").includes("ECG")
      ? "<span class='badge badge-blue'>ECG</span>"
      : "<span class='badge' style='background:rgba(121,192,255,.15);color:#79c0ff'>PPG</span>";
    var accColor = m.accuracy_pct >= 63 ? "var(--green)" : "var(--red)";
    html += "<tr data-sensor='" + m.sensor + "' data-status='" + m.status + "'>";
    html += "<td><span class='rank-num " + rankClass + "'>" + m.rank + "</span></td>";
    html += "<td style='max-width:180px'>" + m.model + "</td>";
    html += "<td>" + sensorBadge + "</td>";
    html += "<td><strong style='color:" + accColor + "'>" + m.accuracy_pct + "%</strong></td>";
    html += "<td>" + m.precision_pct + "%</td><td>" + m.recall_pct + "%</td><td>" + m.f1_score_pct + "%</td>";
    html += "<td>" + statusBadge + "</td></tr>";
  }
  tbody.innerHTML = html;
}

function filterModels(filter, btn) {
  document.querySelectorAll("#tab-models .filter-bar .filter-btn").forEach(function(b) { b.classList.remove("active"); });
  btn.classList.add("active");
  var filtered = leaderboardData;
  if (filter !== "all") {
    filtered = leaderboardData.filter(function(m) { return m.sensor === filter || m.status === filter; });
  }
  updateLeaderboard(filtered);
}

function sortTable(col) {
  if (sortState.col === col) sortState.asc = !sortState.asc;
  else { sortState.col = col; sortState.asc = false; }
  var sorted = leaderboardData.slice().sort(function(a, b) {
    return sortState.asc ? (a[col] > b[col] ? 1 : -1) : (a[col] < b[col] ? 1 : -1);
  });
  updateLeaderboard(sorted);
}

function renderAllAccuracyChart(data) {
  var canvas = document.getElementById("chart-all-accuracy");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  if (allAccuracyChart) { allAccuracyChart.destroy(); allAccuracyChart = null; }

  var sorted = data.slice().sort(function(a, b) { return b.accuracy_pct - a.accuracy_pct; });
  var colors = sorted.map(function(m) {
    return m.status === "SEHAT" ? "rgba(63,185,80,.75)"
      : m.status === "ANOMALI" ? "rgba(248,81,73,.75)"
      : "rgba(210,153,34,.75)";
  });

  allAccuracyChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: sorted.map(function(m) { return m.model; }),
      datasets: [{ label: "Akurasi (%)", data: sorted.map(function(m) { return m.accuracy_pct; }), backgroundColor: colors, borderRadius: 4 }]
    },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: function(c) { return c.raw + "%"; } } } },
      scales: { x: { min: 40, max: 70, grid: { color: document.documentElement.getAttribute("data-theme") === "light" ? "#d0d7de" : "#21262d" }, ticks: { callback: function(v) { return v + "%"; } } }, y: { grid: { display: false }, ticks: { font: { size: 10 } } } }
    }
  });
}

function renderHealthDistChart(data) {
  var canvas = document.getElementById("chart-health-dist");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  if (healthDistChart) { healthDistChart.destroy(); healthDistChart = null; }

  var counts = { SEHAT: 0, ANOMALI: 0, CUKUP: 0 };
  data.forEach(function(m) { if (counts[m.status] !== undefined) counts[m.status]++; });

  healthDistChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Sehat (Stabil)", "Anomali (Bias)", "Cukup / Valid"],
      datasets: [{ data: [counts.SEHAT, counts.ANOMALI, counts.CUKUP], backgroundColor: ["rgba(63,185,80,.8)", "rgba(248,81,73,.8)", "rgba(210,153,34,.8)"], borderColor: ["#3fb950", "#f85149", "#d29922"], borderWidth: 2 }]
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { padding: 16 } } } }
  });
}

function showCM(type, btn) {
  if (btn && btn.parentElement) {
    btn.parentElement.querySelectorAll(".filter-btn").forEach(function(b) { b.classList.remove("active"); });
    btn.classList.add("active");
  }
  
  var cmTypes = ["dl-base", "dl-v2", "dl-v3", "dl-comp"];
  cmTypes.forEach(function(t) {
    var el = document.getElementById("cm-" + t);
    if (el) el.style.display = (type === t) ? "grid" : "none";
  });
}


// ══════════════════════════════════════════════════════════
//  PREPROCESSING TAB
// ══════════════════════════════════════════════════════════

function setSigType(type) {
  currentSigType = type;
  var btnPpg = document.getElementById("sig-type-ppg");
  var btnEcg = document.getElementById("sig-type-ecg");
  if (btnPpg) btnPpg.classList.toggle("active", type === "ppg");
  if (btnEcg) btnEcg.classList.toggle("active", type === "ecg");
}

function handleSignalFile(e) {
  var file = e.target.files[0];
  if (file) processSignalFile(file);
}

function processSignalFile(file) {
  var procInfo = document.getElementById("proc-info");
  var sigPanel = document.getElementById("signal-panel");
  if (sigPanel) sigPanel.classList.remove("visible");
  if (procInfo) procInfo.classList.add("visible");
  var procText = document.getElementById("proc-text");
  if (procText) procText.textContent = "Memproses \"" + file.name + "\" (" + (file.size / 1048576).toFixed(1) + " MB)... Mohon tunggu";

  var formData = new FormData();
  formData.append("file", file);
  formData.append("signal_type", currentSigType);

  fetch("/api/process-signal", { method: "POST", body: formData })
    .then(function(resp) { return resp.json(); })
    .then(function(d) {
      if (procInfo) procInfo.classList.remove("visible");
      if (d.error) { showError("Gagal proses sinyal: " + d.error); return; }
      renderSignalViewer(d);
      if (sigPanel) sigPanel.classList.add("visible");
    })
    .catch(function(err) {
      if (procInfo) procInfo.classList.remove("visible");
      showError("Kesalahan upload sinyal: " + err.message);
    });
}

function renderSignalViewer(d) {
  var meta = document.getElementById("signal-meta-row");
  if (meta) {
    meta.innerHTML =
      "<div class='signal-meta-item'><span class='signal-meta-label'>Tipe: </span><span class='signal-meta-value'>" + d.signal_type.toUpperCase() + "</span></div>" +
      "<div class='signal-meta-item'><span class='signal-meta-label'>Durasi: </span><span class='signal-meta-value'>" + d.duration_sec + " detik</span></div>" +
      "<div class='signal-meta-item'><span class='signal-meta-label'>Sampel: </span><span class='signal-meta-value'>" + d.n_samples.toLocaleString() + "</span></div>" +
      "<div class='signal-meta-item'><span class='signal-meta-label'>Puncak: </span><span class='signal-meta-value'>" + d.n_peaks + "</span></div>";
  }

  var canvas = document.getElementById("chart-signal");
  if (canvas) {
    var ctx = canvas.getContext("2d");
    if (signalChart) { signalChart.destroy(); signalChart = null; }

    var peakSet = {};
    d.peaks_t.forEach(function(pt) { peakSet[Math.round(pt * 10)] = true; });
    var peakData = d.t_axis.map(function(t, i) { return peakSet[Math.round(t * 10)] ? d.clean[i] : null; });

    signalChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: d.t_axis,
        datasets: [
          { label: "Sinyal Mentah", data: d.raw, borderColor: "rgba(121,192,255,.5)", borderWidth: 1.2, pointRadius: 0, tension: 0.3, backgroundColor: "rgba(121,192,255,.05)", fill: true },
          { label: "Sinyal Bersih", data: d.clean, borderColor: "rgba(63,185,80,.9)", borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
          { label: "Puncak", data: peakData, type: "scatter", backgroundColor: "rgba(248,81,73,.8)", pointRadius: 4, borderWidth: 0, showLine: false }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 400 },
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { position: "top", labels: { boxWidth: 12, padding: 16 } } },
        scales: {
          x: { title: { display: true, text: "Waktu (detik)" }, grid: { color: document.documentElement.getAttribute("data-theme") === "light" ? "#d0d7de" : "#21262d" }, ticks: { maxTicksLimit: 12 } },
          y: { title: { display: true, text: "Amplitudo (normalisasi)" }, grid: { color: document.documentElement.getAttribute("data-theme") === "light" ? "#d0d7de" : "#21262d" } }
        }
      }
    });
  }

  var featsGrid = document.getElementById("hrv-extracted-grid");
  if (featsGrid && d.features) {
    var FEAT_LABELS = { MeanNN_ms:"MeanNN (ms)", SDNN_ms:"SDNN (ms)", RMSSD_ms:"RMSSD (ms)", LF:"LF Power (nu)", HF:"HF Power (nu)", LF_HF_Ratio:"LF/HF Ratio" };
    var html = "";
    for (var k in d.features) {
      var v = d.features[k];
      html += "<div class='stat-card' style='padding:.8rem'><span class='stat-label'>" + (FEAT_LABELS[k] || k) + "</span><span class='stat-value' style='font-size:1.2rem'>" + (v != null ? Number(v).toFixed(3) : "<span style='color:var(--text-dim)'>—</span>") + "</span></div>";
    }
    featsGrid.innerHTML = html;
  }
  extractedFeatures = d.features;
}

function sendExtractedToDemo() {
  if (!extractedFeatures) return;
  document.querySelectorAll(".nav-tab").forEach(function(b) { b.classList.remove("active"); });
  document.querySelectorAll(".tab-panel").forEach(function(p) { p.classList.remove("active"); });
  var demoTab = document.querySelector("[data-tab='demo']");
  var demoPanel = document.getElementById("tab-demo");
  if (demoTab)   demoTab.classList.add("active");
  if (demoPanel) demoPanel.classList.add("active");

  var FIELDS = ["MeanNN_ms","SDNN_ms","RMSSD_ms","LF","HF","LF_HF_Ratio"];
  FIELDS.forEach(function(f) {
    var v = extractedFeatures[f];
    if (v != null) {
      var sl = document.getElementById("sl-" + f);
      var ni = document.getElementById("in-" + f);
      if (sl) sl.value = v;
      if (ni) ni.value = Number(v).toFixed(3);
    }
  });
}


// ══════════════════════════════════════════════════════════
//  DEMO TAB
// ══════════════════════════════════════════════════════════

function loadPreset(key) {
  var p = PRESETS[key];
  if (!p) return;
  var FIELDS = ["MeanNN_ms","SDNN_ms","RMSSD_ms","LF","HF","LF_HF_Ratio"];
  FIELDS.forEach(function(f) {
    var sl = document.getElementById("sl-" + f);
    var ni = document.getElementById("in-" + f);
    if (sl && p[f] != null) sl.value = p[f];
    if (ni && p[f] != null) ni.value = Number(p[f]).toFixed(3);
  });
}

function resetForm() {
  var defaults = { MeanNN_ms:700, SDNN_ms:54, RMSSD_ms:69, LF:0.035, HF:0.060, LF_HF_Ratio:1.10 };
  for (var f in defaults) {
    var sl = document.getElementById("sl-" + f);
    var ni = document.getElementById("in-" + f);
    if (sl) sl.value = defaults[f];
    if (ni) ni.value = defaults[f];
  }
  var rp = document.getElementById("result-panel");
  var rph = document.getElementById("result-placeholder");
  if (rp)  rp.classList.remove("visible");
  if (rph) rph.style.display = "block";
}

function syncInput(field) {
  var sl = document.getElementById("sl-" + field);
  var ni = document.getElementById("in-" + field);
  if (sl && ni) ni.value = Number(sl.value).toFixed((field.startsWith("LF") || field === "HF") ? 3 : 1);
}

function syncSlider(field) {
  var sl = document.getElementById("sl-" + field);
  var ni = document.getElementById("in-" + field);
  if (sl && ni) sl.value = ni.value;
}

function runPredict() {
  var FIELDS = ["MeanNN_ms","SDNN_ms","RMSSD_ms","LF","HF","LF_HF_Ratio"];
  var btn = document.getElementById("btn-predict");
  if (btn) { btn.disabled = true; btn.innerHTML = "<div class='spinner'></div> Menganalisis..."; }

  var payload = {};
  FIELDS.forEach(function(f) {
    var el = document.getElementById("in-" + f);
    var v = el ? el.value : null;
    payload[f] = (v !== null && v !== "") ? parseFloat(v) : null;
  });

  fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })
    .then(function(resp) { return resp.json(); })
    .then(function(d) {
      if (d.error) { showError("Error prediksi: " + d.error); return; }
      displayResult(d);
    })
    .catch(function(err) { showError("Gagal menghubungi server: " + err.message); })
    .finally(function() {
      if (btn) { btn.disabled = false; btn.innerHTML = "🔮 Analisis Beban Kognitif"; }
    });
}

function displayResult(d) {
  var placeholder = document.getElementById("result-placeholder");
  var panel = document.getElementById("result-panel");
  if (placeholder) placeholder.style.display = "none";
  if (!panel) return;
  panel.classList.remove("visible");
  void panel.offsetWidth;
  panel.classList.add("visible");

  var isTinggi = d.label === "Tinggi";
  var badge = document.getElementById("result-badge");
  if (badge) badge.className = "result-badge " + (isTinggi ? "tinggi" : "rendah");
  var icon = document.getElementById("result-icon");
  var label = document.getElementById("result-label");
  var sublabel = document.getElementById("result-sublabel");
  if (icon)     icon.textContent     = isTinggi ? "🔴" : "🟢";
  if (label)    label.textContent    = "BEBAN KOGNITIF " + (d.label || "").toUpperCase();
  if (sublabel) sublabel.textContent = isTinggi ? "Kondisi aktivasi tinggi — Math Test / Stroop / IQ" : "Kondisi relaksasi — Baseline / Neutral";

  var confPct  = document.getElementById("conf-pct");
  var confFill = document.getElementById("conf-fill");
  if (confPct)  confPct.textContent = d.confidence + "%";
  if (confFill) setTimeout(function() { confFill.style.width = d.confidence + "%"; }, 100);

  var canvas = document.getElementById("chart-feat-contrib");
  if (canvas && d.ufr_weights) {
    var ctx = canvas.getContext("2d");
    if (featContribChart) { featContribChart.destroy(); featContribChart = null; }
    var ufr = d.ufr_weights;
    var ufrLabels = Object.keys(ufr).map(function(k) { return k.replace("_ms","").replace("_HF_Ratio","/ HF"); });
    featContribChart = new Chart(ctx, {
      type: "bar",
      data: { labels: ufrLabels, datasets: [{ label: "Kontribusi UFR (%)", data: Object.values(ufr), backgroundColor: "rgba(188,140,255,.7)", borderColor: "#bc8cff", borderWidth: 1, borderRadius: 4 }] },
      options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: document.documentElement.getAttribute("data-theme") === "light" ? "#d0d7de" : "#21262d" }, ticks: { callback: function(v) { return v + "%"; } } }, y: { grid: { display: false } } } }
    });
  }

  var interp = document.getElementById("result-interpretation");
  if (interp) {
    interp.innerHTML = isTinggi
      ? "<strong style='color:var(--red)'>Interpretasi:</strong> Profil HRV menunjukkan tanda-tanda aktivasi kognitif tinggi. Nilai HF Power (" + (d.fitur_input && d.fitur_input.HF ? d.fitur_input.HF : "—") + " nu) yang menjadi fitur paling berpengaruh (50,26% UFR) dan MeanNN (" + (d.fitur_input && d.fitur_input.MeanNN_ms ? d.fitur_input.MeanNN_ms : "—") + " ms) mengindikasikan respons otonom yang konsisten dengan beban kerja mental tinggi seperti Math Test, IQ Test, atau Stroop Test."
      : "<strong style='color:var(--green)'>Interpretasi:</strong> Profil HRV menunjukkan tanda-tanda kondisi relaksasi atau beban kognitif rendah. Nilai HF Power (" + (d.fitur_input && d.fitur_input.HF ? d.fitur_input.HF : "—") + " nu) yang tinggi dan MeanNN (" + (d.fitur_input && d.fitur_input.MeanNN_ms ? d.fitur_input.MeanNN_ms : "—") + " ms) mengindikasikan sistem saraf parasimpatik aktif secara seimbang, konsisten dengan kondisi Baseline atau Neutral.";
  }
}

// ── Lightbox ─────────────────────────────────────────────
function openLightbox(src, alt) {
  var img = document.getElementById("lightbox-img");
  var lb  = document.getElementById("lightbox");
  if (img) img.src = src;
  if (img) img.alt = alt || "";
  if (lb)  lb.classList.add("open");
}

function closeLightbox(e) {
  var lb = document.getElementById("lightbox");
  if (lb && (e.target === lb || (e.target && e.target.classList.contains("lightbox-close")))) {
    lb.classList.remove("open");
  }
}
