const $ = (sel) => document.querySelector(sel);

function setView(name) {
  document.querySelectorAll(".nav-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === name);
  });
  ["earth", "forecast", "models", "health"].forEach((v) => {
    const el = $(`#view-${v}`);
    if (!el) return;
    el.classList.toggle("hidden", v !== name);
  });
}

document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => setView(btn.dataset.view));
});

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function fmt(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

async function refreshState() {
  const loc = $("#location").value.trim() || "Birmingham";
  const data = await api(`/api/current?location=${encodeURIComponent(loc)}`);
  const cell = data.earth_state?.cells?.[0];
  $("#latlon").textContent = `${data.location} · ${fmt(data.latitude, 3)}°, ${fmt(data.longitude, 3)}°`;
  const grid = $("#state-grid");
  grid.innerHTML = "";
  const vars = cell?.variables || {};
  const order = ["temperature_2m", "precipitation", "wind_speed_10m"];
  for (const key of order) {
    const q = vars[key];
    if (!q) continue;
    const div = document.createElement("div");
    div.className = "state-item";
    div.innerHTML = `
      <div class="label">${key}</div>
      <div class="value">${fmt(q.estimate)}${q.unit || ""}</div>
      <div class="unc">±${fmt(q.uncertainty)} · ${q.integrity}</div>
    `;
    grid.appendChild(div);
  }
  const chip = $("#assurance-chip");
  chip.textContent = `DATA ${data.data_assurance}`;
  chip.className = `assurance ${data.data_assurance}`;
}

function drawForecast(fc) {
  const canvas = $("#forecast-chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const mean = fc.mean.map((v) => (v === null ? NaN : v));
  const q10 = fc.q10.map((v) => (v === null ? NaN : v));
  const q90 = fc.q90.map((v) => (v === null ? NaN : v));
  const vals = [...mean, ...q10, ...q90].filter((v) => !Number.isNaN(v));
  if (!vals.length) return;

  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const pad = (max - min) * 0.12 || 1;
  const yMin = min - pad;
  const yMax = max + pad;
  const n = mean.length;
  const xAt = (i) => (i / Math.max(n - 1, 1)) * (w - 40) + 20;
  const yAt = (v) => h - 30 - ((v - yMin) / (yMax - yMin)) * (h - 50);

  // band
  ctx.beginPath();
  for (let i = 0; i < n; i++) if (!Number.isNaN(q90[i])) ctx.lineTo(xAt(i), yAt(q90[i]));
  for (let i = n - 1; i >= 0; i--) if (!Number.isNaN(q10[i])) ctx.lineTo(xAt(i), yAt(q10[i]));
  ctx.closePath();
  ctx.fillStyle = "rgba(14, 107, 110, 0.18)";
  ctx.fill();

  // mean
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < n; i++) {
    if (Number.isNaN(mean[i])) continue;
    if (!started) { ctx.moveTo(xAt(i), yAt(mean[i])); started = true; }
    else ctx.lineTo(xAt(i), yAt(mean[i]));
  }
  ctx.strokeStyle = "#073f46";
  ctx.lineWidth = 2.2;
  ctx.stroke();

  // axis labels
  ctx.fillStyle = "#2a4550";
  ctx.font = "12px Sora, sans-serif";
  ctx.fillText(fmt(yMax), 4, 18);
  ctx.fillText(fmt(yMin), 4, h - 12);
}

async function runForecast() {
  const loc = $("#location").value.trim() || "Birmingham";
  const variable = $("#variable").value;
  const hours = $("#hours").value;
  const fc = await api(
    `/api/forecast?location=${encodeURIComponent(loc)}&variable=${encodeURIComponent(variable)}&hours=${hours}`
  );
  drawForecast(fc);
  const chip = $("#assurance-chip");
  chip.textContent = `FORECAST ${fc.assurance}`;
  chip.className = `assurance ${fc.assurance}`;
  $("#forecast-meta").innerHTML = `
    <strong>${fc.location_name}</strong> · ${fc.variable} (${fc.unit}) · horizon ${fc.horizon_hours}h<br/>
    Models: ${fc.model_ids.join(", ")}<br/>
    Assurance: ${fc.assurance} — ${(fc.assurance_reasons || []).join("; ")}<br/>
    Integrity: ${fc.integrity}
  `;
}

async function loadModels() {
  const models = await api("/api/models");
  const sources = await api("/api/sources");
  const mt = $("#models-table");
  mt.innerHTML = `<div class="row head"><div>Model</div><div>Status</div><div>Family</div><div>MAE</div></div>`;
  for (const m of models) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<div>${m.model_id}</div><div>${m.status}</div><div>${m.family}</div><div>${fmt(m.metrics?.mae, 3)}</div>`;
    mt.appendChild(row);
  }
  const st = $("#sources-table");
  st.innerHTML = `<div class="row head"><div>Source</div><div>Status</div><div>Free</div><div>Domain</div></div>`;
  for (const s of sources) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<div>${s.name}</div><div>${s.status}</div><div>${s.free}</div><div>${s.domain}</div>`;
    st.appendChild(row);
  }
}

async function loadHealth() {
  const health = await api("/api/system-health");
  $("#health-json").textContent = JSON.stringify(health, null, 2);
}

async function runVerify() {
  const loc = $("#location").value.trim() || "Birmingham";
  const report = await api(`/api/verification?location=${encodeURIComponent(loc)}&hours=24`);
  $("#verify-json").textContent = JSON.stringify(report, null, 2);
}

$("#refresh-btn").addEventListener("click", () => refreshState().catch(alert));
$("#forecast-btn").addEventListener("click", () => runForecast().catch(alert));
$("#verify-btn").addEventListener("click", () => runVerify().catch(alert));

// Boot
refreshState().catch(() => {});
loadModels().catch(() => {});
loadHealth().catch(() => {});
