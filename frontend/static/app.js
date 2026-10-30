/* NEAR-ZERO X dashboard client — polls /api/state and renders. */
const $ = (id) => document.getElementById(id);
const state = { scenario: null };

const BAND = { safe: "band-safe", caution: "band-caution", warning: "band-warning" };

async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}

async function poll() {
  try {
    const st = await api("/api/state");
    $("conn").className = "chip ok";
    $("conn").textContent = "online";
    render(st);
  } catch (e) {
    $("conn").className = "chip off";
    $("conn").textContent = "offline";
  }
}

function render(st) {
  if (!st.ready) { $("riskValue").textContent = "--"; return; }
  $("mode").textContent = st.mode;
  const sc = st.risk.score;
  $("riskValue").textContent = sc.toFixed(0);
  const band = st.risk.state.toLowerCase();
  const el = $("riskBand");
  el.textContent = band;
  el.className = "gauge-band " + (BAND[band] || "band-safe");

  $("statCol").textContent = st.risk.collision_probability.toFixed(0) + "%";
  $("statTtc").textContent = st.risk.ttc !== null ? st.risk.ttc.toFixed(2) + "s" : "—";
  $("statPmd").textContent = st.risk.predicted_min_distance !== null
    ? st.risk.predicted_min_distance.toFixed(2) + "m" : "—";
  $("statConf").textContent = st.risk.confidence.toFixed(2);

  renderCounterfactual(st);
  renderTracks(st);
  renderEvents(st);
  renderMetrics(st);
  renderManual(st);
  renderFeedStatus(st);
}

function renderCounterfactual(st) {
  const rows = st.counterfactual || [];
  const bestName = rows.filter((r) => r.best).map((r) => r.intervention)[0] || null;
  const badge = $("bestBadge");
  if (bestName) {
    badge.textContent = "BEST: " + bestName;
    badge.className = bestName.startsWith("VEHICLE") ? "badge best" : "badge warn";
  } else {
    badge.textContent = "NO ACTION";
    badge.className = "badge none";
  }
  const body = $("ctBody");
  body.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    if (r.best) tr.className = "best-row";
    tr.innerHTML =
      "<td>" + esc(r.intervention) + "</td>" +
      "<td>" + r.resulting_risk.toFixed(0) + "</td>" +
      "<td>" + r.risk_reduction.toFixed(0) + "</td>" +
      "<td>" + r.intervention_cost.toFixed(2) + "</td>";
    body.appendChild(tr);
  }
}

function renderTracks(st) {
  const body = $("trackBody");
  body.innerHTML = "";
  for (const t of st.tracks || []) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      "<td>#" + t.track_id + "</td>" +
      "<td>" + esc(t.cls) + "</td>" +
      "<td>" + t.speed.toFixed(1) + "</td>" +
      "<td>(" + t.pos[0].toFixed(1) + ", " + t.pos[1].toFixed(1) + ")</td>";
    body.appendChild(tr);
  }
  if (!(st.tracks || []).length) {
    body.innerHTML = "<tr><td colspan=4 class='muted'>no tracks</td></tr>";
  }
}

function renderEvents(st) {
  const list = $("eventList");
  list.innerHTML = "";
  const v = st.verify || {};
  const open = v.open || [];
  for (const e of open) {
    const li = document.createElement("li");
    li.className = "ev-warn";
    li.innerHTML = "<b>" + esc(e.state) + "</b> pair " + e.subject_id + "→" + e.target_id +
      " risk " + e.peak_risk.toFixed(0);
    list.appendChild(li);
  }
  if (v.last_event) {
    const le = v.last_event;
    const cls = le.state === "PREVENTED" ? "ev-prev" : (le.state === "UNRESOLVED" ? "ev-unres" : "ev-warn");
    const li = document.createElement("li");
    li.className = cls;
    li.innerHTML = "<b>#" + le.event_id + " " + esc(le.state) + "</b>" +
      (le.recommended ? " · " + esc(le.recommended) : "") +
      " peak " + (le.peak_risk || 0).toFixed(0);
    list.appendChild(li);
  }
}

function renderMetrics(st) {
  const m = st.metrics || {};
  $("metricsLine").textContent =
    "fps " + (m.fps || 0).toFixed(1) +
    " · perceive " + (m.perceive_ms || 0).toFixed(1) + "ms" +
    " · risk " + (m.risk_ms || 0).toFixed(1) + "ms" +
    " · plan " + (m.plan_ms || 0).toFixed(1) + "ms" +
    " · verify " + (m.verify_ms || 0).toFixed(1) + "ms";
}

const MANUAL = ["NO_ACTION", "STOP_VEHICLE", "SLOW_VEHICLE_30", "SLOW_VEHICLE_50",
                "HUMAN_STOP", "HUMAN_LEFT", "HUMAN_RIGHT", "VEHICLE_LEFT", "VEHICLE_RIGHT"];

function renderManual(st) {
  if (state.buttonsBuilt) return;
  state.buttonsBuilt = true;
  const box = $("manualBtns");
  for (const name of MANUAL) {
    const b = document.createElement("button");
    b.textContent = name.replace("_", "·");
    b.title = "evaluate " + name;
    b.addEventListener("click", async () => {
      await api("/api/intervention", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ intervention: name }),
      });
    });
    box.appendChild(b);
  }
}

function renderFeedStatus(st) {
  const txt = st.best_intervention && st.best_intervention.intervention !== "NO_ACTION"
    ? "RECOMMENDED " + st.best_intervention.intervention
    : (st.risk.state === "warning" ? "WARNING" : "monitoring");
  $("feedstatus").textContent = txt;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

async function init() {
  try {
    const scs = await api("/api/scenarios");
    const sel = $("scenarioSel");
    sel.innerHTML = "";
    for (const s of scs) {
      const o = document.createElement("option");
      o.value = s; o.textContent = s;
      sel.appendChild(o);
    }
    sel.value = "demo_avoided";
    $("btnScenario").addEventListener("click", async () => {
      await api("/api/scenario", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario: sel.value }),
      });
    });
  } catch (e) { /* sim-only controls */ }

  $("btnRule").addEventListener("click", async () => {
    const text = $("ruleText").value.trim();
    if (!text) return;
    const r = await api("/api/rules", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (r.ok) $("ruleText").value = "";
  });

  setInterval(poll, 250);
  poll();
}

init();