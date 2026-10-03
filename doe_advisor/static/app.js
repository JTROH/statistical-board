/* doe-advisor front end. No framework, no build step — same shape as the two
   sibling tools, so there is one less thing to install on a managed laptop.

   The form is the source of truth throughout. The chat box (when a key is
   present) only writes into it; nothing is ever submitted that the scientist
   has not seen in the fields. */

const $ = (id) => document.getElementById(id);

const state = {
  presets: [],
  capabilities: null,
  lastForm: null,
};

/* ------------------------------------------------------------------ */
/* Row builders                                                        */
/* ------------------------------------------------------------------ */

function factorRow(factor = {}) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input class="f-name" placeholder="e.g. pH"></td>
    <td><input class="f-low num" type="number" step="any"></td>
    <td><input class="f-high num" type="number" step="any"></td>
    <td><input class="f-units" placeholder="optional"></td>
    <td><button type="button" class="link" title="Remove this factor">&times;</button></td>`;
  tr.querySelector(".f-name").value = factor.name ?? "";
  tr.querySelector(".f-low").value = factor.low ?? "";
  tr.querySelector(".f-high").value = factor.high ?? "";
  tr.querySelector(".f-units").value = factor.units ?? "";
  tr.querySelector("button").onclick = () => tr.remove();
  return tr;
}

function responseRow(response = {}) {
  const goals = (state.capabilities && state.capabilities.response_goals) || [];
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input class="r-name" placeholder="e.g. titer"></td>
    <td><input class="r-units" placeholder="g/L"></td>
    <td>
      <select class="r-goal">${goals.map((g) => `<option value="${g.value}">${g.label}</option>`).join("")}</select>
      <input class="r-target-value num" type="number" step="any" placeholder="target value" hidden>
    </td>
    <td><input class="r-target num" type="number" step="any" placeholder="blank = unknown"></td>
    <td><input class="r-noise num" type="number" step="any" placeholder="blank = unknown"></td>
    <td><button type="button" class="link" title="Remove this response">&times;</button></td>`;
  tr.querySelector(".r-name").value = response.name ?? "";
  tr.querySelector(".r-units").value = response.units ?? "";
  tr.querySelector(".r-goal").value = response.goal ?? "screen";
  tr.querySelector(".r-target-value").value = response.target_value ?? "";
  tr.querySelector(".r-target").value = response.target_effect ?? "";
  tr.querySelector(".r-noise").value = response.noise_sd ?? "";
  tr.querySelector("button").onclick = () => {
    tr.remove();
    updateEffectNote();
  };
  const syncTargetValue = () => {
    tr.querySelector(".r-target-value").hidden = tr.querySelector(".r-goal").value !== "target";
  };
  tr.querySelector(".r-goal").onchange = () => {
    syncTargetValue();
    updateModelNote();
  };
  tr.addEventListener("input", updateEffectNote);
  syncTargetValue();
  return tr;
}

/* ------------------------------------------------------------------ */
/* Form <-> payload                                                    */
/* ------------------------------------------------------------------ */

const numOrNull = (value) => (value === "" || value == null ? null : Number(value));

function readForm() {
  const factors = [...$("factors").querySelectorAll("tbody tr")].map((tr) => ({
    name: tr.querySelector(".f-name").value.trim(),
    low: numOrNull(tr.querySelector(".f-low").value),
    high: numOrNull(tr.querySelector(".f-high").value),
    units: tr.querySelector(".f-units").value.trim(),
  }));
  const responses = [...$("responses").querySelectorAll("tbody tr")].map((tr) => ({
    name: tr.querySelector(".r-name").value.trim(),
    units: tr.querySelector(".r-units").value.trim(),
    goal: tr.querySelector(".r-goal").value,
    target_value: numOrNull(tr.querySelector(".r-target-value").value),
    target_effect: numOrNull(tr.querySelector(".r-target").value),
    noise_sd: numOrNull(tr.querySelector(".r-noise").value),
  }));
  const named = factors.filter((f) => f.name);
  return {
    factors,
    responses,
    model_order: $("model-order").value,
    max_runs: numOrNull($("max-runs").value),
    n_center_points: numOrNull($("centre").value),
    expected_run_losses: numOrNull($("losses").value),
    hard_ranges: $("hard-ranges").checked,
    title: named.length
      ? `Design options: ${named.map((f) => f.name).join(", ")}`
      : "Experimental design memo",
  };
}

function writeForm(form) {
  if (Array.isArray(form.factors) && form.factors.length) {
    const body = $("factors").querySelector("tbody");
    body.innerHTML = "";
    form.factors.forEach((f) => body.appendChild(factorRow(f)));
  }
  if (Array.isArray(form.responses) && form.responses.length) {
    const body = $("responses").querySelector("tbody");
    body.innerHTML = "";
    form.responses.forEach((r) => body.appendChild(responseRow(r)));
  }
  if (form.model_order) $("model-order").value = form.model_order;
  if (form.max_runs != null) $("max-runs").value = form.max_runs;
  if (form.n_center_points != null) $("centre").value = form.n_center_points;
  if (form.expected_run_losses != null) $("losses").value = form.expected_run_losses;
  $("hard-ranges").checked = Boolean(form.hard_ranges);
  updateModelNote();
  updateEffectNote();
}

/* Signal-to-noise, live, so the scientist sees what their two numbers mean
   before anything is computed. Bands match the memo's primer. */
function updateEffectNote() {
  const first = $("responses").querySelector("tbody tr");
  const note = $("effect-note");
  if (!first) {
    note.textContent = "";
    return;
  }
  const units = first.querySelector(".r-units").value.trim();
  const target = numOrNull(first.querySelector(".r-target").value);
  const noise = numOrNull(first.querySelector(".r-noise").value);
  const u = units ? ` ${units}` : "";
  if (target == null || noise == null || noise <= 0) {
    note.innerHTML =
      "Fill in both <strong>target effect</strong> and <strong>noise SD</strong> and the tool can tell you " +
      "the chance each design has of finding the change you care about.";
    return;
  }
  const ratio = Math.abs(target) / noise;
  let verdict;
  if (ratio >= 2) verdict = "a <strong>clear signal</strong> — the change you care about is at least twice the scatter, so small designs can find it.";
  else if (ratio >= 1) verdict = "a <strong>moderate signal</strong> — about the same size as the scatter, so it takes a careful design with enough runs.";
  else verdict = "a <strong>subtle signal</strong> — smaller than the scatter, so it will take many runs or lower noise to see it.";
  note.innerHTML =
    `Target effect ${target}${u} &divide; noise SD ${noise}${u} = <strong>${ratio.toFixed(2)}</strong> ` +
    `times the noise. That is ${verdict}`;
}

/* ------------------------------------------------------------------ */
/* Rendering results                                                   */
/* ------------------------------------------------------------------ */

const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
const num = (v, dp = 2) => (v == null ? "—" : Number(v).toFixed(dp));

function confoundingLabel(worst) {
  if (worst === 0) return "none";
  if (worst >= 0.99) return "complete";
  return `partial (${worst.toFixed(2)})`;
}

function roleTag(roles) {
  const role = roles[0] || "alternative";
  const label = { recommended: "Recommended", economical: "Cheaper", thorough: "More thorough" }[role] ||
    "Alternative";
  return `<span class="tag ${role}">${label}</span>`;
}

function renderTable(options) {
  const body = $("options-table").querySelector("tbody");
  body.innerHTML = "";
  options.forEach((o) => {
    const tr = document.createElement("tr");
    if (o.roles.includes("recommended")) tr.className = "recommended";
    tr.innerHTML = `
      <td>${roleTag(o.roles)}${o.name}</td>
      <td>${o.n_runs}</td>
      <td>${pct(o.power)}</td>
      <td>${confoundingLabel(o.worst_alias)}</td>
      <td>${num(o.i_value)}</td>
      <td>${o.robustness_applicable ? pct(o.robustness) : "—"}</td>
      <td>${num(o.score, 3)}</td>`;
    body.appendChild(tr);
  });
}

/* Pros and cons are derived here from the engine's numbers, not fetched.
   The narrated prose lives in the memo; this view stays fast and offline. */
function prosAndCons(option, cheapest, dearest, basis) {
  const pros = [];
  const cons = [];
  const u = basis && basis.units ? ` ${basis.units}` : "";
  const sig = (v) => Number(v).toPrecision(3).replace(/\.?0+$/, "");

  if (option.power != null) {
    const p = Math.round(option.power * 100);
    if (option.power >= 0.9) pros.push(`Strong power (${p}%): if the effect is real, this design finds it about ${p} times in 100.`);
    else if (option.power >= 0.8) pros.push(`Adequate power (${p}%), just over the 80% bar — about ${100 - p} campaigns in 100 would still miss a real effect.`);
    else cons.push(`Underpowered (${p}%): if the effect is real, about ${100 - p} campaigns in 100 would miss it — you could run everything and conclude nothing.`);

    const byKind = option.power_by_kind || {};
    [["curvature", "curvature (squared) terms"], ["interaction", "interactions"]].forEach(([kind, words]) => {
      const v = byKind[kind];
      if (v != null && v < 0.8 && byKind.main != null && v < byKind.main) {
        cons.push(`The ${words} are the weak point: ${Math.round(v * 100)}% power, against ${Math.round(byKind.main * 100)}% for the main effects.`);
      }
    });

    if (option.detectable_effect_units != null && basis && basis.target_effect != null) {
      const seen = sig(option.detectable_effect_units);
      if (option.power >= 0.8) {
        pros.push(`Smallest change it can reliably see: about ${seen}${u}. You asked for ${basis.target_effect}${u}, which is larger — you have margin.`);
      } else {
        cons.push(`Smallest change it can reliably see: about ${seen}${u}. Your ${basis.target_effect}${u} is smaller than that, so it will often go unnoticed.`);
      }
    }
    if (option.power_if_noise_high != null && option.power >= 0.8) {
      const worse = Math.round(option.power_if_noise_high * 100);
      const k = Number(option.noise_stress_factor || 1.5).toPrecision(3).replace(/\.?0+$/, "");
      if (worse < 80) cons.push(`Sensitive to your noise estimate: if the real SD is ${k}× what you entered, power falls to ${worse}%.`);
      else pros.push(`Forgiving of a bad noise estimate: even at ${k}× the SD you entered, power stays at ${worse}%.`);
    }
  } else if (option.detectable_effect_sd != null) {
    const inUnits = option.detectable_effect_units != null ? ` — about ${sig(option.detectable_effect_units)}${u}` : "";
    pros.push(
      `No target effect given, so power cannot be computed. At 80% power this design can reliably see a change of about ` +
        `${num(option.detectable_effect_sd)} times your run-to-run noise${inUnits}. Smaller changes would often go unnoticed.`
    );
  }

  if (option.worst_alias === 0) pros.push("Nothing is confounded — every term stands on its own.");
  else if (option.alias_statements.length) cons.push(option.alias_statements[0]);

  if (option.n_runs === cheapest) pros.push(`Cheapest option here at ${option.n_runs} runs.`);
  if (option.n_runs === dearest && cheapest !== dearest) cons.push(`Most expensive at ${option.n_runs} runs.`);
  if (option.detail.three_level) pros.push("Three levels per factor, so it can detect curvature.");
  if (option.detail.replicates) {
    cons.push(`Replication (${option.detail.replicates}x) improves precision but unties nothing that was confounded.`);
  }
  if (option.exceeds_declared_range) {
    cons.push("Some runs sit outside the ranges you declared — check they are physically runnable.");
  }
  if (option.n_center_points === 0) cons.push("No centre points, so it cannot tell you whether the response is curved.");
  if (option.residual_df <= 2) cons.push(`Only ${option.residual_df} degrees of freedom for error — wide error bars.`);
  if (option.robustness_applicable && option.robustness < 1) {
    cons.push(`Fragile: fails in ${pct(1 - option.robustness)} of run-loss scenarios.`);
  } else if (option.robustness_applicable && option.worst_power_after_loss != null && option.worst_power_after_loss < 0.8) {
    cons.push(`Still fits after the run losses you expect, but power can fall to ${pct(option.worst_power_after_loss)}.`);
  } else if (option.robustness_applicable) {
    pros.push("Survives the run losses you expect.");
  }
  return { pros, cons };
}

function renderPowerBasis(basis) {
  const el = $("power-basis");
  if (!basis || basis.noise_sd == null) {
    el.textContent = basis
      ? `No target effect and noise SD were given for ${basis.name}, so the Power column is blank and the ranking rests on the other three axes.`
      : "";
    return;
  }
  const u = basis.units ? ` ${basis.units}` : "";
  const goal = basis.goal_statement ? ` Goal: ${basis.goal_statement}.` : "";
  if (basis.target_effect == null) {
    el.textContent = `Noise SD for ${basis.name}: ${basis.noise_sd}${u}. No target effect was given, so power is not computed; each option says instead how small a change it can see.${goal}`;
    return;
  }
  let runs = "";
  if (basis.runs_for_80_power != null) {
    runs = ` An ideal two-level design reaches 80% power with about ${basis.runs_for_80_power} runs`;
    if (basis.runs_for_80_power_if_noise_high != null) {
      const k = Number(basis.noise_stress_factor || 1.5).toPrecision(3).replace(/\.?0+$/, "");
      runs += `, or about ${basis.runs_for_80_power_if_noise_high} if the noise is really ${k}× larger`;
    }
    runs += " — see the first chart below.";
  } else {
    runs = " No practical run count reaches 80% power at this target effect and noise.";
  }
  el.textContent =
    `Power is for ${basis.name}: a change of ${basis.target_effect}${u} against run-to-run noise of ` +
    `${basis.noise_sd}${u} (${Number(basis.standardised_effect).toFixed(2)} times the noise).${goal}${runs}`;
}

function renderDetails(options, basis) {
  const host = $("details");
  host.innerHTML = "";
  const runs = options.map((o) => o.n_runs);
  const cheapest = Math.min(...runs);
  const dearest = Math.max(...runs);

  options.forEach((option) => {
    const { pros, cons } = prosAndCons(option, cheapest, dearest, basis);
    const div = document.createElement("div");
    div.className = "option-detail";
    div.innerHTML = `
      <h3>${roleTag(option.roles)}${option.name}</h3>
      <p class="meta">
        ${option.n_runs} runs (${option.n_center_points} centre points) &middot;
        ${option.n_model_terms} model terms &middot; ${option.residual_df} residual df &middot;
        D&#8209;efficiency ${num(option.d_efficiency, 3)} &middot; G&#8209;efficiency ${pct(option.g_efficiency)}
      </p>
      <div class="pros-cons">
        <div><h4>In its favour</h4><ul>${pros.map((p) => `<li>${p}</li>`).join("") || "<li>—</li>"}</ul></div>
        <div><h4>Against it</h4><ul class="con">${cons.map((c) => `<li>${c}</li>`).join("") || "<li>—</li>"}</ul></div>
      </div>
      <details>
        <summary class="hint">How this score was reached</summary>
        <ul class="hint">${option.score_explanation.map((l) => `<li>${l}</li>`).join("")}</ul>
      </details>`;
    host.appendChild(div);
  });
}

function renderRejected(rejected) {
  $("rejected-count").textContent = rejected.length ? `(${rejected.length})` : "(none)";
  $("rejected").innerHTML = rejected
    .map((r) => `<li>${r.name} <span class="why">— ${r.reason}</span></li>`)
    .join("");
}

/* ------------------------------------------------------------------ */
/* Actions                                                             */
/* ------------------------------------------------------------------ */

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}

async function postJSON(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({ error: "Unexpected response from the server." }));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

async function runDesign() {
  $("error").hidden = true;
  $("status").textContent = "Scoring candidate designs…";
  $("run").disabled = true;
  try {
    const form = readForm();
    const data = await postJSON("api/design", form);
    state.lastForm = form;

    if (!data.options.length) {
      $("results").hidden = true;
      $("status").textContent = "";
      showError(
        "No design can meet this specification. Either raise the run budget, or ask for a simpler model — " +
          `${data.rejected.length} candidates were considered and all were rejected.`
      );
      return;
    }

    renderTable(data.options);
    renderPowerBasis(data.power_basis);
    renderDetails(data.options, data.power_basis);
    renderRejected(data.rejected);
    $("chart-power").src = data.charts.power || "";
    $("chart-power").hidden = !data.charts.power;
    $("chart-options").src = data.charts.options || "";
    $("chart-fds").src = data.charts.fds || "";
    $("chart-layout").src = data.charts.layout || "";
    $("results").hidden = false;
    $("status").textContent = `${data.n_candidates} candidates considered.`;
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showError(err.message);
    $("status").textContent = "";
  } finally {
    $("run").disabled = false;
  }
}

async function downloadMemo() {
  $("memo-status").textContent = "Building the memo…";
  $("download").disabled = true;
  try {
    const response = await fetch("api/memo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.lastForm || readForm()),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || "Could not build the memo.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "design-memo.pdf";
    link.click();
    URL.revokeObjectURL(url);
    $("memo-status").textContent = "Downloaded.";
  } catch (err) {
    showError(err.message);
    $("memo-status").textContent = "";
  } finally {
    $("download").disabled = false;
  }
}

async function fillFromDescription() {
  $("error").hidden = true;
  $("chat-notes").hidden = true;
  $("chat-fill").disabled = true;
  try {
    const data = await postJSON("api/intake/extract", { text: $("chat-text").value });
    writeForm(data.form || {});
    const notes = (data.form && data.form.notes) || [];
    if (notes.length) {
      $("chat-notes").innerHTML =
        "<strong>Check these — they were assumed, not stated:</strong><ul>" +
        notes.map((n) => `<li>${n}</li>`).join("") +
        "</ul>";
      $("chat-notes").hidden = false;
    }
  } catch (err) {
    showError(err.message);
  } finally {
    $("chat-fill").disabled = false;
  }
}

function updateModelNote() {
  const option = $("model-order").selectedOptions[0];
  let text = option
    ? `Fitting ${option.dataset.label || option.textContent.toLowerCase()}. More ambitious models need more runs.`
    : "";
  const firstGoal = $("responses").querySelector("tbody tr .r-goal");
  if (option && firstGoal && firstGoal.value !== "screen" && option.value !== "quadratic") {
    text +=
      " Your goal is to find a best setting or hit a target. That eventually needs the curvature model; " +
      "this simpler model is fine for a first pass to find which factors matter.";
  }
  $("model-note").textContent = text;
}

function applyPreset(id) {
  const preset = state.presets.find((p) => p.id === id);
  if (!preset) return;
  $("preset-note").textContent = preset.description || "";
  writeForm({
    factors: preset.factors,
    responses: (preset.responses || []).slice(0, 1),
    ...(preset.defaults || {}),
  });
}

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */

async function boot() {
  const [capabilities, presets] = await Promise.all([
    fetch("api/capabilities").then((r) => r.json()),
    fetch("api/presets").then((r) => r.json()),
  ]);
  state.capabilities = capabilities;
  state.presets = presets.presets;

  $("footer-version").textContent = `doe-advisor ${capabilities.version}`;

  $("model-order").innerHTML = capabilities.model_orders
    .map((m) => `<option value="${m.value}" data-label="${m.label}">${m.label}</option>`)
    .join("");
  $("model-order").value = "interaction";

  $("preset").innerHTML = state.presets.map((p) => `<option value="${p.id}">${p.label}</option>`).join("");
  const initial = state.presets.find((p) => p.id !== "blank") || state.presets[0];
  if (initial) {
    $("preset").value = initial.id;
    applyPreset(initial.id);
  }

  // The chat box exists only when it can actually work. An intake shortcut
  // that errors on click is worse than one that is simply absent.
  $("chat-block").hidden = !capabilities.llm;

  $("preset").onchange = (e) => applyPreset(e.target.value);
  $("add-factor").onclick = () => $("factors").querySelector("tbody").appendChild(factorRow());
  $("add-response").onclick = () => $("responses").querySelector("tbody").appendChild(responseRow());
  $("model-order").onchange = updateModelNote;
  $("run").onclick = runDesign;
  $("download").onclick = downloadMemo;
  $("chat-fill").onclick = fillFromDescription;
  updateModelNote();
}

boot().catch((err) => showError(`Could not start: ${err.message}`));
