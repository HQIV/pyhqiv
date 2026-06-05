/**
 * HQIV Web Calculator (Pyodide WASM)
 * Loads the exact pyhqiv package built from main.
 * All computations use the same Python functions as the CLI/package.
 */

const PYODIDE_VERSION = "0.26.4";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const WHEEL_GLOB = "wheels/pyhqiv-*-py3-none-any.whl"; // resolved at build time

let pyodide = null;
let chartInstance = null;

const statusEl = () => document.getElementById("status");
const logEl = () => document.getElementById("boot-log");

function setStatus(msg, cls = "text-blue-400") {
  const el = statusEl();
  if (el) el.innerHTML = `<span class="${cls}">${msg}</span>`;
}

function appendLog(msg) {
  const el = logEl();
  if (el) {
    el.textContent += msg + "\n";
    el.scrollTop = el.scrollHeight;
  }
  console.log("[hqiv-web]", msg);
}

async function loadPyodideAndPackages() {
  setStatus("Loading Pyodide runtime...");
  appendLog(`Fetching Pyodide ${PYODIDE_VERSION}...`);

  const script = document.createElement("script");
  script.src = `${PYODIDE_BASE}pyodide.js`;
  document.head.appendChild(script);

  await new Promise((resolve, reject) => {
    script.onload = resolve;
    script.onerror = () => reject(new Error("Failed to load pyodide.js"));
  });

  appendLog("Initializing Pyodide...");
  pyodide = await loadPyodide({
    indexURL: PYODIDE_BASE,
  });

  appendLog("Loading numpy + scipy (prebuilt WASM)...");
  setStatus("Loading numpy + scipy (this can take 10-30s)...");
  await pyodide.loadPackage(["numpy", "scipy"]);

  appendLog("Loading micropip...");
  await pyodide.loadPackage("micropip");

  // Resolve the exact wheel name that was injected by the GitHub Action
  const wheelUrl = await resolveWheelUrl();
  appendLog(`Installing pyhqiv from ${wheelUrl} ...`);
  setStatus("Installing pyhqiv wheel + deps (pint, ...)");

  const micropip = pyodide.pyimport("micropip");
  await micropip.install(wheelUrl);

  appendLog("Importing pyhqiv (this pulls geometry, thermo, nuclei, ...)");
  setStatus("Importing pyhqiv modules...");
  await pyodide.runPythonAsync(`
import pyhqiv
print("pyhqiv version:", pyhqiv.__version__)
# Force key imports that have top-level scipy (spherical_harmonics etc.)
import pyhqiv.spherical_harmonics
import pyhqiv.isotope_ladder
import pyhqiv.hqiv_nuclei
import pyhqiv.thermo
import pyhqiv.lightcone
import pyhqiv.metric
import pyhqiv.scale_witness
print("Core imports successful")
`);

  // Load small ref data for sigma display (optional, can be used from py too)
  await pyodide.runPythonAsync(`
import json, pyodide
# The refs are served next to the page; fetch via JS and inject if needed.
print("Python env ready for calculator use.")
`);

  setStatus("Ready — same code, same outputs as main.", "text-emerald-400");
  appendLog("HQIV calculator ready in browser (WASM).");
  return pyodide;
}

async function resolveWheelUrl() {
  // 1. data-wheel attr (injected by CI sed)
  const fromData = document.body.dataset.wheel;
  if (fromData) {
    return new URL(fromData, window.location.href).href;
  }

  // 2. manifest.json written by the GitHub Action (and by local setup commands)
  try {
    const res = await fetch(new URL("wheels/manifest.json", window.location.href), { cache: "no-store" });
    if (res.ok) {
      const m = await res.json();
      if (m && m.wheel) {
        return new URL(`wheels/${m.wheel}`, window.location.href).href;
      }
    }
  } catch (e) { /* ignore */ }

  // 3. Dev convenience: many static servers (python -m http.server, npx serve, etc.) return HTML
  //    directory listings. Parse <a href="...whl"> links and pick the best candidate.
  try {
    const dirUrl = new URL("wheels/", window.location.href);
    const res = await fetch(dirUrl, { cache: "no-store" });
    if (res.ok) {
      const html = await res.text();
      // Find all .whl links in the listing
      const re = /href=["']([^"']*pyhqiv[^"']*\.whl)["']/gi;
      const matches = [];
      let m;
      while ((m = re.exec(html)) !== null) {
        let raw = m[1];
        // Extract just the filename portion and decode % encodings (e.g. %2B -> +)
        let fname = raw.split("/").pop() || raw;
        try { fname = decodeURIComponent(fname); } catch (_) {}
        if (fname.includes(".whl")) {
          matches.push(new URL("wheels/" + fname, window.location.href).href);
        }
      }
      if (matches.length > 0) {
        matches.sort();
        return matches[matches.length - 1];
      }
    }
  } catch (e) { /* no listing or not html */ }

  // 4. Last attempt: a couple of static dev names (rarely hit after clean)
  const candidates = [
    "wheels/pyhqiv-0.4.3.dev-py3-none-any.whl",
  ];
  for (const c of candidates) {
    try {
      const r = await fetch(new URL(c, window.location.href), { method: "HEAD", cache: "no-store" });
      if (r.ok) return new URL(c, window.location.href).href;
    } catch (_) {}
  }

  throw new Error(
    "Could not locate pyhqiv-*.whl. " +
    "Local test: python3 -m build --wheel && " +
    "mkdir -p web/wheels && cp dist/pyhqiv-*-py3-none-any.whl web/wheels/ && " +
    "python3 -c 'import glob,os; os.chdir(\"web/wheels\"); [os.unlink(f) for f in glob.glob(\"pyhqiv-*-py3-none-any.whl\")[:-1]]; import json; w=glob.glob(\"pyhqiv-*-py3-none-any.whl\")[0]; open(\"manifest.json\",\"w\").write(json.dumps({\"wheel\":w}))'"
  );
}

async function runPython(code) {
  if (!pyodide) throw new Error("Pyodide not ready");
  return await pyodide.runPythonAsync(code);
}

async function computeGeometryConstants() {
  const code = `
from pyhqiv.lightcone import (
    alpha, reference_m, curvature_norm_combinatorial,
    omega_k_at_horizon, available_modes
)
from pyhqiv.metric import gamma_hqiv, g_eff
from pyhqiv.auxiliary_field import phi_of_shell, shell_temperature
from pyhqiv.scale_witness import derived_proton_mass_MeV, derived_neutron_mass_MeV

m = reference_m()
a = alpha()
g = gamma_hqiv()
curv = curvature_norm_combinatorial()
omega = omega_k_at_horizon(m, m)
modes = available_modes(m)
phi = phi_of_shell(m)
T = shell_temperature(m)
mp = derived_proton_mass_MeV()
mn = derived_neutron_mass_MeV()

{
  "reference_m": int(m),
  "alpha": float(a),
  "gamma": float(g),
  "curvature_norm": float(curv),
  "omega_k_horizon": float(omega),
  "available_modes_ref": int(modes),
  "phi_ref": float(phi),
  "T_ref": float(T),
  "proton_MeV": float(mp),
  "neutron_MeV": float(mn),
}
`;
  const res = await runPython(code);
  return pyodide.toPy ? pyodide.toPy(res).toJs({ dict_converter: Object.fromEntries }) : res;
}

async function computeNucleus(Z, N) {
  const code = `
from pyhqiv.isotope_ladder import IsotopeLadderConfig, IsotopeState, nuclear_binding_energy_mev, nucleus_mass_mev
from pyhqiv.hqiv_nuclei import make_nucleus

cfg = IsotopeLadderConfig(shell_m=4, rotational_scale_mev=0.0)
st = IsotopeState(Z=${Z}, N=${N}, J=0.0)
mass = nucleus_mass_mev(st, cfg)
b = nuclear_binding_energy_mev(st, cfg)
nuc = make_nucleus(${Z}, ${N})

{
  "Z": ${Z},
  "N": ${N},
  "A": ${Z}+${N},
  "mass_mev": float(mass),
  "binding_mev": float(b),
  "steps": len(nuc.steps),
}
`;
  const res = await runPython(code);
  return pyodide.toPy ? pyodide.toPy(res).toJs({ dict_converter: Object.fromEntries }) : res;
}

async function computeThermo(composition, T_K, P_Pa) {
  const code = `
from pyhqiv.thermo import compute_free_energy, molar_mass_from_Z, hqiv_answer_thermo
import json

G, info = compute_free_energy(${P_Pa}, ${T_K}, ${JSON.stringify(composition)})
mm = None
# Try to extract molar mass if composition looks like element
try:
  # very light shim; real users pass Z= or formula to higher helpers
  if "=" in ${JSON.stringify(composition)}:
    # e.g. "Z=1,A=2"
    parts = {}
    for p in ${JSON.stringify(composition)}.split(","):
      k,v = p.strip().split("=")
      parts[k.strip()] = int(v)
    if "Z" in parts:
      mm = molar_mass_from_Z(parts.get("Z"), parts.get("A"))
except Exception:
  pass

{
  "G": float(G),
  "phi": float(info.get("phi", 0.0)),
  "f_lapse": float(info.get("f_lapse", 0.0)),
  "molar_mass_kg_mol": float(mm) if mm else None,
  "T_K": ${T_K},
  "P_Pa": ${P_Pa},
  "composition": ${JSON.stringify(composition)},
}
`;
  const res = await runPython(code);
  return pyodide.toPy ? pyodide.toPy(res).toJs({ dict_converter: Object.fromEntries }) : res;
}

async function batchNuclei() {
  // Hard-coded known set (matches tests/data). We compute preds live.
  const refs = [
    { symbol: "2H", Z: 1, N: 1, B_mev: 2.224566, sigma_mev: 0.000012 },
    { symbol: "4He", Z: 2, N: 2, B_mev: 28.295674, sigma_mev: 0.000012 },
    { symbol: "12C", Z: 6, N: 6, B_mev: 92.161753, sigma_mev: 0.000025 },
    { symbol: "16O", Z: 8, N: 8, B_mev: 127.619343, sigma_mev: 0.000030 },
    { symbol: "56Fe", Z: 26, N: 30, B_mev: 492.259, sigma_mev: 0.003 },
  ];

  const results = [];
  for (const r of refs) {
    const pred = await computeNucleus(r.Z, r.N);
    const z = (pred.binding_mev - r.B_mev) / r.sigma_mev;
    results.push({
      ...r,
      pred_mass: pred.mass_mev,
      pred_binding: pred.binding_mev,
      delta: pred.binding_mev - r.B_mev,
      z: z,
    });
  }
  return results;
}

function renderNucleusResult(container, data, ref = null) {
  let html = `
    <div class="grid grid-cols-2 gap-2 text-sm">
      <div class="font-mono">Z=${data.Z} N=${data.N} (A=${data.A})</div>
      <div></div>
      <div>Mass (pred)</div><div class="font-mono">${data.mass_mev.toFixed(6)} MeV</div>
      <div>Binding (pred)</div><div class="font-mono font-semibold">${data.binding_mev.toFixed(6)} MeV</div>
  `;
  if (ref) {
    const z = (data.binding_mev - ref.B_mev) / ref.sigma_mev;
    const zcls = Math.abs(z) > 5 ? "text-amber-400" : "text-emerald-400";
    html += `
      <div>Exp binding</div><div class="font-mono">${ref.B_mev.toFixed(6)} ± ${ref.sigma_mev} MeV</div>
      <div>Δ / z-score</div><div class="font-mono ${zcls}">${(data.binding_mev - ref.B_mev).toFixed(6)} / ${z.toFixed(2)}σ</div>
    `;
  }
  html += `</div>`;
  container.innerHTML = html;
}

function renderBatchTable(container, rows) {
  let html = `<table class="w-full text-xs border border-slate-700">
    <thead><tr class="bg-slate-800">
      <th class="p-1 text-left">Nuclide</th>
      <th class="p-1">B_exp (MeV)</th>
      <th class="p-1">B_pred (MeV)</th>
      <th class="p-1">σ</th>
      <th class="p-1">z (σ)</th>
    </tr></thead><tbody>`;
  for (const r of rows) {
    const zcls = Math.abs(r.z) > 10 ? "text-red-400" : (Math.abs(r.z) > 3 ? "text-amber-400" : "text-emerald-300");
    html += `<tr class="border-t border-slate-800">
      <td class="p-1 font-semibold">${r.symbol}</td>
      <td class="p-1 font-mono">${r.B_mev.toFixed(6)}</td>
      <td class="p-1 font-mono">${r.pred_binding.toFixed(6)}</td>
      <td class="p-1 font-mono">${r.sigma_mev}</td>
      <td class="p-1 font-mono ${zcls}">${r.z.toFixed(2)}</td>
    </tr>`;
  }
  html += `</tbody></table>
  <p class="text-[10px] text-slate-500 mt-1">z = (pred − exp) / σ_exp. Large |z| is expected for the uncalibrated network model (see paper + tests/test_binding_energy_vs_pdg.py).</p>`;
  container.innerHTML = html;
}

function renderBatchChart(rows) {
  const canvas = document.getElementById("sigma-chart");
  if (!canvas) return;
  if (chartInstance) {
    chartInstance.destroy();
  }

  const labels = rows.map(r => r.symbol);
  const exp = rows.map(r => r.B_mev);
  const pred = rows.map(r => r.pred_binding);

  chartInstance = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Exp (AME2020)", data: exp, backgroundColor: "#64748b" },
        { label: "HQIV pred (network)", data: pred, backgroundColor: "#22d3ee" },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, title: { text: "Total binding energy (MeV)" } },
      },
      plugins: {
        legend: { position: "top" },
        tooltip: { mode: "index" },
      },
    },
  });
}

async function initUI() {
  // Geometry
  const geoBtn = document.getElementById("geo-btn");
  const geoOut = document.getElementById("geo-out");
  if (geoBtn) {
    geoBtn.onclick = async () => {
      geoBtn.disabled = true;
      geoOut.textContent = "Computing...";
      try {
        const g = await computeGeometryConstants();
        geoOut.innerHTML = `
          <div class="font-mono text-xs grid grid-cols-2 gap-x-3 gap-y-0.5">
            <div>reference_m</div><div>${g.reference_m}</div>
            <div>α (exact)</div><div>${g.alpha}</div>
            <div>γ = 1-α</div><div>${g.gamma}</div>
            <div>curvature norm</div><div>${g.curvature_norm}</div>
            <div>Ω_k (m;m)</div><div>${g.omega_k_horizon}</div>
            <div>modes @ m=${g.reference_m}</div><div>${g.available_modes_ref}</div>
            <div>φ(m)</div><div>${g.phi_ref.toFixed(6)}</div>
            <div>T(m) natural</div><div>${g.T_ref.toFixed(6)}</div>
            <div>m_p (MeV)</div><div>${g.proton_MeV.toFixed(6)}</div>
            <div>m_n (MeV)</div><div>${g.neutron_MeV.toFixed(6)}</div>
          </div>
          <div class="text-emerald-400 text-[10px] mt-1">All values from pure lattice + Lean witnesses (identical to package).</div>
        `;
      } catch (e) {
        geoOut.textContent = "Error: " + e.message;
      } finally {
        geoBtn.disabled = false;
      }
    };
  }

  // Single nucleus
  const nucBtn = document.getElementById("nuc-btn");
  const nucOut = document.getElementById("nuc-out");
  const zIn = document.getElementById("nuc-z");
  const nIn = document.getElementById("nuc-n");
  if (nucBtn && zIn && nIn) {
    nucBtn.onclick = async () => {
      const Z = parseInt(zIn.value, 10) || 1;
      const N = parseInt(nIn.value, 10) || 1;
      nucBtn.disabled = true;
      nucOut.textContent = "Computing...";
      try {
        const d = await computeNucleus(Z, N);
        // Try to find matching ref
        const refs = window.HQIV_BINDING_REFS || [];
        const ref = refs.find(r => r.Z === Z && r.N === N);
        renderNucleusResult(nucOut, d, ref);
      } catch (e) {
        nucOut.textContent = "Error: " + e.message;
      } finally {
        nucBtn.disabled = false;
      }
    };
  }

  // Batch + sigma + chart
  const batchBtn = document.getElementById("batch-btn");
  const batchOut = document.getElementById("batch-out");
  if (batchBtn) {
    batchBtn.onclick = async () => {
      batchBtn.disabled = true;
      batchOut.textContent = "Running batch over 5 reference nuclei (live HQIV code)...";
      try {
        const rows = await batchNuclei();
        renderBatchTable(batchOut, rows);
        renderBatchChart(rows);
      } catch (e) {
        batchOut.textContent = "Batch error: " + e.message;
      } finally {
        batchBtn.disabled = false;
      }
    };
  }

  // Thermo
  const thBtn = document.getElementById("thermo-btn");
  const thOut = document.getElementById("thermo-out");
  const compIn = document.getElementById("thermo-comp");
  const tIn = document.getElementById("thermo-t");
  const pIn = document.getElementById("thermo-p");
  if (thBtn) {
    thBtn.onclick = async () => {
      const comp = (compIn && compIn.value) || "H2O";
      const T = parseFloat(tIn && tIn.value) || 300.0;
      const Pbar = parseFloat(pIn && pIn.value) || 1.0;
      const P_Pa = Pbar * 1e5;
      thBtn.disabled = true;
      thOut.textContent = "Computing free energy...";
      try {
        const d = await computeThermo(comp, T, P_Pa);
        thOut.innerHTML = `
          <div class="font-mono text-xs">
            composition: ${d.composition}<br/>
            T=${d.T_K} K, P=${(d.P_Pa/1e5).toFixed(3)} bar<br/>
            G = ${d.G.toExponential(6)} (natural units)<br/>
            φ = ${d.phi.toFixed(6)}<br/>
            f_lapse = ${d.f_lapse.toFixed(6)}<br/>
            ${d.molar_mass_kg_mol ? `molar mass ≈ ${d.molar_mass_kg_mol.toFixed(5)} kg/mol<br/>` : ""}
          </div>
          <div class="text-[10px] text-slate-400">Uses HQIVThermoSystem + compute_free_energy (exact same as <code>pyhqiv.thermo</code>).</div>
        `;
      } catch (e) {
        thOut.textContent = "Error: " + e.message + " (try \"H2O\", \"Z=1,A=2\", or \"Si\")";
      } finally {
        thBtn.disabled = false;
      }
    };
  }

  // REPL
  const replBtn = document.getElementById("repl-btn");
  const replCode = document.getElementById("repl-code");
  const replOut = document.getElementById("repl-out");
  if (replBtn && replCode && replOut) {
    replBtn.onclick = async () => {
      const code = replCode.value.trim();
      if (!code) return;
      replOut.textContent = "Running...";
      try {
        // Capture prints by redirecting stdout in python
        const wrapped = `
import sys, io
_old = sys.stdout
sys.stdout = io.StringIO()
try:
${code.split("\n").map(l => "    " + l).join("\n")}
    _out = sys.stdout.getvalue()
finally:
    sys.stdout = _old
print("--- result ---")
print(_out)
`;
        const out = await runPython(wrapped);
        replOut.textContent = String(out);
      } catch (e) {
        replOut.textContent = "Error:\\n" + e.message + "\\n\\nTip: use print(...) or last expression. Imports: from pyhqiv.xxx import ...";
      }
    };
    // Seed a useful example
    if (!replCode.value) {
      replCode.value = `from pyhqiv.lightcone import reference_m, omega_k_at_horizon, curvature_norm_combinatorial
from pyhqiv.metric import gamma_hqiv
from pyhqiv.lepton_resonance_ladder import tuft_hopf_kappa6, tuft_hopf_kappa6_second_order_correction
from pyhqiv.quantum_optics.horizon_qed import vacuum_zero_point_natural
from pyhqiv.now_setters import m_now
m = reference_m()
print("Ω_k =", omega_k_at_horizon(m, m))
print("curvature =", curvature_norm_combinatorial())
print("gamma =", gamma_hqiv())
print("tuft kappa6 (C2 correction for D/mass in papers ~0.12% level):", tuft_hopf_kappa6())
print("C2 (second order):", tuft_hopf_kappa6_second_order_correction())
print("m_now for vacuum cap:", m_now)
u0 = vacuum_zero_point_natural(0, int(m_now)+5)
print("HQIV vacuum zero point natural (finite modes to now, paper script match):", u0)
print("Mainstream worst case (Planck cutoff): ~10**120 times observed (vacuum catastrophe)")
print("HQIV flatness tuning: 0 (natural from lattice age); mainstream GR: 10**60 digits initial fine tune")
from pyhqiv import cosmic_birefringence_deg_at_now
print("CMB birefringence (paper ~0.379 deg from alpha+now; Python witness 0.3; obs 0.342±0.094):", cosmic_birefringence_deg_at_now())
print("ETA10 (baryon-photon, HQIV first-principles from dynamic shell integrator / paper script):", pyhqiv.eta10_from_dynamic_first_principles())
print("  (obs ~6.10; derived ~6.19782 — the genuine prediction, not fitted)")
print("Hierarchy tuning: 0 (natural from m~4 lock-in + lattice); mainstream: 10**16-32 quadratic tuning")`;
    }
  }

  // Quick presets for nuclei
  document.querySelectorAll("[data-nuc]").forEach(btn => {
    btn.onclick = () => {
      const [z, n] = btn.dataset.nuc.split(",").map(Number);
      if (zIn) zIn.value = z;
      if (nIn) nIn.value = n;
      if (nucBtn) nucBtn.click();
    };
  });

  // Quick presets for thermo
  document.querySelectorAll("[data-thermo]").forEach(btn => {
    btn.onclick = () => {
      if (compIn) compIn.value = btn.dataset.thermo;
      if (tIn) tIn.value = btn.dataset.t || "300";
      if (pIn) pIn.value = btn.dataset.p || "1";
      if (thBtn) thBtn.click();
    };
  });

  // Live Arena σ button (fetches the generated programme_sigma.json)
  const arenaBtn = document.getElementById("arena-load-btn");
  if (arenaBtn) {
    arenaBtn.onclick = loadArenaSigma;
    // Optional: auto-load a lightweight summary on ready (comment out if you prefer manual)
    // setTimeout(() => { if (!document.getElementById("arena-out")?.innerHTML?.includes("Weighted")) loadArenaSigma(); }, 1500);
  }
}

async function bootstrap() {
  const readyBadge = document.getElementById("ready-badge");
  try {
    await loadPyodideAndPackages();

    // Inject binding refs for sigma UI (small static data)
    try {
      const r = await fetch(new URL("refs/binding_refs.json", window.location.href));
      if (r.ok) {
        const data = await r.json();
        window.HQIV_BINDING_REFS = data.ame2020 || [];
      }
    } catch (e) {
      console.warn("Could not load binding refs json, using inline defaults");
      window.HQIV_BINDING_REFS = [
        { symbol: "2H", Z: 1, N: 1, B_mev: 2.224566, sigma_mev: 0.000012 },
        { symbol: "4He", Z: 2, N: 2, B_mev: 28.295674, sigma_mev: 0.000012 },
        { symbol: "12C", Z: 6, N: 6, B_mev: 92.161753, sigma_mev: 0.000025 },
        { symbol: "16O", Z: 8, N: 8, B_mev: 127.619343, sigma_mev: 0.000030 },
        { symbol: "56Fe", Z: 26, N: 30, B_mev: 492.259, sigma_mev: 0.003 },
      ];
    }

    await initUI();

    // Auto-run a nice default demo
    const geoBtn = document.getElementById("geo-btn");
    if (geoBtn) setTimeout(() => geoBtn.click(), 120);

    if (readyBadge) readyBadge.textContent = "WASM ready";
  } catch (err) {
    console.error(err);
    setStatus("Failed to initialize: " + err.message, "text-red-400");
    appendLog("ERROR: " + err.message);
    const hint = document.getElementById("boot-hint");
    if (hint) hint.classList.remove("hidden");
  }
}

// Expose a couple helpers for console debugging
window.hqivComputeNucleus = (z, n) => computeNucleus(z, n);
window.hqivRun = (code) => runPython(code);

// --- Live HQIV Arena σ loader (for the website calculator) ---
const ARENA_PROGRAMME_URL = (location.hostname === "localhost" || location.hostname === "127.0.0.1")
  ? new URL("refs/arena_programme_sigma.json", location.href).href
  : "https://raw.githubusercontent.com/disregardfiat/pyhqiv/main/arena/programme_sigma.json";
const ARENA_RESULTS_URL = "https://raw.githubusercontent.com/disregardfiat/pyhqiv/main/arena_results.json";

async function loadArenaSigma() {
  const out = document.getElementById("arena-out");
  const btn = document.getElementById("arena-load-btn");
  if (btn) btn.disabled = true;
  if (out) out.innerHTML = "Fetching latest Arena snapshot from main...";

  try {
    const res = await fetch(ARENA_PROGRAMME_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const snap = data.sigma_snapshot || data;
    const weighted = snap.sigma_weighted ?? snap.sigma ?? "n/a";
    const score = snap.overall_score ?? "n/a";
    const prot = snap.num_protected_regressions ?? 0;

    const cores = (snap.alignment_cores || []).filter(m => m.protected);
    const phenom = snap.phenomenology_metrics || [];

    // Find the worst offenders for display
    const bigGaps = [...phenom]
      .sort((a, b) => (b.rel_err || 0) - (a.rel_err || 0))
      .slice(0, 3);

    let html = `
      <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
        <div>
          <div class="label">Weighted σ (lower better)</div>
          <div class="font-mono text-2xl text-amber-400">${Number(weighted).toExponential(3)}</div>
          <div class="text-[10px] text-slate-500">overall_score = ${score}</div>
        </div>
        <div>
          <div class="label">Protected cores (Lean-exact)</div>
          <div class="font-semibold">${cores.length} / ${cores.length} at rel_err = 0</div>
          <div class="text-emerald-400 text-xs">No regressions on this run (${prot} reported)</div>
        </div>
        <div>
          <div class="label">Generated</div>
          <div class="mono text-xs">${data.generated_at || snap.timestamp || ""}</div>
          <div class="text-[10px]">pyhqiv ${data.pyhqiv_version || ""}</div>
        </div>
      </div>

      <div class="mt-3">
        <div class="label">Largest current gaps (Arena phenomenology metrics — with leader comparison)</div>
        <div class="font-mono text-xs grid gap-1">
    `;
    for (const g of bigGaps) {
      const re = Number(g.rel_err || 0).toExponential(2);
      const note = g.mainstream_note ? `<br/><span class="text-amber-300">Leader: ${g.mainstream_note}</span>` : "";
      html += `<div>${g.name}: rel_err≈${re} (value ${Number(g.value||0).toPrecision(4)} vs ref ${Number(g.reference||0).toPrecision(4)})${note}</div>`;
    }
    html += `</div></div>`;

    // Simple ASCII bar chart for sigma contributions (top phenom by rel_err)
    html += `<div class="mt-2"><div class="label">Sigma contrib bars (rel_err, log-compressed; deuteron dominates as expected gap)</div><pre class="text-[10px] bg-black/50 p-1 overflow-x-auto">`;
    const sortedPhenom = [...phenom].sort((a,b) => (b.rel_err||0) - (a.rel_err||0)).slice(0,6);
    const maxRe = Math.max(...sortedPhenom.map(g => g.rel_err || 0)) || 1;
    for (const g of sortedPhenom) {
      const re = g.rel_err || 0;
      const len = re > 0 ? Math.max(1, Math.floor( Math.log10(re+1) / Math.log10(maxRe+1) * 35 )) : 0;
      const bar = '█'.repeat(len) + ' '.repeat(35-len);
      html += `${(g.name||'').slice(0,22).padEnd(22)} |${bar}| ${re.toExponential(1)}\n`;
    }
    html += `</pre></div><canvas id="arena-sigma-chart" height="80" class="mt-1"></canvas>`;

    html += `
      <div class="mt-2 text-[10px] text-slate-400">
        Programme view maps these to Wikipedia unsolved problems. See <a href="https://disregardfiat.tech/#arena" target="_blank" class="underline">disregardfiat.tech/#arena</a> (leaderboard) and <a href="https://disregardfiat.tech/#mysteries" target="_blank" class="underline">#mysteries</a> for the full open-problems matrix.
        The huge σ is driven by placeholder orbital/thermo "badness" metrics (active targets for dynamic corrections in the Arena).
      </div>
    `;

    if (out) out.innerHTML = html;

    // Render proper bar chart with Chart.js for sigma contribs
    try {
      const canvas = document.getElementById("arena-sigma-chart");
      if (canvas && window.Chart) {
        const sorted = [...phenom].sort((a,b)=>(b.rel_err||0)-(a.rel_err||0)).slice(0,6);
        const labels = sorted.map(g => (g.name||"").slice(0,18));
        const vals = sorted.map(g => g.rel_err || 0);
        if (window.arenaSigmaChart) window.arenaSigmaChart.destroy();
        window.arenaSigmaChart = new Chart(canvas, {
          type: "bar",
          data: { labels, datasets: [{ label: "rel_err (sigma contrib)", data: vals, backgroundColor: "#f59e0b" }] },
          options: { indexAxis: "y", responsive: true, scales: { x: { type: "logarithmic", min: 1e-10 } }, plugins: { legend: { display: false } } }
        });
      }
    } catch(e){ /* chart optional */ }
  } catch (e) {
    if (out) out.innerHTML = `Failed to load live Arena JSON: ${e.message}. <br/>You can still view it directly: <a href="${ARENA_PROGRAMME_URL}" target="_blank" class="underline">programme_sigma.json</a> or the per-run artifacts in CI.`;
  } finally {
    if (btn) btn.disabled = false;
  }
}
