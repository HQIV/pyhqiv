# pyhqiv — Horizon-Quantized Informational Vacuum (HQIV) Calculator

[![PyPI version](https://badge.fury.io/py/pyhqiv.svg)](https://badge.fury.io/py/pyhqiv)
[![CI](https://github.com/disregardfiat/pyhqiv/actions/workflows/ci.yml/badge.svg)](https://github.com/disregardfiat/pyhqiv/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18794889.svg)](https://doi.org/10.5281/zenodo.18794889)

> **⚠️ Experimental status.** All features in this package are experimental. APIs and numerical results may change. Public contribution and feedback are greatly appreciated — please open issues or pull requests on [GitHub](https://github.com/disregardfiat/pyhqiv).

**pyhqiv** is the clean, first-principles Python calculator for the HQIV framework (discrete null-lattice combinatorics + horizon monogamy + octonionic carriers). It exactly mirrors the Lean formalization in [HQIV/hqiv-lean](https://github.com/HQIV/hqiv-lean) and the paper series in `HQIV_LEAN/papers/`.

It is designed as the **usable calculator**:
- **src/pyhqiv/** contains **only pure geometry + functional code** (no physics constants except cube diagonals √3, 2π phase, naturals).
- All scale/anchors (masses, T ladder, etc.) come from Lean witnesses (via `lean_witnesses` + `scale_witness`).
- Applied modules (thermo, orbital, nuclei, etc.) take **minimal inputs** (just A/Z or composition for elements/isotopes/compounds) — everything derives from the foundation + local conditions.
- Comparisons and benchmarks live in tests with **explicit error bars from source material** (PDG, Planck, literature sigmas, paper tables).
- The **HQIV Arena** (with hqiv-lean) is the improvement engine: submit pure dynamic functions (second-order terms, corrections) or features + tests; CI ensures they beat σ everywhere with no protected regressions.

See the foundational paper: Ettinger, Steven Jr. *Horizon-Quantized Informational Vacuum (HQIV)...* Zenodo 2026 (DOI above) and the full paper series at `HQIV_LEAN/papers/`.

## Citation

Use the Zenodo record or the repo button (CITATION.cff):

```bibtex
@misc{ettinger2026hqiv,
  author       = {Ettinger, Steven Jr},
  title        = {Horizon-Quantized Informational Vacuum (HQIV): A Unified Framework from Causal Horizon Monogamy and Discrete Null-Lattice Combinatorics},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18794889},
  url          = {https://doi.org/10.5281/zenodo.18794889}
}
```

## Installation

```bash
pip install pyhqiv
```

From source (editable, for development/arena):

```bash
git clone https://github.com/disregardfiat/pyhqiv.git && cd pyhqiv
pip install -e ".[dev]"
```

Optional extras: `ase`, `mda`, `qutip`, `jax`, `pyvista`, `cosmology`, `all`.

## The Clean Calculator Rules (important for users & contributors)

- **src/pyhqiv/ is pure and const-free** (except geometry). No masses, T_CMB, α_EM, hbar_SI, etc. in .py source.
- Use `from pyhqiv.scale_witness import ...` or `load_lean_witnesses()` for anchors (derivedProtonMass, referenceM, resonance k's, etc.). Local conditions (CMB now, earth surface g/ε0, H0, etc.) live in `tests/setup_defaults.py` + `src/pyhqiv/local_conditions.json` (for tests/applied use).
- **Minimal inputs**: For thermo/phase/allotropes/masses/bindings → just Z (atomic number), optional A (mass number), or formula/stoich. The foundation (lightcone + metric + auxiliary + fluid) + witnesses do the rest.
- All paper comparisons / benchmarks have tests with **error bars from the source** (see `tests/test_all_paper_comparisons_with_errors.py`, `test_thermo.py`, `test_binding_energy_vs_pdg.py`, etc.). Gaps are expected and scored via z/σ.
- **Arena is how you improve it**: New pure functions (no consts) or features must come with tests + error bars. CI (Lean alignment, pytest, sigma scoring) decides merges. See below.

## Foundation (the core — use these directly)

```python
from pyhqiv.lightcone import (
    alpha,                    # exactly 3/5
    available_modes,          # 4*(m+2)*(m+1) from lattice axiom
    curvature_norm_combinatorial,  # 6^7 * sqrt(3) pure geometry
    omega_k_at_horizon, omega_k_partial,
    reference_m, new_modes, shell_shape, ...
)
from pyhqiv.metric import (
    gamma_hqiv,               # exactly 2/5 = 1 - alpha
    hqvm_lapse,               # 1 + Φ + φ * t
    g_eff, three_minus_gamma, hqvm_friedmann_residual, ...
)
from pyhqiv.auxiliary_field import (
    phi_of_shell, shell_temperature,  # φ(m) = 2/(m+1), T(m)=1/(m+1) (T_Pl=1 natural)
    phi_of_temperature, ...
)
from pyhqiv.scale_witness import (
    ScaleWitness, defaultScaleWitness,  # proton_lockin / codata_alpha / cmb_now
    derived_proton_mass_MeV, derived_neutron_mass_MeV,
    local_cmb_temperature_K, local_earth_surface_g,
    molar_mass_from_Z,  # just A/Z → kg/mol from anchors
    xi_g_for_witness, load_local_conditions,
)
from pyhqiv.lean_witnesses import load_lean_witnesses  # single source of truth (Lean export + overlay)
from pyhqiv.thermodynamic_fundamentals import (
    horizon_entropy_counting, entropy_increment_per_shell,  # S ∝ cum modes, ΔS > 0
    second_law_arrow_holds, temperature_at_shell,
    local_equilibrium_proxy, ...
)
```

**Geometry-only repro** (paper numbers, no scales):

```python
from pyhqiv.lightcone import reference_m, omega_k_at_horizon, curvature_norm_combinatorial
from pyhqiv.metric import gamma_hqiv

m = reference_m()  # 4
print("Ω_k(self at horizon) =", omega_k_at_horizon(m, m))  # == 1.0 (theorem)
print("curvature norm =", curvature_norm_combinatorial())  # 6^7√3
print("gamma =", gamma_hqiv())  # 0.4
```

## Key Applied Modules (everything flows from A/Z + foundation)

### Thermodynamics, Phase, Allotropes, Specific Heat, etc. (`pyhqiv.thermo`)
Full first-principles from the axiom (no DAC/ref data). Inputs are just composition (Z/A or formula).

```python
from pyhqiv.thermo import (
    HQIVThermoSystem, compute_free_energy, hqiv_answer_thermo,
    PhaseDiagramGenerator, HQIVHydrogen,
    molar_mass_from_Z, allotrope_theta_modifier,
    theta_local_from_density, phi_from_rho_T,
    TESTABLE_PREDICTIONS,
)

# Just A/Z
M_H2 = molar_mass_from_Z(Z=1, A=2)
M_H2O = 2 * molar_mass_from_Z(1, 1) + molar_mass_from_Z(8, 16)
print("M_H2O (kg/mol) from anchors:", M_H2O)

# Allotropes for same Z (different packing → different effective Θ/ρ)
mod_ice = allotrope_theta_modifier("ice_ih")
mod_diamond = allotrope_theta_modifier("diamond")
mod_graphite = allotrope_theta_modifier("graphite")

# Full system + phase + free energy (composition string or Z-based)
sys = HQIVThermoSystem(P_Pa=1e5, T_K=300.0, composition="Z=1,A=2")  # or "H2O"
G, info = compute_free_energy(1e5, 300.0, "H2O")
print("G, phi, f:", G, info["phi"], info["f_lapse"])

# Answerer (parses questions)
print(hqiv_answer_thermo("metallic hydrogen transition at 300 K"))  # GPa
print(hqiv_answer_thermo("silicon melting at 10 GPa"))  # K

# Phase stability, testable predictions, etc.
```

See `tests/test_thermo.py` for A/Z-driven cases (H2, H2O, ice allotrope ~272 K, Si melt, blackbody heat proxies from papers, conductivity/phase stubs) with error bars.

### Orbital / Flyby Anomalies / SPARC Galaxy Rotation (`pyhqiv.orbital`)
Live HQIV corrections for the orbital_flyby and octonionic_action papers.

```python
from pyhqiv.orbital import (
    hqiv_galaxy_rotation_point,
    hqiv_flyby_inertia_screen,
    hqiv_inertia_factor, rindler_denominator,
)

# Galaxy (SPARC-style, exponential disk + inertia + Rindler)
pt = hqiv_galaxy_rotation_point(
    radius=10.0, disk_total_mass=5e9, disk_scale_length=1.8,
    observed_v=110.0, phi_shell=0
)
print("a_bary, f_inertia, a_hqiv:", pt["a_bary"], pt["f_inertia"], pt["a_hqiv"])

# Flyby screen (direction-dependent, polar fiber, m_shell)
screen = hqiv_flyby_inertia_screen(a_loc=9.8, phi=2.0, h_z=0.5, h=1.0, h_ref=1.0, rho_pol=0.5, m_shell=0)
```

Benchmarks in the master comparison test use live code + paper literature sigmas/error bars (NEAR/Galileo/etc. anomalies, M33 etc. flat curves).

### Other High-Value Modules
- `pyhqiv.fluid`: `f_inertia(a, φ)`, `g_vac_vector`, `eddy_viscosity` (core for modified NS, inertia screen, flyby/galaxy).
- `pyhqiv.isotope_ladder` / `pyhqiv.hqiv_nuclei`: Binding, masses, Q-values, half-lives from A/Z + network (Lean mirrors). Used for thermo/nuclear.
- `pyhqiv.sm_mass_ladder`, `pyhqiv.sm_gr_unification`: Geometric SM masses/couplings from electron anchor + Lean.
- `pyhqiv.state`, `pyhqiv.carrier` (So8Carrier), `pyhqiv.regimes`: Unified shell/metric/carrier + galactic/blackhole/quantum façades.
- `pyhqiv.so8_generators`: Lean-certified so(8) (28 dim).
- Response/semiconductors/crystal/ase: materials (conductivity, band gaps, defects, relaxation) with HQIV φ/lapse corrections.
- `pyhqiv.thermodynamic_fundamentals`: Entropy from modes, 2nd-law arrow, equilibrium proxies.

Full list in `src/pyhqiv/__init__.py` and the module docstrings (Lean citations everywhere).

## Quickstarts

**Pure geometry / paper foundations (no scales):**
```python
from pyhqiv.lightcone import reference_m, omega_k_at_horizon, curvature_norm_combinatorial
from pyhqiv.metric import gamma_hqiv
m = reference_m()
print(omega_k_at_horizon(m, m), curvature_norm_combinatorial(), gamma_hqiv())
```

**Thermo for real materials (just A/Z):**
```python
from pyhqiv.thermo import molar_mass_from_Z, HQIVThermoSystem, hqiv_answer_thermo
M = molar_mass_from_Z(6)  # carbon
print(hqiv_answer_thermo("diamond vs graphite density or ice melt"))
sys = HQIVThermoSystem(1e5, 300.0, composition="Z=6")  # or allotrope-aware
```

**Orbital benchmarks (live):**
```python
from pyhqiv.orbital import hqiv_galaxy_rotation_point, hqiv_flyby_inertia_screen
# ... as above
```

See `examples/` (many still reference older surface; the clean ones are in tests + the paper repro scripts).

## Web / WASM Calculator

A live calculator that runs the **exact same package** (identical inputs → identical outputs, including σ comparisons) entirely in the browser via Pyodide + WebAssembly.

- Live at: https://disregardfiat.github.io/pyhqiv/ (after first `main` push + Pages enabled)
- Built **only on pushes to `main`** (see `.github/workflows/web.yml`). Never on PRs.
- Features:
  - Pure geometry + Lean witness constants
  - Nucleus binding energies + masses via the isotope ladder (same code as `isotope_ladder`)
  - z-score / σ display against the AME2020 references used in `tests/test_binding_energy_vs_pdg.py`
  - Thermo free energy, φ, lapse etc. (same `compute_free_energy`)
  - Full Python REPL for any public API
  - Optional charts (Chart.js) comparing predictions vs experiment

**Local test of the web UI:**
```bash
python -m build
mkdir -p web/wheels
cp dist/pyhqiv-*-py3-none-any.whl web/wheels/
python -m http.server -d web 8080
```
Open http://localhost:8080. The JS auto-detects the wheel.

See `web/README.md` and `web/main.js` for how the bridge works (very thin — most logic stays in the real `src/pyhqiv/` modules).

## Package Layout (current clean rebuild)

| Path | Description |
|------|-------------|
| `lightcone.py` | Discrete null lattice, α=3/5, modes, curvature norm, Ω_k, shell shapes (Lean OctonionicLightCone) |
| `metric.py` | HQVM lapse (1+Φ+φ t), γ=2/5, G_eff(φ)=φ^α, Friedmann (Lean HQVMetric) |
| `auxiliary_field.py` | φ(m), T(m) ladder (Lean AuxiliaryField + SM_GR_Unification) |
| `scale_witness.py` | ScaleWitness enum (proton_lockin default), derived masses, local conditions (CMB, earth, H0...), molar_mass_from_Z |
| `thermo.py` | HQIVThermoSystem, free energy, phase diagrams, EOS, H2 metallic, allotrope modifiers, answerer, specific heat proxies (A/Z driven) |
| `orbital.py` | Galaxy rotation (SPARC), flyby inertia screens, Rindler, corrections (live for paper benchmarks) |
| `thermodynamic_fundamentals.py` | Lattice entropy S(m), 2nd-law arrow, equilibrium proxies, blackbody finite sums |
| `fluid.py` | f_inertia, g_vac, eddy viscosity (core modified NS / inertia screen) |
| `isotope_ladder.py`, `hqiv_nuclei.py` | A/Z → masses, binding, Q, half-lives, caustics (Lean mirrors) |
| `sm_mass_ladder.py`, `sm_gr_unification.py` | Geometric SM masses/couplings from electron anchor |
| `lean_witnesses.py` | Loader for Lean JSON (single source + overlay) |
| `so8_generators.py`, `carrier.py`, `state.py`, `regimes/` | so(8), carriers, unified HQIVState, galactic/blackhole/quantum façades |
| `response.py`, `crystal.py`, `semiconductors.py` (if present), `ase_interface.py` | Materials response, conductivity, PBC, ASE calculator with HQIV corrections |
| (others) | nuclear, modified_maxwell, quantum_*, etc. |

Legacy surface is in `bak/` + `docs/legacy_api_inventory.md`. The rebuild is the forward path.

## Tests, Reproducibility & Paper Coverage

```bash
pip install -e ".[dev]"
pytest tests/ -q
# or focused
pytest tests/test_all_paper_comparisons_with_errors.py -q
pytest tests/test_thermo.py -q
```

- `tests/test_all_paper_comparisons_with_errors.py`: Master aggregator for **every** numerical comparison vs experiment in the papers (flyby anomalies with lit sigmas, SPARC curves, masses, bindings, ice/Si melt, blackbody heat ratios, thermo phase/allotrope, etc.). All with explicit error bars + z-scores from PDG/Planck/literature/paper sources. Live calculator code where possible.
- `tests/test_thermo.py`, `test_binding_energy_vs_pdg.py`, `test_hadron_masses_with_errors.py`, `test_lepton_resonance.py`, `test_isotope_ladder.py`, `test_nuclear*.py`, `test_paper_numbers.py`, `test_sm_mass_ladder.py`, etc.: Specific coverage with error bars.
- Many use z-scores / loose envelopes (model gaps are real; Arena improves them).

Reproduce paper numbers / figures with current examples (update as modules stabilize) or the paper script bundles in `HQIV_LEAN/papers/*/scripts/`.

## HQIV Arena & How to Submit PRs / Contribute

pyhqiv + hqiv-lean = branch-based, CI-driven physics improvement platform.

**To participate (recommended even for humans):**

1. `hqiv-arena login` (GitHub PAT with repo scope).
2. `hqiv-arena clone` (or `hqiv-arena clone my-workspace`) — sets up symlinks to HQIV_LEAN.
3. `cd .../pyhqiv; hqiv-arena setup`
4. `hqiv-arena run` — runs alignment + tests + sigma scoring locally.
5. Make changes:
   - **New pure function / correction** (second-order terms etc.): put in `src/pyhqiv/` (no constants!). Use foundation + witnesses only.
   - **New feature** (e.g. better allotrope, conductivity tensor, new orbital channel): add the code + **new tests** with error bars from source material. Inputs minimal (A/Z where applicable).
   - Register metrics if scoring impact: `from pyhqiv.arena.metrics import register_metric, Metric` (see examples in tests or `arena/metrics.py`).
6. `hqiv-arena submit --note-file progress.md --model "YourName/Agent"` — creates branch, PR, runs full CI (Lean cert + 5 gates).
7. Only improving, aligned changes that beat baseline σ (no protected regressions on Ω_k, lapse, proton anchor, so(8)=28, derived masses, flyby/SPARC/thermo residuals, etc.) merge. Leaderboard updates on main.

**Gates (see `.github/workflows/hqiv-arena.yml` and CONTRIBUTING.md):**
1. Lean certificate (lake build, no sorrys).
2. Lean ↔ Python alignment (`scripts/validate_hqiv_alignment.py`).
3. Full pytest.
4. Sigma-everywhere scoring (deltas vs baseline; protected regressions penalized).
5. Leaderboard + badges on merge.

**Templates & details:**
- `arena/templates/new_benchmark_test.py.template`
- `CONTRIBUTING.md` (full workflow, adding benchmarks, badges)
- `arena/SKILL.md` (hqiv-arena CLI reference)
- Live: https://disregardfiat.tech/#arena (pulls `arena/leaderboard.json`)

Main is sacred. Serious work on branches. Use `hqiv-arena sync` / `reset` to stay at frontier.

**License note:** MIT with Government Use Restriction (see full text in repo).

## Reproducibility & Paper Numbers

The calculator reproduces Lean/paper foundations (Ω_k(N;N)=1, curvature norm = 6^7√3, α=3/5, γ=2/5, referenceM=4, so(8)=28, derived masses, etc.) via witnesses + pure code. See `tests/test_paper_numbers.py` and the master comparison test.

For full paper tables/figures, use the scripts bundles shipped with each paper on Zenodo (or the Lean targets).

## Further Reading

- Papers + scripts: `HQIV_LEAN/papers/` (thermodynamics_arrow, orbital_flyby, octonionic_action, tuft_sm_lagrangian, nucleon_binding, etc.)
- Lean: `HQIV_LEAN/hqiv-lean/Hqiv/Geometry/`, `Physics/`, etc.
- hqiv_lab (allotropes/condensed phase): `HQIV_LEAN/hqiv_lab/`
- Docs in this repo: `docs/`
- Arena CLI skill: `arena/SKILL.md`

Contributions that add coverage (more paper comparisons with error bars, new pure modules) or improve σ are especially welcome.

Happy σ reduction!