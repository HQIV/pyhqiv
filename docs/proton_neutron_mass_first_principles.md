# Proton and neutron mass: first-principles audit

Analysis of the current implementation for **emergent** proton/neutron mass from bound quarks, with **no constants** other than unit conversions. All non-derived numbers and epsilons are flagged; first-principles improvements are suggested.

**Design principle (docs/binding_energy_walkthrough.md):** The **total energy state** of any system lives in its **8×8 matrix**. There is no separate Coulomb force or Coulomb energy; what we call Coulomb is encoded in that state. Geometry (quark stand-off, equilibrium distances) should come from **PDEs for the energy well** (minimize \(E_{\rm tot} = \int (\rho c^2 + \hbar c/\Delta x) d^3x\) or solve the resulting Euler–Lagrange equations), not from force-based relaxation.

---

## 1. Current flow (proton/neutron mass)

| Step | Where | What |
|------|--------|------|
| Quark masses | `constants.py` | `M_U_MEV_QCD`, `M_D_MEV_QCD` → radii r_u, r_d = ħc/(m_q c²) |
| Quark geometry | `subatomic`, `relax_quark_positions` | **Placeholder:** equilibrium from Coulomb force + touching (to be replaced by PDE-derived well) |
| Free Θ | `_quark_geometry_theta_m("uud"/"udd")` | μ = Σr/√(Σr²), Θ = L×8×μ |
| Coulomb energy | `_quark_coulomb_energy_mev` | **To remove:** E_Coul added by hand; target: all energy from 8×8 only |
| **Nucleon energy** | `proton_energy_mev`, `neutron_energy_mev` | **Implemented:** E from 8×8 composite only (Θ_eff from L×effective_modes; T_QCD at epoch). No separate E_Coul. |
| NuclearConfig | `nuclear.py` | Uses `nucleon_energies_mev(t_cmb, epoch="now")` for E_free (layer 0) |
| Elsewhere | `nuclear.py`, tests, examples | Still use **M_PROTON_MEV**, **M_NEUTRON_MEV** for node masses and phase lift where not yet switched |

**Target:** Geometry from PDE-based energy well (replace force-based relax). Replace M_PROTON_MEV/M_NEUTRON_MEV with emergent values from the 8×8 path everywhere.

**Parameter-free scale (implemented):** Nucleon masses from **T_QCD + Fano at "now"** — no fit to 938/939. **Note:** \(T_{\rm lock}\) in the cosmology section is the lock-in *epoch* value; the scale **at "now"** can be different. Use `t_qcd_now_gev` when the lattice/paper defines a QCD temperature at the today hypersurface; default uses \(T_{\rm lock}\). \(E_{\rm scale} = T_{\rm QCD\,now} \times 1000/\sqrt{3}\) MeV; L from that and 8; \(E = \hbar c/(L \times \text{effective\_modes})\). See `proton_energy_mev(..., t_qcd_now_gev=None, epoch="now")`, `_energy_scale_mev_from_t_qcd_fano(t_qcd_now_gev=None)`.

---

## 2. Constants and epsilons (inventory)

### 2.1 Input physics constants (should become emergent or stay as “unit / SM”)

| Constant | Location | Value | Role | First-principles status |
|----------|----------|--------|------|--------------------------|
| **M_PROTON_MEV** | `constants.py` | 938.272 | Proton rest mass (MeV) | **Should be emergent**: use `proton_energy_mev()` everywhere |
| **M_NEUTRON_MEV** | `constants.py` | 939.565 | Neutron rest mass (MeV) | **Should be emergent**: use `neutron_energy_mev()` everywhere |
| **M_U_MEV_QCD** | `constants.py` | 2.2 | Up-quark mass → r_u, quark radii | Doc: "in full HQIV derived from mass equation at now" — **placeholder** |
| **M_D_MEV_QCD** | `constants.py` | 4.7 | Down-quark mass → r_d | Same — **placeholder** until mass equation at lattice "now" |
| ALPHA_EM_INV | `constants.py` | 137.036 | 1/α (used in E_Coul placeholder) | **Layer 0 target:** no separate Coulomb; energy from 8×8 only. Keep only if needed for other (e.g. EM) unit conversion. |
| HBAR_C_MEV_FM | `constants.py` | 197.3 | ħc (MeV·fm) | Unit conversion; keep |
| T_CMB_K / 2.725 | `constants.py`, `hqiv_scalings` | 2.725 | "Now" / lapse scale | Paper "now"; keep |

### 2.2 Tuning / magic numbers (not from axiom or unit conversion)

| Constant | Location | Value | Role | First-principles improvement |
|----------|----------|--------|------|------------------------------|
| **_COULOMB_RELAX_SCALE** | `horizon_network.py` | 0.001 | Coulomb force scale in relax (quark + nucleon) | **Remove:** No Coulomb force. Geometry from PDE energy-well minimization; energy from 8×8 only (see binding_energy_walkthrough §6.1, §6.5.1). |
| **R_EQ_SCALE** | `horizon_network.py` | 1.4/1.2 | r_eq = scale×(r_i+r_j) for graph connection | Derive from `equilibrium_separation_two_horizons(r1,r2,L)` so scale emerges from V_eff |
| **_LAMBDA_COH_FACTOR** | `horizon_network.py` | 2.0 | λ_coh = factor × (r1+r2) for trace term | Paper or algebra; or remove if trace term is derived |
| **quark_state_matrix scale** | `subatomic.py` | 0.1 (u), 0.12 (d) | Flavor strength in 8×8: I + scale*gen | Derive from m_u/m_d or from algebra (e.g. hypercharge block), not ad hoc |
| **sub_scale = L*1e-4** | `subatomic._constituent_horizons_m` | 1e-4 | Sub-nucleon Θ scale (unused in nucleon E path) | Remove or replace: constituent Θ should come from quark merge (Θ from μ and L), not L×1e-4 |
| **hqiv_scalings** | `hqiv_scalings.py` | 2.0e-43, 3.71e54, 1.38e-23, 1.2e20 | TAU_TICK, MACROSCOPIC_DECAY_SCALE, PHI_CRIT, LATTICE_BASE factor | Prefer expressions from paper (T_CMB, L_Pl, lapse) so no bare magic numbers |

### 2.3 Numerical guards (epsilons)

| Pattern | Location (examples) | Purpose | Recommendation |
|----------|----------------------|----------|-----------------|
| **1e-30** | subatomic, horizon_network, nuclear, … | Avoid div-by-zero (denom, clip) | Keep as machine/stable constant; consider single `EPS_DENOM` in constants or utils |
| **1e-20, 1e-35** | horizon_network, nuclear | Lower bounds on r, lattice_base | Same |
| **1e-300** | lattice.py | log(T_Pl/T) guard | Same |

These are numerical only; document once (e.g. "EPS_DENOM = 1e-30 for safe division") so they are not mistaken for physics.

---

## 3. First-principles improvements (concise)

1. **Nucleon mass everywhere**
   - Replace **M_PROTON_MEV** / **M_NEUTRON_MEV** with `proton_energy_mev(t_cmb)` / `neutron_energy_mev(t_cmb)` (or cached values from `nucleon_energies_mev`) in:
     - `build_nucleon_matrix_with_phase` (E_mev for phase θ),
     - `minimize_nucleon_configuration` (node masses),
     - `_binding_energy_via_network` (radii r_p, r_n from E_p, E_n via r = ħc/E),
     - tests/examples that build particles with `mass_mev`.
   - Keep M_PROTON_MEV/M_NEUTRON_MEV only as **optional** reference/validation (e.g. tests that compare emergent vs PDG).

2. **Quark masses**
   - Keep M_U_MEV_QCD, M_D_MEV_QCD as placeholders until "mass equation at now" exists; document clearly: "Input: current quark masses; target: from lattice/HQIV mass equation."
   - Optionally allow passing quark masses into `proton_energy_mev`/`neutron_energy_mev` so the only free inputs are (t_cmb, m_u, m_d) and unit conversions.

3. **No Coulomb force; energy and geometry from 8×8 + PDEs**
   - **Remove** the Coulomb force and the E_Coul term entirely. Total energy lives in the 8×8 state; what we call Coulomb is part of the algebraic invariants (hypercharge, trace(M@Δ)).
   - **Geometry** (quark stand-off, binding angles, nucleon positions): replace force-based `relax_quark_positions` / `relax_nucleon_positions` with **PDE-based** determination of the energy well: minimize \(E_{\rm tot} = \int (\rho c^2 + \hbar c/\Delta x) d^3x\) over the configuration, or solve the Euler–Lagrange PDEs. See docs/binding_energy_walkthrough.md §6.5.1.
   - Nucleon energy: E from composite 8×8 only (e.g. Θ_eff from 8 + trace(M@Δ), E = ħc/Θ_eff).

4. **R_EQ_SCALE**
   - Derive connection threshold from `equilibrium_separation_two_horizons(r1, r2, lattice_base_m)` (or equivalent min of V_eff) so the "1.4/1.2" factor is not hard-coded.

5. **Quark 8×8 flavor scale**
   - Replace `scale = 0.1 if flavor == "u" else 0.12` in `quark_state_matrix` with something from:
     - algebra (e.g. hypercharge or mass ratio from m_u, m_d), or
     - a single dimensionless ratio (m_d/m_u) and a base scale from ħc/lattice, so no ad hoc 0.1/0.12.

6. **_constituent_horizons_m**
   - Not used in the nucleon-energy path (proton_energy_mev uses Θ from μ and Coulomb). Either remove, or repurpose so constituent Θ come from the same merge/μ logic (no L×1e-4).

7. **Single epsilon constant**
   - Add e.g. `EPS_DENOM = 1e-30` (and optionally `EPS_DENOM_SQ`) in constants or utils; use it everywhere a denominator is clamped to avoid confusion with physics constants.

---

## 4. Summary table (existence report)

| Type | Existence |
|------|------------|
| **Proton/neutron mass constants** | M_PROTON_MEV, M_NEUTRON_MEV in constants.py for reference/PDG. Nucleon energies use first-principles path (proton_energy_mev, neutron_energy_mev from 8×8); still replace M_* in node masses and phase lift where used. |
| **Quark mass constants** | Yes: M_U_MEV_QCD, M_D_MEV_QCD in constants.py; used in subatomic.py (radii, quark_nodes_for_nucleon). Documented as placeholder for "mass equation at now". |
| **Relaxation / geometry constants** | Yes: _COULOMB_RELAX_SCALE (0.001), R_EQ_SCALE (1.4/1.2), _LAMBDA_COH_FACTOR (2.0) in horizon_network.py. |
| **Flavor scale in 8×8** | Yes: 0.1 (u), 0.12 (d) in subatomic.quark_state_matrix. |
| **Sub-nucleon scale factor** | Yes: 1e-4 in _constituent_horizons_m (subatomic.py); unused in nucleon E path. |
| **Epsilon (1e-30 etc.)** | Yes: many occurrences across subatomic, horizon_network, nuclear, entanglement, utils, etc., for denominator clipping. |
| **hqiv_scalings magic numbers** | Yes: 2.0e-43, 3.71e54, 1.38e-23, 1.616e-35*1.2e20 in hqiv_scalings.py. |

Implementing the changes in §3 would make proton and neutron masses **emergent from bound quarks** with no constants other than unit conversions and the chosen inputs (t_cmb, and optionally m_u, m_d until the mass equation is in place).
