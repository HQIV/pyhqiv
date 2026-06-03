# HQIV LEAN consistency and engine constants

This document verifies consistency between the formal development in **HQIV_LEAN** (`~/Repos/HQIV_LEAN`) and the **pyhqiv** engine in this repo, and enforces that **no physics constants enter the engine** other than unit conversions on an interaction layer. Quantities that appear in Lean but not here are **implementation earmarks**.

## 1. Single axiom and core relations

| Item | HQIV_LEAN | pyhqiv | Status |
|------|-----------|--------|--------|
| **E_tot = m c² + ħ c/Δx** | Referenced in comments (informational-energy axiom); not a formal definition | Docstrings and `effective_horizon_from_energy_mev` (Δx ∝ ħc/E) | ✓ Narrative alignment |
| **φ = 2/Θ_local** | `AuxiliaryField`: φ(m) = 2/T(m), T(m) = 1/(m+1) ⇒ φ(m) = 2(m+1) | `utils.phi_from_theta_local`: φ = 2c²/Θ (SI); lattice T = T_Pl/(m+1) | ✓ Same relation (natural vs SI) |
| **Lapse (fluid)** | — | `fluid.f_inertia`: f = a/(a + k φ), k = 1/6 | ✓ Paper particle action |
| **ADM lapse** | `HQVMetric`: N = 1 + Φ + φ t | `phase.adm_lapse_compression_factor`: 1 + γ(φ/c²)(δ̇θ′/c) | ✓ Same structure |
| **Phase-lift φ/6** | `PhaseLiftDelta`: phaseLiftCoeff m = φ(m)/6 | `F_INERTIA_K = 1/6` in fluid | ✓ |

## 2. Derived constants (no free parameters in Lean)

| Constant | Lean | pyhqiv | Status |
|----------|------|--------|--------|
| **α** | α = 3/5 (OctonionicLightCone) | `ALPHA = 0.60` | ✓ |
| **γ** | γ = 1 − α = 2/5 (HQVMetric) | `GAMMA = 0.40` | ✓ |
| **α_GUT** | 1/(cubeDirections × octonionImaginaryDim) = 1/42 | `coupling.ALPHA_GUT` = 1/42 | ✓ Implemented in `coupling` |
| **1/α_eff(φ)** | (1/α_GUT)(1 + c·α·log(φ+1)) (SM_GR_Unification, Schrodinger) | `coupling.one_over_alpha_eff`, `coupling.alpha_eff_shell(m)` | ✓ Implemented in `coupling` |
| **Curvature norm** | 6^7 × √3 (cubeDirections^7 × unitCubeHalfDiagonal) | `COMBINATORIAL_INVARIANT = 6**7 * sqrt(3)` | ✓ |
| **shell_shape(m)** | (1/(m+1))(1 + α log(m+1)) | `curvature_imprint_delta_E`: (1/(m+1))(1 + α ln(T_Pl/T)), T = T_Pl/(m+1) | ✓ Same formula |
| **G_eff(φ)** | φ^α (natural units) | Lattice / cosmology: same scaling | ✓ |
| **T(m)** | T_Pl/(m+1), T_Pl = 1 | T = T_Pl_GeV/(m+1) with T_Pl from constants | ✓ (unit conversion only) |

## 3. Unit conversions only (interaction layer)

These may appear in the codebase **only** for converting between natural and SI/experimental units, or for comparison with experiment. They must **not** be used inside the core engine to set coupling strengths or scales that Lean derives.

| Constant | Lean | pyhqiv | Intended use |
|----------|------|--------|--------------|
| **c_SI, ℏ_SI, G_SI** | Forces.lean | C_SI, HBAR_SI (G_SI not in pyhqiv) | Unit conversion only ✓ |
| **1/α_EM(M_Z) ≈ 127.9** | Witness in SM_GR_Unification | — | Comparison with `one_over_alpha_eff(φ_now)`; not an engine input |
| **ALPHA_EM_INV = 137.036** | — | constants.py | **Interaction layer only.** Engine should use `coupling.alpha_eff_shell(m)` or `coupling.one_over_alpha_eff(phi)`; reserve 137 for tests/comparison. |
| **T_CMB (K), m_proton/neutron (MeV)** | Witnesses / bounds | T_CMB_K, nucleon from first principles | Input to `evolve_to_cmb` / comparison; nucleon masses from `proton_energy_mev`, `neutron_energy_mev` (first principles). ✓ |

## 4. Constants audit: engine vs interaction layer

- **Allowed in engine:** Paper-derived structural numbers that Lean proves from the axiom (α, γ, α_GUT, 6^7√3, φ/6, lattice formulas). Unit conversion factors (c, ℏ, GeV↔K, etc.) when used purely for dimensions.
- **Interaction layer only:** CODATA/PDG values used to *compare* output to experiment (e.g. 1/α_EM, 127.9, 137.036). These must not fix coupling strengths inside the engine; the engine should use the derived 1/α_eff(φ).
- **Status:** Engine uses `coupling.alpha_eff_shell(m)` (and `alpha_eff_at_now()`) for Coulomb/EM coupling; `ALPHA_EM_INV` remains only in constants for tests and comparison.

## 5. Binding and composite masses (Lean vs pyhqiv)

| Lean (BoundStates, Schrodinger) | pyhqiv | Status |
|---------------------------------|--------|--------|
| E_bind = Σ_k w_k · alphaEffAtShell m k | Horizon network V_eff, merge_constituents; EM coupling from alpha_eff_shell(m) | ✓ |
| M_nucleon = M_constituent − E_bind_QCD | nucleon_energies_mev() from first principles | ✓ |
| M_nucleus = A·M_nucleon_avg − E_bind_nuclear | binding_energy_mev, E_free − E_nucleus | ✓ |
| E_bind_atomic magnitude μ Z² α_eff²/2 | opposing_fields_energy_mev, _equilibrium_* use alpha_eff_shell(M_TRANS) | ✓ |
| coulombStrengthShell m = alphaEffShell m | coupling.coulomb_strength_shell(m), alpha_eff_shell(m) | ✓ |

## 6. Lattice and curvature

| Lean | pyhqiv | Status |
|------|--------|--------|
| latticeSimplexCount m = (m+2)(m+1) | discrete_mode_count: 8*binom(m+2,2) = 4(m+2)(m+1) | ✓ (octonion factor 8) |
| cumLatticeSimplexCount hockey-stick | cumulative_mode_count | ✓ |
| x/θ from Planck distances (n+1)/(N+1) | x_over_theta_from_horizons(n, N) | ✓ |
| Ω_k(n; N) = Ω_k_true · (∫₀ⁿ/∫₀ᴺ) · (x/θ) | omega_k_at_horizon(n, N) | ✓ |
| Friedmann (3−γ)H² = 8π G_eff(φ)(ρ_m+ρ_r) | Cosmology background | ✓ |

## 7. Implementation earmarks (in Lean, not yet in pyhqiv)

1. ~~**α_eff(φ) in engine:**~~ **Done.** Engine uses `alpha_eff_shell(m)` and `one_over_alpha_eff(phi)`; `ALPHA_EM_INV` only in constants for interaction-layer comparison.
2. ~~**Binding coupling at shell m:**~~ **Done.** Nuclear and subatomic use `alpha_eff_shell(M_TRANS)` or optional shell; `opposing_fields_energy_mev(m_trans=...)`.
3. ~~**Coulomb strength API:**~~ **Done.** `coupling.coulomb_strength_shell(m)`, `alpha_eff_at_now()`.
4. **E_bind_from_network form:** Align binding formulas with Lean’s structural form E_bind = Σ w_k · bindingCouplingAtShell m k (weights from 8×8; coupling from φ(m)).
5. ~~**Lapse-corrected Schrödinger:**~~ **Done.** `phase.hqvm_lapse(Phi, phi, t)` returns N = 1+Φ+φt; use for iℏ ∂_t ψ = N H ψ in coordinate time.
6. ~~**Now condition:**~~ **Done.** Documented in `coupling`: "now" = shell M_TRANS, φ_natural = 2(M_TRANS+1); `alpha_eff_at_now()`.
7. **Equivalent β coefficients:** Lean derives equivalent one-loop β from O-Maxwell; no need in engine unless we want to compare running with SM; keep as optional comparison.

## 8. Files to touch for full alignment

- **constants.py:** Keep `ALPHA_EM_INV` with comment “Interaction layer / comparison only; engine uses coupling.one_over_alpha_eff / alpha_eff_shell.”
- **coupling.py (new):** `ALPHA_GUT`, `one_over_alpha_eff(phi, c)`, `alpha_eff_shell(m, c)`, `phi_of_shell_natural(m)`.
- **subatomic.py:** Uses `alpha_eff_shell(shell)` (default M_TRANS). ✓
- **nuclear.py:** Uses `alpha_eff_shell(M_TRANS)`; `opposing_fields_energy_mev(m_trans=...)`. ✓
- **phase.py:** `hqvm_lapse(Phi, phi, t)` = 1+Φ+φt. ✓
- **lattice.py:** Already consistent; optional: re-export `phi_of_shell_natural` from coupling for clarity.

---

**Summary:** Core formulas (φ = 2/Θ, α, γ, curvature shape, G_eff, lapse) are consistent. The main correction is to **derive EM coupling from 1/α_eff(φ)** in the engine and reserve **ALPHA_EM_INV** for the interaction layer. Earmarks above list what to add or refactor for full Lean alignment.
