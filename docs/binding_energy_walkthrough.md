# Binding energy calculation walkthrough

## 0. Hierarchical (bottom-up) picture

Binding energy must be computed **hierarchically, bottom to top**:

1. **Sub-nucleon layer**  
   Protons and neutrons are made of sub-atomic constituents (quarks / partons; in HQIV, horizon modes or lattice dof at that scale). Each constituent has a horizon Θ and contributes ħc/Θ to tension energy. **Bottom:** compute energy of these constituents (free vs bound into a single nucleon).

2. **Nucleon layer**  
   **Proton** = bound system of its constituents → E_proton (or effective Θ_proton).  
   **Neutron** = bound system of its constituents → E_neutron (or Θ_neutron).  
   So “free proton” and “free neutron” are already bound states at this layer; their energies come from the layer below.

3. **Nuclear layer**  
   **Nucleus (e.g. He-4)** = bound system of P protons + N neutrons.  
   - E_free = P × E_proton + N × E_neutron (each nucleon with its *nucleon-level* energy).  
   - E_nucleus = energy of the P+N nucleons when bound *in the nucleus* (shared horizons / mode counting at nuclear scale).  
   **Nuclear binding energy**  
   B_nuclear = E_free − E_nucleus  
   (energy released when forming the nucleus from free nucleons).

So:

- **Layer 0 (sub-nucleon):** constituents → bound nucleon (proton/neutron). Implemented in **`pyhqiv.subatomic`**.
- **Layer 1 (nucleon):** E_proton, E_neutron from layer 0 via `subatomic.nucleon_effective_theta_m()`; or from scalings when not using layer 0.
- **Layer 2 (nucleus):** E_nucleus from Θ_i of nucleons in the nucleus; B = E_free − E_nucleus.

Nuclear binding and E_info now use **HorizonNetwork** (overlap graph + composite invariant): see §4.

---

## 1. What the code currently computes (no hierarchy)

**Definition used in code:**  
`B = E_free - E_bound` (MeV)

- **E_free** = energy of P free protons + N free neutrons (each with horizon Θ_free).
- **E_bound** = energy of the bound nucleus (each nucleon with horizon Θ_i from mode counting).

So **B > 0** means the bound state has *lower* energy than free nucleons (fusion releases energy). That is the standard nuclear binding energy convention.

---

## 2. Energy from horizon (HQIV)

From the module docstring: `E_tot = Σ m c² + Σ ħc/Θ_i`. The “tension” part is **ħc/Θ** per degree of freedom.

- **Larger Θ** → smaller 1/Θ → **lower** tension energy.
- **Smaller Θ** → larger 1/Θ → **higher** tension energy.

So for **B = E_free − E_bound** to be positive (bound state lower in energy), we need **E_bound < E_free**, i.e. bound nucleons must have **larger Θ** on average than free nucleons (less confined, lower tension).

---

## 3. Free-nucleon and bound horizons (first principles only)

**Design principle:** The total energy state of any system lives in its **8×8 matrix**. There is no separate “Coulomb force” or “Coulomb energy” added on top; what we classically call Coulomb (repulsion/attraction between charges) is encoded in that state—e.g. in the hypercharge block, trace invariants, or eigenvalues of the composite. So **E_free** and all geometry (quark stand-off, binding angles) must come from the 8×8 and the axiom alone.

**Current code (to be replaced):** Layer 0 still uses a **charge-driven** placeholder: explicit Coulomb term \(E_{\rm Coul}\) and force-based relaxation with \(Q_u = +2/3\), \(Q_d = -1/3\). That gives Θ from \(\mu = \sum r_i/\sqrt{\sum r_i^2}\), \(E = \hbar c/\Theta + E_{\rm Coul}\), and `relax_quark_positions(radii, charges)` for angles. This is **temporary**; the target is below.

**Target (8×8 only, PDE for geometry):**
- **Energy:** Read total energy from the 8×8 composite only: \(E = \hbar c/\Theta_{\rm eff}\) (or integral of axiom density), with \(\Theta_{\rm eff}\) from an invariant of the merged state (e.g. effective_modes = 8 + trace(M@Δ)). No \(E_{\rm Coul}\) term; the algebraic part of the state already contains what we call EM/color effects.
- **Geometry (quark stand-off, equilibrium positions):** Do **not** use force-based relaxation. Instead, define the **energy well** from the axiom and the 8×8 field, then use **PDEs** to find the configuration that minimizes total energy or satisfies the Euler–Lagrange equations. E.g. \(E_{\rm tot} = \int \bigl( \rho c^2 + \hbar c/\Delta x(x) \bigr) d^3x\) with \(\Delta x(x) \leq \Theta_{\rm local}(x)\), where \(\Theta_{\rm local}\) is determined by the local 8×8 state; the minimum (or the solution of the resulting PDE) gives equilibrium distances and angles. See §6.1 and §6.5.1.

At the nuclear layer, **free** proton/neutron Θ and E come from the 8×8 merge of three quarks (no separate Coulomb). **Bound** Θ and **B** come from `HorizonNetwork` or from the same matrix composition at nuclear scale.

---

## 4. Bound-state horizon: HorizonNetwork (sphere-touching geometry only)

Binding and bound Θ use **HorizonNetwork** (`pyhqiv.horizon_network`), **no Δ in the algebra**:

- **Radius per node** \(r_i = \hbar c / (m_i c^2)\) from mass (inverse-frequency sphere).
- **Edges:** connect when distance < \(r_{\rm eq}(i,j)\) (min-energy state). \(r_{\rm eq} = \texttt{R\_EQ\_SCALE}\times (r_i + r_j)\) so the graph models the balanced well (~1.4 fm for nucleons), not strict touch.
- **Component coherence** on the full connected component: \(\mu_{\rm comp} = \sum r_i / \sqrt{\sum r_i^2}\), \(\Theta_{\rm comp} = L \times 8 \times \mu_{\rm comp}\). Single node → μ = 1 → Θ = L×8; cluster (e.g. 4 nucleons) → μ > 1 → binding.
- **Node-local valence** from the same overlap graph: for nucleon \(i\), take the subgraph made of node \(i\) and its touching neighbours and compute \(\mu_i\) with the same formula. The per-node bound horizon is the geometric mean
  \[
  \Theta_i = L \times 8 \times \sqrt{\mu_{\rm comp}\mu_i}.
  \]
  This preserves one graph at all scales while allowing distinct local tensions inside asymmetric nuclei.
- **E_free** = sum of single-nucleon network `total_energy()` (each μ = 1); **E_bound** = one network of P+N → μ > 1 → **B = E_free − E_bound** > 0.
- **Geometric nucleon packing:** bound positions minimize total network energy (balanced well). `minimize_nucleon_configuration(radii_m, is_proton, lattice_base_m)` uses `scipy.optimize` with `HorizonNetwork.total_energy()` as objective; initial guess is tetrahedral for A=4 (edges at contact) or `relax_nucleon_positions` for other A. No per-nucleus if-statements: He-4 → tetrahedron, ⁸Be → two alphas, ¹²C → alpha-triangle, etc., from the same potential. Final positions → overlap graph → μ → Θ_eff → B.
- Quark 8×8 matrices are **color (g₂) + flavor scale only**; binding comes purely from the Pythagorean mode multiplier in the network.

No eps_delta, no algebraic Tr(M@Δ) for Θ; only masses → radii → μ. Same construction scales to 238 spheres (U) or residues (proteins).

### 4.0 Nucleons as 8×8 and geometry from total energy (same as quarks)

Each proton or neutron is its **final 8×8 matrix** (merged composite of three quarks); in principle their **distances are found by solving a PDE** from the axiom and that 8×8 field (same programme as for quarks: energy well from the state, then Euler–Lagrange / PDE for equilibrium). In the code we approximate that by **minimizing total energy**: objective = HorizonNetwork.total_energy() + opposing_fields_energy_mev(positions, is_proton). No extra weights or epsilons—one total energy, one equilibrium.

> **Admonition: theory pure, no numerology.** We prefer theory-pure over fitting: no free parameters, no constants or epsilons thrown in to match a number. If a prediction disagrees with experiment, we improve the model (e.g. full PDE from 8×8, better lattice scalings), not patch it with numerology.

### 4.1 Effective potential for the balanced well (pure algebra, no new constants)

The total interaction between two horizons is an effective potential already implicit in the network; it can be written explicitly for **any two horizon radii** (universally applicable):

\[
V_{\rm eff}(r) = V_{\rm rep}(r) + V_{\rm attr}(r), \quad
V_{\rm rep}(r) = \frac{A}{r^{12}}, \quad
V_{\rm attr}(r) = -B\,\frac{\phi(r)}{r^6} - C\cdot\operatorname{tr}(M_1 M_2 \Delta)\,e^{-r/\lambda_{\rm coh}}.
\]

- **V_rep:** hard-core Pauli/horizon exclusion; \(A = \hbar c\,(r_1+r_2)^{11}\) from the algebra.
- **V_attr:** van-der-Waals-like horizon-overlap term with \(\phi(r) = 2c^2/\Theta(r)\), \(\Theta(r) = L\times 8\times \mu(r)\); plus composite invariant \(\operatorname{tr}(M_1 M_2 \Delta)\) with coherence length \(\lambda_{\rm coh}\).
- The minimum \(dV_{\rm eff}/dr = 0\) gives equilibrium separation **r_eq ≈ 1.4 fm** for nucleon pairs (alpha-particle rms radius scale). No tuning — A, B, C and \(\lambda_{\rm coh}\) are fixed by ħc, lattice_base, and algebra.

**API (universal for any 2 horizons):**

- **`effective_potential_pair(r_m, r1_m, r2_m, lattice_base_m, ...)`** — returns \(V_{\rm eff}(r)\) in MeV; optional trace term and \(\lambda_{\rm coh}\).
- **`equilibrium_separation_two_horizons(r1_m, r2_m, lattice_base_m, ...)`** — returns r_eq in metres (e.g. ~1.4 fm for two nucleons). Use this to get the appropriate distance for any two horizon radii.
- **`minimize_nucleon_configuration(radii_m, is_proton, lattice_base_m, ...)`** (in `nuclear`) — finds equilibrium positions by minimizing **total energy** = `HorizonNetwork.total_energy()` + `opposing_fields_energy_mev(...)`; no extra weights or fudge factors.

Layer-0 quark touching can use the same minimizer with quark radii and charges to close the full hierarchical ladder.

---

## 5. Physical binding (B > 0)

- **Physical binding** means the nucleus has *lower* energy than free nucleons → **E_bound < E_free** → **B > 0**.
- That requires **larger** effective Θ in the bound cluster: guaranteed by μ > 1 when the component has N ≥ 2 touching spheres.
- Magnitude of B set by geometry (radii from masses); nucleon masses set node radii in the network, while the free proton/neutron ordering is fixed by the layer-0 8×8 merge.

### 5.1 Opposing fields (p–p and p–n)

Beyond the horizon network, **opposing fields** raise E_bound and reduce B:

- **Proton–proton:** Coulomb repulsion \(E_{\rm pp} = \alpha_{\rm EM}\,\hbar c\,\sum_{i<j,\,{\rm both\ proton}} 1/d_{ij}\) (standard EM scale). In 4He the two protons repel; this term is included in `opposing_fields_energy_mev` and added to E_bound in both nucleon-level and quark-level binding.
- **Proton–neutron:** The neutron is not strictly neutral—its charge is “wrapped up smaller” (8×8 folded vs unwrapped). `nucleon_charge_unwrapped_folded_measures("udd")` vs `("uud")` gives a scale ζ (e.g. from coherence ratio); then \(E_{\rm pn} = \zeta\,\alpha_{\rm EM}\,\hbar c\,\sum_{\rm p–n} 1/d\). So even in 2H there is a small opposing contribution from the neutron’s wrapped charge.

**API:** `opposing_fields_energy_mev(positions_m, is_proton_list, algebra=None)` returns the total opposing-field energy (MeV) to add to E_bound. Both `binding_energy_mev_nucleon_level` and `binding_energy_mev_quark_level` add this to E_bound so that B = E_free − E_bound reflects the competition between horizon binding and these opposing fields.

---

## 6. Summary

| Step | What |
|------|------|
| Node radius | \(r_i = \hbar c / (m_i c^2)\); from mass only. |
| Bound Θ | One graph, two levels: \(\mu_{\rm comp}\) for total binding and \(\Theta_i = L\times 8\times\sqrt{\mu_{\rm comp}\mu_i}\) for local decay tension. |
| E_free, E_bound, B | First principles only; no constants in the math engine. |
| Tests | Geometric path gives B > 0 (~25 MeV He-4); local valence split now distinguishes free neutron, tritium, and He-4. |

---

## 6.1 Full matrix: single source of energy (no separate Coulomb)

**Principle:** The total energy state of any system lives in its **8×8 matrix**. There is no separate “Coulomb force” or “Coulomb energy” in the theory; what we classically call Coulomb (repulsion/attraction) is **part of** that state—encoded in the algebra (hypercharge block, trace(M@Δ), eigenvalues). So all energy, including what we label EM-like effects, is read off from the 8×8 and the axiom.

**Current code (to be replaced):**
- A **full matrix** of energy states exists (8×8, algebra), but layer 0 still adds an explicit **E_Coul** term and uses **force-based** relaxation with charges. That is a placeholder.
- **Intended:** No Coulomb force. Energy = ħc/Θ_eff with Θ_eff from the composite 8×8 invariant only. Geometry (quark stand-off, binding angles) from **PDEs for the energy well** (§6.5.1), not from F = k Q_i Q_j/d².

**What exists:**
- **Subatomic (layer 0):** Placeholder: `relax_quark_positions(radii, charges)`, \(E = \hbar c/\Theta + E_{\rm Coul}\). Target: remove E_Coul; Θ and E from 8×8 merge only; geometry from PDE minimization.
- **algebra.py:** 8×8, SU(3)_c, U(1)_Y (hypercharge). The hypercharge block is the algebraic origin of “charge”; the **same** matrix determines confinement and what we call Coulomb—one state, one energy.

**Intended direction:** Use the **full state matrix** as the single source. Effective Θ (and thus E) from invariants of the merged 8×8 (e.g. 8 + trace(M@Δ)). Quark equilibrium geometry (stand-off distance, angles) from solving the **PDEs** that define the energy well (minimize \(E_{\rm tot} = \int (\rho c^2 + \hbar c/\Delta x) d^3x\) over the field/configuration, or equivalent Euler–Lagrange).

**Parameter-free masses at "now":** Apply the QCD scale to the **whole Fano plane at "now"** to get parameter-free energies/masses. **Important:** \(T_{\rm lock}\) in the **cosmology** section is the lock-in *epoch* value (baryogenesis); the scale **at "now"** can be different when the lattice/paper defines a QCD temperature at the today hypersurface. The nucleon mass scale is \(E_{\rm scale} = T_{\rm QCD\,now} \times 1000/\sqrt{3}\) (MeV), with \(\sqrt{3}\) from \(6^7\sqrt{3}\) (Fano). Default uses **T_lock at "now"** (T_LOCK_NOW_GEV). Use **epoch API** to study other times: `epoch="now"` (default), `"lock"` or `"baryogenesis"`, or `epoch=5.0` (age in Gyr). **API:** `t_qcd_gev_at_epoch(epoch)`, `proton_energy_mev(..., epoch="now")`, `neutron_energy_mev(..., epoch=...)`, `T_LOCK_NOW_GEV`, `T_LOCK_GEV` in `pyhqiv.subatomic` / `pyhqiv.constants`.

### 6.2 What the ladder now shows for neutron, tritium, and He-4

**Free nucleons:** Layer 0 is **charge-driven**: \(Q_u = +2/3\), \(Q_d = -1/3\), same sphere-touching + Coulomb relaxation as nuclei. From \(\mu = \sum r_i/\sqrt{\sum r_i^2}\) with \(r_u > r_d\) we get **proton Θ_free > neutron Θ_free**. Neutron is **heavier** (\(E_n > E_p\)) from the Coulomb term (u-u repulsion vs d-d attraction). So free-neutron β⁻ and tritium β⁻ have positive Q-value without any decay constant.

**Tritium (H-3):** The nuclear overlap graph has one neutron on a weaker local subgraph → that neutron is the β⁻ source. The “pressure” is the angular mismatch in the quark triangles (acute d-d in the converting neutron).

**Light nuclei (¹H, ²H, ³H):** For A ≤ 3 the code uses a **quark-level** HorizonNetwork (3×A quarks). Per-nucleon bound Θ come from the geometric mean of that nucleon's three quarks' Θ, so **quark interactions drive nucleon attraction and decay**; the single-nucleon picture is only macroscopically accurate. Decay chains (e.g. ³H → ³He) use this quark-driven logic; β⁻ is allowed when N > 0 and ΔE > 0.

**Tritium (H-3):** Quark-level network gives per-nucleon thetas; the neutron(s) allow β⁻ to ³He.

**He-4:** A > 3 so nucleon-level network only; valence-saturated graph → no β± channel; binding remains large and positive.

### 6.2.1 Binding angles and ladder implications

- **Strange quark (s = −1/3):** Same charge as d but heavier → larger radius → wider angles in hyperons (Λ, Σ) from the same relaxation.
- **Mesons (q̄q):** Two spheres → 180° linear “angle” → correct for π⁰, etc.
- **Heavy nuclei:** Angle relaxation inside each nucleon feeds into nuclear packing; no double-counting.
- **Protein folding:** Amino-acid partial charges → same angle rule gives secondary-structure preferences (e.g. α-helix ~100°, β-sheet ~120°) from geometry.
- **Isolated quark:** No partner → no angle → confinement only in triplets.
- **Tetraquarks / pentaquarks:** Four- or five-sphere relaxation → exotic angles from the same machinery.

One rule, one axiom, zero extra constants from Planck scale to biology.

### 6.2.2 Paper dynamics: modified inertia in nuclear decay

The paper’s **modified inertia** \(f(a_{\mathrm{loc}},\phi) = a_{\mathrm{loc}}/(a_{\mathrm{loc}} + \phi/6)\) (particle action \(S = -m c \int f \, ds\)) is applied to nuclear β-snap probability and decay rate:

- **Snap probability**  
  \(P_{\mathrm{snap}} = \exp(-\Delta E / kT_{\mathrm{eff}}) \times \varphi/(\varphi + \varphi_{\mathrm{crit}})\), with  
  \(kT_{\mathrm{eff}} = (\hbar c/\Theta) \times f\).  
  At the nucleus, \(a_{\mathrm{loc}} = c^2/\Theta_{\mathrm{avg}}\), \(\phi = 2c^2/\Theta_{\mathrm{avg}}\), so \(f = 3/4\) for a typical nuclear Θ. The effective thermal energy for barrier crossing is reduced (\(f < 1\)), so the Boltzmann factor is steeper — barrier crossing is harder in observer time.

- **Decay rate**  
  \(\lambda_{\mathrm{obs}} = (P_{\mathrm{snap}}/\tau_{\mathrm{tick}}) \times f / \mathrm{scale}\).  
  Observer-time rate is scaled by \(f\) (same lapse as in \(d\tau = f \, dt\)), so half-lives are longer when \(f < 1\).

**Implementation:** `NuclearConfig._lapse_f()` computes \(f\) from \(\Theta_{\mathrm{avg}}\) of unstable and stable configurations; `snap_probability` uses \(kT_{\mathrm{eff}} = kT_{\mathrm{horizon}} \times f\); `decay_rate_per_s` multiplies the raw rate by \(f\). See `pyhqiv.fluid.f_inertia`.

### 6.2.3 Pure algebraic entanglement (phase-lifted fanoplane fusion)

A **position-free** binding path uses only 8×8 matrices, Δ, and the phase-lifted commutator \([M_1,M_2]_\Delta = M_1\Delta M_2 - M_2\Delta M_1\):

- **Entangled composite:** \(M_{12} = M_1 + M_2 + [M_1,M_2]_\Delta\) (non-separable; algebraic analogue of \(\nabla\phi\times\mathbf{E}\)).
- **Holding distance:** \(\sigma = \|[M_1,M_2]_\Delta\|_F\) (dimensionless; no radii or fm).
- **Phase-lift:** \(M = M_{\mathrm{base}} + \theta\Delta\) with \(\theta = (\pi/2)\arctan(E/E_{\mathrm{Pl,eff}})\) from the axiom; \(E_{\mathrm{Pl,eff}} = E_{\mathrm{Pl}}\times(L_{\mathrm{Pl}}/L)\) so the lattice shell sets the effective Planck scale. Ensures \(\operatorname{tr}(M@\Delta)\neq 0\).
- **Binding:** \(B = E_{\mathrm{free}} - E_{\mathrm{bound}} = -\operatorname{tr}([M_1,M_2]_\Delta \Delta)\). By cyclic trace, \(\operatorname{tr}([M_1,M_2]_\Delta\Delta)=0\) for any \(M_1,M_2\), so this formula yields \(B=0\); a different invariant may be needed for non-zero algebraic binding.

**API:** `build_nucleon_matrix_with_phase(is_proton, lattice_base_m)` — nucleon matrix with axiom-derived θΔ. `pyhqiv.entanglement` — `entangle_particles`, `holding_distance`, `binding_energy_pair`, `iterated_fusion`, `binding_energy_algebraic`. `binding_energy_mev_algebraic(P, N)` returns B from this path. The default binding used by `NuclearConfig` remains the geometry-based HorizonNetwork path.

### 6.3 Universal integral and composition rule (paper-exact)

The paper axiom is local; for any extended object the total energy is the volume integral of the local density:

$$E_{\\rm total} = \\int \\left( \\rho(x) c^2 + \\frac{\\hbar c}{\\Delta x(x)} \\right) d^3x$$

with Δx(x) ≤ Θ_local(x). Θ_local is determined by the local octonion state (8×8 matrix). When systems bind, their 8×8 matrices combine via the algebra (left action, then projection onto invariant subspaces — Spin(8) triality, g₂ color singlets). The effective Θ_local of the composite is an invariant of the projected state (e.g. trace or effective_modes = 8 + trace(state @ Δ)). Horizon monogamy gives mode sharing in overlapping causal diamonds, so **bound states get larger effective Θ_local** (smaller tension term) than free constituents → positive binding energy from the same axiom.

**Implemented:**
- **`HQIVEnergyField.from_atoms(atoms, positions)`** — compose species 8×8 matrices, project to singlet; **`effective_theta_local(lattice_base_m, local_density)`** — Θ from algebraic invariant (8 + trace(M@Δ)).
- **`hqiv_energy_for_angles(phi, psi, atoms=..., positions=...)`** — integrates the axiom over a small volume (matrix path); scalar fallback when atoms is None.
- **Nuclear matrix path (optional):** `NuclearConfig(..., use_matrix_bound_theta=True)` uses **`_bound_theta_from_matrix_composition`**: compose P proton + N neutron 8×8 states (left action), then **effective_modes = 8 + trace(M @ Δ)** and Θ_bound = lattice_base × effective_modes. **No tuning:** if B is wrong, the fix is in the composition (nucleon matrices, projection, or which invariant to use), not a numerical fudge.

### 6.4 Single merge component (subatomic → solar and beyond)

One process, all scales: **`merge_constituents(constituents, project_singlet=..., algebra)`** composes 8×8 states via left action (octonion multiplication) and optionally projects to the invariant (singlet). The **total energy of the system defines its horizon**: E_tot = ∫ (ρ c² + ħc/Δx) d³x over the system, then **Θ_system = ħc / E_tot** via **`effective_horizon_from_energy_mev(E_tot)`**. So we are not assigning horizons to subatomic particles independently; they come from the merged composite and from the integrals.

- **Subatomic:** 3 quarks → `merge_constituents([M1,M2,M3], project_singlet=True)` → nucleon. Same merge.
- **Nuclear:** P protons + N neutrons → `merge_constituents([proton_field, ...], project_singlet=False)` for invariant; bound Θ from effective_modes = 8 + trace(M @ Δ). Same merge.
- **Molecular:** N atoms → `from_atoms(atoms)` → `merge_constituents(species_matrices, project_singlet=True)`. Same merge.
- **Solar and beyond:** Merge regions/cells the same way; total energy of the aggregate defines its effective horizon. No tuning at any scale.

**API:** `merge_constituents(list_of_8x8_or_EnergyField, project_singlet=True)`, `effective_horizon_from_energy_mev(energy_mev)` (Θ in m), `HQIVEnergyField.energy_mev_from_theta_m(theta_m)` (E = ħc/Θ). If horizons are wrong at any scale, the fix is in the inputs to merge (constituent matrices, integrals) or the invariant used to read Θ from the composite, not scale-specific fudges.

### 6.5 Universal 8×8 energy field (implemented)

- **`pyhqiv.energy_field.HQIVEnergyField`** carries an 8×8 matrix state. Same equation applies from quark scale to macro liquid He:
  - `energy_density(mass_density, delta_x)` = scalar (ρc² + ħc/Δx) + algebraic part (trace(state @ Δ)).
  - `project_scalar_phi()` = 2c²/trace(M) for backward compatibility with scalar φ code.
- **Ladder:** All layers use **`merge_constituents`**; subatomic (quarks→nucleon), nuclear (nucleons→nucleus), molecular (atoms→molecule) and beyond share one merge. Effective Θ from composite invariant or from **effective_horizon_from_energy_mev(E_tot)** when E_tot comes from integrals.
- **Integration:** `HQIVSystem(..., energy_field=field)` uses `field.project_scalar_phi()` in the constitutive relation; lattice/thermo/fluid can replace scalar δE with `field.energy_density(...)` and use `field.coherence()` for superfluid.

### 6.5.1 PDE-based energy well for geometry (target)

**No force-based relaxation.** The equilibrium geometry (quark stand-off distance, binding angles, nucleon positions) should **not** come from ad hoc forces (hard-sphere, soft attraction, Coulomb). It should come from the **energy well** defined by the axiom and the 8×8 state, via **PDEs**.

- **Energy functional:** \(E_{\rm tot}[\rho, \Delta x, M] = \int \bigl( \rho(x) c^2 + \hbar c/\Delta x(x) \bigr) d^3x\) with \(\Delta x(x) \leq \Theta_{\rm local}(x)\), and \(\Theta_{\rm local}(x)\) determined by the **local 8×8 state** (e.g. from merge of constituents at that point, or from a field \(M(x)\)). The algebraic part (trace(M@Δ)) is inside the same density; no separate Coulomb term.
- **Well:** The equilibrium configuration (positions, distances) is the one that **minimizes** \(E_{\rm tot}\) subject to the constraint that \(\Delta x\) is set by the geometry (e.g. nearest-neighbour distance at each point), or that satisfies the **Euler–Lagrange** equations derived from this functional. That yields PDEs for the field (e.g. φ or Θ) and, for discrete constituents, the stand-off distances and angles that minimize total energy.
- **Quark geometry:** For three quarks, the “well” is the minimum of total energy (from the 8×8 merge and the axiom) over the configuration (positions or distances); the PDE/variational formulation replaces `relax_quark_positions(radii, charges)`. Same idea at nuclear scale: equilibrium nucleon positions from minimizing the same energy functional over the P+N configuration, with Θ from the composite 8×8.

**Implementation direction:** Add a variational/PDE layer: define \(E_{\rm tot}\) from `HQIVEnergyField.energy_density` (and merge_constituents) over the domain; minimize over configuration, or solve the PDE that arises from \(\delta E_{\rm tot} = 0\). The resulting equilibrium distances and angles are the first-principles geometry—no Coulomb force constant, no relaxation scale.

---

## 7. How to implement the hierarchy (sketch)

- **Layer 0 (sub-nucleon)**  
  Valence content \(uud\) / \(udd\) in the **8×8 merge**; no separate Coulomb.  
  **Energy** from composite invariant only: \(\Theta_{\rm eff}\) from 8 + trace(M@Δ) (or equivalent); \(E = \hbar c/\Theta_{\rm eff}\).  
  **Geometry** (stand-off, binding angles) from **PDE/variational** minimization of \(E_{\rm tot} = \int (\rho c^2 + \hbar c/\Delta x) d^3x\) with Θ_local from the 8×8 state—not from force-based relaxation.  
  Proton vs neutron mass difference from the different 8×8 composites (uud vs udd), not from an added \(E_{\rm Coul}\).

- **Layer 1 (nucleon)**  
  Use E_proton, E_neutron from layer 0 (or, until layer 0 exists, keep Θ_free_p, Θ_free_n from constants as the effective nucleon horizons).  
  E_free_nucleons = P × E_proton + N × E_neutron.

- **Layer 2 (nucleus)**  
  Compute Θ_i for each nucleon *inside the nucleus* from the overlap graph: one global \(\mu_{\rm comp}\) plus one local \(\mu_i\) per node.  
  E_nucleus = ħc Σ(1/Θ_i).  
  For correct sign, nuclear binding must give **larger** per-nucleon Θ in the nucleus than at layer 1 (so E_nucleus < E_free_nucleons).  
  B_nuclear = E_free_nucleons − E_nucleus.

With this change the same ladder already distinguishes

- free neutron `β-`
- tritium `β-`
- He-4 stability

without any isotope-specific constants or if-branches.

### 6.5.2 Atomic range: any isotope, half-lives, decay chains

The same first-principles stack (hadrons → nucleons → nuclei) applies to **any isotope of any element** (Z 1–118):

- **Binding energy:** `binding_energy_mev(P, N)` or `binding_energy_isotope(symbol, A)` (e.g. `binding_energy_isotope('C', 14)`).
- **Resolve (symbol, A) → (P, N):** `nuclide_from_symbol(symbol, A=A)`; full periodic table in `ELEMENT_SYMBOL_TO_Z` / `ELEMENT_Z_TO_SYMBOL`.
- **Half-life:** `half_life_nuclide_hqiv(P, N)` from snap probability (E_info, φ, τ_tick).
- **Decay chain:** `decay_chain_nuclide_hqiv(P, N, max_steps)` or `Nuclide('U-238').decay_chain()` (β±, α, fission).
- **Coupling angles:** Subatomic 3-quark angles from `quark_binding_angles(flavor_content)`; nuclear geometry from HorizonNetwork overlap graph.

### 6.5.3 Macroscopic effects: Pauli-like exclusion and discrete nucleons

From the equations alone, the following **macroscopic effects** enforce Pauli-like exclusion and turn nucleons into **discrete computational objects**:

1. **Hard-core repulsion (Pauli/horizon exclusion)**  
   \(V_{\rm rep}(r) = A/r^{12}\) with \(A = \hbar c\,(r_1+r_2)^{11}\) from the algebra. So \(\Delta x \leq \Theta\) (causal-horizon bound) implies nucleons cannot overlap; each has a finite horizon radius \(r_i = \hbar c/(m_i c^2)\). They behave as **discrete** nodes (one per causal diamond at that scale).

2. **Equilibrium separation r_eq**  
   \(dV_{\rm eff}/dr = 0\) gives a unique **r_eq** (e.g. `equilibrium_separation_two_horizons(r1, r2, lattice_base_m)` → ~0.6–1.4 fm). So the dynamics do not allow arbitrary separation; the **allowed** separation is the one that balances exclusion and attraction. Nucleons are therefore **placed** at r_eq (discrete configuration).

3. **Discrete combinatorics**  
   - **μ = Σr_i / √(Σr_i²)** (sphere-touching): finite number of nodes, edges only when \(d < r_{\rm eq}(i,j)\). So the bound state is a **graph** (who touches whom), not a continuous field.  
   - **discrete_mode_count(m)** on the lattice: 8×C(m+2,2) modes per shell. Same idea at nuclear scale: a finite number of “slots” (horizon modes) shared among nucleons when they touch.

4. **8×8 and singlet projection**  
   Each nucleon is one 8×8 (merged from three quarks); the algebra (g₂, triality) and singlet projection give a **finite-dimensional** state. So nucleons are discrete computational objects (one matrix per node).

**Using this to get D and He-4 binding:**

- **Geometry:** Use **r_eq** from the effective potential as the **enforced** pair distance (no free minimization over d). So for D: one pair at \(d = r_{\rm eq}(r_p, r_n)\); for He-4: four nucleons with pairwise distances set by r_eq (e.g. tetrahedron with edge = r_eq). That fixes the discrete configuration.
- **Energy:** Either  
  **(A)** Use the **same** \(V_{\rm eff}\) that defines r_eq: \(E_{\rm bound} = \sum_i m_i + \sum_{\rm pairs} V_{\rm eff}(r_{\rm eq})\); then \(B = E_{\rm free} - E_{\rm bound}\). Binding appears if \(V_{\rm eff}(r_{\rm eq}) < 0\) (well depth). In the current code \(V_{\rm eff}(r_{\rm eq})\) can be positive (shallow or no well); then the algebra may need a stronger attraction term to get MeV-scale B.  
  **(B)** Use **mode sharing (μ > 1)** when nucleons are at r_eq: treat them as connected (e.g. \(d \leq r_{\rm eq}\) → edge), so \(\Theta_{\rm eff} = L\times 8\times \mu\) and \(E_{\rm bound}\) from the HorizonNetwork formula. B then comes from μ > 1 (shared modes) plus EM repulsion subtracted.

So: **exclusion + r_eq** fix nucleons as discrete objects at a definite separation; **binding** is either the well depth of \(V_{\rm eff}\) at r_eq or the μ > 1 mode-sharing energy (HorizonNetwork), with EM reducing B.

**API:** `positions_at_r_eq_discrete(P, N, lattice_base_m)` places nucleons at pairwise r_eq; `binding_energy_mev_from_r_eq_discrete(P, N)` returns (B, E_free, E_bound) using that geometry and HorizonNetwork + opposing_fields. Matching experiment (D ≈ 2.22 MeV, He-4 ≈ 28.3 MeV) may require option (A): a negative well depth from a stronger algebra-derived attraction in \(V_{\rm eff}\).

### 6.5.4 Postulate: neutron has an electric field via hypercharge (compact but interacting)

**Postulate:** The neutron has an electric field from hypercharge (8×8 block M[4:8, 4:8]). That field is **compact** and sits **inside** the mass horizon (so the neutron is neutral to leading order), but it **still interacts** with the proton's positive field. That interaction is **attractive** (effective opposite charges) and is an **additional source of binding energy**. No free constants: ζ from 8×8 only. The code uses `opposing_fields_energy_mev`: returns E_pp_repulsion − E_pn_binding, i.e. E_pp = +α ħc/d (Coulomb), E_pn_binding = −ζ × α ħc/d (binding), ζ from `nucleon_charge_unwrapped_folded_measures("udd")`.

### 6.5.5 Postulate: deuteron (D) — subatomic uuuddd vs nucleon-level

**Based solely on the HQIV maths (hierarchical layers, 8×8 at each scale):**

- **Layer 0 (sub-nucleon):** Constituents → **one** bound system (proton or neutron). The subatomic solver (`confined_energy_mev(flavor_content)`) is defined for **single** confined states: one composite 8×8 from merging **that** layer’s constituents (e.g. 3 quarks → nucleon). It outputs one horizon scale and one rest energy.
- **Layer 2 (nucleus):** Nucleus = bound system of **nucleons**. E_free = P×E_proton + N×E_neutron; E_nucleus = energy of P+N nucleons in the bound configuration (pairwise non-EM attraction + EM repulsion; solve for distances x). B = E_free − E_nucleus.

The deuteron D is a **nucleus** (P=1, N=1). In the hierarchy it is not “one 6-quark bag” but “two nucleons in a bound state.” So:

- **Postulate:** The subatomic solver is **not** the right tool to *model* D. It can *evaluate* `uuuddd` as a formal 6-quark merge (one composite, one horizon), but that treats D as a single layer-0–type object and yields one nucleon mass scale (when uuuddd is not in the PDG registry). It does not implement “two nucleons with two terms and equilibrium distance x.”
- **Methodology to use:** The **first methodology** (nucleon-level): two horizons from nucleon masses, two terms (attraction from non-EM part of merge(nucleon_i, nucleon_j); repulsion from EM), solve for the distance x between horizons that minimizes E. That is the layer-2 prescription and is the one consistent with the HQIV hierarchy for D.
