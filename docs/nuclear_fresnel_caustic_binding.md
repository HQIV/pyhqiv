# Nuclear binding: intended Fresnel–caustic algorithm (spec)

This document is the **target** physics narrative for **algorithmic** nuclear binding in HQIV. It replaces the interim story “sum PDG nucleon masses for E_free + HorizonNetwork gap” as the **design authority** once the caustic pipeline is wired as the default.

**Lean alignment (packaging, no new axioms in-file):**

- `HQIV_LEAN/Hqiv/Physics/NuclearAndAtomicSpectra.lean` — `single_nucleon_caustic`, `barbell_ring_caustic` (mode count ↔ caustic strength), `V_nuclear`, dipole `V_mag`, β width / half-life scaffolding.
- `HQIV_LEAN/Hqiv/Physics/BoundStates.lean` — hierarchical `E_bind_nuclear_from_network` (structural form for the 8×8 network layer).

**Prior Python sketch (not imported by default):** `bak/pyhqiv/nuclear_binding_first_principles.py` — Fresnel reflectance, catacaustic amplitude, valley search bands, magnetic scale from 8×8 proton/neutron measures, tetrahedral/barbell coordinates.

---

## 1. Single nucleon: Fresnel caustic and valley

Each nucleon is treated as a **compact horizon** (radius scale set by its mass and 8×8 state, so **proton and neutron have slightly different effective diameters** and **different magnetic moments** from the octonionic / triality sector).

- **Quantum modes** are **attenuated and refracted** at the nucleon boundary in the same **Fresnel** sense as light bending past a sphere: reflectance/transmission at the horizon interface sets the strength of **deflection into a caustic**.
- The **caustic valley** (line of stationary phase / enhanced mode density) lies **outside** the hard horizon, at a distance of order **twice the nucleon radius** \( \sim 2r \) from the centre (the exact factor comes from the caustic geometry in the Fresnel construction, not from a fitted SEMF parameter).
- The **caustic amplitude** as a function of separation \(r\) from the centre uses the Fresnel boundary condition plus a **catacaustic** geometric factor (e.g. \(\sin^3\theta\) weighting) and **screening** so the field never blows up; see the `_caustic_amplitude` structure in `bak/pyhqiv/nuclear_binding_first_principles.py`.

**Binding contribution:** overlap of another nucleon’s degrees of freedom with **this** valley lowers informational tension (shared modes) → **negative** potential energy relative to isolated nucleons.

---

## 2. Proton–proton alignment

**Protons** carry the dominant dipole / hypercharge structure in the 8×8 block; where geometry allows, they **align end-to-end** (moment axes along the internuclear axis) so dipole–dipole and valley overlap are **consistent with spin statistics** (fermionic antisymmetry in the overall wavefunction is enforced at the composite level, not by ad hoc potentials).

---

## 3. Deuterium **²H** (D): \(Z=1\), \(N=1\))

- Configuration: **one proton, one neutron**.
- **Each nucleon sits in the other’s Fresnel caustic valley** (~\(2r\)) — the **barbell** limit of two touching horizons.
- Binding energy: **algorithmic** from **valley–valley overlap integrals** (mode counting + curvature imprint on the discrete ladder), **not** from tabulated \(B_D\).
- No charged second proton → no Coulomb barrier between two “ends” of the barbell in the same way as **pp**; the **p–n** channel is the stable pair.

---

## 4. Tritium **³H** (\(Z=1\), \(N=2\))

- Ground shape: **barbell** (p–n or n–n–p topology in the same valley language) with a **toroidal ring caustic** enhanced by **`new_modes` at one shell higher** (Lean: `barbell_ring_caustic`).
- The **third nucleon (second neutron)** sits preferentially in the **ring caustic** encircling the barbell, not in a random spherical shell.
- **Screening** (auxiliary-field / \(\Delta\phi\)) and **spin statistics** open a **β⁻ channel** to **³He** (\(Z=2\), \(N=1\)).
- **Q-value** (energy of the initial state above the ³He core + leptons) **correlates with half-life** via the same informational snap / width picture as in `NuclearAndAtomicSpectra` (β rate ∝ phase space × \(|\mathcal{M}|^2\), HQIV width from horizon tension).

---

## 5. Hydrogen-4 **⁴H** (\(Z=1\), \(N=3\)) — unbound / narrow

- **Allowed as a compound** in the sense of **quantum numbers and intermediate structure**, but **binding energy and spin statistics** imply **immediate neutron emission** as the **dominant** channel (single-particle width large).
- **Uncertainty width** ↔ **very short half-life** for the dineutron / three-body breakup.
- **β⁻ decay** remains **kinematically possible** but **strongly suppressed relative to neutron ejection** (branching ratio ≪ 1): the **same** width formalism must predict **two channels** with **vastly different** partial widths.

---

## 6. Helium-4 **⁴He** (\(Z=2\), \(N=2\))

- **Four nucleons** in the **mutual caustic network**: each sits in the others’ valleys (tetrahedral / compact **four-body** minimum in 3D).
- **High symmetry** → large **mode sharing** and **deep binding** relative to D and ³H.
- **No β±** at the nuclear level for this closed \(Z=N\) light α-like system in the same valence picture (consistent with observed stability).

---

## 7. What the current `pyhqiv.nuclear` module does instead (interim)

Until the caustic pipeline is the default:

- **`binding_energy_mev` / `binding_energy_mev_functional`:** minimizer **disabled**; returns **B = 0** while **E_free** uses **fixed nucleon masses** (witness/constant anchors) — **not** the intended algorithm.
- **`NuclearConfig` / `_binding_energy_via_network`:** **HorizonNetwork** overlap graph + **PDG-scale** masses on nodes — useful **interim** binding gap, **not** the Fresnel valley geometry described above.
- **Algebraic / quark-expanded paths:** partial views of the 8×8 stack; they do **not** yet implement the **2r valley + barbell + ring** placement rules.

---

## 8. Implementation checklist (for code, not done here)

1. **Nucleon radii** \(r_p \neq r_n\) and **dipole vectors** from 8×8 (already sketched via `nucleon_charge_unwrapped_folded_measures` / magnetic scale in bak).
2. **Valley centre** at **~2r** from each centre; **pairwise overlap** of mode density → **binding density** integrated to **B**.
3. **D:** axis through two centres, separation minimizing **total** caustic + dipole + screening (`V_nuclear`-style sum in Lean).
4. **³H:** impose **ring** degree of freedom for the third nucleon; compute **β⁻ Q** to ³He → feed **decay width**.
5. **⁴H:** compute **neutron emission** partial width vs **β** partial width.
6. **⁴He:** **tetrahedral** minimum of four **mutual** valley overlaps.
7. **E_free:** eventually from **layer-0→1** nucleon energies (constituent → nucleon), **not** from bare PDG paste-in for the nuclear layer; until then document any witness use as **temporary anchor**.

---

## 9. Relation to tests

- Tests that compare **`binding_energy_mev`** to experimental **D / ⁴He** bands are **invalid** while §7 applies (B ≡ 0 or network interim).
- Target tests: **internal consistency** (valley depth, overlap monotonicity, ³H β Q vs width ordering, ⁴H branching **n ≫ β**), plus **eventual** comparison to data once the caustic integral matches MeV scale without SEMF fitting.
