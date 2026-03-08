# Qualified statements: what we can state from first principles

This document states what **pyhqiv** can assert from the HQIV framework (gold-standard functional method, decay snap formula, horizon coupling). Each statement has an **API** and an **experiment reference** so results can be compared.

---

## 1. Free neutron half-life

**Statement:** The free neutron (0, 1) is unstable to β⁻ decay. We predict a half-life in seconds from the snap probability (Boltzmann × φ-damping × modified inertia) and ΔE_info between neutron and proton configurations.

**API:**

```python
from pyhqiv.nuclear import half_life_nuclide_hqiv, nuclide_from_symbol

# By (P, N)
t_half_s = half_life_nuclide_hqiv(0, 1)  # seconds, or None if stable

# By nuclide name (neutron has no element symbol; use (0,1))
# For isotopes: nuclide_from_symbol("H", A=2) → (1, 1) for deuterium
```

**Experiment:** PDG free neutron half-life ≈ **879.4 ± 0.6 s** (β⁻ → p + e⁻ + ν̄ₑ).

**Qualified:** We state that the code returns a **finite half-life in seconds** for (0, 1); the **magnitude** can be compared to 879 s. No tuning of the decay formula to match this number.

---

## 2. C-14 half-life and decay chain

**Statement:** Carbon-14 (6, 8) undergoes β⁻ decay to nitrogen-14 (7, 7). We predict the dominant decay mode, the daughter (P, N), the full decay chain until stable, and a half-life in seconds.

**API:**

```python
from pyhqiv.nuclear import (
    half_life_nuclide_hqiv,
    decay_chain,
    decay_chain_nuclide_hqiv,
    nuclide_from_symbol,
)
from pyhqiv.utils import decay_chain_nuclide  # NuclideDecay with half_life_seconds, decay_mode, chain

# (P, N) for C-14
P, N = nuclide_from_symbol("C", A=14)  # (6, 8)

# Half-life (seconds)
t_half_s = half_life_nuclide_hqiv(6, 8)

# Decay chain: list of (P, N) from (6,8) until stable
chain = decay_chain(6, 8, max_steps=20)

# Full summary: half-life, mode, daughter (P,N), chain
t_half, mode, dP, dN, chain = decay_chain_nuclide_hqiv(6, 8)
# Expect mode "β⁻", daughter (7, 7), chain [(6,8), (7,7), ...]
```

**Experiment:** PDG C-14 β⁻ to N-14; half-life ≈ **5700 yr** (≈ 1.8×10¹¹ s).

**Qualified:** We state that for (6, 8) the code returns an **allowed β⁻ snap** to (7, 7), a **decay chain** that includes (7, 7), and a **finite half-life in seconds**. The **magnitude** can be compared to 5700 yr; the **mode and daughter** are β⁻ and N-14.

---

## 3. Coupling angles for heavy water (D₂O)

**Statement:** Heavy water has two deuterons (²H) and one oxygen. The same horizon-coupling and bond-angle machinery that gives Θ_local for H and O can be used with **isotope-dependent Θ** so that D (mass 2) has a different effective horizon than H (mass 1). Bonding angles and angle-deficit energy then differ from H₂O in a well-defined way.

**API:**

```python
from pyhqiv.utils import theta_for_atom
from pyhqiv.nuclear import nuclide_from_symbol, binding_energy_isotope

# Deuterium: Z=1, A=2 → use mass_amu=2 for Θ scaling
theta_D_ang = theta_for_atom("H", coordination=2, mass_amu=2)  # Å
theta_O_ang = theta_for_atom("O", coordination=2)

# Binding energy for ²H (for consistency with nuclear layer)
B_D = binding_energy_isotope("H", 2)  # MeV

# Molecular bonding angles (from Molecule / Atom) use these Θ when
# atoms are built with species and optional mass; angle-deficit energy
# is then first-principles for D₂O geometry.
```

**Experiment:** D–O–D angle in D₂O is very close to H–O–H in H₂O (~104.5°); bond length D–O is slightly shorter than H–O. Isotope effects on zero-point and coupling can be compared to experiment.

**Qualified:** We state that **Θ_local for deuterium** is available via `theta_for_atom("H", coordination=..., mass_amu=2)` and is **isotope-dependent** (heavier → larger Θ in the current scaling). **Coupling angles** for D₂O (and their energy deficit) use the same bond-graph and angle-deficit logic as H₂O; the numerical prediction can be compared to structural data.

---

## 4. Gold-standard binding (any nucleus)

**Statement:** For any (P, N) we return a **BindingResult**: B_mev, E_free_mev, E_bound_mev, positions_m, theta_p_m, theta_n_m from the functional method (mass → Θ_eff, minimize, horizon coupling). Tests run against this as the gold standard.

**API:**

```python
from pyhqiv.nuclear import binding_energy_mev_functional, BindingResult, binding_energy_mev

res = binding_energy_mev_functional(2, 2)  # 4He
assert res.E_bound_mev <= res.E_free_mev + 1e-6  # self-consistency
B = max(res.B_mev, 0.0)
# Or scalar:
B = binding_energy_mev(2, 2)
```

**Qualified:** We state **self-consistency** (B = E_free − E_bound) and that the **structure** (positions, thetas) is determined by the same method for ²H, ⁴He, C-12, etc. Magnitude of B can be compared to PDG per nucleus.

---

## 5. Summary table

| Quantity | API | Experiment reference |
|----------|-----|----------------------|
| Free neutron t₁/₂ | `half_life_nuclide_hqiv(0, 1)` | PDG ≈ 879.4 s |
| C-14 t₁/₂ | `half_life_nuclide_hqiv(6, 8)` | ≈ 5700 yr |
| C-14 decay chain | `decay_chain(6, 8)`, `decay_chain_nuclide_hqiv(6, 8)` | β⁻ → N-14 |
| Heavy-water Θ (D) | `theta_for_atom("H", coordination=2, mass_amu=2)` | D₂O structure |
| Binding (any nucleus) | `binding_energy_mev_functional(P, N)` | PDG B per isotope |

All of the above are **parameter-free** from the paper constants (T_CMB, lattice, snap formula); where we quote experiment we do so for comparison only, not as an input.
