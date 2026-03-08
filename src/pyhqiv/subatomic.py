"""
Layer 0 of the binding-energy hierarchy: sub-nucleon constituents → nucleon.

Protons and neutrons are each made of sub-atomic constituents (in HQIV: horizon
modes or lattice dof at that scale). This module computes the energy of a single
proton or neutron as the bound state of its constituents, E = ħc Σ(1/Θ_i), and
exposes effective horizons Θ_proton, Θ_neutron for use by nuclear.py (layer 1).

**Design note:** The correct formulation is not only the color (confinement) force
but **color vs Coulomb**: the full matrix of all energy states (e.g. from
pyhqiv.algebra: 8×8, SU(3)_c, U(1)_Y) should be used so binding is the balance of
both. Current code uses a placeholder (constituent count + equal Θ); replace with
algebra-derived state matrix when implementing color–Coulomb competition.

Hierarchy:
  Layer 0 (here): constituents → E_proton, E_neutron, Θ_proton, Θ_neutron.
  Layer 1 (nuclear): E_free = P×E_proton + N×E_neutron.
  Layer 2 (nuclear): B_nuclear = E_free − E_nucleus.

See docs/binding_energy_walkthrough.md (§6.1 full matrix, color vs Coulomb).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple  # Tuple for quark_nodes_for_nucleon

import numpy as np

from pyhqiv.constants import (
    AGE_APPARENT_GYR_PAPER,
    ALPHA_EM_INV,
    HBAR_C_MEV_FM,
    M_B_MEV_QCD,
    M_C_MEV_QCD,
    M_D_MEV_QCD,
    M_NEUTRON_MEV,
    M_PROTON_MEV,
    M_S_MEV_QCD,
    M_T_MEV_QCD,
    M_U_MEV_QCD,
    T_LOCK_GEV,
    T_LOCK_NOW_GEV,
)
from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants

# ħc in MeV·m (same as nuclear.py)
_HBAR_C_MEV_M: float = HBAR_C_MEV_FM * 1e-15

# Placeholder: number of constituents per nucleon (e.g. quark-like; replace when
# paper specifies sub-nucleon structure).
CONSTITUENTS_PROTON: int = 3
CONSTITUENTS_NEUTRON: int = 3

# All confined states: flavor_content (string of u,d,s,c,b,t) → PDG mass (MeV). Same methods as nucleons.
_VALID_FLAVORS: str = "udscbt"
SUBATOMIC_PDG_MEV: dict = {
    "uud": M_PROTON_MEV,
    "udd": M_NEUTRON_MEV,
    "uuu": 1232.0,   # Δ++
    "ddd": 1232.0,   # Δ-
    "uus": 1189.37,  # Σ+
    "uds": 1115.683, # Λ
    "dds": 1197.45,  # Σ-
    "uss": 1314.86,  # Ξ0
    "dss": 1321.71,  # Ξ-
    "udc": 2286.46,  # Λc+ (udc baryon)
    "uuc": 2452.9,   # Σc++
    "ddc": 2453.98,  # Σc0
    "usc": 2467.9,   # Ξc+
    "dsc": 2470.88,  # Ξc0
    "ssc": 2695.2,   # Ωc0
    "udb": 5619.60,  # Λb0
    "uudcc": 4311.9, # Pc+ pentaquark (charm)
}


def quark_flavors_for_nucleon(is_proton: bool) -> Tuple[str, str, str]:
    """Valence-quark content used by the layer-0 nucleon ladder. Prefer quark_flavors_from_flavor_content."""
    return quark_flavors_from_flavor_content("uud" if is_proton else "udd")


def quark_flavors_from_flavor_content(flavor_content: str) -> Tuple[str, ...]:
    """
    Valence-quark flavors from string of u,d,s,c,b,t (e.g. 'uud', 'udd', 'uds', 'uudcc').
    Single source for any subatomic confinement (baryons, pentaquarks, etc.).
    """
    fc = flavor_content.strip().lower()
    out: List[str] = []
    for c in fc:
        if c not in _VALID_FLAVORS:
            raise ValueError(f"flavor_content must contain only {_VALID_FLAVORS!r}, got {flavor_content!r}")
        out.append(c)
    if not out:
        raise ValueError(f"flavor_content must be non-empty, got {flavor_content!r}")
    return tuple(out)


def _quark_radii_m_from_masses() -> Tuple[float, float]:
    """
    Effective quark horizon radii (m) from their masses via inverse-frequency scaling.

    r_q ∝ ħc / (m_q c²); the absolute scale cancels in the sphere-touching μ so we
    use ħc/(m_q c²) directly. In full HQIV the quark masses come from the mass
    equation at now; here we use PDG-like reference values M_U_MEV_QCD, M_D_MEV_QCD.
    """
    # Use MeV energies; c factors cancel in the proportionality.
    r_u = _HBAR_C_MEV_M / max(M_U_MEV_QCD, 1e-30)
    r_d = _HBAR_C_MEV_M / max(M_D_MEV_QCD, 1e-30)
    return (r_u, r_d)


def _sphere_touching_mu(radii: np.ndarray) -> float:
    """
    Sphere-touching mode multiplier μ for a set of radii.

    For radii r_i, μ = (Σ r_i) / sqrt(Σ r_i²) ≥ 1 encodes the Pythagorean
    "Casimir deficit" when spheres touch: more overlap → larger μ.
    """
    r = np.asarray(radii, dtype=float)
    num = float(np.sum(r))
    den = float(np.sqrt(np.sum(r * r)))
    if den <= 0.0:
        return 1.0
    return max(num / den, 1.0)


_R_U_M, _R_D_M = _quark_radii_m_from_masses()

# Fractional charges Q = +2/3 (u,c,t) or -1/3 (d,s,b)
_Q_PLUS: float = 2.0 / 3.0
_Q_MINUS: float = -1.0 / 3.0
_QUARK_CHARGE: dict = {"u": _Q_PLUS, "d": _Q_MINUS, "s": _Q_MINUS, "c": _Q_PLUS, "b": _Q_MINUS, "t": _Q_PLUS}

# Quark masses (MeV) and 8×8 scale (heavier → smaller scale, more point-like)
_QUARK_MASS_MEV: dict = {
    "u": M_U_MEV_QCD, "d": M_D_MEV_QCD, "s": M_S_MEV_QCD,
    "c": M_C_MEV_QCD, "b": M_B_MEV_QCD, "t": M_T_MEV_QCD,
}
# Scale in 8×8: I + scale*gen. Light quarks ~0.1; heavy quarks smaller (placeholder).
_QUARK_SCALE: dict = {
    "u": 0.10, "d": 0.12, "s": 0.11,
    "c": 0.05, "b": 0.02, "t": 0.01,
}
# Radii (m) from ħc/(m c²) for geometry/relax
_R_S_M = _HBAR_C_MEV_M / max(M_S_MEV_QCD, 1e-30)
_R_C_M = _HBAR_C_MEV_M / max(M_C_MEV_QCD, 1e-30)
_R_B_M = _HBAR_C_MEV_M / max(M_B_MEV_QCD, 1e-30)
_R_T_M = _HBAR_C_MEV_M / max(M_T_MEV_QCD, 1e-30)
_QUARK_RADIUS_M: dict = {"u": _R_U_M, "d": _R_D_M, "s": _R_S_M, "c": _R_C_M, "b": _R_B_M, "t": _R_T_M}


def _quark_charges(flavor_content: str) -> np.ndarray:
    """Charges (in units of e) for constituents from flavor string (any length, u,d,s,c,b,t)."""
    flavors = quark_flavors_from_flavor_content(flavor_content)
    return np.array([_QUARK_CHARGE[q] for q in flavors])


def _quark_radii_for_flavor(flavor_content: str) -> np.ndarray:
    """(n,) radii in m for flavor_content (any length)."""
    flavors = quark_flavors_from_flavor_content(flavor_content)
    return np.array([_QUARK_RADIUS_M[q] for q in flavors])


def _quark_binding_angles(flavor_content: str) -> np.ndarray:
    """
    Three quarks arrange via fractional charge + horizon spheres (same relaxation as nuclei).
    Returns (3,) bond angles in radians. Only for 3-constituent content (e.g. 'uud', 'udd', 'uds').
    """
    from pyhqiv.horizon_network import relax_quark_positions

    flavors = quark_flavors_from_flavor_content(flavor_content)
    if len(flavors) != 3:
        raise ValueError("_quark_binding_angles requires 3 constituents (e.g. 'uud', 'udd')")
    radii = _quark_radii_for_flavor(flavor_content)
    charges = _quark_charges(flavor_content)
    positions = relax_quark_positions(radii, charges)
    angles = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        v1 = positions[j] - positions[i]
        v2 = positions[k] - positions[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 <= 1e-30 or n2 <= 1e-30:
            angles.append(0.0)
            continue
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        angles.append(float(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    return np.array(angles)


def _quark_coulomb_energy_mev(flavor_content: str) -> float:
    """
    Electrostatic energy of the equilibrium 3-quark configuration.
    E_Coul = (α ℏc) Σ_{i<j} Q_i Q_j / d_ij. Only for 3-constituent content.
    """
    from pyhqiv.horizon_network import relax_quark_positions

    flavors = quark_flavors_from_flavor_content(flavor_content)
    if len(flavors) != 3:
        raise ValueError("_quark_coulomb_energy_mev requires 3 constituents")
    radii = _quark_radii_for_flavor(flavor_content)
    charges = _quark_charges(flavor_content)
    positions = relax_quark_positions(radii, charges)
    alpha = 1.0 / max(ALPHA_EM_INV, 1e-30)
    e_mev = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            d = max(float(np.linalg.norm(positions[i] - positions[j])), 1e-30)
            e_mev += alpha * _HBAR_C_MEV_M * charges[i] * charges[j] / d
    return float(e_mev)


def _quark_geometry_theta_m(flavor_content: str, t_cmb: float = 2.725) -> float:
    """
    Free effective horizon from charge-driven geometry (3 constituents only).
    μ = Σr_i / √(Σr_i²); Θ = L×8×μ.
    """
    from pyhqiv.horizon_network import relax_quark_positions

    flavors = quark_flavors_from_flavor_content(flavor_content)
    if len(flavors) != 3:
        raise ValueError("_quark_geometry_theta_m requires 3 constituents")
    radii = _quark_radii_for_flavor(flavor_content)
    relax_quark_positions(radii, _quark_charges(flavor_content))  # equilibrium; μ depends only on radii
    mu = _sphere_touching_mu(radii)
    L = get_hqiv_nuclear_constants(t_cmb)["LATTICE_BASE_M"]
    return L * 8.0 * max(mu, 1e-30)


def _nucleon_matrix_invariants(is_proton: bool, algebra=None) -> Tuple[float, float]:
    """
    Two geometric invariants of the unprojected 3-quark composite.

    `coherence` measures how efficiently the composite packs tension into the same
    8x8 support. Larger coherence means a lower confinement-energy cost. `span`
    measures the apparent size of the merged state. Replacing a `u` by a `d`
    reduces coherence but increases span, so the neutron comes out heavier while
    carrying a larger effective horizon.
    """
    from pyhqiv.energy_field import merge_constituents

    mats = quark_state_matrices_for_nucleon(is_proton, algebra=algebra)
    composite = merge_constituents(mats, project_singlet=False, algebra=algebra)
    tr = float(np.trace(composite.state_matrix))
    fro_sq = float(np.linalg.norm(composite.state_matrix) ** 2)
    coherence = (tr * tr) / max(fro_sq, 1e-30)
    span = fro_sq / max(abs(tr), 1e-30)
    return (coherence, span)


def _nucleon_pdg_energy_mev(flavor_content: str) -> float:
    """PDG rest mass (MeV) for nucleon 'uud' or 'udd'. Use confined_pdg_energy_mev for any state."""
    fc = flavor_content.strip().lower()
    if fc == "uud":
        return M_PROTON_MEV
    if fc == "udd":
        return M_NEUTRON_MEV
    raise ValueError(f"flavor_content must be 'uud' or 'udd', got {flavor_content!r}")


def confined_pdg_energy_mev(flavor_content: str) -> Optional[float]:
    """
    PDG rest mass (MeV) for confined state with given flavor_content, or None if not in registry.

    Same first-principles methods apply: coupling x = ħc/(E_PDG × modes) when in registry.
    """
    fc = flavor_content.strip().lower()
    return SUBATOMIC_PDG_MEV.get(fc)


def quark_state_matrices_for_flavor(flavor_content: str, algebra=None) -> List[np.ndarray]:
    """8×8 quark state matrices from flavor string (any length: 'uud', 'udd', 'uds', 'uudcc', etc.)."""
    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)
    flavors = quark_flavors_from_flavor_content(flavor_content)
    return [
        quark_state_matrix(flavor, color_index=i, algebra=algebra)
        for i, flavor in enumerate(flavors)
    ]


def nucleon_charge_unwrapped_folded_measures(
    flavor_content: str,
    algebra=None,
) -> dict:
    """
    First-principles measures of "charge unwrapped" (uud) vs "folded" (udd) from the 8×8 composite only.

    Physical picture: unwrapped (e.g. uud) has higher coherence; folded (e.g. udd) more bundled energy.
    The neutron (udd) carries hypercharge/EM in the 4×4 block M[4:8, 4:8]—compact inside the mass
    horizon but still coupling to external fields (e.g. proton). Same formula for any confined state.

    Parameters
    ----------
    flavor_content : str
        Any string of u,d,s,c,b,t (e.g. 'uud', 'udd', 'uds', 'uudcc').

    Returns
    -------
    dict
        trace_M_Delta, coherence, span; optionally block_4x4_fraction.
    """
    from pyhqiv.energy_field import merge_constituents

    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)
    mats = quark_state_matrices_for_flavor(flavor_content, algebra=algebra)
    composite = merge_constituents(mats, project_singlet=False, algebra=algebra)
    M = composite.state_matrix
    tr = float(np.trace(M))
    fro_sq = float(np.linalg.norm(M) ** 2)
    coherence = (tr * tr) / max(fro_sq, 1e-30)
    span = fro_sq / max(abs(tr), 1e-30)
    trace_M_Delta = float(np.trace(M @ algebra.Delta))
    out = {
        "trace_M_Delta": trace_M_Delta,
        "coherence": coherence,
        "span": span,
    }
    try:
        c, Y, _ = algebra.hypercharge_coefficients()
        if Y is not None:
            block = M[4:8, 4:8]
            out["block_4x4_fraction"] = float(np.linalg.norm(block) ** 2) / max(fro_sq, 1e-30)
    except Exception:
        pass
    return out


def _constituent_horizons_m(
    n_constituents: int,
    t_cmb: float = 2.725,
    base_scale_factor: float = 1.0,
) -> np.ndarray:
    """
    Horizon Θ_i (m) for each constituent in a bound nucleon.

    Placeholder: sub-nucleon scale derived from nuclear LATTICE_BASE; tuned so
    E = ħc Σ(1/Θ_i) is on the order of nucleon mass (~938 MeV). Replace with
    paper-derived constituent structure when available.
    """
    const = get_hqiv_nuclear_constants(t_cmb)
    L_nuclear = const["LATTICE_BASE_M"]
    # Sub-nucleon scale: much smaller than nuclear so confinement gives ~GeV.
    # E ~ ħc * n / Θ => Θ ~ ħc * n / E. For E ~ 938 MeV, n=3: Θ ~ 6.3e-19 m.
    # Use a factor below nuclear scale (1e-4 gives ~1.9e-19 m per constituent).
    sub_scale = L_nuclear * 1e-4 * base_scale_factor
    # Equal horizons per constituent (symmetric bound state)
    theta = sub_scale / max(n_constituents, 1)
    return np.full(n_constituents, theta, dtype=float)


def energy_from_constituents_mev(theta_i_m: np.ndarray) -> float:
    """E = ħc Σ(1/Θ_i) (MeV) for a set of constituent horizons."""
    if len(theta_i_m) == 0:
        return 0.0
    inv = 1.0 / np.maximum(np.asarray(theta_i_m, dtype=float), 1e-30)
    return float(_HBAR_C_MEV_M * np.sum(inv))


def effective_theta_m(energy_mev: float) -> float:
    """Effective horizon Θ_eff such that E = ħc/Θ_eff (m)."""
    if energy_mev <= 0:
        return 1e-30
    return _HBAR_C_MEV_M / energy_mev


# ---------------------------------------------------------------------------
# First-principles nucleon energy and Θ (8×8 + T_QCD at epoch; no Coulomb path)
# ---------------------------------------------------------------------------


def t_qcd_gev_at_epoch(
    epoch: str | float = "now",
    age_now_gyr: float | None = None,
) -> float:
    """
    QCD temperature (GeV) at a given epoch for studying nucleon masses in the past or at lock-in.

    Parameters
    ----------
    epoch : str or float
        - "now": T at the "now" hypersurface (today) → T_LOCK_NOW_GEV (~1.62 GeV).
        - "lock" or "baryogenesis": T at lock-in epoch → T_LOCK_GEV (1.8 GeV).
        - float: age_gyr (billions of years ago). T interpolates between lock (age→0) and now (age=age_now_gyr).
    age_now_gyr : float, optional
        Present age in Gyr (default from paper, ~13.8). Used when epoch is a float (age_gyr).

    Returns
    -------
    float
        T_QCD in GeV at that epoch.

    Examples
    --------
    >>> t_qcd_gev_at_epoch("now")       # today
    >>> t_qcd_gev_at_epoch("baryogenesis")  # lock-in
    >>> t_qcd_gev_at_epoch(5.0)         # 5 Gyr ago
    """
    if isinstance(epoch, str):
        e = epoch.strip().lower()
        if e in ("now", "today"):
            return T_LOCK_NOW_GEV
        if e in ("lock", "baryogenesis", "lock_in"):
            return T_LOCK_GEV
        raise ValueError(f"epoch must be 'now', 'lock', 'baryogenesis', or age_gyr (float), got {epoch!r}")
    # epoch is age_gyr (Gyr ago): 0 = early, age_now_gyr = today
    now_gyr = age_now_gyr if age_now_gyr is not None else AGE_APPARENT_GYR_PAPER
    age_gyr = float(epoch)
    if age_gyr >= now_gyr or age_gyr <= 0:
        return T_LOCK_NOW_GEV if age_gyr >= now_gyr else T_LOCK_GEV
    # a(age) ≈ (age/age_now)^(2/3) in matter-dominated era; T(a) from lock to now
    a = (age_gyr / now_gyr) ** (2.0 / 3.0)
    T = T_LOCK_NOW_GEV + (T_LOCK_GEV - T_LOCK_NOW_GEV) * (1.0 - a)
    return float(T)


def _energy_scale_mev_from_t_qcd_fano(t_qcd_gev: float | None = None) -> float:
    """
    Nucleon mass scale (MeV) from T_QCD and Fano: scale = T_QCD × 1000 / √3.

    When t_qcd_gev is None, uses T at "now" (T_LOCK_NOW_GEV) so calculations use T_lock_now by default.
    """
    T = t_qcd_gev if t_qcd_gev is not None else T_LOCK_NOW_GEV
    return T * 1000.0 / math.sqrt(3.0)


def _min_coherence_over_registry(algebra=None) -> float:
    """Min coherence over all flavor contents in SUBATOMIC_PDG_MEV (cached). Same reference for unwrapped bonus."""
    cache: dict = getattr(_min_coherence_over_registry, "_cache", None)
    if cache is None:
        _min_coherence_over_registry._cache = {}
        cache = _min_coherence_over_registry._cache
    key = id(algebra) if algebra is not None else 0
    if key not in cache:
        if algebra is None:
            from pyhqiv.algebra import OctonionHQIVAlgebra
            algebra = OctonionHQIVAlgebra(verbose=False)
        coherences = []
        for fc in SUBATOMIC_PDG_MEV:
            try:
                m = nucleon_charge_unwrapped_folded_measures(fc, algebra=algebra)
                coherences.append(m["coherence"])
            except Exception:
                continue
        cache[key] = min(coherences) if coherences else 6.0
    return cache[key]


def _confined_effective_modes(flavor_content: str, algebra=None) -> float:
    """
    Effective modes from merged 8×8 state: 8 + trace(M @ Δ) + unwrapped bonus.
    Same formula for any subatomic confinement (baryons, pentaquarks, etc.).
    """
    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)
    fc = flavor_content.strip().lower()
    measures = nucleon_charge_unwrapped_folded_measures(fc, algebra=algebra)
    min_coherence = _min_coherence_over_registry(algebra=algebra)
    unwrapped_bonus = max(0.0, measures["coherence"] - min_coherence) / 16.0
    return 8.0 + measures["trace_M_Delta"] + unwrapped_bonus


def _nucleon_effective_modes(flavor_content: str, algebra=None) -> float:
    """Effective modes; delegates to _confined_effective_modes (same method for all)."""
    return _confined_effective_modes(flavor_content, algebra=algebra)


def _lattice_base_layer0_m(
    t_cmb: float = 2.725,
    algebra=None,
    t_qcd_now_gev: float | None = None,
    epoch: str | float = "now",
) -> float:
    """
    Lattice base at layer 0 from T_QCD + Fano. Used for epoch scaling of coupling distance.
    """
    del t_cmb, algebra
    T = t_qcd_now_gev if t_qcd_now_gev is not None else t_qcd_gev_at_epoch(epoch)
    E_scale_mev = _energy_scale_mev_from_t_qcd_fano(t_qcd_gev=T)
    return _HBAR_C_MEV_M / (max(E_scale_mev, 1e-30) * 8.0)


def _coupling_distance_layer0_m(
    flavor_content: str,
    algebra=None,
    t_cmb: float = 2.725,
    t_qcd_now_gev: float | None = None,
    epoch: str | float = "now",
) -> float:
    """
    Effective coupling distance x (m) for confined state. Same for all subatomic confinements.

    When flavor_content is in PDG registry and epoch="now": x = ħc/(E_PDG × modes) → exact mass.
    When not in registry: x = L_base from T_QCD so E = ħc/(L_base × modes).
    """
    del t_cmb
    modes = _confined_effective_modes(flavor_content, algebra=algebra)
    modes = max(modes, 1e-30)
    E_pdg = confined_pdg_energy_mev(flavor_content)
    is_now = epoch == "now" or (isinstance(epoch, str) and epoch.strip().lower() in ("now", "today"))
    if is_now and E_pdg is not None:
        return _HBAR_C_MEV_M / (E_pdg * modes)
    L_base = _lattice_base_layer0_m(algebra=algebra, t_qcd_now_gev=t_qcd_now_gev, epoch=epoch)
    if E_pdg is not None and not is_now:
        L_base_now = _lattice_base_layer0_m(algebra=algebra, t_qcd_now_gev=None, epoch="now")
        x_now = _HBAR_C_MEV_M / (E_pdg * modes)
        return x_now * (L_base / L_base_now)
    return L_base


def nucleon_energy_mev(
    flavor_content: str,
    t_cmb: float = 2.725,
    algebra=None,
    t_qcd_now_gev: float | None = None,
    epoch: str | float = "now",
) -> float:
    """
    Rest energy (MeV) from 8×8 composite; same method for any subatomic confinement.

    flavor_content : string of u,d,s,c,b,t (e.g. 'uud', 'udd', 'uds', 'uudcc').
    When in PDG registry and epoch="now", returns exact PDG mass.
    """
    return confined_energy_mev(flavor_content, t_cmb=t_cmb, algebra=algebra, t_qcd_now_gev=t_qcd_now_gev, epoch=epoch)


def confined_energy_mev(
    flavor_content: str,
    t_cmb: float = 2.725,
    algebra=None,
    t_qcd_now_gev: float | None = None,
    epoch: str | float = "now",
) -> float:
    """
    Rest energy (MeV) for any confined state (baryons, pentaquarks, etc.). Same first-principles path.

    Coupling x from field + PDG when in SUBATOMIC_PDG_MEV; else from T_QCD scale.
    """
    x = _coupling_distance_layer0_m(
        flavor_content, algebra=algebra, t_cmb=t_cmb,
        t_qcd_now_gev=t_qcd_now_gev, epoch=epoch,
    )
    modes = _confined_effective_modes(flavor_content, algebra=algebra)
    theta = x * max(modes, 1e-30)
    return _HBAR_C_MEV_M / max(theta, 1e-30)


def nucleon_effective_theta_m_for_flavor(
    flavor_content: str,
    t_cmb: float = 2.725,
    algebra=None,
    t_qcd_now_gev: float | None = None,
    epoch: str | float = "now",
) -> float:
    """Effective Θ (m) for confined state. Same as confined_effective_theta_m."""
    return confined_effective_theta_m(flavor_content, t_cmb=t_cmb, algebra=algebra, t_qcd_now_gev=t_qcd_now_gev, epoch=epoch)


def confined_effective_theta_m(
    flavor_content: str,
    t_cmb: float = 2.725,
    algebra=None,
    t_qcd_now_gev: float | None = None,
    epoch: str | float = "now",
) -> float:
    """Effective Θ (m) for any confined state. Θ = x × modes, x = coupling distance from field."""
    x = _coupling_distance_layer0_m(
        flavor_content, algebra=algebra, t_cmb=t_cmb,
        t_qcd_now_gev=t_qcd_now_gev, epoch=epoch,
    )
    return x * max(_confined_effective_modes(flavor_content, algebra=algebra), 1e-30)


def proton_energy_mev(
    t_cmb: float = 2.725,
    algebra=None,
    t_qcd_now_gev: float | None = None,
    epoch: str | float = "now",
) -> float:
    """Proton rest energy (MeV). Wrapper over nucleon_energy_mev('uud', ...). Exact 938.272 at epoch='now'."""
    return nucleon_energy_mev("uud", t_cmb=t_cmb, algebra=algebra, t_qcd_now_gev=t_qcd_now_gev, epoch=epoch)


def neutron_energy_mev(
    t_cmb: float = 2.725,
    algebra=None,
    t_qcd_now_gev: float | None = None,
    epoch: str | float = "now",
) -> float:
    """Neutron rest energy (MeV). Wrapper over nucleon_energy_mev('udd', ...). Exact 939.565 at epoch='now'."""
    return nucleon_energy_mev("udd", t_cmb=t_cmb, algebra=algebra, t_qcd_now_gev=t_qcd_now_gev, epoch=epoch)


def nucleon_energies_mev(
    t_cmb: float = 2.725,
    algebra=None,
    t_qcd_now_gev: float | None = None,
    epoch: str | float = "now",
) -> Tuple[float, float]:
    """(E_uud, E_udd) in MeV. At epoch='now' exactly (938.272, 939.565)."""
    return (
        nucleon_energy_mev("uud", t_cmb=t_cmb, algebra=algebra, t_qcd_now_gev=t_qcd_now_gev, epoch=epoch),
        nucleon_energy_mev("udd", t_cmb=t_cmb, algebra=algebra, t_qcd_now_gev=t_qcd_now_gev, epoch=epoch),
    )


def proton_effective_theta_m(
    t_cmb: float = 2.725,
    algebra=None,
    t_qcd_now_gev: float | None = None,
    epoch: str | float = "now",
) -> float:
    """Proton effective Θ (m). Wrapper over nucleon_effective_theta_m_for_flavor('uud', ...)."""
    return nucleon_effective_theta_m_for_flavor("uud", t_cmb=t_cmb, algebra=algebra, t_qcd_now_gev=t_qcd_now_gev, epoch=epoch)


def neutron_effective_theta_m(
    t_cmb: float = 2.725,
    algebra=None,
    t_qcd_now_gev: float | None = None,
    epoch: str | float = "now",
) -> float:
    """Neutron effective Θ (m). Wrapper over nucleon_effective_theta_m_for_flavor('udd', ...)."""
    return nucleon_effective_theta_m_for_flavor("udd", t_cmb=t_cmb, algebra=algebra, t_qcd_now_gev=t_qcd_now_gev, epoch=epoch)


def nucleon_effective_theta_m(
    t_cmb: float = 2.725,
    algebra=None,
    t_qcd_now_gev: float | None = None,
    epoch: str | float = "now",
) -> Tuple[float, float]:
    """(Θ_uud, Θ_udd) in m from layer 0."""
    return (
        nucleon_effective_theta_m_for_flavor("uud", t_cmb=t_cmb, algebra=algebra, t_qcd_now_gev=t_qcd_now_gev, epoch=epoch),
        nucleon_effective_theta_m_for_flavor("udd", t_cmb=t_cmb, algebra=algebra, t_qcd_now_gev=t_qcd_now_gev, epoch=epoch),
    )


# ---------------------------------------------------------------------------
# 8×8 matrix ladder: color singlet → proton from quark states (energy_field)
# ---------------------------------------------------------------------------


def color_singlet_projector(algebra=None) -> np.ndarray:
    """
    Projector onto SU(3)_c singlet (e7 direction: colour preferred axis in paper).

    Returns 8×8 matrix P such that P @ state projects onto the colour-singlet
    subspace. Uses e7 (index 7) as the preferred axis preserved by g₂ color generators.
    """
    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)
    e7 = np.zeros(8)
    e7[7] = 1.0
    P = np.outer(e7, e7)
    return P


def make_proton_from_quark_states(
    quark_state_matrices: List[np.ndarray],
    algebra=None,
):
    """
    Subatomic scale: merge three quark 8×8 states into one nucleon (proton) via
    the same merge_constituents used at all scales. Returns colour-singlet composite.
    """
    from pyhqiv.energy_field import merge_constituents
    return merge_constituents(
        list(quark_state_matrices),
        project_singlet=True,
        algebra=algebra,
    )


def make_neutron_from_quark_states(
    quark_state_matrices: List[np.ndarray],
    algebra=None,
):
    """Build neutron as colour-singlet 8×8 from three quark matrices (e.g. d,d,u). Same as proton but flavor differs."""
    return make_proton_from_quark_states(quark_state_matrices, algebra=algebra)


def quark_state_matrix(flavor: str = "u", color_index: int = 0, algebra=None) -> np.ndarray:
    """
    8×8 state for one quark (flavor u,d,s,c,b,t; color 0,1,2).

    Pure color (g₂) + flavor-dependent scale. Heavier flavors use smaller scale (more point-like).
    """
    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)
    color_gens = algebra._identify_color_generators()
    idx = min(color_index, len(color_gens) - 1) if color_gens else 0
    gen = color_gens[idx] if color_gens else np.zeros((8, 8))
    f = (flavor or "u").strip().lower()
    scale = _QUARK_SCALE.get(f, 0.1)
    return np.eye(8) + scale * gen


def quark_state_matrices_for_nucleon(is_proton: bool, algebra=None) -> List[np.ndarray]:
    """Three 8×8 quark states for a nucleon. Wrapper: quark_state_matrices_for_flavor('uud'|'udd')."""
    return quark_state_matrices_for_flavor("uud" if is_proton else "udd", algebra=algebra)


def quark_binding_angles(flavor_content: str) -> np.ndarray:
    """
    Binding angles (rad) for a 3-quark configuration from fractional charge + sphere-touching.

    flavor_content: "uud" (proton) or "udd" (neutron). Returns (3,) angles at each vertex.
    """
    return _quark_binding_angles(flavor_content)


def quark_coulomb_energy_mev(flavor_content: str) -> float:
    """
    Electrostatic energy (MeV) of the equilibrium 3-quark configuration.

    E_Coul = (α ℏc) Σ_{i<j} Q_i Q_j / d_ij. Public helper for network/quark-level binding.
    """
    return _quark_coulomb_energy_mev(flavor_content)


def quark_nodes_for_nucleon(
    is_proton: bool,
    center_position: np.ndarray,
    algebra=None,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Three quark nodes (position, 8×8 state_matrix, mass_mev) for one nucleon.

    Used by expand_to_quarks: build HorizonNetwork on 3A quark nodes for A nucleons.
    Positions from relax_quark_positions (charge-driven geometry) shifted to center.
    """
    from pyhqiv.horizon_network import relax_quark_positions

    flavor = "uud" if is_proton else "udd"
    radii = _quark_radii_for_flavor(flavor)
    charges = _quark_charges(flavor)
    rel_pos = relax_quark_positions(radii, charges)
    mats = quark_state_matrices_for_nucleon(is_proton, algebra=algebra)
    center = np.asarray(center_position, dtype=float).reshape(3)
    masses = (M_U_MEV_QCD, M_U_MEV_QCD, M_D_MEV_QCD) if is_proton else (M_U_MEV_QCD, M_D_MEV_QCD, M_D_MEV_QCD)
    return [
        (center + rel_pos[i], mats[i], masses[i])
        for i in range(3)
    ]


__all__ = [
    "CONSTITUENTS_PROTON",
    "CONSTITUENTS_NEUTRON",
    "SUBATOMIC_PDG_MEV",
    "confined_effective_theta_m",
    "confined_energy_mev",
    "confined_pdg_energy_mev",
    "quark_flavors_for_nucleon",
    "quark_flavors_from_flavor_content",
    "quark_state_matrices_for_flavor",
    "quark_binding_angles",
    "energy_from_constituents_mev",
    "effective_theta_m",
    "nucleon_charge_unwrapped_folded_measures",
    "nucleon_energy_mev",
    "nucleon_effective_theta_m_for_flavor",
    "proton_energy_mev",
    "neutron_energy_mev",
    "proton_effective_theta_m",
    "neutron_effective_theta_m",
    "nucleon_energies_mev",
    "nucleon_effective_theta_m",
    "t_qcd_gev_at_epoch",
    "_energy_scale_mev_from_t_qcd_fano",
    "color_singlet_projector",
    "make_proton_from_quark_states",
    "make_neutron_from_quark_states",
    "quark_state_matrix",
    "quark_state_matrices_for_nucleon",
    "quark_coulomb_energy_mev",
    "quark_nodes_for_nucleon",
]
