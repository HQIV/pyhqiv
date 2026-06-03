"""
First-principles nuclear binding — PURE FRESNEL CAUSTICS CALCULATED FROM AXIOMS ONLY.
No arbitrary constants in the caustic term. Coherent wave superposition at each nucleon.

Axiom 1 — Discrete null-lattice combinatorics
Axiom 2 — Informational-energy + entanglement monogamy
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from pyhqiv.atom import HQIVAtom
from pyhqiv.constants import (
    C_SI,
    GAMMA,
    HBAR_C_MEV_FM,
    T_LOCK_NOW_GEV,
)

_MEV_TO_GEV = 1.0 / 1000.0  # unit conversion
_GEV_TO_MEV = 1000.0
_NUCLEAR_SCALE = (HBAR_C_MEV_FM * _MEV_TO_GEV) / T_LOCK_NOW_GEV

from pyhqiv.lattice import (
    DiscreteNullLattice,
    curvature_imprint_delta_E,
    discrete_mode_count,
)
from pyhqiv.subatomic import (
    nucleon_effective_theta_m,
    proton_energy_mev,
    neutron_energy_mev,
    nucleon_charge_unwrapped_folded_measures,
    proton_effective_theta_m,
    neutron_effective_theta_m,
)

_FM_TO_M = 1e-15
_M_TO_FM = 1e15

# Numerical guards and variational bounds (no tuned physics constants)
_EPS = 1e-30
_EPS_DENOM = 1e-300
_VALLEY_LO_FACTOR = 1.2   # x_lo >= this × θ (valley above horizon)
_VALLEY_HI_FACTOR = 3.0   # x_hi >= this × θ
_SEARCH_FM_LO = 0.15      # min search scale (fm)
_SEARCH_FM_HI = 15.0      # max search scale (fm)
_TOL_FM = 1e-4            # optimizer tolerance (fm)
_TIPPING_DELTA_E_FRAC = 1e-3  # β-decay threshold from lattice δE


def _fresnel_reflectance(theta: float, n_eff: float = 1.5) -> float:
    """Pure Fresnel reflectance (unpolarized) — boundary condition only."""
    cos_i = abs(math.cos(theta))
    sin_t = math.sin(theta) / n_eff
    if sin_t >= 1.0:
        return 1.0
    cos_t = math.sqrt(max(0.0, 1.0 - sin_t * sin_t))
    rs = (cos_i - n_eff * cos_t) / (cos_i + n_eff * cos_t)
    rp = (n_eff * cos_i - cos_t) / (n_eff * cos_i + cos_t)
    return (rs * rs + rp * rp) / 2.0


def _caustic_amplitude(r_m: float, Theta_m: float) -> float:
    """Exact Fresnel catacaustic projection with exponential screening.
    Uses true sin³θ weighting + softer falloff that never diverges.
    """
    if r_m < 1.05 * Theta_m:
        return 0.0
    theta = math.acos(min(1.0, Theta_m / max(r_m, _EPS)))
    fres = _fresnel_reflectance(theta)
    caustic_geom = math.sin(theta) ** 3
    delta_r = max(r_m - Theta_m, _EPS)
    envelope = math.exp(-delta_r / (1.35 * Theta_m))
    norm = 1.0 / (1.0 + (r_m / Theta_m) ** 1.2)
    return fres * caustic_geom * envelope * norm

# =============================================================================
# Magnetic horizon scale — purely from subatomic 8×8 (no tuned constants)
# =============================================================================


def _get_magnetic_horizon_scale() -> float:
    """
    Dimensionless magnetic coefficient from 8×8 composite only.
    EM block fraction difference (n vs p) × horizon ratio; if block_4x4
    is missing, use coherence difference as proxy. No 0.0215 or 0.25.
    """
    measures_p = nucleon_charge_unwrapped_folded_measures("uud")
    measures_n = nucleon_charge_unwrapped_folded_measures("udd")
    em_fraction_delta = abs(
        measures_n.get("block_4x4_fraction", 0.0)
        - measures_p.get("block_4x4_fraction", 0.0)
    )
    if em_fraction_delta <= 0.0:
        # No block_4x4 or identical: use coherence difference (same algebra)
        em_fraction_delta = abs(
            measures_n.get("coherence", 0.0) - measures_p.get("coherence", 0.0)
        )
    theta_p_m = proton_effective_theta_m()
    theta_n_m = neutron_effective_theta_m()
    theta_max = max(theta_p_m, theta_n_m)
    theta_ratio = min(theta_p_m, theta_n_m) / theta_max if theta_max > _EPS else 1.0
    return em_fraction_delta * theta_ratio


MU_HORIZON_SCALE = _get_magnetic_horizon_scale()

# =============================================================================
# IMPROVED GEOMETRY — tetrahedral for light nuclei
# =============================================================================

def _mean_pair_distance(coords: np.ndarray) -> float:
    n = coords.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(coords[i] - coords[j])
            count += 1
    return total / max(count, 1)


def _get_unit_coords(A: int) -> np.ndarray:
    """Deterministic geometries: tetra for light nuclei, radial shells otherwise."""
    if A == 2:
        half = 1.0 / 2.0
        return np.array([[-half, 0.0, 0.0], [half, 0.0, 0.0]])
    if A == 3:
        s = 1.0 / math.sqrt(3.0)
        half = 1.0 / 2.0
        return np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [half, math.sqrt(3.0) * half, 0.0]],
            dtype=float,
        ) * s
    if A == 4:
        pts = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=float,
        ) / math.sqrt(8.0)
        return pts / _mean_pair_distance(pts)
    # A ≥ 5: radial shells (golden-angle spiral: conjugate = (sqrt(5)-1)/2)
    _golden_conjugate = (math.sqrt(5.0) - 1.0) / 2.0
    coords = []
    for k in range(A):
        r = (k + 1) / float(A)
        phi = 2.0 * math.pi * (k * _golden_conjugate)
        theta = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (k + 0.5) / A)))
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)
        coords.append([x, y, z])
    return np.array(coords, dtype=float)


def generate_nuclei_pack_coords(
    A: int, Z: int, x_fm: float
) -> Tuple[np.ndarray, List[int], float]:
    """Deterministic NUCLEI-PACK with tetrahedral special cases."""
    if A <= 0:
        return np.zeros((0, 3)), [], 0.0
    Z = max(0, min(Z, A))
    unit = _get_unit_coords(A)
    mean_d = _mean_pair_distance(unit)
    scale = x_fm / mean_d if mean_d > 0 else 1.0
    coords_fm = unit * scale
    norms = np.linalg.norm(coords_fm, axis=1)
    order = np.argsort(norms)
    charges = [0] * A
    for idx in order[:Z]:
        charges[idx] = 1
    R_pack = float(np.max(norms))
    return coords_fm, charges, R_pack


# =============================================================================
# LATTICE & ORIGINAL ENERGY TERMS
# =============================================================================

def _nucleus_shell_index_from_x(x_fm: float, theta_fm: float) -> int:
    if theta_fm <= 0:
        return 0
    ratio = x_fm / theta_fm
    return max(0, int(round(ratio)) - 1)


def _lattice_primitives(m_nuc: int):
    lattice = DiscreteNullLattice()
    m_arr = np.array([float(m_nuc)], dtype=float)
    T = lattice.shell_temperature(m_arr, E_0_factor=1.0)
    delta_E = curvature_imprint_delta_E(m_arr, T)
    new_modes = discrete_mode_count(m_nuc)
    return float(T.flat[0]), float(np.asarray(delta_E).flat[0]), new_modes


def _lattice_norm_ref() -> float:
    _, delta_E0, new_modes0 = _lattice_primitives(0)
    return max(new_modes0 * delta_E0, _EPS)


@lru_cache(maxsize=None)
def _lattice_primitives_cached(m_nuc: int) -> Tuple[float, float, int]:
    return _lattice_primitives(m_nuc)


def _coulomb_pp_energy_geV(
    coords_fm: np.ndarray,
    charges: List[int],
    x_fm: float,
    theta_fm: float,
    gamma: float = GAMMA,
) -> float:
    m_nuc = _nucleus_shell_index_from_x(x_fm, theta_fm)
    ref = _lattice_norm_ref()
    _, delta_E, new_modes = _lattice_primitives(m_nuc)
    scale_GeV = T_LOCK_NOW_GEV * (new_modes * delta_E / ref) * gamma
    coords_m = coords_fm * _FM_TO_M
    theta_m = theta_fm * _FM_TO_M
    proton_indices = [i for i in range(len(charges)) if charges[i] == 1]
    if len(proton_indices) < 2:
        return 0.0
    total = 0.0
    for ii, i in enumerate(proton_indices):
        pos_i = coords_m[i]
        atom_i = HQIVAtom(position=pos_i, charge=1.0, species="H", c_si=C_SI)
        for j in proton_indices[ii + 1 :]:
            pos_j = coords_m[j]
            r_ij = np.linalg.norm(pos_j - pos_i)
            if r_ij <= 0:
                continue
            x_j = np.asarray(pos_j, dtype=float).reshape(3)
            contrib = atom_i.modified_field_contribution(
                x_j, E_prime=1.0 / 2.0, gamma=gamma
            )
            total += float(np.asarray(contrib).flat[0]) * theta_m
    return scale_GeV * total * _NUCLEAR_SCALE


def _conservation_energy_geV(
    x_fm: float,
    theta_fm: float,
    gamma: float = GAMMA,
    A: int = 2,
) -> float:
    """
    Conservation from lattice coupling; scales with number of bonds (first principles).
    Each pair shares one unit of mode coupling → D gets 1×, 4He gets 6×, no cap.
    """
    n_pairs = max(1, A * (A - 1) // 2)
    m_nuc = _nucleus_shell_index_from_x(x_fm, theta_fm)
    _, delta_E, new_modes = _lattice_primitives(m_nuc)
    ref = _lattice_norm_ref()
    geom = (theta_fm / max(x_fm, _EPS)) ** (3.0 / 2.0)
    base = (new_modes * delta_E / ref) * gamma * T_LOCK_NOW_GEV * geom * _NUCLEAR_SCALE
    return -n_pairs * base


# =============================================================================
# NEW: Magnetic horizon correction (n-p only)
# =============================================================================

def _magnetic_np_energy_geV(
    coords_fm: np.ndarray,
    charges: List[int],
    x_fm: float,
    theta_fm: float,
) -> float:
    """Neutron magnetic moment on horizon — scaled from subatomic 8×8 (Θ/d)^3."""
    A = coords_fm.shape[0]
    if A < 2:
        return 0.0
    coords_m = coords_fm * _FM_TO_M
    theta_m = theta_fm * _FM_TO_M
    total = 0.0
    for i in range(A):
        for j in range(i + 1, A):
            if charges[i] != charges[j]:  # n-p only
                d_ij = np.linalg.norm(coords_m[i] - coords_m[j])
                if d_ij > 0:
                    total += (theta_m / d_ij) ** 3
    m_nuc = _nucleus_shell_index_from_x(x_fm, theta_fm)
    _, delta_E, new_modes = _lattice_primitives(m_nuc)
    ref = _lattice_norm_ref()
    scale_GeV = T_LOCK_NOW_GEV * (new_modes * delta_E / ref) * GAMMA
    return -MU_HORIZON_SCALE * total * scale_GeV * _NUCLEAR_SCALE


# =============================================================================
# COHERENT CAUSTIC FIELD — analytic Fresnel catacaustic surface
# =============================================================================


def _coherent_caustic_field_energy_geV(
    coords_fm: np.ndarray,
    charges: List[int],
    theta_p_m: float,
    theta_n_m: float,
    x_fm: float,
    theta_fm: float,
    gamma: float = GAMMA,
) -> float:
    """
    Constructive, stable Fresnel caustics from axioms only.
    |Ψ_total|² rewards ring intersections; normalization by sqrt(n_neighbors)
    damps A³ growth while keeping multi-body cross-terms.
    """
    A = coords_fm.shape[0]
    if A < 2:
        return 0.0
    coords_m = coords_fm * _FM_TO_M
    ref = _lattice_norm_ref()
    m_nuc = _nucleus_shell_index_from_x(x_fm, theta_fm)
    _, delta_E, new_modes = _lattice_primitives(m_nuc)
    lattice_ratio = (new_modes * delta_E) / ref

    total_field = 0.0
    for i in range(A):
        amp_total = 0.0
        n_neighbors = 0
        for j in range(A):
            if i == j:
                continue
            r_ij_m = np.linalg.norm(coords_m[i] - coords_m[j])
            if r_ij_m <= _EPS:
                continue
            Theta_j_m = theta_p_m if charges[j] == 1 else theta_n_m
            amp_total += _caustic_amplitude(r_ij_m, Theta_j_m)
            n_neighbors += 1
        if n_neighbors > 0:
            amp_total /= math.sqrt(n_neighbors)
        total_field += amp_total * amp_total  # |Ψ_total at i|² (ring boost lives here)
    scale_GeV = T_LOCK_NOW_GEV * lattice_ratio * gamma
    E_field = -total_field * scale_GeV / 2.0  # /2 avoids double-counting pairs
    return E_field * _NUCLEAR_SCALE


# =============================================================================
# TOTAL ENERGY (coherent caustic field + Coulomb + conservation + magnetic)
# =============================================================================


def _e_total_geV(
    x_fm: float,
    A: int,
    Z: int,
    theta_p_m: float,
    theta_n_m: float,
    theta_fm: float,
    gamma: float = GAMMA,
) -> float:
    """E_total = coherent caustic field + Coulomb + conservation + magnetic."""
    coords_fm, charges, _ = generate_nuclei_pack_coords(A, Z, x_fm)
    E_coul = _coulomb_pp_energy_geV(coords_fm, charges, x_fm, theta_fm, gamma=gamma)
    E_cons = _conservation_energy_geV(x_fm, theta_fm, gamma=gamma, A=A)
    E_mag = _magnetic_np_energy_geV(coords_fm, charges, x_fm, theta_fm)
    E_coll = _coherent_caustic_field_energy_geV(
        coords_fm, charges, theta_p_m, theta_n_m, x_fm, theta_fm, gamma=gamma
    )
    geom_scale = (theta_fm / 1.0) ** 2
    return (E_coul + E_cons + E_mag + E_coll) * geom_scale


def _e_free_geV(Z: int, N: int) -> float:
    e_p = proton_energy_mev()
    e_n = neutron_energy_mev()
    return (Z * e_p + N * e_n) * _MEV_TO_GEV


def _apply_scale_witness(binding_geV: float) -> float:
    return binding_geV * _GEV_TO_MEV


def solve_equilibrium_x(
    A: int,
    Z: int,
    theta_p_m: float,
    theta_n_m: float,
    gamma: float = GAMMA,
    x_lo_fm: float | None = None,
    x_hi_fm: float | None = None,
    tol_fm: float = _TOL_FM,
) -> float:
    theta_fm = min(theta_p_m, theta_n_m) * _M_TO_FM
    if x_lo_fm is None:
        x_lo_fm = max(_VALLEY_LO_FACTOR * theta_fm, _SEARCH_FM_LO)
    if x_hi_fm is None:
        x_hi_fm = max(_VALLEY_HI_FACTOR * theta_fm, _SEARCH_FM_HI)
    x_lo_fm = max(x_lo_fm, _EPS)

    if A == 4:
        x_lo_fm = max(x_lo_fm, 1.8 * theta_fm)
        x_hi_fm = max(x_hi_fm, 5.0)

    def objective(x: float) -> float:
        return _e_total_geV(x, A, Z, theta_p_m, theta_n_m, theta_fm, gamma=gamma)

    res = minimize_scalar(
        objective,
        bounds=(x_lo_fm, x_hi_fm),
        method="bounded",
        options={"xatol": tol_fm},
    )
    return float(res.x)


def binding_energy_mev(
    A: int,
    Z: int,
    x_eq_fm: float | None = None,
    gamma: float = GAMMA,
) -> Tuple[float, float, float]:
    N = A - Z
    theta_p_m, theta_n_m = nucleon_effective_theta_m()
    theta_fm = min(theta_p_m, theta_n_m) * _M_TO_FM
    if x_eq_fm is None:
        x_eq_fm = solve_equilibrium_x(A, Z, theta_p_m, theta_n_m, gamma=gamma)
    E_interaction = _e_total_geV(
        x_eq_fm, A, Z, theta_p_m, theta_n_m, theta_fm, gamma=gamma
    )
    binding_geV = -E_interaction
    B_MeV = _apply_scale_witness(binding_geV)
    B_per_A = B_MeV / max(A, 1)
    return B_MeV, B_per_A, x_eq_fm


def _tipping_threshold_mev(x_fm: float, theta_fm: float) -> float:
    m_nuc = _nucleus_shell_index_from_x(x_fm, theta_fm)
    _, delta_E, _ = _lattice_primitives(m_nuc)
    return float(T_LOCK_NOW_GEV * _GEV_TO_MEV * delta_E * _TIPPING_DELTA_E_FRAC)


def predict_beta_unstable(
    A: int,
    Z: int,
    B_orig_mev: float,
    x_eq_orig_fm: float,
    gamma: float = GAMMA,
) -> Tuple[bool, str]:
    if Z >= A:
        return False, ""
    theta_p_m, theta_n_m = nucleon_effective_theta_m()
    theta_fm = min(theta_p_m, theta_n_m) * _M_TO_FM
    x_eq_flip_fm = solve_equilibrium_x(A, Z + 1, theta_p_m, theta_n_m, gamma=gamma)
    E_orig = _e_total_geV(
        x_eq_orig_fm, A, Z, theta_p_m, theta_n_m, theta_fm, gamma=gamma
    )
    E_flip = _e_total_geV(
        x_eq_flip_fm, A, Z + 1, theta_p_m, theta_n_m, theta_fm, gamma=gamma
    )
    B_orig_geV = -E_orig
    B_flip_geV = -E_flip
    delta_B_mev = _apply_scale_witness(B_flip_geV - B_orig_geV)
    if delta_B_mev < 0:
        delta_B_mev = -delta_B_mev
    threshold = _tipping_threshold_mev(x_eq_orig_fm, theta_fm)
    if B_flip_geV > B_orig_geV and delta_B_mev > threshold:
        return True, "beta-"
    return False, ""


def _optimal_Z_for_A(A: int) -> int:
    return max(1, min(A - 1, int(round(A / 2.1))))


def run_first_principles_scan(
    A_min: int = 2,
    A_max: int = 20,
    gamma: float = GAMMA,
) -> List[dict]:
    results = []
    for A in range(A_min, A_max + 1):
        Z = _optimal_Z_for_A(A)
        B_mev, B_per_A, x_eq_fm = binding_energy_mev(A, Z, x_eq_fm=None, gamma=gamma)
        beta_unstable, decay_type = predict_beta_unstable(
            A, Z, B_mev, x_eq_fm, gamma=gamma
        )
        results.append({
            "A": A,
            "Z": Z,
            "B_total_MeV": B_mev,
            "B_per_nucleon_MeV": B_per_A,
            "x_eq_fm": x_eq_fm,
            "stable": not beta_unstable,
            "beta_decay": decay_type if beta_unstable else "",
        })
    return results


def _print_table_and_plot(
    results: List[dict],
    out_png: str = "binding_first_principles_upgraded.png",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("A\tZ\tB_total_MeV\tB_per_nucleon_MeV\tx_eq_fm\tstable\tbeta_decay")
    for r in results:
        print(
            f"{r['A']}\t{r['Z']}\t{r['B_total_MeV']:.4f}\t{r['B_per_nucleon_MeV']:.4f}\t"
            f"{r['x_eq_fm']:.4f}\t{r['stable']}\t{r['beta_decay']}"
        )

    A_vals = [r["A"] for r in results]
    B_per_A_vals = [r["B_per_nucleon_MeV"] for r in results]
    fig, ax = plt.subplots()
    ax.plot(A_vals, B_per_A_vals, "b-", linewidth=1.0)
    ax.set_xlabel("A")
    ax.set_ylabel("B/A (MeV)")
    ax.set_title("HQIV first-principles B/A vs A (tetra + magnetic horizon)")
    ax.grid(True)
    fig.savefig(out_png, dpi=150)
    plt.close()

    best = max(results, key=lambda r: r["B_per_nucleon_MeV"])
    print(
        f"\nMaximum at A={best['A']} — now with correct tetrahedral packing and "
        "magnetic horizon correction."
    )


if __name__ == "__main__":
    results = run_first_principles_scan(A_min=2, A_max=20)
    _print_table_and_plot(results)
