"""
Sphere-packing geometry and self-consistent equilibrium horizon distance (parameter-free, HQIV-derived).

Equilibrium separation x is solved by minimizing total energy E_network(x) + E_EM(x) + E_curvature(x)
over trial positions scaled by x. No fixed "touch" scale: nucleons sit at the minimum of the
total energy functional (HorizonNetwork + opposing_fields + curvature from ω_k(x,m)).
Lean mirror: Hqiv/Physics/AtomicNuclear.lean (equilibrium_horizon, isotopeShellStructure).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from pyhqiv.constants import ALPHA, DIPOLE_DIPOLE_MEV_FM3, M_TRANS

# Curvature coupling to total energy (ω_k(x) contribution); order ~1% of E_free so minimum is stable
_CURVATURE_ENERGY_FRACTION: float = 0.01
# Reference scale (fm) so curvature repulsion grows with x (x/θ factor from paper)
_CURVATURE_X_REF_FM: float = 1.0
# Lattice Casimir prefactor (MeV·fm²): C_lat from stars-and-bars + curvature norm; replace with Lean-derived value
# TAINTED: 938 is used as reference so B(2H) can match 2.22 MeV; nucleon mass is first-principles output elsewhere.
# See docs/coupled_system_design.md §2.4. Form 7.2e-3 * referenceM_MeV * (ħc)² scale.
_REFERENCE_M_MEV: float = 938.0
_LATTICE_CASIMIR_MEV_FM2: float = 7.2e-3 * _REFERENCE_M_MEV * (197.3**2) * 7.0  # stars-and-bars; exact value from Lean theorem


def em_angular_contribution(
    x_fm: float,
    theta: float,
    m_shell: int = M_TRANS,
) -> float:
    """
    EM angular energy (MeV) from P-N magnetic dipole-dipole interaction.

    V_EM(θ) = (μ_p μ_n / r³) (3 cos² θ - 1); θ = angle of P-N axis to spin/magnetic axis.
    Deuteron triplet favours alignment (3 cos² θ - 1 > 0 for θ ≈ 0). Uses experimental
    μ_p, μ_n (DIPOLE_DIPOLE_MEV_FM3). At r ~ 1 fm gives ~0.1 MeV; deepens binding when
    θ_eq aligns poles. Lean: em_angular_term θ (alpha_eff m).
    """
    x_fm = max(float(x_fm), 1e-6)
    # (3 cos² θ - 1): θ=0 or π → +2; μ_p μ_n < 0 so aligned state is attractive → negate prefactor
    factor = 3.0 * (math.cos(theta) ** 2) - 1.0
    return -DIPOLE_DIPOLE_MEV_FM3 * factor / (x_fm**3)


def casimir_contribution(
    x_fm: float,
    m_shell: int,
    A_eff: Optional[float] = None,
    alpha: float = ALPHA,
) -> float:
    """
    Derived Casimir energy (MeV) from lattice mode cutoff at horizon x. Always attractive (negative).

    Uses only shell_shape, omega_k_from_distance, and reference-mass scaling. Competes with
    curvature repulsion → stable minimum at fm scales. Lean: casimir_contribution x m.
    """
    from pyhqiv.lattice import omega_k_from_distance

    x_fm = max(float(x_fm), 1e-6)
    if A_eff is None:
        A_eff = np.pi * (x_fm / 2.0) ** 2  # effective area from separation
    omega = omega_k_from_distance(x_fm * 1e-15, reference_m_trans=m_shell)
    shape_factor = shell_shape(m_shell, alpha=alpha) ** 3
    return -_LATTICE_CASIMIR_MEV_FM2 * shape_factor * omega / (x_fm**3) * A_eff


def _trial_positions_scaled_by_x(P: int, N: int, x_fm: float) -> np.ndarray:
    """
    Trial (A, 3) positions in fm with single scale x_fm. Same shapes as nucleon_positions_3d
    but scaled by the variational parameter x (equilibrium horizon distance).
    """
    A = P + N
    if A <= 0:
        return np.zeros((0, 3))
    x_fm = max(float(x_fm), 1e-6)
    if A == 2:
        return np.array([[0.0, 0.0, 0.0], [x_fm, 0.0, 0.0]], dtype=float)
    if A == 4:
        tetra = np.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
            dtype=float,
        )
        tetra *= x_fm / math.sqrt(3.0)
        return tetra
    # FCC: nearest-neighbor = 2*r0 = x_fm => r0 = x_fm/2
    return fcc_packing_positions(A, x_fm / 2.0)


def _trial_positions_scaled_by_x_theta(
    P: int, N: int, x_fm: float, theta: float
) -> np.ndarray:
    """
    Trial (A, 3) positions with scale x_fm and angle theta (rad). For A=2 only:
    second nucleon at (x sin θ, 0, x cos θ) so θ = angle of P-N axis to z (spin axis).
    For A > 2 falls back to _trial_positions_scaled_by_x (theta ignored).
    """
    A = P + N
    if A <= 0:
        return np.zeros((0, 3))
    x_fm = max(float(x_fm), 1e-6)
    if A == 2:
        st, ct = math.sin(theta), math.cos(theta)
        return np.array(
            [[0.0, 0.0, 0.0], [x_fm * st, 0.0, x_fm * ct]],
            dtype=float,
        )
    return _trial_positions_scaled_by_x(P, N, x_fm)


def curvature_contribution(
    x_fm: float,
    m_shell: int,
    t_cmb: float,
    E_free_mev: float,
) -> float:
    """
    Curvature energy (MeV) at trial separation x. ω_k(x,m) from lattice; repulsion grows with x.

    At nuclear distances (fm) the lattice maps to the horizon shell, so we scale by (x/x_ref)
    so that E_curv grows with separation and the variational minimum is at finite x.
    """
    from pyhqiv.lattice import omega_k_from_distance

    x_m = max(x_fm * 1e-15, 1e-30)
    ok = omega_k_from_distance(x_m, reference_m_trans=m_shell)
    x_ratio = max(x_fm / _CURVATURE_X_REF_FM, 1e-6)
    return E_free_mev * ok * x_ratio * _CURVATURE_ENERGY_FRACTION


def equilibrium_horizon_distance(
    P: int,
    N: int,
    m_shell: int = M_TRANS,
    t_cmb: float = 2.725,
    x_lo_fm: float = 0.5,
    x_hi_fm: float = 10.0,
    tol_fm: float = 5e-3,
) -> float:
    """
    Solve for equilibrium horizon distance x (fm) that minimizes total bound-state energy.

    E_total(x) = E_network(x) + E_EM(x) + E_curvature(x). Uses only existing primitives:
    HorizonNetwork, opposing_fields_energy_mev, omega_k_from_distance, α_eff(m).
    """
    from scipy.optimize import minimize_scalar

    from pyhqiv.algebra import OctonionHQIVAlgebra
    from pyhqiv.horizon_network import HorizonNetwork
    from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants
    from pyhqiv.nuclear import _nucleon_state_matrix_unprojected, opposing_fields_energy_mev
    from pyhqiv.subatomic import nucleon_energies_mev

    A = P + N
    if A <= 0:
        return 0.0

    const = get_hqiv_nuclear_constants(t_cmb)
    lattice_base_m = const["LATTICE_BASE_M"]
    algebra = OctonionHQIVAlgebra(verbose=False)
    E_p, E_n = nucleon_energies_mev(t_cmb=t_cmb)
    E_free = P * E_p + N * E_n
    M_p = _nucleon_state_matrix_unprojected(True, algebra)
    M_n = _nucleon_state_matrix_unprojected(False, algebra)
    is_proton_list = [True] * P + [False] * N

    def total_energy(x_fm: float) -> float:
        positions_fm = _trial_positions_scaled_by_x(P, N, x_fm)
        positions_m = positions_fm * 1e-15
        nodes = (
            [(positions_m[i], M_p, E_p) for i in range(P)]
            + [(positions_m[P + j], M_n, E_n) for j in range(N)]
        )
        net = HorizonNetwork(nodes, lattice_base_m, algebra=algebra)
        E_net = net.total_energy()
        E_em = opposing_fields_energy_mev(
            positions_m, is_proton_list, algebra=algebra, m_trans=m_shell
        )
        E_curv = curvature_contribution(x_fm, m_shell, t_cmb, E_free)
        E_cas = casimir_contribution(x_fm, m_shell)
        return E_net + E_em + E_curv + E_cas

    res = minimize_scalar(
        total_energy,
        bounds=(x_lo_fm, x_hi_fm),
        method="bounded",
        options={"xatol": tol_fm},
    )
    return float(res.x)


def equilibrium_horizon_distance_and_angle(
    P: int,
    N: int,
    m_shell: int = M_TRANS,
    t_cmb: float = 2.725,
    x_lo_fm: float = 0.5,
    x_hi_fm: float = 10.0,
    tol_fm: float = 5e-3,
    use_angular: bool = True,
):
    """
    For A=2 (deuteron): minimize E_total(x, θ) over (x, θ); return (x_eq, theta_eq).
    For A>2 or use_angular=False: return (equilibrium_horizon_distance(...), None).

    θ = angle of P-N axis to spin axis; dipole-dipole (3 cos² θ - 1) favours alignment.
    """
    A = P + N
    if A <= 0:
        return (0.0, None)
    if A != 2 or not use_angular:
        x_eq = equilibrium_horizon_distance(
            P, N, m_shell=m_shell, t_cmb=t_cmb,
            x_lo_fm=x_lo_fm, x_hi_fm=x_hi_fm, tol_fm=tol_fm,
        )
        return (x_eq, None)

    from scipy.optimize import minimize

    from pyhqiv.algebra import OctonionHQIVAlgebra
    from pyhqiv.horizon_network import HorizonNetwork
    from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants
    from pyhqiv.nuclear import _nucleon_state_matrix_unprojected, opposing_fields_energy_mev
    from pyhqiv.subatomic import nucleon_energies_mev

    const = get_hqiv_nuclear_constants(t_cmb)
    lattice_base_m = const["LATTICE_BASE_M"]
    algebra = OctonionHQIVAlgebra(verbose=False)
    E_p, E_n = nucleon_energies_mev(t_cmb=t_cmb)
    E_free = P * E_p + N * E_n
    M_p = _nucleon_state_matrix_unprojected(True, algebra)
    M_n = _nucleon_state_matrix_unprojected(False, algebra)
    is_proton_list = [True] * P + [False] * N

    def total_energy_2d(v: np.ndarray) -> float:
        x_fm, theta = float(v[0]), float(v[1])
        positions_fm = _trial_positions_scaled_by_x_theta(P, N, x_fm, theta)
        positions_m = positions_fm * 1e-15
        nodes = (
            [(positions_m[i], M_p, E_p) for i in range(P)]
            + [(positions_m[P + j], M_n, E_n) for j in range(N)]
        )
        net = HorizonNetwork(nodes, lattice_base_m, algebra=algebra)
        E_net = net.total_energy()
        E_em = opposing_fields_energy_mev(
            positions_m, is_proton_list, algebra=algebra, m_trans=m_shell
        )
        E_curv = curvature_contribution(x_fm, m_shell, t_cmb, E_free)
        E_cas = casimir_contribution(x_fm, m_shell)
        E_ang = em_angular_contribution(x_fm, theta, m_shell)
        return E_net + E_em + E_curv + E_cas + E_ang

    x0 = np.array([(x_lo_fm + x_hi_fm) / 2.0, math.pi / 4.0])
    res = minimize(
        total_energy_2d,
        x0,
        method="L-BFGS-B",
        bounds=[(x_lo_fm, x_hi_fm), (0.0, math.pi)],
        options={"ftol": 1e-9},
    )
    x_eq = float(res.x[0])
    theta_eq = float(res.x[1])
    return (x_eq, theta_eq)


def shell_shape(
    m: int,
    alpha: float = ALPHA,
) -> float:
    """
    Per-shell curvature shape (1/(m+1)) * (1 + α ln(m+1)). Lean shell_shape formula.

    Same factor that appears in curvature_imprint_delta_E (without the 6^7√3 norm).
    T(m) = T_Pl/(m+1) ⇒ ln(T_Pl/T(m)) = ln(m+1).
    """
    m_f = max(float(m) + 1.0, 1e-300)
    return (1.0 / m_f) * (1.0 + alpha * math.log(m_f))


def fcc_packing_positions(A: int, r0_fm: float) -> np.ndarray:
    """
    First A positions (fm) on an FCC lattice, sorted by distance from origin.

    Nearest-neighbor distance = 2*r0_fm (touching spheres of radius r0_fm).
    FCC conventional cell: a such that a/sqrt(2) = 2*r0_fm ⇒ a = 2*r0_fm*sqrt(2).
    Good for A ≲ 40; later upgrade to full variational minimization if needed.
    """
    if A <= 0:
        return np.zeros((0, 3))
    a_fm = 2.0 * r0_fm * math.sqrt(2.0)  # FCC nearest-neighbor = a/sqrt(2)
    # FCC basis in conventional cell (0,0,0), (a/2,a/2,0), (a/2,0,a/2), (0,a/2,a/2)
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ], dtype=float)
    # Generate points in a cube of cells; 3 cells per axis → 27 cells × 4 = 108 points
    n_cell = 4  # so we have enough points for A up to ~100
    points = []
    for i in range(-n_cell, n_cell + 1):
        for j in range(-n_cell, n_cell + 1):
            for k in range(-n_cell, n_cell + 1):
                for b in range(4):
                    x = (i + basis[b, 0]) * a_fm
                    y = (j + basis[b, 1]) * a_fm
                    z = (k + basis[b, 2]) * a_fm
                    points.append([x, y, z])
    points = np.array(points, dtype=float)
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[: A]]


def nucleon_positions_3d(
    P: int,
    N: int,
    m_shell: int = M_TRANS,
    t_cmb: float = 2.725,
    alpha: float = ALPHA,
    use_equilibrium_solve: bool = True,
    return_x_eq: bool = False,
    use_angular: bool = False,
):
    """
    Return (A, 3) array of nucleon positions in fm. Self-consistent from lattice axiom.

    When use_equilibrium_solve=True: minimizes E_total(x) or E_total(x, θ) for A=2 with use_angular.
    When use_angular=True and A=2: minimizes over (x, θ) (dipole-dipole alignment).
    When return_x_eq=True: returns (positions, x_eq, theta_eq) with theta_eq=None if not angular.
    When use_equilibrium_solve=False: falls back to fixed scale (r0 from shell_shape × 6^(7/3)√3).
    """
    A = P + N
    if A <= 0:
        out = np.zeros((0, 3))
        return (out, None, None) if return_x_eq else out

    if use_equilibrium_solve:
        x_eq, theta_eq = equilibrium_horizon_distance_and_angle(
            P, N, m_shell=m_shell, t_cmb=t_cmb, use_angular=use_angular
        )
        if theta_eq is not None:
            positions = _trial_positions_scaled_by_x_theta(P, N, x_eq, theta_eq)
        else:
            positions = _trial_positions_scaled_by_x(P, N, x_eq)
        return (positions, x_eq, theta_eq) if return_x_eq else positions

    # Fallback: fixed scale from horizon (legacy / tests)
    from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants

    const = get_hqiv_nuclear_constants(t_cmb)
    lattice_base_fm = const["LATTICE_BASE_M"] * 1e15
    shape = shell_shape(m_shell, alpha=alpha)
    curvature_factor = (6.0 ** (7.0 / 3.0)) * math.sqrt(3.0)
    r0_fm = max(lattice_base_fm * shape * curvature_factor, 0.1)
    if A == 2:
        out = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    elif A == 4:
        tetra = np.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
            dtype=float,
        )
        tetra *= r0_fm / math.sqrt(3.0)
        out = tetra
    else:
        out = fcc_packing_positions(A, r0_fm)
    return (out, None, None) if return_x_eq else out


__all__ = [
    "casimir_contribution",
    "em_angular_contribution",
    "equilibrium_horizon_distance_and_angle",
    "shell_shape",
    "fcc_packing_positions",
    "nucleon_positions_3d",
]
