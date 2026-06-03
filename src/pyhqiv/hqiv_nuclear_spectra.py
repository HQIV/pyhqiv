"""
Nuclear / atomic spectral scaffolding packaged from existing HQIV geometry + SM witnesses.

Lean source:
  `HQIV_LEAN/Hqiv/Physics/NuclearAndAtomicSpectra.lean`

Implements ``R_m``, ``modes`` (alias of ``available_modes``), ``mu_neutron``, ``V_mag``,
``V_nuclear``, and beta half-life scalars **as definitional combinations** — same
re-exports as Lean (γ from metric, α_EM from SM–GR witnesses, modes from light cone).
"""

from __future__ import annotations

import math

from pyhqiv.hqiv_bound_states import NetworkWeight28, e_bind_nuclear_from_network
from pyhqiv.lightcone import available_modes, new_modes
from pyhqiv.metric import gamma_hqiv
from pyhqiv.sm_gr_unification import alpha_EM_at_MZ


def R_m(m: int) -> float:
    """Lean: ``R_m m = (m : ℝ) + 1`` (horizon shell radius index)."""
    if m < 0:
        raise ValueError("shell index m must be non-negative")
    return float(m) + 1.0


def modes(m: int) -> float:
    """Lean: ``modes m = Hqiv.available_modes m``."""
    return float(available_modes(m))


def single_nucleon_caustic_mode_identity(m: int) -> tuple[float, float]:
    """
    Lean: ``single_nucleon_caustic`` — numerically check ``modes m = 4(m+2)(m+1)`` and ``R_m = m+1``.

    Returns (modes_m, R_m).
    """
    mm = modes(m)
    rm = R_m(m)
    algebraic = 4.0 * (float(m) + 2.0) * (float(m) + 1.0)
    if not math.isclose(mm, algebraic, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("available_modes(m) inconsistent with Lean closed form")
    return mm, rm


def barbell_ring_new_modes_identity(m: int) -> float:
    """
    Lean: ``barbell_ring_caustic`` — ``new_modes (m+1) = 8 * (m+2)``.

    Returns the Python ``new_modes(m+1)`` value after checking the identity.
    """
    lhs = new_modes(m + 1)
    rhs = 8.0 * (float(m) + 2.0)
    if not math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("new_modes(m+1) inconsistent with Lean barbell identity")
    return lhs


def mu_neutron() -> float:
    """
    Lean: ``mu_neutron = -gamma_HQIV * m_proton / (2 * m_proton)``.

    The proton mass witness cancels; numerically this is ``-gamma_HQIV / 2`` (same as Lean
    after ``field_simp`` / definitional equality), so no witness JSON is required here.
    """
    return -gamma_hqiv() / 2.0


def V_mag(
    mu_i: float,
    mu_j: float,
    r: float,
    s_para_i: float,
    s_para_j: float,
    s_dot: float,
    delta_phi_mag: float,
) -> float:
    """Lean: ``V_mag`` — dipole–dipole scalar + screening ``Δφ_mag``."""
    if r == 0.0:
        raise ValueError("separation r must be non-zero")
    r3 = r**3
    return -(mu_i * mu_j / r3) * (3.0 * (s_para_i * s_para_j) - s_dot) + delta_phi_mag


def V_nuclear(
    m: int,
    z_eff: float,
    mu_i: float,
    mu_j: float,
    r: float,
    s_para_i: float,
    s_para_j: float,
    s_dot: float,
    delta_phi_mag: float,
) -> float:
    """
    Lean: ``V_nuclear m Z_eff ... = -γ * modes m / R_m m + α_EM * Z_eff / r + V_mag``.
    """
    horizon = -gamma_hqiv() * modes(m) / R_m(m)
    coulomb = alpha_EM_at_MZ() * float(z_eff) / float(r)
    return horizon + coulomb + V_mag(mu_i, mu_j, r, s_para_i, s_para_j, s_dot, delta_phi_mag)


def e_bind_nuclear_shell(m: int, weights: NetworkWeight28, c: float = 1.0) -> float:
    """Lean: ``E_bind_nuclear_shell`` alias for ``E_bind_nuclear_from_network``."""
    return e_bind_nuclear_from_network(m, weights, c)


def beta_decay_rate_scalar(
    _particle_tag: str,
    m_e: float,
    matrix_element_M: float,
    *,
    g_fermi: float,
) -> float:
    """
    Lean: ``beta_decay_rate _ m_e ℳ = G_F² * m_e⁵ * ℳ²`` (width scaffold).

    ``g_fermi`` is Fermi's constant in the **same unit system** as ``m_e`` and ``ℳ``
    (Lean leaves ``G_F`` abstract; nothing is hardcoded here).
    """
    return beta_decay_rate_with_gf(g_fermi, m_e, matrix_element_M)


def beta_decay_rate_with_gf(g_fermi: float, m_e: float, matrix_element_M: float) -> float:
    """Scalar width ``G_F² m_e⁵ |ℳ|²`` (Lean ``beta_decay_rate`` shape)."""
    gf = float(g_fermi)
    me = float(m_e)
    mm = float(matrix_element_M)
    return (gf**2) * (me**5) * (mm**2)


def half_life_from_width(width: float) -> float:
    """Lean: ``half_life_from_width Γ = log 2 / Γ`` (same units as 1/width)."""
    g = float(width)
    if g <= 0.0:
        raise ValueError("width must be positive")
    return math.log(2.0) / g


def gamma_transition_energy(e_i: float, e_f: float, v_mag_corr: float) -> float:
    """Lean: ``gamma_transition_energy E_i E_f V_mag_corr = (E_i - E_f) + V_mag_corr``."""
    return (float(e_i) - float(e_f)) + float(v_mag_corr)


def transition_frequency(e_i: float, e_f: float, v_mag_corr: float, h_planck: float) -> float:
    """Lean: ``transition_frequency`` = ((E_i - E_f) + V_mag_corr) / h."""
    return gamma_transition_energy(e_i, e_f, v_mag_corr) / float(h_planck)


__all__ = [
    "R_m",
    "barbell_ring_new_modes_identity",
    "beta_decay_rate_scalar",
    "beta_decay_rate_with_gf",
    "e_bind_nuclear_shell",
    "gamma_transition_energy",
    "half_life_from_width",
    "modes",
    "mu_neutron",
    "single_nucleon_caustic_mode_identity",
    "transition_frequency",
    "V_mag",
    "V_nuclear",
]
