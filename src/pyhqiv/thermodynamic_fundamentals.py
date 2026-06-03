"""
Thermodynamic fundamentals (HQIV emergent thermo proxies).

This module is an aggregation layer that exposes the thermodynamic content
already derived elsewhere in the rebuild:

- Microscopic 2nd-law arrow from discrete light-cone combinatorics:
  `S(m) ∝ cumLatticeSimplexCount(m)` and hence `ΔS(m) ∝ latticeSimplexCount(m) > 0`.

- Equilibrium vs non-equilibrium (Clausius equality proxy):
  in the homogeneous limit, local equilibrium corresponds to the compatibility
  constraint that the gravitational action vanishes:
    `S_HQVM_grav = 0  <->  HQVM Friedmann residual = 0`.

  This is the emergent-GR/thermo closure proxy in the current preliminary stack.

Notes
-----
This rebuild currently provides *computable proxies* rather than a full
δQ = ∫T_ab k^a k^b and area-law geometry implementation.
"""

from __future__ import annotations

import math

from pyhqiv.action import s_hqvm_grav, s_hqvm_grav_zero_holds
from pyhqiv.auxiliary_field import shell_temperature
from pyhqiv.gr_from_maxwell import hqvm_friedmann_power_residual
from pyhqiv.lightcone import cumulative_lattice_simplex_count, lattice_simplex_count
from pyhqiv.metric import g_eff, three_minus_gamma, hqvm_friedmann_residual


def horizon_entropy_counting(m: int, *, k_b: float = 1.0) -> float:
    """
    Discrete horizon entropy in counting units:
      S(m) = k_B * cumLatticeSimplexCount(m).
    """
    if m < 0:
        raise ValueError("m must be non-negative")
    return k_b * float(cumulative_lattice_simplex_count(m))


def entropy_increment_per_shell(m: int, *, k_b: float = 1.0) -> float:
    """
    Discrete increment:
      ΔS(m) = S(m) - S(m-1) = k_B * latticeSimplexCount(m).
    """
    if m <= 0:
        raise ValueError("need m>=1 for an increment")
    return k_b * float(lattice_simplex_count(m))


def temperature_at_shell(m: int) -> float:
    """
    Natural-unit horizon temperature ladder:
      T(m) = 1/(m+1)  (with T_Pl = 1).
    """
    return shell_temperature(m)


def second_law_arrow_holds(m_max: int) -> bool:
    """
    Micro 2nd law proxy:
    ΔS(m) > 0 for all integer shells in [1, m_max].
    """
    if m_max < 1:
        return True
    for m in range(1, m_max + 1):
        if entropy_increment_per_shell(m) <= 0.0:
            return False
    return True


def local_equilibrium_proxy(phi_auxiliary: float, rho_m: float, rho_r: float, *, atol: float = 1e-12) -> bool:
    """
    Local equilibrium proxy (Clausius equality equality case):
      S_HQVM_grav(phi, rho_m, rho_r) == 0.
    """
    return s_hqvm_grav_zero_holds(phi_auxiliary, rho_m, rho_r, atol=atol)


def entropy_production_proxy(phi_auxiliary: float, rho_m: float, rho_r: float) -> float:
    """
    Non-equilibrium proxy:
      entropy production strength ∝ |S_HQVM_grav| = |Friedmann residual|.
    """
    return abs(s_hqvm_grav(phi_auxiliary, rho_m, rho_r))


def equilibrium_rho_total_for_phi(phi_auxiliary: float) -> float:
    """
    Given phi, compute the equilibrium total density that makes the Friedmann residual vanish:
      rho_total = (3-gamma) * phi^2 / (8π G_eff(phi)).
    """
    return three_minus_gamma() * (phi_auxiliary**2) / (8.0 * math.pi * g_eff(phi_auxiliary))


def clausius_residual_proxy(phi_auxiliary: float, rho_m: float, rho_r: float) -> float:
    """
    A dimensionless-ish "residual" proxy for deviation from local equilibrium.
    In the preliminary stack, we use the magnitude of the Friedmann residual itself.
    """
    return abs(hqvm_friedmann_residual(phi_auxiliary, rho_m, rho_r))


def clausius_equality_proxy_power_form(phi_auxiliary: float, rho_m: float, rho_r: float) -> float:
    """
    Same deviation computed from the explicit power-form residual:
      |(13/5)φ^2 - 8π φ^α (rho_total)|.
    """
    return abs(hqvm_friedmann_power_residual(phi_auxiliary, rho_m, rho_r))


__all__ = [
    "entropy_increment_per_shell",
    "horizon_entropy_counting",
    "local_equilibrium_proxy",
    "second_law_arrow_holds",
    "temperature_at_shell",
    "entropy_production_proxy",
    "equilibrium_rho_total_for_phi",
    "clausius_residual_proxy",
    "clausius_equality_proxy_power_form",
]

