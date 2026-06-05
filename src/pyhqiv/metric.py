"""
HQIV metric and lapse layer for the fresh pyhqiv rebuild.

This module mirrors:

- `HQIV_LEAN/Hqiv/Geometry/HQVMetric.lean`

It is the next step above the light-cone and auxiliary-field foundations:
the ADM lapse, time-angle, metric coefficients, gamma split, varying-G, and
the homogeneous Friedmann relation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pyhqiv.lightcone import alpha


def hqvm_lapse(phi_newtonian: float, phi_auxiliary: float, t: float) -> float:
    """
    ADM lapse N = 1 + Phi + phi * t.

    Lean reference:
    `Hqiv.Geometry.HQVMetric.HQVM_lapse`
    """
    return 1.0 + phi_newtonian + phi_auxiliary * t


def time_angle(phi_auxiliary: float, t: float) -> float:
    """
    Horizon time-angle delta theta' = phi * t.

    Lean reference:
    `Hqiv.Geometry.HQVMetric.timeAngle`
    """
    return phi_auxiliary * t


def two_pi() -> float:
    """
    One full phase period.

    Lean reference:
    `Hqiv.Geometry.HQVMetric.twoPi`
    """
    return 2.0 * math.pi


def hqvm_g_tt(lapse: float) -> float:
    """
    Time-time metric coefficient g_tt = -N^2.

    Lean reference:
    `Hqiv.Geometry.HQVMetric.HQVM_g_tt`
    """
    return -(lapse**2)


def hqvm_spatial_coeff(scale_factor: float, phi_newtonian: float) -> float:
    """
    Spatial conformal coefficient a^2 (1 - 2 Phi).

    Lean reference:
    `Hqiv.Geometry.HQVMetric.HQVM_spatial_coeff`
    """
    return (scale_factor**2) * (1.0 - 2.0 * phi_newtonian)


def gamma_hqiv() -> float:
    """
    Entanglement-monogamy complement of alpha: gamma = 1 - alpha.

    Lean reference:
    `Hqiv.Geometry.HQVMetric.gamma_HQIV`
    """
    return 1.0 - alpha()


def g0() -> float:
    """
    Reference Newton coupling in natural units.

    Lean reference:
    `Hqiv.Geometry.HQVMetric.G0`
    """
    return 1.0


def h0() -> float:
    """
    Reference Hubble scale in natural units.

    Lean reference:
    `Hqiv.Geometry.HQVMetric.H0`
    """
    return 1.0


def h_of_phi(phi_auxiliary: float) -> float:
    """
    Homogeneous-limit identification H(phi) = phi.

    Lean reference:
    `Hqiv.Geometry.HQVMetric.H_of_phi`
    """
    return phi_auxiliary


def g_eff(phi_auxiliary: float) -> float:
    """
    Effective Newton coupling G_eff(phi) = G0 * (H/H0)^alpha.

    Lean reference:
    `Hqiv.Geometry.HQVMetric.G_eff`
    """
    h_ratio = h_of_phi(phi_auxiliary) / h0()
    return g0() * (h_ratio**alpha())


def rho_total(rho_m: float, rho_r: float) -> float:
    """
    Homogeneous matter+radiation density.

    Lean reference:
    `Hqiv.Geometry.HQVMetric.rho_total`
    """
    return rho_m + rho_r


def three_minus_gamma() -> float:
    """
    Friedmann prefactor 3 - gamma.
    """
    return 3.0 - gamma_hqiv()


def friedmann_lhs(phi_auxiliary: float) -> float:
    """
    Left-hand side of the homogeneous HQVM Friedmann relation.
    """
    return three_minus_gamma() * (h_of_phi(phi_auxiliary) ** 2)


def friedmann_rhs(phi_auxiliary: float, rho_m: float, rho_r: float) -> float:
    """
    Right-hand side of the homogeneous HQVM Friedmann relation.
    """
    return 8.0 * math.pi * g_eff(phi_auxiliary) * rho_total(rho_m, rho_r)


def hqvm_friedmann_residual(phi_auxiliary: float, rho_m: float, rho_r: float) -> float:
    """
    Signed residual of the homogeneous HQVM Friedmann equation.

    Lean reference:
    `Hqiv.Geometry.HQVMetric.HQVM_Friedmann_eq`
    """
    return friedmann_lhs(phi_auxiliary) - friedmann_rhs(phi_auxiliary, rho_m, rho_r)


def hqvm_friedmann_holds(
    phi_auxiliary: float,
    rho_m: float,
    rho_r: float,
    *,
    atol: float = 1e-12,
) -> bool:
    """
    Numerical predicate for the homogeneous HQVM Friedmann equation.
    """
    return abs(hqvm_friedmann_residual(phi_auxiliary, rho_m, rho_r)) <= atol


@dataclass(frozen=True)
class HQVMMetricSnapshot:
    """
    Convenience bundle for the metric evaluated at one point.
    """

    phi_newtonian: float
    phi_auxiliary: float
    t: float
    lapse: float
    time_angle_value: float
    g_tt: float


def build_metric_snapshot(phi_newtonian: float, phi_auxiliary: float, t: float) -> HQVMMetricSnapshot:
    """
    Evaluate the core lapse/metric quantities together.
    """
    lapse = hqvm_lapse(phi_newtonian, phi_auxiliary, t)
    return HQVMMetricSnapshot(
        phi_newtonian=phi_newtonian,
        phi_auxiliary=phi_auxiliary,
        t=t,
        lapse=lapse,
        time_angle_value=time_angle(phi_auxiliary, t),
        g_tt=hqvm_g_tt(lapse),
    )


__all__ = [
    "HQVMMetricSnapshot",
    "build_metric_snapshot",
    "friedmann_lhs",
    "friedmann_rhs",
    "g0",
    "g_eff",
    "gamma_hqiv",
    "h0",
    "h_of_phi",
    "hqvm_friedmann_holds",
    "hqvm_friedmann_residual",
    "hqvm_g_tt",
    "hqvm_lapse",
    "hqvm_spatial_coeff",
    "rho_total",
    "three_minus_gamma",
    "time_angle",
    "two_pi",
]
