"""
Action layer for the fresh pyhqiv rebuild.

This module mirrors the Lean math in:

- `Hqiv/Physics/Action.lean`

In the preliminary rebuild, we implement the only fully-evaluable
non-placeholder content:

1. The gravitational action functional `S_HQVM_grav`.
2. The equivalence between `S_HQVM_grav = 0` and the homogeneous Friedmann
   equation residual being zero.

The full O-Maxwell action functional is kept at the "placeholder"
level (because the discrete manifold potentials are not yet represented
in Python).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pyhqiv.metric import g_eff, gamma_hqiv, hqvm_friedmann_holds, hqvm_friedmann_residual


def s_hqvm_grav(phi_auxiliary: float, rho_m: float, rho_r: float) -> float:
    """
    Gravitational action constraint form of Friedmann.

    Lean reference:
    `Hqiv.Physics.Action.S_HQVM_grav`
      (3 - gamma_HQIV) * phi^2 - 8π * G_eff(phi) * (rho_m + rho_r)
    """
    return (3.0 - gamma_hqiv()) * (phi_auxiliary**2) - 8.0 * math.pi * g_eff(phi_auxiliary) * (
        rho_m + rho_r
    )


def s_hqvm_grav_zero_holds(
    phi_auxiliary: float,
    rho_m: float,
    rho_r: float,
    *,
    atol: float = 1e-12,
) -> bool:
    """
    Numeric predicate corresponding to `S_HQVM_grav = 0`.
    """
    return abs(s_hqvm_grav(phi_auxiliary, rho_m, rho_r)) <= atol


def friedmann_from_action_equivalence_holds(
    phi_auxiliary: float,
    rho_m: float,
    rho_r: float,
    *,
    atol: float = 1e-12,
) -> bool:
    """
    Checks the Lean equivalence:
      S_HQVM_grav = 0  <->  HQVM_Friedmann_eq
    """
    left = s_hqvm_grav_zero_holds(phi_auxiliary, rho_m, rho_r, atol=atol)
    right = hqvm_friedmann_holds(phi_auxiliary, rho_m, rho_r, atol=atol)
    return (left and right) or ((not left) and (not right))


@dataclass(frozen=True)
class ActionSnapshot:
    phi_auxiliary: float
    rho_m: float
    rho_r: float
    s_grav: float
    friedmann_residual: float


def build_action_snapshot(phi_auxiliary: float, rho_m: float, rho_r: float) -> ActionSnapshot:
    """
    Evaluate the gravitational action constraint and the Friedmann residual.
    """
    sgrav = s_hqvm_grav(phi_auxiliary, rho_m, rho_r)
    residual = hqvm_friedmann_residual(phi_auxiliary, rho_m, rho_r)
    return ActionSnapshot(
        phi_auxiliary=phi_auxiliary,
        rho_m=rho_m,
        rho_r=rho_r,
        s_grav=sgrav,
        friedmann_residual=residual,
    )


__all__ = [
    "ActionSnapshot",
    "build_action_snapshot",
    "friedmann_from_action_equivalence_holds",
    "s_hqvm_grav",
    "s_hqvm_grav_zero_holds",
]

