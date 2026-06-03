"""
GR-from-Maxwell layer for the fresh pyhqiv rebuild.

This module mirrors the Lean correspondence in:

- `Hqiv/Physics/GRFromMaxwell.lean`

In the preliminary Python rebuild, the "Maxwell -> GR" correspondence
collapses (in the homogeneous limit) to the statement that the same
φ and α determine:

  (3 - gamma) φ^2 = 8π * G_eff(φ) * (rho_m + rho_r)

and equivalently the power form:

  (13/5) φ^2 = 8π * (φ^α) * (rho_m + rho_r)
"""

from __future__ import annotations

import math

from pyhqiv.lightcone import alpha
from pyhqiv.metric import hqvm_friedmann_residual


def o_maxwell_determines_hqvm_gr_homogeneous_equivalence(
    phi_auxiliary: float,
    rho_m: float,
    rho_r: float,
    *,
    atol: float = 1e-12,
) -> bool:
    """
    Numeric analogue of the Lean theorem
    `O_Maxwell_determines_HQVM_GR_homogeneous`.

    Checks the residual computed using the metric layer.
    """
    residual = hqvm_friedmann_residual(phi_auxiliary, rho_m, rho_r)
    return abs(residual) <= atol


def hqvm_friedmann_power_residual(
    phi_auxiliary: float,
    rho_m: float,
    rho_r: float,
) -> float:
    """
    Residual of the explicit power-form Friedmann equation:

      (13/5) φ^2 - 8π φ^α (rho_m + rho_r)
    """
    return (13.0 / 5.0) * (phi_auxiliary**2) - 8.0 * math.pi * (phi_auxiliary**alpha()) * (
        rho_m + rho_r
    )


__all__ = [
    "hqvm_friedmann_power_residual",
    "o_maxwell_determines_hqvm_gr_homogeneous_equivalence",
]

