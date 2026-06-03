"""
O-Maxwell derived couplings (no beta running).

This ports the Lean *φ-correction* part of `Hqiv.Physics.SM_GR_Unification.lean`
(“Effective EM coupling from O-Maxwell (no beta function)”).

In HQIV:
  1/alpha_eff(φ) = (1/alpha_GUT) * (1 + c * alpha * log(φ + 1))
and shell Coulomb strength is identified with alpha_eff at that shell.
"""

from __future__ import annotations

import math

from pyhqiv.auxiliary_field import phi_of_shell
from pyhqiv.lightcone import alpha, cube_directions, octonion_imaginary_dim


def alpha_gut() -> float:
    """
    alpha_GUT = 1 / 42 derived from the structural product 6*7.

    Lean reference:
    `alpha_GUT = 1/42` (derived from cubeDirections × octonionImaginaryDim).
    """
    return 1.0 / float(cube_directions() * octonion_imaginary_dim())


def one_over_alpha_eff(phi: float, c: float = 1.0) -> float:
    """
    Effective inverse fine-structure constant from O-Maxwell φ-correction.

    Lean reference:
    `one_over_alpha_eff φ c = (1/alpha_GUT) * (1 + c*alpha*log(φ+1))`
    """
    if phi <= -1.0:
        raise ValueError("phi must satisfy phi > -1 so that log(phi+1) is defined")
    return (1.0 / alpha_gut()) * (1.0 + c * alpha() * math.log(phi + 1.0))


def alpha_eff_from_phi(phi: float, c: float = 1.0) -> float:
    """
    alpha_eff(φ) = 1 / one_over_alpha_eff(φ).
    """
    inv = one_over_alpha_eff(phi, c=c)
    if inv == 0.0:
        raise ZeroDivisionError("one_over_alpha_eff evaluated to 0")
    return 1.0 / inv


def alpha_eff_at_shell(m: int, c: float = 1.0) -> float:
    """
    alpha_eff(m) from the shell auxiliary field value phi_of_shell(m).
    """
    return alpha_eff_from_phi(phi_of_shell(m), c=c)


def coulomb_strength_shell(m: int, c: float = 1.0) -> float:
    """
    Coulomb strength at shell m, identified with alpha_eff(m) in the Lean scaffold.
    """
    return alpha_eff_at_shell(m, c=c)


def expected_ground_energy_at_shell(m: int, Z: int, mu: float, c: float = 1.0) -> float:
    """
    Shell-resolved ground-state energy magnitude form from BoundStates.lean:

      E = - mu * Z^2 * (alpha_eff(m))^2 / 2
    """
    a_eff = alpha_eff_at_shell(m, c=c)
    return -mu * float(Z) ** 2 * (a_eff**2) / 2.0


__all__ = [
    "alpha_eff_at_shell",
    "alpha_eff_from_phi",
    "alpha_gut",
    "coulomb_strength_shell",
    "expected_ground_energy_at_shell",
    "one_over_alpha_eff",
]

