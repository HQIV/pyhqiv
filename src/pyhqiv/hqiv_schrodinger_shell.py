"""
Shell-dependent effective couplings from the HQIV Schrödinger / O-Maxwell layer.

Lean source:
  `HQIV_LEAN/Hqiv/QuantumMechanics/Schrodinger.lean`
    `oneOverAlphaEffShell`, `alphaEffShell`, `coulombStrengthShell`
  `HQIV_LEAN/Hqiv/Physics/SM_GR_Unification.lean`
    `one_over_alpha_eff`, `alpha_GUT`, lattice `alpha`

All numerics for α_GUT use the Lean witness export; α = 3/5 and φ(m) use the same
Python definitions as `lightcone` / `auxiliary_field` (Lean-derived, not fitted).
"""

from __future__ import annotations

import math

from pyhqiv import auxiliary_field as aux
from pyhqiv import lightcone
from pyhqiv.sm_gr_unification import alpha_gut


def one_over_alpha_eff(phi: float, c: float = 1.0) -> float:
    """
    Lean: ``Hqiv.one_over_alpha_eff φ c = (1/α_GUT) * (1 + c * α * log(φ+1))``.

    Parameters
    ----------
    phi
        Auxiliary field value (e.g. ``phi_of_shell m``).
    c
        Fano-plane normalisation (Lean default 1).
    """
    if phi <= -1.0:
        raise ValueError("phi must satisfy phi > -1 for log(phi+1)")
    ag = alpha_gut()
    if ag == 0.0:
        raise ZeroDivisionError("alpha_GUT witness is zero")
    a = lightcone.alpha()
    return (1.0 / ag) * (1.0 + c * a * math.log(phi + 1.0))


def one_over_alpha_eff_shell(m: int, c: float = 1.0) -> float:
    """Lean: ``oneOverAlphaEffShell m c = one_over_alpha_eff (phi_of_shell m) c``."""
    phi = aux.phi_of_shell(m)
    return one_over_alpha_eff(phi, c)


def alpha_eff_shell(m: int, c: float = 1.0) -> float:
    """Lean: ``alphaEffShell m c = (oneOverAlphaEffShell m c)⁻¹``."""
    inv = one_over_alpha_eff_shell(m, c)
    if inv == 0.0:
        raise ZeroDivisionError("one_over_alpha_eff_shell returned zero")
    return 1.0 / inv


def coulomb_strength_shell(m: int, c: float = 1.0) -> float:
    """Lean: ``coulombStrengthShell m c = alphaEffShell m c`` (natural units)."""
    return alpha_eff_shell(m, c)


__all__ = [
    "alpha_eff_shell",
    "coulomb_strength_shell",
    "one_over_alpha_eff",
    "one_over_alpha_eff_shell",
]
