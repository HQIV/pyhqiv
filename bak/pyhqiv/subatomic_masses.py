"""
First-principles quark mass ladder from HQIV axioms.

This module implements the exact Lean-translated mass ladder described in
`Hqiv/Physics/SM_GR_Unification.lean`:

- Auxiliary field:      φ(m)          = 2(m + 1)
- Lattice multiplicity: lattice(m)    = 4(m + 2)(m + 1)
- Shell shape:          shell_shape(m) = [1 + (3/5) ln(m + 1)] / (m + 1)
- Mass factor:          mass_factor(m) = φ(m) × lattice(m) × shell_shape(m)
- Proton normalisation: norm = m_p / mass_factor(4)
- Effective mass:       m_eff(m)      = mass_factor(m) × norm

Here `m` is the shell index in the discrete null lattice; referenceM = 4 is
the QCD lock-in cluster used in the paper and Lean proofs. The single
laboratory anchor is the proton mass m_p ≈ 938.272 MeV at "now".

Quark masses are then assigned by mapping each flavour to a shell m_q via the
octonionic / triality geometry (three generations, six quarks). At this
stage we keep the projection factors proj_q = 1.0 as placeholders; in the
full HQIV picture they are inner products between the quark octonion
generator and the auxiliary field at that horizon and can be computed from
`OctonionHQIVAlgebra` closure data.

This entire module is **PDG-free**: the only reference scale is m_p from the
HQIV ladder itself; PDG values remain confined to the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from pyhqiv.constants import M_PROTON_MEV


def _phi(m: float) -> float:
    """Auxiliary-field VEV φ(m) = 2(m+1) (natural units)."""
    return 2.0 * (m + 1.0)


def _lattice(m: float) -> float:
    """Stars-and-bars multiplicity 4(m+2)(m+1) (octonionic light-cone factor)."""
    return 4.0 * (m + 2.0) * (m + 1.0)


def _shell_shape(m: float) -> float:
    """
    shell_shape(m) = [1 + (3/5) ln(m+1)] / (m+1)

    with α = 3/5 derived from the lattice (hockey-stick identity in Lean).
    """
    mp1 = m + 1.0
    if mp1 <= 0.0:
        mp1 = 1.0
    return (1.0 + (3.0 / 5.0) * float(np.log(mp1))) / mp1


def mass_factor(m: float) -> float:
    """
    Dimensionless mass factor at shell m from φ, lattice, and shell_shape.
    """
    return _phi(m) * _lattice(m) * _shell_shape(m)


_REF_SHELL: float = 4.0  # referenceM = 4 (QCD overlap cluster)
_REF_MASS_FACTOR: float = mass_factor(_REF_SHELL)
_NORM: float = M_PROTON_MEV / _REF_MASS_FACTOR


def m_eff_shell(m: float) -> float:
    """
    Effective/constituent mass scale at shell m (MeV).

    This is the direct Lean translation:
        m_eff(m) = mass_factor(m) × norm
    where norm is chosen so that m_eff(4) = m_p (the proton witness).
    """
    return mass_factor(m) * _NORM


@dataclass(frozen=True)
class QuarkMassLadder:
    """
    First-principles quark mass ladder at "now".

    Attributes
    ----------
    shell_index : dict
        q -> m_q shell assignment in the discrete lattice.
    projection : dict
        q -> dimensionless geometric projection factor (octonion overlap).
        Currently set to 1.0 for all flavours as a placeholder; in the full
        HQIV picture these are computed from OctonionHQIVAlgebra closure
        data (inner products with the auxiliary field).
    masses_mev : dict
        q -> m_q (MeV) given by projection[q] * m_eff_shell(shell_index[q]).
    """

    shell_index: Dict[str, float]
    projection: Dict[str, float]
    masses_mev: Dict[str, float]


def build_quark_mass_ladder() -> QuarkMassLadder:
    """
    Construct the six-quark mass ladder from the shell-based mass_factor.

    Current minimal shell assignment (can be refined from triality geometry):

      - u, d: m = 2  (light cluster below referenceM)
      - s:    m = 3
      - c:    m = 5
      - b:    m = 8
      - t:    m = 55 (far horizon; matches top scale in the sandbox ladder)

    All projection factors are set to 1.0; code that needs refined values
    can override them using the same API.
    """
    shell_index: Dict[str, float] = {
        "u": 2.0,
        "d": 2.0,
        "s": 3.0,
        "c": 5.0,
        "b": 8.0,
        "t": 55.0,
    }
    projection: Dict[str, float] = {q: 1.0 for q in shell_index}
    masses_mev: Dict[str, float] = {
        q: projection[q] * m_eff_shell(shell_index[q]) for q in shell_index
    }
    return QuarkMassLadder(shell_index=shell_index, projection=projection, masses_mev=masses_mev)


# Default ladder at "now" (proton-anchored).
QUARK_MASS_LADDER_NOW: QuarkMassLadder = build_quark_mass_ladder()


__all__ = [
    "mass_factor",
    "m_eff_shell",
    "QuarkMassLadder",
    "build_quark_mass_ladder",
    "QUARK_MASS_LADDER_NOW",
]

