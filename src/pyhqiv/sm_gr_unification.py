"""
SM–GR unification “now” witnesses (no beta-running).

This module ports only the *computable witness outputs* from:
  `HQIV_LEAN/Hqiv/Physics/SM_GR_Unification.lean`

without relying on the beta-running engine.

It currently exports:
- couplings at `M_Z`: 1/alpha_EM(M_Z), alpha_EM(M_Z), sin^2(theta_W)(M_Z), alpha_s(M_Z)
- mass/scale witnesses: M_Pl, M_Z, m_electron, m_proton, m_neutron

Notes
-----
The Lean file derives these numbers from the O-Maxwell + lattice pipeline.
In this preliminary Python rebuild we treat them as first-principles
*witnesses* aligned with Lean’s definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyhqiv.lean_witnesses import load_lean_witnesses

def alpha_gut() -> float:
    """
    alpha_GUT (Lean witness export).
    """
    w = load_lean_witnesses()
    return w.get_float("alpha_GUT")


def one_over_alpha_EM_at_MZ() -> float:
    """
    one_over_alpha_EM_at_MZ (Lean witness export).
    """
    w = load_lean_witnesses()
    return w.get_float("one_over_alpha_EM_at_MZ")


def alpha_EM_at_MZ() -> float:
    """
    alpha_EM_at_MZ = 1 / one_over_alpha_EM_at_MZ.
    """
    inv = one_over_alpha_EM_at_MZ()
    if inv == 0.0:
        raise ZeroDivisionError("one_over_alpha_EM_at_MZ is zero")
    return 1.0 / inv


def sin2thetaW_at_MZ() -> float:
    """
    sin2thetaW_at_MZ (Lean witness export).
    """
    w = load_lean_witnesses()
    return w.get_float("sin2thetaW_at_MZ")


def alpha_s_at_MZ() -> float:
    """
    alpha_s_at_MZ (Lean witness export).
    """
    w = load_lean_witnesses()
    return w.get_float("alpha_s_at_MZ")


def M_Pl_natural() -> float:
    """
    Planck mass in natural Planck units: M_Pl_natural = 1.
    """
    return 1.0


def M_Z_natural() -> float:
    """
    M_Z_natural = M_Z_GeV / M_Pl_GeV (Lean witness export).
    """
    w = load_lean_witnesses()
    return w.get_float("M_Z_GeV") / w.get_float("M_Pl_GeV")


def m_electron_natural() -> float:
    """
    m_electron_natural = m_electron_GeV / M_Pl_GeV (Lean witness export).
    """
    w = load_lean_witnesses()
    m_electron_GeV = (
        w.get_float("m_electron_MeV") * w.get_float("MEV_TO_EV") / w.get_float("GEV_TO_EV")
    )
    return m_electron_GeV / w.get_float("M_Pl_GeV")


def m_proton_MeV_central() -> float:
    """
    m_proton_MeV_central (Lean witness export).
    """
    w = load_lean_witnesses()
    return w.get_float("derivedProtonMass_MeV")


def m_neutron_MeV_central() -> float:
    """
    m_neutron_MeV_central (Lean witness export).
    """
    w = load_lean_witnesses()
    return w.get_float("derivedNeutronMass_MeV")


@dataclass(frozen=True)
class SMConstantsAtNow:
    """
    Minimal bundle of SM–GR “now” witnesses.
    """

    alpha_GUT: float
    one_over_alpha_EM: float
    alpha_EM: float
    sin2thetaW: float
    alpha_s: float
    M_Pl: float
    M_Z: float
    m_electron_natural: float
    m_proton_MeV: float
    m_neutron_MeV: float


def sm_constants_at_now() -> SMConstantsAtNow:
    """
    Bundle the Lean witness outputs.
    """
    return SMConstantsAtNow(
        alpha_GUT=alpha_gut(),
        one_over_alpha_EM=one_over_alpha_EM_at_MZ(),
        alpha_EM=alpha_EM_at_MZ(),
        sin2thetaW=sin2thetaW_at_MZ(),
        alpha_s=alpha_s_at_MZ(),
        M_Pl=M_Pl_natural(),
        M_Z=M_Z_natural(),
        m_electron_natural=m_electron_natural(),
        m_proton_MeV=m_proton_MeV_central(),
        m_neutron_MeV=m_neutron_MeV_central(),
    )


__all__ = [
    "SMConstantsAtNow",
    "alpha_EM_at_MZ",
    "alpha_s_at_MZ",
    "alpha_gut",
    "m_electron_natural",
    "m_neutron_MeV_central",
    "m_proton_MeV_central",
    "M_Pl_natural",
    "M_Z_natural",
    "one_over_alpha_EM_at_MZ",
    "sm_constants_at_now",
    "sin2thetaW_at_MZ",
]

