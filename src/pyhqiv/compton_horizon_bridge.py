"""
Compton clock ↔ horizon quarter-angle (π/2) in the HQIV **time-warping** picture.

Lean proves ``horizonQuarterPeriod = twoPi / 4 = π/2`` as the natural quarter-turn of
the horizon phase cycle (``ModifiedMaxwell.lean``: ``horizonQuarterPeriod_eq_pi_div_two``).
That is a **pure angle in the time-angle / lapse** sector — aligned with “warping is
time, not space” at the level of the HQVM phase story.

A massive particle has **intrinsic** Compton angular frequency ``ω = m c² / ħ`` (rest
frequency). A **quarter period** of that oscillation in **time** is

    ``Δt_{π/2} = (π/2) / ω``,

i.e. exactly **one radian-quarter of phase** at the Compton rate. Identifying the
**same** ``π/2`` with ``horizon_quarter_period()`` links:

* **Horizon phase** quarter-turn (rad), and
* **Rest-mass clock** quarter-cycle (s),

without introducing spatial curvature — only **time** / phase scaling.

This module computes ``ω`` and ``Δt_{π/2}`` from mass in MeV. It does **not** prove the
identification from axioms; it implements the **scaling check** you asked for.
"""

from __future__ import annotations

import math

from pyhqiv.forces import hbar_si
from pyhqiv.modified_maxwell import horizon_quarter_period

# 1 MeV in joules (exact, SI 2019).
from pyhqiv.scale_witness import load_local_conditions as _lc
MEV_TO_J = float(_lc()["compton_MeV_to_J"])


def horizon_quarter_angle_rad() -> float:
    """``twoPi/4 = π/2`` — same as ``ModifiedMaxwell.horizonQuarterPeriod``."""
    return horizon_quarter_period()


def compton_angular_frequency_rad_s(m_mev: float) -> float:
    """``ω = E/ħ`` with ``E = m c²`` and ``m`` given as mass in MeV."""
    if m_mev <= 0.0:
        raise ValueError("mass must be positive")
    e_j = float(m_mev) * MEV_TO_J
    return e_j / hbar_si()


def compton_frequency_hz(m_mev: float) -> float:
    """Ordinary frequency ``f = ω / (2π)``."""
    return compton_angular_frequency_rad_s(m_mev) / (2.0 * math.pi)


def compton_quarter_period_seconds(m_mev: float) -> float:
    """
    Time to accumulate **π/2** radians of phase at the Compton angular frequency:

        ``Δt = horizon_quarter_angle / ω = (π/2) / ω``.

    This is the direct **π/2 ↔ Compton** time scale (light-speed / rest-clock version).
    """
    omega = compton_angular_frequency_rad_s(m_mev)
    return horizon_quarter_angle_rad() / omega


def compton_quarter_period_s_from_pdg_electron() -> float:
    """Convenience: electron mass in MeV from PDG central (``lepton_resonance_ladder``)."""
    from pyhqiv.lepton_resonance_ladder import PDG_ELECTRON_MEV

    return compton_quarter_period_seconds(PDG_ELECTRON_MEV)


__all__ = [
    "MEV_TO_J",
    "compton_angular_frequency_rad_s",
    "compton_frequency_hz",
    "compton_quarter_period_seconds",
    "compton_quarter_period_s_from_pdg_electron",
    "horizon_quarter_angle_rad",
]
