"""
Spin-statistics layer for the fresh pyhqiv rebuild.

This mirrors the key computable content of:
- `HQIV_LEAN/Hqiv/Physics/SpinStatistics.lean`

Lean’s abstract chain derives the practical sign rule:
  - half-integer spin modes: identical exchange phase = -1
  - integer spin modes: identical exchange phase = +1

It also includes the standard energy-time uncertainty based lifetime:
  tau = ħ / ΔE  and  t_{1/2} = ln(2) * tau
where ΔE is in MeV.
"""

from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass


class SpinClass(str, Enum):
    INTEGER = "integer"
    HALF_INTEGER = "halfInteger"


def two_pi_phase(spin_class: SpinClass) -> complex:
    """
    2π-rotation phase:
      - integer: +1
      - half-integer: -1
    """
    if spin_class == SpinClass.INTEGER:
        return 1.0 + 0.0j
    if spin_class == SpinClass.HALF_INTEGER:
        return -1.0 + 0.0j
    raise ValueError(f"unknown spin class: {spin_class}")


def exchange_phase_identical(spin_class: SpinClass) -> complex:
    """
    Spin-statistics sign rule for identical modes:
      exchangePhase(m,m) = twoPiPhase(m)
    """
    return two_pi_phase(spin_class)


def spin_statistics_exchange_sign_matches_two_pi(spin_class: SpinClass, exchange_phase: complex) -> bool:
    """
    Convenience predicate checking:
      exchange_phase == two_pi_phase(spin_class)
    """
    return exchange_phase == two_pi_phase(spin_class)


# -----------------------------
# Resonance lifetime utilities
# -----------------------------

def hbar_MeV_s() -> float:
    """
    Reduced Planck constant in MeV·s.

    Value from Lean witness (no literal in this .py source).
    """
    from pyhqiv.lean_witnesses import load_lean_witnesses

    return load_lean_witnesses().get_float("hbar_MeV_s")


def resonance_lifetime(delta_E_mev: float) -> float:
    """
    Mean lifetime τ = ħ / ΔE with ΔE in MeV.
    """
    if delta_E_mev == 0.0:
        raise ValueError("delta_E_mev must be non-zero")
    return hbar_MeV_s() / delta_E_mev


def resonance_half_life(delta_E_mev: float) -> float:
    """
    Half-life t_{1/2} = ln(2) * τ.
    """
    return math.log(2.0) * resonance_lifetime(delta_E_mev)


@dataclass(frozen=True)
class ResonanceTimes:
    delta_E_mev: float
    lifetime_s: float
    half_life_s: float


def build_resonance_times(delta_E_mev: float) -> ResonanceTimes:
    """
    Build a small bundle of lifetime quantities.
    """
    tau = resonance_lifetime(delta_E_mev)
    return ResonanceTimes(
        delta_E_mev=delta_E_mev,
        lifetime_s=tau,
        half_life_s=math.log(2.0) * tau,
    )


__all__ = [
    "ResonanceTimes",
    "SpinClass",
    "build_resonance_times",
    "exchange_phase_identical",
    "hbar_MeV_s",
    "resonance_half_life",
    "resonance_lifetime",
    "spin_statistics_exchange_sign_matches_two_pi",
    "two_pi_phase",
]

