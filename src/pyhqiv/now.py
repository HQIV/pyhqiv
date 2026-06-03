"""
Preliminary present-day ("now") module for the fresh pyhqiv rebuild.

This module mirrors the current Lean path:

- `HQIV_LEAN/Hqiv/Geometry/Now.lean`
- `HQIV_LEAN/Hqiv/Geometry/UniverseAge.lean`

The module intentionally separates:

1. the framework-natural definition of "now" (`H = H0 = 1` in natural units), and
2. an observational witness shell inferred from a supplied temperature.

That matches the Lean distinction between the dynamical `phi = 1` present slice
and the separate shell ladder used to map a temperature witness to a real shell.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyhqiv.auxiliary_field import phi_of_real_shell, shell_temperature
from pyhqiv.lightcone import curvature_integral, omega_k_partial, qcd_shell, reference_m
from pyhqiv.metric import h0


def h0_ref() -> float:
    """
    Reference Hubble value at the natural present slice.

    Lean reference:
    `Hqiv.Geometry.Now.H0_ref`
    """
    return h0()


def now_condition(phi_value: float, *, atol: float = 0.0) -> bool:
    """
    Framework-natural "now" condition: phi = H0 = 1.

    Lean reference:
    `Hqiv.Geometry.Now.nowCondition`
    """
    return abs(phi_value - h0_ref()) <= atol


def shell_index_for_temperature(temperature_natural: float) -> float:
    """
    Real shell index associated with a temperature witness.

    Lean reference:
    `Hqiv.Geometry.Now.shellIndexForTemperature`
    """
    if temperature_natural <= 0.0:
        raise ValueError("temperature witness must be positive")
    return (1.0 / temperature_natural) - 1.0


def temperature_from_shell_index(shell_index_real: float) -> float:
    """
    Inverse of `shell_index_for_temperature` on the real ladder.

    Lean reference:
    `Hqiv.Geometry.Now.shellIndexForTemperature_inv`
    """
    if shell_index_real <= -1.0:
        raise ValueError("real shell index must satisfy s > -1")
    return 1.0 / (shell_index_real + 1.0)


def apparent_age(t: float) -> float:
    """
    Homogeneous-limit apparent age.

    Lean reference:
    `Hqiv.Geometry.UniverseAge.apparentAge`
    """
    return t


def wall_clock_age_homogeneous(phi_value: float, t: float) -> float:
    """
    Homogeneous-limit wall-clock age t + phi * t^2 / 2.

    Lean reference:
    `Hqiv.Geometry.UniverseAge.wallClockAgeHomogeneous`
    """
    return t + phi_value * (t**2) / 2.0


def age_ratio_homogeneous(phi_value: float, t: float) -> float:
    """
    Homogeneous-limit wall/apparent age ratio.

    Lean reference:
    `Hqiv.Geometry.UniverseAge.ageRatioHomogeneous`
    """
    if t == 0.0:
        raise ValueError("age ratio is undefined at t = 0")
    return wall_clock_age_homogeneous(phi_value, t) / apparent_age(t)


def age_ratio_at_now(t: float) -> float:
    """
    Homogeneous-limit age ratio on the natural present slice.

    Lean reference:
    `Hqiv.Geometry.UniverseAge.ageRatio_at_now`
    """
    return age_ratio_homogeneous(h0_ref(), t)


@dataclass(frozen=True)
class TemperatureWitness:
    """
    Observational witness mapped onto the real shell ladder.

    The witness is not the definition of "now"; it is an externally supplied
    temperature that can be located on the shell ladder and compared to the
    framework-natural present slice.
    """

    temperature_natural: float
    shell_index_real: float
    lower_shell: int
    lower_shell_temperature: float
    lower_shell_phi: float
    lower_shell_omega_k_partial: float


@dataclass(frozen=True)
class PreliminaryNow:
    """
    Minimal bundle for the current rebuild stage.

    It contains the framework-natural "now" definition plus the derived
    reference-shell geometry from the light-cone, and optionally a temperature
    witness projected onto the shell ladder.
    """

    natural_now_phi: float
    reference_shell: int
    qcd_transition_shell: int
    reference_curvature_integral: float
    witness: TemperatureWitness | None = None


def build_temperature_witness(temperature_natural: float) -> TemperatureWitness:
    """
    Project an observational temperature onto the real shell ladder.
    """
    shell_real = shell_index_for_temperature(temperature_natural)
    lower_shell = max(0, int(shell_real))
    return TemperatureWitness(
        temperature_natural=temperature_natural,
        shell_index_real=shell_real,
        lower_shell=lower_shell,
        lower_shell_temperature=shell_temperature(lower_shell),
        lower_shell_phi=phi_of_real_shell(shell_real),
        lower_shell_omega_k_partial=omega_k_partial(lower_shell),
    )


def build_preliminary_now(*, temperature_natural: float | None = None) -> PreliminaryNow:
    """
    Build the preliminary "now" bundle from the Lean light-cone stack.
    """
    witness = None
    if temperature_natural is not None:
        witness = build_temperature_witness(temperature_natural)
    ref_shell = reference_m()
    return PreliminaryNow(
        natural_now_phi=h0_ref(),
        reference_shell=ref_shell,
        qcd_transition_shell=qcd_shell(),
        reference_curvature_integral=curvature_integral(ref_shell),
        witness=witness,
    )


__all__ = [
    "PreliminaryNow",
    "TemperatureWitness",
    "age_ratio_at_now",
    "age_ratio_homogeneous",
    "apparent_age",
    "build_preliminary_now",
    "build_temperature_witness",
    "h0_ref",
    "now_condition",
    "shell_index_for_temperature",
    "temperature_from_shell_index",
    "wall_clock_age_homogeneous",
]
