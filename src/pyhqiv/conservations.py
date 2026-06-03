"""
Conservations layer for the fresh pyhqiv rebuild.

This module mirrors the Lean math in:

- `Hqiv/Conservations.lean`

In the current preliminary Python stack, the only non-placeholder
conservation we can evaluate directly (without a full manifold) is the
phase/spin conservation via the periodic horizon time-angle:

    timeAngle(φ, t) = φ * t

with period 2π and the forced interval containment:

    if t ∈ [0, 2π/φ] and φ > 0 then timeAngle(φ, t) ∈ [0, 2π].
"""

from __future__ import annotations

from dataclasses import dataclass

from pyhqiv.metric import hqvm_lapse, h0, time_angle, two_pi


def structure_from_o_dim() -> int:
    """
    Dimension of the structure from O-counting.

    Lean reference:
    `Hqiv.conservations.structure_from_O_dim` and `structure_from_O_dim = 28`.
    """
    return 28


def phase_conservation_numeric(phi_auxiliary: float, t: float) -> tuple[bool, bool, bool]:
    """
    Numeric checks corresponding to `metric_forces_phase_conservation`.

    Returns (timeAngle(phi,0)=0, timeAngle(phi,2π/phi)=2π, and
    timeAngle(phi,t) within [0,2π] when t is in [0,2π/phi]).
    """
    if phi_auxiliary <= 0.0:
        raise ValueError("phi_auxiliary must be > 0 for phase-periodicity checks")
    t_period = two_pi() / phi_auxiliary
    cond0 = time_angle(phi_auxiliary, 0.0) == 0.0
    cond_period = time_angle(phi_auxiliary, t_period) == two_pi()
    in_interval = (0.0 <= t) and (t <= t_period)
    if not in_interval:
        return cond0, cond_period, False
    val = time_angle(phi_auxiliary, t)
    return cond0, cond_period, (0.0 <= val) and (val <= two_pi())


def lapse_forces_time_angle_term(phi_newtonian: float, phi_auxiliary: float, t: float) -> float:
    """
    Evaluates the forced decomposition:
      HQVM_lapse(Φ, φ, t) = 1 + Φ + timeAngle(φ, t)

    Lean reference:
    `lapse_forces_time_angle_as_horizon_term` and `lapse_decompose`.
    """
    return hqvm_lapse(phi_newtonian, phi_auxiliary, t)


@dataclass(frozen=True)
class ConservationCheck:
    structure_dim: int
    phase_cond0: bool
    phase_cond_period: bool
    phase_interval_cond: bool


def build_conservation_check(phi_auxiliary: float, t: float) -> ConservationCheck:
    """
    Convenience bundle for phase conservation checks.
    """
    cond0, cond_period, cond_interval = phase_conservation_numeric(phi_auxiliary, t)
    return ConservationCheck(
        structure_dim=structure_from_o_dim(),
        phase_cond0=cond0,
        phase_cond_period=cond_period,
        phase_interval_cond=cond_interval,
    )


__all__ = [
    "ConservationCheck",
    "build_conservation_check",
    "lapse_forces_time_angle_term",
    "phase_conservation_numeric",
    "structure_from_o_dim",
]

