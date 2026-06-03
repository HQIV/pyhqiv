"""
Preliminary modified-Maxwell layer for the fresh pyhqiv rebuild.

This module mirrors the current Lean scaffold in:

- `HQIV_LEAN/Hqiv/Physics/ModifiedMaxwell.lean`

It intentionally keeps the same status as the Lean file:

- the O-sector inhomogeneous equation is present,
- the H restriction and classic-Maxwell comparison are present,
- manifold/divergence details remain placeholders,
- the phase-horizon tipping angle is implemented directly.
"""

from __future__ import annotations

import math
from typing import Callable

from pyhqiv.auxiliary_field import phi_of_temperature, shell_temperature
from pyhqiv.lightcone import alpha
from pyhqiv.metric import two_pi


def _require_octonion_component(component: int) -> None:
    if component < 0 or component > 7:
        raise ValueError("octonion component must be in the range 0..7")


def _require_spacetime_index(index: int) -> None:
    if index < 0 or index > 3:
        raise ValueError("spacetime index must be in the range 0..3")


def grad_phi(spacetime_index: int) -> float:
    """
    Placeholder for the spacetime gradient of phi.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.grad_φ`
    """
    _require_spacetime_index(spacetime_index)
    return 0.0


def div_mu(_field: Callable[[int], float]) -> float:
    """
    Placeholder divergence operator.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.div_μ`
    """
    return 0.0


def g_rr(_x: float) -> float:
    """
    Placeholder radial metric component in the flat limit.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.g_rr`
    """
    return 1.0


def current_o(component: int, spacetime_index: int) -> float:
    """
    Placeholder O-sector current.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.J_O`
    """
    _require_octonion_component(component)
    _require_spacetime_index(spacetime_index)
    return 0.0


def emergent_maxwell_inhomogeneous_o(
    component: int,
    spacetime_index: int,
    *,
    current_fn: Callable[[int, int], float] = current_o,
    grad_phi_fn: Callable[[int], float] = grad_phi,
) -> float:
    """
    Preliminary inhomogeneous O-sector Maxwell residual.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.emergentMaxwellInhomogeneous_O`
    """
    _require_octonion_component(component)
    _require_spacetime_index(spacetime_index)
    shell_t = shell_temperature(spacetime_index)
    phi_correction = alpha() * math.log(phi_of_temperature(shell_t)) * grad_phi_fn(spacetime_index)
    return -4.0 * math.pi * current_fn(component, spacetime_index) - phi_correction


def in_h(component: int) -> bool:
    """
    Quaternionic-subalgebra membership test.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.inH`
    """
    _require_octonion_component(component)
    return component < 4


def emergent_maxwell_inhomogeneous_h(
    spacetime_index: int,
    *,
    current_fn: Callable[[int, int], float] = current_o,
    grad_phi_fn: Callable[[int], float] = grad_phi,
) -> float:
    """
    Restriction of the O-sector equation to the EM component in H.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.emergentMaxwellInHomogeneous_H`
    """
    return emergent_maxwell_inhomogeneous_o(
        0,
        spacetime_index,
        current_fn=current_fn,
        grad_phi_fn=grad_phi_fn,
    )


def classic_maxwell_inhomogeneous(
    spacetime_index: int,
    *,
    current_fn: Callable[[int, int], float] = current_o,
) -> float:
    """
    Classic inhomogeneous Maxwell source term for the H restriction.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.classicMaxwellInhomogeneous`
    """
    _require_spacetime_index(spacetime_index)
    return 4.0 * math.pi * current_fn(0, spacetime_index)


def spatial_indices() -> tuple[int, int, int]:
    """
    Spatial directions when axis 0 is treated as time.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.spatialIndices`
    """
    return (1, 2, 3)


def maxwell3d_div_e() -> float:
    """
    Placeholder 3D div(E) relation.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.maxwell3D_div_E`
    """
    return 0.0


def maxwell3d_curl_b_minus_dedt() -> tuple[float, float, float]:
    """
    Placeholder 3D curl(B) - dE/dt relation.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.maxwell3D_curl_B_minus_dE_dt`
    """
    return (0.0, 0.0, 0.0)


def charge_conservation_o(
    component: int,
    spacetime_index: int,
    *,
    current_fn: Callable[[int, int], float] = current_o,
    grad_phi_fn: Callable[[int], float] = grad_phi,
) -> float:
    """
    Placeholder divergence of the inhomogeneous O-equation.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.charge_conservation_O`
    """
    _require_octonion_component(component)
    _require_spacetime_index(spacetime_index)
    return div_mu(
        lambda mu: emergent_maxwell_inhomogeneous_o(
            component,
            mu,
            current_fn=current_fn,
            grad_phi_fn=grad_phi_fn,
        )
    )


def horizon_quarter_period() -> float:
    """
    Quarter-turn of the horizon phase.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.horizonQuarterPeriod`
    """
    return two_pi() / 4.0


def delta_theta_prime(e_prime: float) -> float:
    """
    Phase-horizon tipping angle driven by local electric energy.

    Lean reference:
    `Hqiv.Physics.ModifiedMaxwell.delta_theta_prime`
    """
    return math.atan(e_prime) * horizon_quarter_period()


__all__ = [
    "charge_conservation_o",
    "classic_maxwell_inhomogeneous",
    "current_o",
    "delta_theta_prime",
    "div_mu",
    "emergent_maxwell_inhomogeneous_h",
    "emergent_maxwell_inhomogeneous_o",
    "g_rr",
    "grad_phi",
    "horizon_quarter_period",
    "in_h",
    "maxwell3d_curl_b_minus_dedt",
    "maxwell3d_div_e",
    "spatial_indices",
]
