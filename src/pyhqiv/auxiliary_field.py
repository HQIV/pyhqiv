"""
Auxiliary-field layer built directly on the light-cone ladder.

This module mirrors:

- `HQIV_LEAN/Hqiv/Geometry/AuxiliaryField.lean`
- the real-shell bridge in `HQIV_LEAN/Hqiv/Physics/SM_GR_Unification.lean`

Only the minimal field relations needed to continue into a preliminary `now`
module are implemented here.
"""

from __future__ import annotations

import math

from pyhqiv.lightcone import alpha


def phi_temperature_coefficient() -> float:
    """
    Coefficient in phi = 2 / T.

    Lean reference:
    `Hqiv.Geometry.AuxiliaryField.phiTemperatureCoeff`
    """
    return 2.0


def t_pl_natural() -> float:
    """
    Planck temperature in natural units.

    Lean reference:
    `Hqiv.Geometry.AuxiliaryField.T_Pl`
    """
    return 1.0


def shell_temperature(m: int) -> float:
    """
    HQIV shell temperature ladder T(m) = T_Pl / (m + 1).

    Lean reference:
    `Hqiv.Geometry.AuxiliaryField.T`
    """
    if m < 0:
        raise ValueError("shell index must be non-negative")
    return t_pl_natural() / float(m + 1)


def phi_of_temperature(temperature: float) -> float:
    """
    Continuous auxiliary field phi(T) = 2 / T.

    Lean reference:
    `Hqiv.Geometry.AuxiliaryField.phi_of_T`
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return phi_temperature_coefficient() / temperature


def phi_of_shell(m: int) -> float:
    """
    Discrete auxiliary field value on shell m.

    Lean reference:
    `Hqiv.Geometry.AuxiliaryField.phi_of_shell`
    """
    return phi_of_temperature(shell_temperature(m))


def shell_temperature_factor(shell_index_real: float) -> float:
    """
    Real-shell temperature factor s + 1.

    Lean reference:
    `Hqiv.Physics.SM_GR_Unification.shellTemperatureFactor`
    """
    if shell_index_real <= -1.0:
        raise ValueError("real shell index must satisfy s > -1")
    return shell_index_real + 1.0


def phi_of_real_shell(shell_index_real: float) -> float:
    """
    Real-shell auxiliary field extension phi(s) = 2 * (s + 1).

    Lean reference:
    `Hqiv.Physics.SM_GR_Unification.phi_of_real_shell`
    """
    return phi_temperature_coefficient() * shell_temperature_factor(shell_index_real)


def shell_shape_real(shell_index_real: float) -> float:
    """
    Continuous shell-shape extension on the real shell ladder.

    Lean reference:
    `Hqiv.Physics.SM_GR_Unification.shellShapeReal`
    """
    factor = shell_temperature_factor(shell_index_real)
    return (1.0 / factor) * (1.0 + alpha() * math.log(factor))


def phase_lift_coeff_real(shell_index_real: float) -> float:
    """
    Continuous phase-lift coefficient phi(s) / 6.

    Lean reference:
    `Hqiv.Physics.SM_GR_Unification.phaseLiftCoeffReal`
    """
    return phi_of_real_shell(shell_index_real) / 6.0


def shell_mass_geometry_factor(shell_index_real: float) -> float:
    """
    Preliminary geometric mass factor on the real shell ladder.

    Lean reference:
    `Hqiv.Physics.SM_GR_Unification.shellMassGeometryFactor`
    """
    return (
        (phi_of_real_shell(shell_index_real) / phi_temperature_coefficient())
        * shell_shape_real(shell_index_real)
        * (1.0 + phase_lift_coeff_real(shell_index_real))
    )


__all__ = [
    "phi_of_real_shell",
    "phi_of_shell",
    "phi_of_temperature",
    "phi_temperature_coefficient",
    "phase_lift_coeff_real",
    "shell_mass_geometry_factor",
    "shell_shape_real",
    "shell_temperature",
    "shell_temperature_factor",
    "t_pl_natural",
]
