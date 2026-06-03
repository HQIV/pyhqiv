"""
Preliminary force-sector assignment and unit-system helpers.

This module mirrors the current Lean scaffold in:

- `HQIV_LEAN/Hqiv/Physics/Forces.lean`

The main job here is to map octonion components into the EM / Weak / Strong
sectors and provide the minimum unit conversions used by the same Lean layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pyhqiv.modified_maxwell import emergent_maxwell_inhomogeneous_o


class ForceSector(str, Enum):
    """
    Gauge-sector assignment from the O-structure.

    Lean reference:
    `Hqiv.Physics.Forces.ForceSector`
    """

    EM = "EM"
    WEAK = "Weak"
    STRONG = "Strong"


def o_component_to_sector(component: int) -> ForceSector:
    """
    Map an O-component into EM / Weak / Strong.

    Lean reference:
    `Hqiv.Physics.Forces.O_component_to_sector`
    """
    if component < 0 or component > 7:
        raise ValueError("octonion component must be in the range 0..7")
    if component < 4:
        return ForceSector.EM if component == 0 else ForceSector.WEAK
    return ForceSector.STRONG


def time_axis() -> int:
    """
    Distinguished time axis for the 4-index spacetime scaffold.

    Lean reference:
    `Hqiv.Physics.Forces.timeAxis`
    """
    return 0


class UnitSystem(str, Enum):
    """
    Preliminary unit-system tag.

    Lean reference:
    `Hqiv.Physics.Forces.UnitSystem`
    """

    METRIC = "Metric"
    SI = "SI"


def _local_conv(key: str) -> float:
    from pyhqiv.scale_witness import load_local_conditions as _lc

    return float(_lc()[key])


def c_si() -> float:
    return _local_conv("C_SI_local")


def hbar_si() -> float:
    return _local_conv("hbar_SI_J_s")


def g_si() -> float:
    return _local_conv("G_SI_m3_per_kg_s2")


def length_natural_to_si(inv_gev: float) -> float:
    """
    Convert GeV^-1 to metres.

    Lean reference:
    `Hqiv.Physics.Forces.length_natural_to_SI`
    """
    return inv_gev * _local_conv("length_natural_to_SI_m_per_GeVinv")


def time_natural_to_si(inv_gev: float) -> float:
    """
    Convert GeV^-1 to seconds.

    Lean reference:
    `Hqiv.Physics.Forces.time_natural_to_SI`
    """
    return inv_gev * _local_conv("time_natural_to_SI_s_per_GeVinv")


def energy_natural_to_si_j(gev: float) -> float:
    """
    Convert GeV to joules.

    Lean reference:
    `Hqiv.Physics.Forces.energy_natural_to_SI_J`
    """
    return gev * _local_conv("energy_natural_to_SI_J_per_GeV")


def e_field_natural_to_si(e_natural: float) -> float:
    """
    Convert GeV^2 electric-field scale into V/m.

    Lean reference:
    `Hqiv.Physics.Forces.E_field_natural_to_SI`
    """
    return e_natural * _local_conv("e_field_natural_to_SI_V_per_m_per_GeV2")


def force_natural_to_si(f_natural: float) -> float:
    """
    Convert GeV^2 force scale into newtons.

    Lean reference:
    `Hqiv.Physics.Forces.force_natural_to_SI`
    """
    return f_natural * _local_conv("force_natural_to_SI_N_per_GeV2")


@dataclass(frozen=True)
class ValueInUnits:
    """
    Minimal value wrapper tagged by unit system.

    Lean reference:
    `Hqiv.Physics.Forces.ValueInUnits`
    """

    system: UnitSystem
    value: float

    def to_real(self) -> float:
        return self.value


def in_metric(value: float) -> ValueInUnits:
    return ValueInUnits(system=UnitSystem.METRIC, value=value)


def in_si(value: float) -> ValueInUnits:
    return ValueInUnits(system=UnitSystem.SI, value=value)


def emergent_maxwell_inhomogeneous_o_metric(component: int, spacetime_index: int) -> float:
    """
    O-sector Maxwell residual in natural/metric units.

    Lean reference:
    `Hqiv.Physics.Forces.emergentMaxwellInhomogeneous_O_metric`
    """
    return emergent_maxwell_inhomogeneous_o(component, spacetime_index)


def emergent_maxwell_inhomogeneous_o_si(component: int, spacetime_index: int) -> float:
    """
    O-sector Maxwell residual in SI units.

    Lean reference:
    `Hqiv.Physics.Forces.emergentMaxwellInhomogeneous_O_SI`

    In the current Lean scaffold this shares the same zero set as the metric
    version, so we keep the same residual until the field/current conversions
    are expanded further.
    """
    return emergent_maxwell_inhomogeneous_o_metric(component, spacetime_index)


__all__ = [
    "ForceSector",
    "UnitSystem",
    "ValueInUnits",
    "c_si",
    "e_field_natural_to_si",
    "emergent_maxwell_inhomogeneous_o_metric",
    "emergent_maxwell_inhomogeneous_o_si",
    "energy_natural_to_si_j",
    "force_natural_to_si",
    "g_si",
    "hbar_si",
    "in_metric",
    "in_si",
    "length_natural_to_si",
    "o_component_to_sector",
    "time_axis",
    "time_natural_to_si",
]
