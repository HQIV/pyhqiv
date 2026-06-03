"""
Horizon-phase and Compton-clock scales (time-warping / quarter-turn identification).

Lean references:

- ``Hqiv.Geometry.ModifiedMaxwell`` (``horizonQuarterPeriod``)
- Compton scaling is a classical SI check layered in :mod:`pyhqiv.compton_horizon_bridge`

:mod:`pyhqiv.lightcone` supplies the derived reference shell ``referenceM``.
"""

from __future__ import annotations

from pyhqiv.compton_horizon_bridge import (
    compton_quarter_period_s_from_pdg_electron,
    compton_quarter_period_seconds,
    horizon_quarter_angle_rad,
)
from pyhqiv.lightcone import reference_m


def blackhole_horizon_quarter_angle_rad() -> float:
    """``π/2`` quarter-turn in the horizon phase story (matches ``horizon_quarter_period``)."""
    return horizon_quarter_angle_rad()


def blackhole_compton_quarter_period_s(*, mass_mev: float | None = None) -> float:
    """
    Time for ``π/2`` radians of phase at the Compton frequency for ``mass_mev``.

    If ``mass_mev`` is omitted, uses the PDG electron anchor from
    :mod:`pyhqiv.compton_horizon_bridge`.
    """
    if mass_mev is None:
        return compton_quarter_period_s_from_pdg_electron()
    return compton_quarter_period_seconds(mass_mev)


def blackhole_reference_shell_m() -> int:
    """Discrete reference shell from the light-cone ladder (``referenceM``)."""
    return reference_m()


__all__ = [
    "blackhole_compton_quarter_period_s",
    "blackhole_horizon_quarter_angle_rad",
    "blackhole_reference_shell_m",
]
