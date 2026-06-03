"""Horizon π/2 ↔ Compton quarter-period consistency."""

import math

from pyhqiv.compton_horizon_bridge import (
    compton_angular_frequency_rad_s,
    compton_quarter_period_seconds,
    horizon_quarter_angle_rad,
)
from pyhqiv.lepton_resonance_ladder import PDG_ELECTRON_MEV


def test_quarter_period_times_omega_equals_pi_over_two() -> None:
    m = PDG_ELECTRON_MEV
    w = compton_angular_frequency_rad_s(m)
    dt = compton_quarter_period_seconds(m)
    assert math.isclose(dt * w, horizon_quarter_angle_rad(), rel_tol=0.0, abs_tol=1e-12)


def test_horizon_quarter_angle_is_pi_over_two() -> None:
    assert math.isclose(horizon_quarter_angle_rad(), math.pi / 2.0, rel_tol=0.0, abs_tol=1e-12)
