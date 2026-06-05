"""Smoke tests for :mod:`pyhqiv.regimes` façades."""

from __future__ import annotations

import numpy as np
import pytest

from pyhqiv import HQIVState, So8Carrier, metric
from pyhqiv import lightcone as lc
from pyhqiv.regimes import (
    blackhole_compton_quarter_period_s,
    blackhole_horizon_quarter_angle_rad,
    blackhole_reference_shell_m,
    born_probs_from_real_state,
    evolve_so8_carrier_expm,
    galactic_g_eff,
    galactic_gamma_hqiv,
    galactic_metric_summary,
    quantum_lepton_coherence_snapshot,
)
from pyhqiv.regimes.quantum import evolve_so8_vector_expm


def test_galactic_facade_matches_metric() -> None:
    phi = 0.7
    assert galactic_g_eff(phi) == metric.g_eff(phi)
    assert galactic_gamma_hqiv() == metric.gamma_hqiv()
    st = HQIVState.from_snapshot(m=3, phi_auxiliary=phi)
    assert galactic_metric_summary(st) == st.metric_summary()


def test_blackhole_facade_reference_shell() -> None:
    assert blackhole_reference_shell_m() == lc.reference_m()
    assert blackhole_horizon_quarter_angle_rad() > 1.0


def test_blackhole_compton_positive() -> None:
    assert blackhole_compton_quarter_period_s() > 0.0
    assert blackhole_compton_quarter_period_s(mass_mev=105.66) > 0.0


def test_quantum_evolve_preserves_norm() -> None:
    c = So8Carrier.from_unit_axis(0)
    coeffs = np.zeros(28, dtype=np.float64)
    coeffs[0] = 0.3
    out = evolve_so8_carrier_expm(c, coeffs, dt=0.05)
    assert np.isclose(np.linalg.norm(out.psi), 1.0, rtol=1e-10)


def test_born_probs_sum_to_one() -> None:
    p = born_probs_from_real_state(np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert p.shape == (8,)
    assert np.isclose(p.sum(), 1.0)


def test_quantum_coherence_snapshot() -> None:
    snap = quantum_lepton_coherence_snapshot()
    assert snap.m_tau >= 0


def test_quantum_coherence_snapshot_custom_t_cmb() -> None:
    snap = quantum_lepton_coherence_snapshot(t_cmb_natural=2e-32)
    assert snap.observer_shell_cmb > 0.0


def test_evolve_so8_vector_expm_matches_carrier_path() -> None:
    c = So8Carrier.from_unit_axis(2)
    coeffs = np.zeros(28, dtype=np.float64)
    coeffs[2] = 0.15
    dt = 0.07
    psi1 = evolve_so8_vector_expm(c.psi, coeffs, dt, c.generators)
    c2 = evolve_so8_carrier_expm(c, coeffs, dt)
    assert np.allclose(psi1, c2.psi, atol=1e-14)


def test_born_probs_zero_vector_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        born_probs_from_real_state(np.zeros(8))


def test_regimes_subpackage_exports() -> None:
    import pyhqiv.regimes as r

    assert hasattr(r, "galactic_g_eff")
    assert "evolve_so8_vector_expm" in r.__all__
    assert hasattr(r, "evolve_so8_vector_expm")
    from pyhqiv.regimes import quantum as rq

    assert rq.evolve_so8_vector_expm is r.evolve_so8_vector_expm


def test_blackhole_horizon_quarter_matches_pi_half() -> None:
    import math

    assert math.isclose(blackhole_horizon_quarter_angle_rad(), math.pi / 2.0, rel_tol=0.0, abs_tol=1e-15)
