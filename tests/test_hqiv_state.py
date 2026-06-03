"""Golden checks: :class:`pyhqiv.state.HQIVState` matches direct module calls."""

from __future__ import annotations

import pytest

from pyhqiv import auxiliary_field as af
from pyhqiv import lightcone as lc
from pyhqiv import metric
from pyhqiv.carrier import So8Carrier
from pyhqiv.state import HQIVState


def test_hqiv_state_curvature_matches_lightcone() -> None:
    for m in (0, 1, 4, 10):
        st = HQIVState.from_snapshot(m=m, horizon_n=12)
        s = st.curvature_summary()
        assert s["shell_shape"] == lc.shell_shape(m)
        assert s["curvature_integral_through_m"] == lc.curvature_integral(m + 1)
        assert s["omega_k_partial"] == lc.omega_k_partial(m)
        assert s["omega_k_at_horizon"] == lc.omega_k_at_horizon(m, 12)
        assert s["alpha"] == lc.alpha()
        assert s["gamma_hqiv"] == metric.gamma_hqiv()


def test_hqiv_state_phi_aux_default_is_shell() -> None:
    m = 4
    st = HQIVState.from_snapshot(m=m)
    assert st.phi_aux() == af.phi_of_shell(m)


def test_hqiv_state_metric_matches_metric_module() -> None:
    st = HQIVState.from_snapshot(m=3, phi_newtonian=0.01, phi_auxiliary=0.5, t=0.2)
    phi_a = 0.5
    snap = metric.build_metric_snapshot(0.01, phi_a, 0.2)
    ms = st.metric_summary()
    assert ms["lapse"] == snap.lapse
    assert ms["g_tt"] == snap.g_tt
    assert ms["time_angle"] == snap.time_angle_value
    assert ms["g_eff"] == metric.g_eff(phi_a)


def test_hqiv_state_with_shell() -> None:
    st = HQIVState.from_snapshot(m=2)
    st2 = st.with_shell(5)
    assert st2.m == 5
    assert st.m == 2


def test_hqiv_state_carrier_optional() -> None:
    c = So8Carrier.from_unit_axis(0)
    st = HQIVState.from_snapshot(m=0, carrier=c)
    assert st.carrier is c
    d = st.as_dict()
    assert d["has_carrier"] is True


def test_hqiv_state_negative_m_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        HQIVState.from_snapshot(m=-1)


def test_hqiv_state_negative_horizon_rejected() -> None:
    with pytest.raises(ValueError, match="horizon_n"):
        HQIVState.from_snapshot(m=0, horizon_n=-1)


def test_hqiv_state_curvature_omits_horizon_key_when_unset() -> None:
    st = HQIVState.from_snapshot(m=2)
    s = st.curvature_summary()
    assert "omega_k_at_horizon" not in s
    assert s["omega_k_partial"] == lc.omega_k_partial(2)


def test_hqiv_state_with_shell_rejects_negative() -> None:
    st = HQIVState.from_snapshot(m=1)
    with pytest.raises(ValueError, match="non-negative"):
        st.with_shell(-3)


def test_hqiv_state_as_dict_no_carrier() -> None:
    st = HQIVState.from_snapshot(m=0)
    d = st.as_dict()
    assert d["has_carrier"] is False
    assert "curvature" in d and "metric" in d


def test_hqiv_state_witnesses_loads() -> None:
    st = HQIVState.from_snapshot(m=0)
    w = st.witnesses()
    assert w.data is not None
    assert len(w.data) > 0
