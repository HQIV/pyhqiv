"""Regression tests for Lean-aligned bound-state and nuclear spectra ports."""

from __future__ import annotations

import math

import pytest

from pyhqiv import auxiliary_field as aux
from pyhqiv.hqiv_bound_states import (
    SO8_DIM,
    NetworkWeight28,
    NuclidePN,
    binding_coupling_at_shell,
    e_bind_from_network,
    e_bind_nuclear_from_network,
    m_nucleus_from_network,
    network_weight_for_nuclide,
)
from pyhqiv.hqiv_nuclear_spectra import (
    R_m,
    V_nuclear,
    barbell_ring_new_modes_identity,
    beta_decay_rate_scalar,
    beta_decay_rate_with_gf,
    modes,
    mu_neutron,
    single_nucleon_caustic_mode_identity,
)
from pyhqiv.hqiv_schrodinger_shell import one_over_alpha_eff, one_over_alpha_eff_shell
from pyhqiv.metric import gamma_hqiv


def test_so8_dim_matches_lean() -> None:
    assert SO8_DIM == 28


def test_phi_of_shell_matches_auxiliary_field() -> None:
    m = 4
    assert aux.phi_of_shell(m) == aux.phi_temperature_coefficient() * (m + 1)


def test_one_over_alpha_eff_matches_closed_form_at_phi() -> None:
    phi = 10.0
    c = 1.0
    from pyhqiv.lightcone import alpha
    from pyhqiv.sm_gr_unification import alpha_gut

    ag = alpha_gut()
    expected = (1.0 / ag) * (1.0 + c * alpha() * math.log(phi + 1.0))
    assert math.isclose(one_over_alpha_eff(phi, c), expected, rel_tol=1e-12, abs_tol=0.0)


def test_binding_coupling_independent_of_generator_index() -> None:
    m = 3
    c0 = binding_coupling_at_shell(m, 0, 1.0)
    c27 = binding_coupling_at_shell(m, 27, 1.0)
    assert c0 == c27


def test_e_bind_from_network_uniform() -> None:
    m = 2
    w = NetworkWeight28.uniform(1.0 / SO8_DIM)
    e = e_bind_from_network(m, w, 1.0)
    a = binding_coupling_at_shell(m, 0, 1.0)
    assert math.isclose(e, a, rel_tol=1e-12, abs_tol=0.0)


def test_nuclide_network_weight_sums_to_available_modes() -> None:
    from pyhqiv.lightcone import available_modes

    m = 5
    w = network_weight_for_nuclide(m, NuclidePN(2, 2))
    assert math.isclose(sum(w.w), available_modes(m), rel_tol=1e-12, abs_tol=0.0)


def test_m_nucleus_from_network_structure() -> None:
    m = 1
    w = NetworkWeight28.uniform(1.0 / SO8_DIM)
    eb = e_bind_nuclear_from_network(m, w, 1.0)
    mn = m_nucleus_from_network(m, 4, 2, 940.0, w, 1.0)
    assert math.isclose(mn, 4.0 * 940.0 - eb, rel_tol=1e-9, abs_tol=0.0)


def test_single_nucleon_caustic_identity() -> None:
    mm, rm = single_nucleon_caustic_mode_identity(3)
    assert math.isclose(rm, 4.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mm, modes(3), rel_tol=0.0, abs_tol=1e-9)


def test_barbell_ring_identity() -> None:
    v = barbell_ring_new_modes_identity(2)
    assert v > 0.0 and math.isfinite(v)


def test_mu_neutron_is_minus_gamma_over_two() -> None:
    assert math.isclose(mu_neutron(), -gamma_hqiv() / 2.0, rel_tol=1e-12, abs_tol=0.0)


def test_V_nuclear_finite() -> None:
    v = V_nuclear(
        m=4,
        z_eff=2.0,
        mu_i=1e-3,
        mu_j=1e-3,
        r=1e-15,
        s_para_i=0.5,
        s_para_j=0.5,
        s_dot=0.25,
        delta_phi_mag=0.0,
    )
    assert math.isfinite(v)


def test_one_over_alpha_eff_shell_positive() -> None:
    v = one_over_alpha_eff_shell(0, 1.0)
    assert v > 0.0 and math.isfinite(v)


def test_beta_decay_rate_scalar_matches_with_gf() -> None:
    gf = 1.1e-5
    me = 0.5
    mell = 2.0
    tag = "n"
    assert math.isclose(
        beta_decay_rate_scalar(tag, me, mell, g_fermi=gf),
        beta_decay_rate_with_gf(gf, me, mell),
        rel_tol=0.0,
        abs_tol=0.0,
    )
