"""Lean SurfaceWaveSelfClock + lock-in shell steps; PDG checks via ``lepton_resonance_ladder``."""

import math

from pyhqiv.lepton_resonance_ladder import (
    lepton_resonance_relative_errors_vs_pdg,
    resonance_k_mu_e,
    resonance_k_tau_mu,
)
from pyhqiv.surface_wave_self_clock import (
    SELF_CLOCK_DEG_AT_NOW_DEFAULT,
    compton_quarter_turn_at_T_lockin,
    cosmic_birefringence_deg_at_now,
    cosmic_birefringence_rad_at_now,
    geometric_resonance_step_lepton_mu_e,
    geometric_resonance_step_lepton_tau_mu,
    lepton_electron_shell,
    lepton_heavy_vertex_shell,
    lepton_muon_shell,
    lepton_tau_birth_phase_matches_quarter_turn_at_lockin,
    mexican_hat_veff,
    self_clock_cumulative_rapidity_cosmic_now,
    self_clock_phase,
    self_clock_phase_with_cosmic_now,
    self_clock_rapidity_update_additivity,
)


def test_self_clock_rapidity_update_holds() -> None:
    """Lean ``selfClock_rapidity_update``: identity holds for any base η."""
    assert self_clock_rapidity_update_additivity(4, 0.1, 2.0, 0.05)
    assert self_clock_rapidity_update_additivity(81, -1.0, 1.5, 0.3)


def test_self_clock_rapidity_update_holds_with_cosmic_now_as_base() -> None:
    """Same proved identity with η = β_rad (0.3° default self-clock at ``now``)."""
    eta_now = self_clock_cumulative_rapidity_cosmic_now()
    assert math.isclose(cosmic_birefringence_deg_at_now(), SELF_CLOCK_DEG_AT_NOW_DEFAULT)
    assert self_clock_rapidity_update_additivity(4, eta_now, 2.0, 0.05)
    assert self_clock_rapidity_update_additivity(81, eta_now, 1.5, 0.02)


def test_tau_birth_phase_equals_quarter_turn_at_lockin() -> None:
    assert lepton_tau_birth_phase_matches_quarter_turn_at_lockin()


def test_self_clock_phase_at_zero_matches_compton_quarter() -> None:
    m = lepton_heavy_vertex_shell()
    assert math.isclose(self_clock_phase(m, 0.0), compton_quarter_turn_at_T_lockin(), abs_tol=1e-12)


def test_cosmic_now_birefringence_default_0p3_deg_in_witness() -> None:
    assert math.isclose(cosmic_birefringence_deg_at_now(), 0.3, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(
        cosmic_birefringence_rad_at_now(),
        math.radians(0.3),
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert self_clock_cumulative_rapidity_cosmic_now() == cosmic_birefringence_rad_at_now()


def test_self_clock_phase_with_cosmic_now_adds_birefringence_rapidity() -> None:
    m = 4
    beta = cosmic_birefringence_rad_at_now()
    assert math.isclose(
        self_clock_phase_with_cosmic_now(m, 0.0),
        self_clock_phase(m, beta),
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        self_clock_phase_with_cosmic_now(m, 0.1),
        self_clock_phase(m, 0.1 + beta),
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_mexican_hat_decreases_along_shells_small_delta() -> None:
    lam, Phi, phi = 1e-9, 0.0, 1.0
    t = 1.0
    m_lo, m_hi = 4, 81
    v_lo = mexican_hat_veff(lam, Phi, phi, t, m_lo)
    v_hi = mexican_hat_veff(lam, Phi, phi, t, m_hi)
    assert v_hi < v_lo


def test_mexican_hat_increases_in_time_fixed_shell() -> None:
    lam, Phi, phi = 1e-9, 0.0, 1.0
    m = 81
    t1, t2 = 0.0, 1.0
    v1 = mexican_hat_veff(lam, Phi, phi, t1, m)
    v2 = mexican_hat_veff(lam, Phi, phi, t2, m)
    assert v1 < v2


def test_charged_lepton_resonance_tau_anchor_not_pdg_subpercent() -> None:
    """``ChargedLeptonResonance`` τ anchor does not recover PDG µ/e (regression guard)."""
    err = lepton_resonance_relative_errors_vs_pdg()
    assert err["rel_err_muon"] > 0.25
    assert err["rel_err_electron"] > 0.25


def test_lockin_shell_ordering_matches_lean() -> None:
    a, b, c = lepton_heavy_vertex_shell(), lepton_muon_shell(), lepton_electron_shell()
    assert a < b < c


def test_lockin_geometric_steps_positive() -> None:
    assert geometric_resonance_step_lepton_tau_mu() > 0.0
    assert geometric_resonance_step_lepton_mu_e() > 0.0


def test_lockin_geometric_steps_match_charged_lepton_k_factors() -> None:
    """Lean: ``geometricResonanceStep`` = detuned outer / detuned inner = ``resonance_k_*``."""
    assert geometric_resonance_step_lepton_tau_mu() == resonance_k_tau_mu()
    assert geometric_resonance_step_lepton_mu_e() == resonance_k_mu_e()
