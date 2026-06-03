"""Dynamic Rindler refinement: auxiliary φ_aux offsets reduce τ-anchor PDG residuals."""

import math

from pyhqiv.dynamic_rindler_lepton import (
    delta_auxiliary_phi_per_shell,
    dynamic_rindler_phi_scale,
    lepton_masses_gev_from_tau_anchor_dynamic_rindler,
    lepton_resonance_relative_errors_vs_pdg_dynamic_rindler,
    resonance_k_mu_e_dynamic_rindler,
    resonance_k_tau_mu_dynamic_rindler,
)
from pyhqiv.lepton_resonance_ladder import (
    lepton_resonance_relative_errors_vs_pdg,
    resonance_k_mu_e,
    resonance_k_tau_mu,
)
from pyhqiv.lightcone import alpha


def test_default_scale_is_alpha_third_without_witness_override() -> None:
    """No Lean export for ``dynamic_rindler_phi_scale`` → single hypothesis ``s = α/3``."""
    assert math.isclose(dynamic_rindler_phi_scale(), float(alpha()) / 3.0, rel_tol=0.0, abs_tol=1e-15)


def test_bare_k_factors_unchanged_at_zero_aux_scale() -> None:
    """``scale=0`` recovers bare ``effCorrected(0,·)`` ratios."""
    assert resonance_k_tau_mu_dynamic_rindler(scale=0.0) == resonance_k_tau_mu()
    assert resonance_k_mu_e_dynamic_rindler(scale=0.0) == resonance_k_mu_e()


def test_dynamic_rindler_reduces_tau_anchor_pdg_errors() -> None:
    bare = lepton_resonance_relative_errors_vs_pdg()
    dyn = lepton_resonance_relative_errors_vs_pdg_dynamic_rindler()
    assert dyn["rel_err_muon"] < bare["rel_err_muon"]
    assert dyn["rel_err_electron"] < bare["rel_err_electron"]
    assert dyn["rel_err_muon"] < 0.06
    assert dyn["rel_err_electron"] < 0.02


def test_delta_tau_is_zero_mu_e_nonzero() -> None:
    d_tau, d_mu, d_e = delta_auxiliary_phi_per_shell()
    assert d_tau == 0.0
    assert d_mu > 0.0
    assert d_e > d_mu


def test_mass_chain_self_consistent() -> None:
    m_e, m_mu, m_tau = lepton_masses_gev_from_tau_anchor_dynamic_rindler()
    k_tm = resonance_k_tau_mu_dynamic_rindler()
    k_me = resonance_k_mu_e_dynamic_rindler()
    assert math.isclose(m_tau / m_mu, k_tm, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(m_mu / m_e, k_me, rel_tol=0.0, abs_tol=1e-9)
