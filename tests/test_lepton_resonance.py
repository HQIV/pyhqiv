"""Lean-aligned lepton resonance: k factors, masses vs PDG, single-δ obstruction."""

import math

from pyhqiv.lepton_resonance_ladder import (
    REF_K_MU_E,
    REF_K_TAU_MU,
    eff_corrected,
    effective_surface_at_shell,
    lepton_masses_gev_from_electron_anchor,
    lepton_masses_gev_from_tau_anchor,
    lepton_resonance_masses_gev_triple,
    lepton_resonance_relative_errors_vs_pdg,
    lepton_resonance_relative_errors_vs_pdg_electron_anchor,
    pdg_single_delta_compat_residual,
    resonance_k_mu_e,
    resonance_k_tau_mu,
)


def test_k_factors_match_lean_reference() -> None:
    assert math.isclose(resonance_k_tau_mu(), REF_K_TAU_MU, rel_tol=0.0, abs_tol=1e-10)
    assert math.isclose(resonance_k_mu_e(), REF_K_MU_E, rel_tol=0.0, abs_tol=1e-10)


def test_triple_matches_tau_anchor_and_witness_ratios() -> None:
    me, mmu, mt = lepton_resonance_masses_gev_triple()
    me2, mmu2, mt2 = lepton_masses_gev_from_tau_anchor()
    assert math.isclose(me, me2, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(mmu, mmu2, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(mt, mt2, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(mt / mmu, resonance_k_tau_mu(), rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mmu / me, resonance_k_mu_e(), rel_tol=0.0, abs_tol=1e-12)


def test_eff_corrected_zero_matches_effective_surface() -> None:
    for m in (4, 81, 16336):
        assert math.isclose(eff_corrected(0.0, m), effective_surface_at_shell(m), abs_tol=0.0)


def test_tau_anchor_pdg_residuals_large_for_charged_lepton_geometry() -> None:
    """Lock-in shells (Lean ``ChargedLeptonResonance``) do not match PDG µ/e with τ anchor."""
    err = lepton_resonance_relative_errors_vs_pdg()
    assert err["rel_err_muon"] > 0.25
    assert err["rel_err_electron"] > 0.25


def test_electron_anchor_mu_near_pdg_tau_not() -> None:
    """e anchor: ``k_{µe}`` can track PDG µ; ``k_{τµ}`` then overshoots PDG τ."""
    err = lepton_resonance_relative_errors_vs_pdg_electron_anchor()
    assert err["rel_err_muon"] < 0.005
    assert err["rel_err_tau"] > 0.35


def test_e_tau_anchors_inverse() -> None:
    m_e_e, m_mu_e, m_tau_e = lepton_masses_gev_from_electron_anchor()
    m_e_t, m_mu_t, m_tau_t = lepton_masses_gev_from_tau_anchor(m_tau_e)
    assert math.isclose(m_e_e, m_e_t, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(m_mu_e, m_mu_t, rel_tol=0.0, abs_tol=1e-12)


def test_pdg_single_delta_obstruction_nonzero() -> None:
    """One δ cannot fix both PDG mass ratios; Lean compat residual ≠ 0."""
    assert abs(pdg_single_delta_compat_residual()) > 1.0
