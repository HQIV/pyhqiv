"""Two-scale table + resonance mass quality vs PDG."""

from pyhqiv.lepton_two_scale_diagnostics import (
    compton_over_classical_ratio,
    lepton_two_scale_table,
    resonance_mass_quality_report,
)


def test_lambda_over_r_is_one_over_alpha_for_all_generations() -> None:
    rows = lepton_two_scale_table()
    target = compton_over_classical_ratio()
    for r in rows:
        assert abs(r.lambda_over_r - target) < 1e-9


def test_shell_surface_decreases_with_heavier_generation_mass() -> None:
    """Rows are (e, µ, τ) with shells (16336, 81, 4): surface ∝ (m+1)(m+2) drops toward τ."""
    rows = lepton_two_scale_table()
    assert rows[0].shell_surface_leading > rows[1].shell_surface_leading > rows[2].shell_surface_leading


def test_resonance_masses_not_all_subpercent_vs_pdg() -> None:
    q = resonance_mass_quality_report()
    assert not q.masses_good_subpercent
