import math

from pyhqiv.lean_witnesses import load_lean_witnesses
from pyhqiv.sm_embedding import sm_hypercharge_weight, sm_sector_multiplicity
from pyhqiv.sm_gr_unification import m_electron_natural
from pyhqiv.sm_mass_ladder import (
    sm_mass_from_geometry,
    sm_mass_from_geometry_eV,
)


def test_muon_tau_from_resonance_witness_match_lean_export() -> None:
    w = load_lean_witnesses()
    m_pl = w.get_float("M_Pl_GeV")
    assert math.isclose(
        sm_mass_from_geometry("muon"),
        w.get_float("m_mu_from_resonance") / m_pl,
        rel_tol=0.0,
        abs_tol=1e-21,
    )
    assert math.isclose(
        sm_mass_from_geometry("tau"),
        w.get_float("m_tau_from_resonance") / m_pl,
        rel_tol=0.0,
        abs_tol=1e-21,
    )


def test_electron_normalization_matches_lean() -> None:
    # Corrected normalization at the electron horizon:
    # the ladder's electron entry matches the electron witness (no ×2 prefactor).
    got = sm_mass_from_geometry("electron")
    expected = m_electron_natural()
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=0.0)


def test_electron_mass_in_eV_scale() -> None:
    got_eV = sm_mass_from_geometry_eV("electron")
    w = load_lean_witnesses()
    expected_eV = w.get_float("m_electron_MeV") * w.get_float("MEV_TO_EV")
    assert abs(got_eV - expected_eV) < 1e-6


def test_hypercharge_weights_and_sector_multiplicities() -> None:
    assert sm_sector_multiplicity("electron") == 1.0
    assert sm_sector_multiplicity("up") == 3.0
    assert sm_sector_multiplicity("nu_e") == 1.0

    assert sm_hypercharge_weight("electron") == 1.0
    assert sm_hypercharge_weight("up") == -2.0 / 3.0
    assert sm_hypercharge_weight("down") == 1.0 / 3.0
    assert sm_hypercharge_weight("nu_e") == 0.0


def test_other_masses_are_finite_and_positive() -> None:
    # Charged leptons
    assert sm_mass_from_geometry("muon") > 0.0
    assert sm_mass_from_geometry("tau") > 0.0

    # Quarks (geometric weights are positive per Lean)
    for q in ("up", "down", "strange", "charm", "bottom", "top"):
        assert sm_mass_from_geometry(q) > 0.0

    # Neutrinos: Lean states nonnegative; allow a tiny negative due to float error.
    for nu in ("nu_e", "nu_mu", "nu_tau"):
        assert sm_mass_from_geometry(nu) >= -1e-30


def test_generations_are_not_degenerate_in_decimal_mode() -> None:
    # With electron-horizon "now" at small m_base, order-unity triality offsets
    # must yield non-degenerate generation masses without any 1/m suppression.
    e = sm_mass_from_geometry("electron")
    mu = sm_mass_from_geometry("muon")
    tau = sm_mass_from_geometry("tau")
    assert mu != e
    assert tau != e
    assert tau != mu

