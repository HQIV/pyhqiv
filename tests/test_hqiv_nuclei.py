"""Python mirror of ``HQIV_LEAN/Hqiv/Physics/HQIVNuclei.lean`` (computable checks)."""

from __future__ import annotations

import math

from pyhqiv.hqiv_nuclei import (
    SPECTRA_DEUTERON_BINDING_MEV,
    casimir_energy_surface,
    deuteron_binding_scale,
    ladder_path_deuteron_lean,
    ladder_path_from_ZN,
    ladder_path_helium4_lean,
    new_modes_succ_identity,
    spherical_harmonic_cumulative_count,
    spin_statistics_half_life_matches_resonance,
    vacuum_mode_density,
    valley_count_from_A,
)


def test_spherical_harmonic_cumulative_count() -> None:
    assert spherical_harmonic_cumulative_count(3) == 16.0


def test_casimir_equals_modes_times_phi_over_two() -> None:
    m = 4
    from pyhqiv.auxiliary_field import phi_of_shell
    from pyhqiv.lightcone import available_modes

    ce = casimir_energy_surface(m)
    assert math.isclose(ce, available_modes(m) * phi_of_shell(m) / 2.0, rel_tol=1e-12)


def test_deuteron_binding_scale_matches_gamma_modes_over_rm() -> None:
    m = 4
    from pyhqiv.hqiv_nuclear_spectra import R_m, modes
    from pyhqiv.metric import gamma_hqiv

    s = deuteron_binding_scale(m)
    assert math.isclose(s, gamma_hqiv() * modes(m) / R_m(m), rel_tol=1e-12)


def test_valley_count_two_times_A_minus_one() -> None:
    assert valley_count_from_A(4) == 6
    assert valley_count_from_A(1) == 0


def test_ladder_paths_lean_examples() -> None:
    assert ladder_path_deuteron_lean() == ("proton", ["bind_neutron"])
    assert ladder_path_helium4_lean() == (
        "proton",
        ["bind_neutron", "bind_proton", "bind_neutron"],
    )
    seed, steps = ladder_path_from_ZN(2, 2)
    assert seed == "proton" and len(steps) == 3


def test_toroidal_ring_closure_numeric() -> None:
    m = 2
    lhs, rhs = new_modes_succ_identity(m)
    assert math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=1e-9)


def test_vacuum_mode_density() -> None:
    m = 4
    from pyhqiv.hqiv_nuclear_spectra import R_m
    from pyhqiv.lightcone import available_modes

    assert math.isclose(vacuum_mode_density(m), available_modes(m) / R_m(m), rel_tol=1e-12)


def test_spectra_deuteron_anchor() -> None:
    # Value from Lean witness; test allows tiny jitter + records source error bar via setup

    # deuteron binding is Lean spectra anchor (not PDG measurement for this test);
    # we still assert closeness to the committed Lean-derived value.
    assert abs(SPECTRA_DEUTERON_BINDING_MEV - 2.224575) < 1e-6
    # If a future contribution changes the Lean value, the witness update + this tolerance keeps CI green.


def test_spin_statistics_half_life_identity() -> None:
    assert spin_statistics_half_life_matches_resonance(1.0)
