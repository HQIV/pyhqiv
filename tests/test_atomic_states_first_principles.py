"""
Tests for first-principles atomic energy levels (atomic_states_first_principles).

- Structure tests: H/He run, ground most bound, n=2 less bound, He > H, binding > 0.
- Comparison table: prints I1_exp, binding_HQIV, diff, ratio (smoke test).
- Agreement test: first ionization HQIV/experiment ratio must lie in (RATIO_MIN, RATIO_MAX);
  currently the model overpredicts by ~2 orders of magnitude, so that test fails until fixed.
"""

import pytest

from pyhqiv.atomic_states_first_principles import (
    electron_mass_and_wave_packet,
    find_atomic_energy_levels,
    ground_state_binding_eV,
)
from pyhqiv.constants import M_E_MEV_REF

# First ionization I1 (eV) — reference only; no calibration in module
FIRST_IONIZATION_EV = {
    1: 13.5984, 2: 24.5874, 6: 11.2603, 7: 14.5341, 8: 13.6181,
    11: 5.1391, 12: 7.6462, 15: 10.4867, 16: 10.3600, 20: 6.1132,
    26: 7.9025, 29: 7.7264, 30: 9.3942,
}

_Z_TO_SYMBOL = {
    1: "H", 2: "He", 6: "C", 7: "N", 8: "O", 11: "Na", 12: "Mg",
    15: "P", 16: "S", 20: "Ca", 26: "Fe", 29: "Cu", 30: "Zn",
}

# Agreement band: binding_HQIV / I1_exp must be in (RATIO_MIN, RATIO_MAX).
# Within factor of 2 of experiment. Current model fails (ratio ~10²–10⁵).
RATIO_MIN, RATIO_MAX = 0.5, 2.0


def test_atomic_levels_run_hydrogen():
    """Hydrogen: levels return, ground state most bound."""
    levels = find_atomic_energy_levels(Z=1, max_n=2, N_electrons=1)
    assert len(levels) >= 2
    n_ground, l_ground, _, _, e_ground = levels[0]
    assert n_ground == 1 and l_ground == 0
    assert e_ground < 0
    assert all(row[4] >= e_ground for row in levels)


def test_atomic_levels_run_helium():
    """Helium: levels return, ground 1s."""
    levels = find_atomic_energy_levels(Z=2, max_n=2, N_electrons=2)
    assert len(levels) >= 2
    n_ground, l_ground, _, _, e_ground = levels[0]
    assert n_ground == 1 and l_ground == 0
    assert e_ground < 0


def test_ground_state_binding_positive():
    """Ground-state binding positive for H and C."""
    b_H = ground_state_binding_eV(1, N_electrons=1, max_n=2)
    b_C = ground_state_binding_eV(6, N_electrons=6, max_n=2)
    assert b_H > 0 and b_C > 0


def test_ordering_vs_z():
    """Binding increases with Z (He > H)."""
    b_H = ground_state_binding_eV(1, N_electrons=1, max_n=2)
    b_He = ground_state_binding_eV(2, N_electrons=2, max_n=2)
    assert b_He > b_H


def test_n2_less_bound_than_n1():
    """First excited (n=2) less bound than ground (n=1) for hydrogen."""
    levels = find_atomic_energy_levels(Z=1, max_n=2, N_electrons=1)
    ground_energy = levels[0][4]
    n2_levels = [row for row in levels if row[0] == 2]
    if n2_levels:
        assert n2_levels[0][4] > ground_energy


def test_electron_mass_and_wave_packet_returns():
    """electron_mass_and_wave_packet returns (m_e_mev, modes, rho); m_e = M_E_MEV_REF (lepton sector)."""
    m_e, modes, rho = electron_mass_and_wave_packet(1, 0, 0, 0.5)
    assert m_e == M_E_MEV_REF, "Electron is lepton; must use reference value until lepton Fano sector derived"
    assert m_e > 0
    assert isinstance(modes, list) and len(modes) >= 1
    assert callable(rho) and rho(1.0) >= 0


@pytest.mark.slow
def test_protein_atoms_run():
    """Protein atoms C, N, O: module runs, bound levels."""
    for Z in (6, 7, 8):
        levels = find_atomic_energy_levels(Z=Z, max_n=2, N_electrons=Z)
        assert len(levels) >= 1
        assert levels[0][4] < 0


@pytest.mark.slow
def test_heavier_elements_run():
    """S, P, Fe, Cu, Zn: run (slow)."""
    for Z in (15, 16, 26, 29, 30):
        binding = ground_state_binding_eV(Z, N_electrons=Z, max_n=2)
        assert binding > 0


def test_comparison_table_prints():
    """Smoke test: comparison table runs and produces diff/ratio for several Z."""
    z_list = [1, 6, 7, 8, 11, 12, 20]
    rows = []
    for Z in z_list:
        I1 = FIRST_IONIZATION_EV.get(Z)
        if I1 is None:
            continue
        binding = ground_state_binding_eV(Z, N_electrons=Z, max_n=2)
        diff = binding - I1
        ratio = binding / I1 if I1 else 0.0
        rows.append((_Z_TO_SYMBOL.get(Z, str(Z)), Z, I1, binding, diff, ratio))
    assert len(rows) >= 5
    print("\n--- Experiment vs HQIV calculation (first ionization, eV) ---")
    print("Element   Z    I1_exp(eV)   binding_HQIV(eV)   diff(HQIV-exp)   ratio(HQIV/exp)")
    for sym, z, i1, b, diff, ratio in rows:
        print(f"  {sym:3}     {z:2}    {i1:10.4f}   {b:16.2f}   {diff:14.2f}   {ratio:12.3f}")
    print("Binding = modified Coulomb + conservation; Casimir nuclear pairs only.\n")


def test_first_ionization_agreement_with_experiment():
    """HQIV first ionization must be within factor 2 of experiment (ratio in [0.5, 2])."""
    # H and He are the minimal set; add C if we want to stress heavier.
    z_list = [1, 2]
    failures = []
    for Z in z_list:
        I1 = FIRST_IONIZATION_EV[Z]
        binding = ground_state_binding_eV(Z, N_electrons=Z, max_n=2)
        ratio = binding / I1
        if not (RATIO_MIN <= ratio <= RATIO_MAX):
            failures.append(
                f"{_Z_TO_SYMBOL[Z]}(Z={Z}): ratio={ratio:.3f} not in [{RATIO_MIN}, {RATIO_MAX}] "
                f"(I1_exp={I1:.4f} eV, binding_HQIV={binding:.2f} eV)"
            )
    assert not failures, "First ionization outside agreement band:\n" + "\n".join(failures)
