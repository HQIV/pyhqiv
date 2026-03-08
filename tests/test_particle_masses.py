"""
Tests that package particle masses match test-folder reference data (PDG).

Single source of truth for mass data: tests/data/particle_masses_pdg.py.
Quarks: constants.M_*_MEV_QCD. Hadrons: subatomic.SUBATOMIC_PDG_MEV and
confined_energy_mev(flavor_content).
"""

import numpy as np
import pytest

from tests.data.particle_masses_pdg import (
    HADRON_LABELS,
    HADRON_MASSES_MEV,
    QUARK_MASSES_MEV,
)


def test_quark_masses_constants_match_test_data():
    """Quark masses in pyhqiv.constants match tests/data reference."""
    from pyhqiv.constants import (
        M_B_MEV_QCD,
        M_C_MEV_QCD,
        M_D_MEV_QCD,
        M_S_MEV_QCD,
        M_T_MEV_QCD,
        M_U_MEV_QCD,
    )
    ref = {
        "u": M_U_MEV_QCD,
        "d": M_D_MEV_QCD,
        "s": M_S_MEV_QCD,
        "c": M_C_MEV_QCD,
        "b": M_B_MEV_QCD,
        "t": M_T_MEV_QCD,
    }
    for q, mass in ref.items():
        assert q in QUARK_MASSES_MEV, f"quark {q} in test data"
        assert abs(mass - QUARK_MASSES_MEV[q]) < 1e-9, (
            f"quark {q}: constants {mass} vs test data {QUARK_MASSES_MEV[q]}"
        )


def test_hadron_masses_subatomic_registry_matches_test_data():
    """SUBATOMIC_PDG_MEV matches tests/data hadron masses."""
    from pyhqiv.subatomic import SUBATOMIC_PDG_MEV
    for flavor, ref_mev in HADRON_MASSES_MEV.items():
        assert flavor in SUBATOMIC_PDG_MEV, f"hadron {flavor} in registry"
        assert abs(SUBATOMIC_PDG_MEV[flavor] - ref_mev) < 1e-9, (
            f"{flavor} ({HADRON_LABELS.get(flavor, flavor)}): "
            f"registry {SUBATOMIC_PDG_MEV[flavor]} vs test data {ref_mev}"
        )
    for flavor in SUBATOMIC_PDG_MEV:
        assert flavor in HADRON_MASSES_MEV, f"registry {flavor} in test data"


def test_confined_energy_mev_matches_test_data_for_all_hadrons():
    """confined_energy_mev(flavor) returns test-data mass for each hadron at epoch='now'."""
    from pyhqiv.subatomic import confined_energy_mev
    for flavor, ref_mev in HADRON_MASSES_MEV.items():
        e = confined_energy_mev(flavor)
        assert np.isfinite(e) and e > 0, f"{flavor} finite positive"
        assert abs(e - ref_mev) < 1e-6, (
            f"{flavor} ({HADRON_LABELS.get(flavor, flavor)}): "
            f"confined_energy_mev {e} vs test data {ref_mev}"
        )


def test_proton_neutron_masses_match_test_data():
    """M_PROTON_MEV / M_NEUTRON_MEV match test-data uud/udd."""
    from pyhqiv.constants import M_NEUTRON_MEV, M_PROTON_MEV
    assert abs(M_PROTON_MEV - HADRON_MASSES_MEV["uud"]) < 1e-9
    assert abs(M_NEUTRON_MEV - HADRON_MASSES_MEV["udd"]) < 1e-9


def test_quark_masses_positive_and_ordered():
    """Quark masses in test data: positive; heavy > light."""
    assert set(QUARK_MASSES_MEV) == {"u", "d", "s", "c", "b", "t"}
    for q, m in QUARK_MASSES_MEV.items():
        assert m > 0 and np.isfinite(m), f"quark {q} positive finite"
    assert QUARK_MASSES_MEV["s"] > QUARK_MASSES_MEV["d"] > QUARK_MASSES_MEV["u"]
    assert QUARK_MASSES_MEV["c"] > QUARK_MASSES_MEV["s"]
    assert QUARK_MASSES_MEV["b"] > QUARK_MASSES_MEV["c"]
    assert QUARK_MASSES_MEV["t"] > QUARK_MASSES_MEV["b"]
