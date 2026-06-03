"""
Validate published hadron masses against explicit uncertainty bands.
"""

import numpy as np
import pytest

from tests.data.hadron_masses_pdg_with_errors import HADRON_MASSES_WITH_ERRORS_MEV


def test_subatomic_registry_covers_all_published_hadrons():
    """SUBATOMIC registry includes all published hadron keys in test data."""
    subatomic = pytest.importorskip("pyhqiv.subatomic")
    for flavor in HADRON_MASSES_WITH_ERRORS_MEV:
        assert flavor in subatomic.SUBATOMIC_PDG_MEV, f"missing hadron key: {flavor}"


def test_subatomic_registry_within_published_error_bars():
    """Registry mass values lie within published uncertainty bars."""
    subatomic = pytest.importorskip("pyhqiv.subatomic")
    for flavor, (central, err) in HADRON_MASSES_WITH_ERRORS_MEV.items():
        got = subatomic.SUBATOMIC_PDG_MEV[flavor]
        assert np.isfinite(got), f"{flavor} mass must be finite"
        assert abs(got - central) <= err, (
            f"{flavor}: got {got} MeV, expected {central} +/- {err} MeV"
        )


def test_confined_energy_mev_within_published_error_bars():
    """confined_energy_mev(flavor) is within published uncertainty bars at now."""
    subatomic = pytest.importorskip("pyhqiv.subatomic")
    for flavor, (central, err) in HADRON_MASSES_WITH_ERRORS_MEV.items():
        got = subatomic.confined_energy_mev(flavor)
        assert np.isfinite(got) and got > 0, f"{flavor} energy must be positive finite"
        assert abs(got - central) <= err, (
            f"{flavor}: confined_energy_mev {got} MeV, expected {central} +/- {err} MeV"
        )
