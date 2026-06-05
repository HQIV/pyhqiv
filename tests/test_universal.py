"""Tests for HQIVUniversalSystem and binding calibration (PDG/NIST targets)."""

import numpy as np
import pytest

pytest.importorskip(
    "pyhqiv.constants",
    reason="universal test uses legacy constants + other modules (bak/)",
)

from pyhqiv.constants import M_NEUTRON_MEV, M_PROTON_MEV
from pyhqiv.horizon_network import mean_field_mu
from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants
from pyhqiv.universal_system import HQIVUniversalSystem

from pyhqiv.nuclear import NuclearConfig


def test_universal_system_deuteron_like():
    """Deuteron-like 2 nucleons: total energy positive; when bound (E < E_free), binding per particle > 0."""
    const = get_hqiv_nuclear_constants()
    L = const["LATTICE_BASE_M"]
    particles = [
        {"position": np.zeros(3), "state_matrix": np.eye(8), "mass_mev": M_PROTON_MEV, "type": "proton"},
        {"position": np.array([2e-15, 0, 0]), "state_matrix": np.eye(8), "mass_mev": M_NEUTRON_MEV, "type": "neutron"},
    ]
    us = HQIVUniversalSystem(particles, lattice_base_m=L, expand_to_quarks=False)
    E = us.total_energy_mev()
    E_free = M_PROTON_MEV + M_NEUTRON_MEV
    assert E > 0
    B_per = us.binding_per_particle()
    if E < E_free:
        assert B_per > 0


def test_nuclear_config_deuteron_binding_scale():
    """Deuteron: gold-standard functional result is self-consistent; NuclearConfig B non-negative."""
    from pyhqiv.nuclear import binding_energy_mev_functional

    deut = NuclearConfig(1, 1)
    assert deut._binding_energy_mev >= 0
    res = binding_energy_mev_functional(1, 1)
    assert np.isfinite(res.B_mev) and np.isfinite(res.E_free_mev) and np.isfinite(res.E_bound_mev)
    np.testing.assert_allclose(res.B_mev, res.E_free_mev - res.E_bound_mev, rtol=1e-9)


def test_mean_field_mu():
    """Mean-field μ = sqrt(1 + avg_neighbors); increases with density."""
    mu_low = mean_field_mu(0.01, r_n=1.2e-15)
    mu_high = mean_field_mu(0.5, r_n=1.2e-15)
    assert mu_low >= 1.0
    assert mu_high > mu_low


def test_universal_expand_to_quarks():
    """With expand_to_quarks=True, 2 nucleons → 6 nodes."""
    const = get_hqiv_nuclear_constants()
    L = const["LATTICE_BASE_M"]
    particles = [
        {"position": np.zeros(3), "state_matrix": np.eye(8), "mass_mev": M_PROTON_MEV, "type": "proton"},
        {"position": np.array([2e-15, 0, 0]), "state_matrix": np.eye(8), "mass_mev": M_NEUTRON_MEV, "type": "neutron"},
    ]
    us = HQIVUniversalSystem(particles, lattice_base_m=L, expand_to_quarks=True)
    assert len(us.net.nodes) == 6
    assert us.total_energy_mev() > 0
