"""Basic chain / readout consistency for HEP (pyhqiv)."""

from __future__ import annotations

import pyhqiv.hep_decay_chain as hep


def test_particle_masses_heavy_and_light():
    m_p = hep.particle_mass_mev("p")
    assert abs(m_p - 938.27) < 0.1
    m_d = hep.particle_mass_mev("D_plus")
    assert 1700 < m_d < 1900
    m_j = hep.particle_mass_mev("Jpsi")
    assert 2900 < m_j < 3100


def test_decay_edges_open_for_known():
    p = hep.build_particle("delta_p")
    edges = hep.edges_from_particle(p)
    assert edges
    assert any("pi_plus" in e.daughters for e in edges)


def test_chain_export_has_sigma():
    j = hep.export_chain_json(["D_plus", "Jpsi"], with_sigma=True)
    assert "particles" in j
    assert any("mass_sigma_mev" in p for p in j["particles"])
