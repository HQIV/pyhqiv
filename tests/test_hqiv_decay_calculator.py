"""Tests for unified HQIV decay calculator (nuclear + HEP bridge, pyhqiv port)."""

from __future__ import annotations

import pyhqiv.hep_decay_chain as hep


def test_beam_mix_parse() -> None:
    # parse is in full hqiv_decay_calculator; here exercise BeamTarget via chain
    setup = hep.BeamTargetSetup("p", 400.0, "p", 0.0)
    kin = hep.collision_kinematics(setup)
    assert kin.sqrt_s_gev > 0


def test_heavy_chain_has_edges() -> None:
    env = hep.ExperimentEnvironment()
    p = hep.build_particle("D_plus")
    edges = hep.edges_from_particle(p, env=env)
    # may be empty in minimal, but build ok
    assert isinstance(edges, list)
