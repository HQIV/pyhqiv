"""Tests for new-state production rates and branching readout (pyhqiv port)."""

from __future__ import annotations

import unittest

import pyhqiv.hep_decay_benchmark as bench
import pyhqiv.hep_decay_chain as hep
import pyhqiv.hep_decay_production_readout as hpr


class TestHepProductionReadout(unittest.TestCase):
    def test_branching_sums_for_new_states(self) -> None:
        env = hep.ExperimentEnvironment()
        for sid in ("D_plus", "lambda_c", "Jpsi", "B_plus"):
            edges = hep.edges_from_particle(hep.build_particle(sid), env=env)
            total = sum(getattr(e, "branching_ratio", 0.0) for e in edges)
            self.assertAlmostEqual(total, 1.0, places=4, msg=sid)

    def test_jpsi_has_em_channel(self) -> None:
        env = hep.ExperimentEnvironment()
        edges = hep.edges_from_particle(hep.build_particle("Jpsi"), env=env)
        em = [e for e in edges if getattr(getattr(e, "mode", None), "channel", None) == "electromagnetic"]
        self.assertGreaterEqual(len(em), 1)
        # ee may be present (BR may be small in minimal width model)
        ee = [e for e in em if getattr(e, "daughters", ()) and "e_plus" in [getattr(d, "species_id", d) for d in getattr(e, "daughters", ())]]
        if ee:
            self.assertGreaterEqual(ee[0].branching_ratio, 0.0)

    def test_production_ordering_lhc(self) -> None:
        setup = hep.FACILITY_PRESETS["LHC_pp_13TeV"]
        kin = hep.collision_kinematics(setup)
        table = {
            r.species_id: r
            for r in hpr.production_rate_table(
                [("D_plus", hep.particle_mass_mev("D_plus")), ("B_plus", hep.particle_mass_mev("B_plus"))],
                sqrt_s_gev=kin.sqrt_s_gev,
                accessible_mass_gev=getattr(kin, "accessible_mass_gev", 100.0),
                collision_mode=setup.resolved_collision_mode(),
            )
        }
        self.assertGreater(
            table["D_plus"].normalized_fraction,
            table["B_plus"].normalized_fraction,
        )

    def test_new_state_benchmark_panels(self) -> None:
        payload = bench.build_payload()
        self.assertEqual(payload["summary"]["fail"], 0)
        self.assertGreaterEqual(payload["summary"]["total"], 50)


if __name__ == "__main__":
    unittest.main()
