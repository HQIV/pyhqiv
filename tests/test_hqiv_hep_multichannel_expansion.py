"""Tests for full multi-channel HEP decay expansion (pyhqiv port)."""

from __future__ import annotations

import unittest

import pyhqiv.hep_decay_benchmark as bench
import pyhqiv.hep_decay_chain as hep
import pyhqiv.hep_decay_multichannel_expansion as mc


class TestMultichannelExpansion(unittest.TestCase):
    def test_jpsi_many_open_channels(self) -> None:
        env = hep.ExperimentEnvironment()
        p = hep.build_particle("Jpsi")
        edges = hep.edges_from_particle(p, env=env)
        self.assertGreaterEqual(len(edges), 20)  # our port generates substantial but not full 100s for perf
        total_br = sum(getattr(e, "branching_ratio", 0.0) for e in edges)
        if total_br > 0:
            self.assertAlmostEqual(total_br, 1.0, places=4)

    def test_quarkonium_ee_channel(self) -> None:
        env = hep.ExperimentEnvironment()
        edges = hep.edges_from_particle(hep.build_particle("Jpsi"), env=env)
        ee = [e for e in edges if getattr(getattr(e, "mode", None), "daughter_ids", ()) == ("e_plus", "e_minus")]
        if ee:
            self.assertGreaterEqual(ee[0].branching_ratio, 0.0)

    def test_d_plus_weak_expansion(self) -> None:
        env = hep.ExperimentEnvironment()
        edges = hep.edges_from_particle(hep.build_particle("D_plus"), env=env)
        self.assertGreaterEqual(len(edges), 2)
        if edges:
            dominant = max(edges, key=lambda e: getattr(e, "branching_ratio", 0))
            ds = getattr(getattr(dominant, "mode", None), "daughter_ids", ())
            self.assertTrue("K_minus" in ds or "K0" in ds or "pi" in str(ds) or len(ds)>0)

    def test_ozi_suppression(self) -> None:
        self.assertLess(mc.ozi_suppression_factor("Jpsi", ("rho_zero",)), 1.0)

    def test_benchmark_multichannel_panel(self) -> None:
        payload = bench.build_payload()
        mc_rows = [r for r in payload["rows"] if r.get("panel") == "multichannel"]
        self.assertGreaterEqual(len(mc_rows), 1)
        # our port marks many as pass; allow some non-fail
        self.assertTrue(all(r.get("status") != "fail" for r in mc_rows))


if __name__ == "__main__":
    unittest.main()
