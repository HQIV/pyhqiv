"""Regression tests for HEP decay observations vs predictions benchmark (pyhqiv)."""

from __future__ import annotations

import pyhqiv.hep_decay_benchmark as bench


def test_full_benchmark_passes_without_known_gaps() -> None:
    payload = bench.build_payload()
    assert payload["summary"]["fail"] == 0
    assert payload["summary"].get("known_gap", 0) == 0
    assert payload["summary"]["pass"] >= 40


def test_observations_file_loads() -> None:
    obs = bench.load_json(bench.DEFAULT_OBSERVATIONS)
    assert obs.get("mass_panel")
    assert obs.get("decay_channels")


def test_payload_builds_with_summary() -> None:
    payload = bench.build_payload()
    assert payload["summary"]["total"] > 0
    assert payload["summary"]["pass"] > 0
    assert payload["comparison_policy"]


def test_proton_and_neutron_mass_tight() -> None:
    payload = bench.build_payload()
    rows = {r["case_id"]: r for r in payload["rows"] if r["panel"] == "mass"}
    assert rows["p"]["status"] == "pass"
    assert rows["n"]["status"] == "pass"
    assert abs(rows["p"]["error"]) < 1e-2


def test_lhc_kinematics_passes() -> None:
    payload = bench.build_payload()
    rows = {r["case_id"]: r for r in payload["rows"] if r["panel"] == "kinematics"}
    # may be present
    if "LHC_pp_13TeV" in rows:
        assert rows["LHC_pp_13TeV"]["status"] == "pass"


def test_delta_decay_topology_open() -> None:
    payload = bench.build_payload()
    open_rows = [r for r in payload["rows"] if "delta_p" in r.get("case_id", "")]
    if open_rows:
        assert any(r["status"] == "pass" for r in open_rows)
