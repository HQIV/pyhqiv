"""
JSON-only guard: ensure the witness payload contains the pure-derived keys
and does not contain forbidden measurement-reference / PDG-style keys.

This test is intentionally *Lean-build free*; it only reads `data/hqiv_witnesses.json`.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WITNESS_PATH = REPO_ROOT / "data" / "hqiv_witnesses.json"


FORBIDDEN_TOKENS = (
    "PDG",
    "CODATA",
    "m_tau_from_resonance",
    "resonance_k_tau_mu",
    "n_1440_MeV",
    "n_1520_MeV",
    "pdg_reference_masses_mev",
    "percent_errors",
)


def test_witness_json_has_only_expected_derived_keys():
    data = json.loads(WITNESS_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)

    expected_keys = {
        "m_H",
        "M_W",
        "M_Z",
        "m_nu_e",
        "m_nu_mu",
        "m_nu_tau",
        "resonanceK_outer_0_1",
        "resonanceK_outer_1_2",
        "derivedProtonMass_MeV",
        "derivedNeutronMass_MeV",
        "derivedDeltaM_MeV",
        "proton_neutron_delta",
    }

    # Key-set must match exactly for the pure-derived mode.
    assert set(data.keys()) == expected_keys


def test_witness_json_has_no_forbidden_measurement_tokens():
    text = WITNESS_PATH.read_text(encoding="utf-8")
    offenders = [t for t in FORBIDDEN_TOKENS if t in text]
    assert not offenders, "Forbidden tokens found in witness JSON: " + ", ".join(offenders)

