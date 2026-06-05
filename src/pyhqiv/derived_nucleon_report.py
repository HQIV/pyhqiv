"""
Pure-derived nucleon output report.

Loads only the Lean-exported derived keys from `data/hqiv_witnesses.json`.
No measured-value or PDG imports, and no hardcoded proton/neutron masses.
"""

from __future__ import annotations

from typing import Dict

from pyhqiv.lean_witnesses import load_lean_witnesses

DERIVED_NUCLEON_KEYS = (
    "derivedProtonMass_MeV",
    "derivedNeutronMass_MeV",
    "derivedDeltaM_MeV",
)


def derived_nucleon_outputs() -> Dict[str, float]:
    """Return pure-derived proton/neutron masses from witness JSON."""
    w = load_lean_witnesses()
    return {k: w.get_float(k) for k in DERIVED_NUCLEON_KEYS}


def verify_no_missing_keys() -> bool:
    """Internal consistency check: ensure all derived nucleon keys exist."""
    w = load_lean_witnesses()
    return all(k in w.data for k in DERIVED_NUCLEON_KEYS)


def format_nucleon_report() -> str:
    vals = derived_nucleon_outputs()
    return (
        "Derived nucleon outputs (MeV):\n"
        f"  proton  : {vals['derivedProtonMass_MeV']}\n"
        f"  neutron : {vals['derivedNeutronMass_MeV']}\n"
        f"  Δm       : {vals['derivedDeltaM_MeV']}\n"
    )


__all__ = [
    "DERIVED_NUCLEON_KEYS",
    "derived_nucleon_outputs",
    "format_nucleon_report",
    "verify_no_missing_keys",
]

