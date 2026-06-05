"""
Pure-derived gauge/lepton output report.

Loads only the derived keys exported by `scripts/export_witnesses.lean`.
"""

from __future__ import annotations

from typing import Dict

from pyhqiv.lean_witnesses import load_lean_witnesses

DERIVED_KEYS = (
    "m_H",
    "M_W",
    "M_Z",
    "m_nu_e",
    "m_nu_mu",
    "m_nu_tau",
    "resonanceK_outer_0_1",
    "resonanceK_outer_1_2",
    "proton_neutron_delta",
)


def derived_gauge_outputs() -> Dict[str, float]:
    """Return pure-derived gauge/lepton outputs from witness JSON."""
    w = load_lean_witnesses()
    return {k: w.get_float(k) for k in DERIVED_KEYS}


def verify_pure_derived_keys_only() -> bool:
    """
    Internal consistency check: all required pure-derived keys exist.
    """
    vals = derived_gauge_outputs()
    return all(k in vals for k in DERIVED_KEYS)


__all__ = ["DERIVED_KEYS", "derived_gauge_outputs", "verify_pure_derived_keys_only"]
