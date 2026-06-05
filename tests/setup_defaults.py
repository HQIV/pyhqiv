"""
Test-only setup defaults and local conditions with explicit error bars from source material.

Rules (per user + arena):
- src/pyhqiv/ contains ZERO scale/measurement constants except geometry (pi, sqrt(3), naturals).
- All scale witnesses, local conditions (proton anchor for comparison, CMB now, earth surface g,
  earth vacuum eps0/mu0, etc.) live in data (witnesses.json, local_conditions.json) + this test helper.
- Every test observable that compares to experiment MUST record its (central, +/- err) from the
  authoritative source (PDG, CODATA, Planck, etc.) here or sibling *_with_errors.py .
- New arena features MUST come with new tests that carry error bars; CI enforces sigma improvement.

This module is NOT imported by src/; only tests, examples (for repro), and arena scoring use it.
"""

from __future__ import annotations

from typing import Dict, Tuple

from pyhqiv.scale_witness import (
    load_local_conditions,
    local_cmb_temperature_K,
    local_cmb_temperature_uncertainty_K,
)


def get_local_cmb() -> Tuple[float, float, str]:
    """(T_K, +/-unc_K, source)"""
    load_local_conditions()
    t = local_cmb_temperature_K()
    unc = local_cmb_temperature_uncertainty_K()
    src = "Planck 2018 (https://arxiv.org/abs/1807.06209); value/unc from local_conditions.json"
    return t, unc, src


def get_local_proton_for_comparison() -> Tuple[float, float, str]:
    """Comparison value only (primary is Lean derivedProtonMass)."""
    local = load_local_conditions()
    val = float(local["local_proton_mass_MeV_for_comparison"])
    unc = float(local.get("local_proton_mass_uncertainty_MeV", 6e-7))
    src = "PDG 2024 / CODATA; see local_conditions.json"
    return val, unc, src


def get_earth_g() -> Tuple[float, float, str]:
    local = load_local_conditions()
    val = float(local["earth_surface_g_m_per_s2"])
    unc = float(local.get("earth_surface_g_uncertainty", 1e-5))
    src = "BIPM / CODATA conventional standard gravity"
    return val, unc, src


def get_earth_vacuum_permittivity() -> Tuple[float, float, str]:
    local = load_local_conditions()
    val = float(local["earth_surface_vacuum_eps0_F_per_m"])
    unc = float(local.get("earth_surface_vacuum_eps0_uncertainty", 1e-21))
    src = "CODATA 2018"
    return val, unc, src


# Aggregated for tests that want a single "local conditions bundle with errors"
LOCAL_CONDITIONS_WITH_ERRORS: Dict[str, Tuple[float, float, str]] = {
    "cmb_T0_K": get_local_cmb(),
    "proton_MeV_comparison": get_local_proton_for_comparison(),
    "earth_g": get_earth_g(),
    "earth_eps0": get_earth_vacuum_permittivity(),
}


# For arena / new tests: when you add a feature that produces a number compared to exp,
# add the (value, err, "Source YYYY (doi or bib)") tuple here or in data/*.py and register
# a metric that computes |pred - central| / err  (or chi2 style) and aim to drive it down.

__all__ = [
    "get_local_cmb",
    "get_local_proton_for_comparison",
    "get_earth_g",
    "get_earth_vacuum_permittivity",
    "LOCAL_CONDITIONS_WITH_ERRORS",
]
