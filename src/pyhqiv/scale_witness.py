"""
Scale witness and local conditions loader for the HQIV calculator.

- ScaleWitness selects the active anchor (default proton_lockin from Lean derived mass).
- All Lean-derived anchors come from lean_witnesses (single source; no literals).
- Local conditions (CMB now, earth surface, vacuum) are loaded from local_conditions.json
  (data file, not .py source). These are for tests, applied modules, and "local condition"
  use cases only — never baked into core geometry or pure functions.
- Core src/ modules must contain zero measurement or scale literals except
  geometry necessities (pi, sqrt(3) for cube diagonal, natural-unit 1.0, 2pi phase).

This mirrors the structure in HQIV_LEAN/papers/.../scripts/hqiv_scale_witness.py
but is data-driven and integrated with pyhqiv.lean_witnesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import json
import math

import importlib.resources as resources
from functools import lru_cache

from pyhqiv.lean_witnesses import load_lean_witnesses, LeanWitnesses


class ScaleWitness(str, Enum):
    """Active single-scale discipline for readouts and applied pipelines."""
    PROTON_LOCKIN = "proton_lockin"
    CODATA_ALPHA = "codata_alpha"
    CMB_NOW = "cmb_now"


DEFAULT_SCALE_WITNESS: ScaleWitness = ScaleWitness.PROTON_LOCKIN


def _default_local_conditions_path() -> Path:
    """Packaged local_conditions.json next to witnesses.json."""
    try:
        p = resources.files("pyhqiv").joinpath("local_conditions.json")
        return Path(str(p))
    except Exception:
        # Fallback for editable installs / symlinks
        here = Path(__file__).resolve().parent
        return here / "local_conditions.json"


@lru_cache(maxsize=1)
def load_local_conditions(path: Path | None = None) -> dict[str, Any]:
    p = path or _default_local_conditions_path()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("local_conditions.json must be an object")
    return data


@lru_cache(maxsize=1)
def get_scale_witness_bundle() -> dict[str, Any]:
    """Return merged witness + local for convenience (derived first)."""
    w: LeanWitnesses = load_lean_witnesses()
    local = load_local_conditions()
    # derived come from Lean (overlay wins); fallbacks only if key absent in this witness set
    try:
        proton = w.get_float("derivedProtonMass_MeV")
    except Exception:
        proton = float(local["local_proton_mass_MeV_for_comparison"])
    try:
        neutron = w.get_float("derivedNeutronMass_MeV")
    except Exception:
        neutron = float(local["local_neutron_mass_MeV_for_comparison"])
    return {
        "scale_witness_default": str(DEFAULT_SCALE_WITNESS),
        "reference_m": w.get_int("referenceM"),
        "derived_proton_mass_MeV": proton,
        "derived_neutron_mass_MeV": neutron,
        "cmb_temperature_now_K": float(local["cmb_temperature_now_K"]),
        "cmb_temperature_uncertainty_K": float(local["cmb_temperature_uncertainty_K"]),
        "earth_g_m_s2": float(local["earth_surface_g_m_per_s2"]),
        "earth_eps0": float(local["earth_surface_vacuum_eps0_F_per_m"]),
        "earth_mu0": float(local["earth_surface_vacuum_mu0_N_per_A2"]),
    }


def local_cmb_temperature_K() -> float:
    """CMB temperature as local condition (earth-like now). Source + unc in local_conditions.json."""
    local = load_local_conditions()
    return float(local["cmb_temperature_now_K"])


def local_cmb_temperature_uncertainty_K() -> float:
    local = load_local_conditions()
    return float(local["cmb_temperature_uncertainty_K"])


def local_earth_surface_g() -> float:
    local = load_local_conditions()
    return float(local["earth_surface_g_m_per_s2"])


def xi_g_for_witness(witness: ScaleWitness | str) -> float:
    """ξ_G / lock-in coordinate for the chosen witness (Lean consistent)."""
    w = str(witness)
    if w == ScaleWitness.PROTON_LOCKIN.value:
        return float(load_lean_witnesses().get_int("referenceM") + 1)
    if w == ScaleWitness.CMB_NOW.value:
        local = load_local_conditions()
        return float(local["cmb_now_shallow_xi"])  # shallow comparison chart (not the brace)
    return float(load_lean_witnesses().get_int("referenceM") + 1)


# Convenience re-exports for Lean-derived (always prefer over local)
def derived_proton_mass_MeV() -> float:
    try:
        return load_lean_witnesses().get_float("derivedProtonMass_MeV")
    except Exception:
        # last resort from local comparison only (value + error bar live in local_conditions.json)
        local = load_local_conditions()
        return float(local["local_proton_mass_MeV_for_comparison"])


def derived_neutron_mass_MeV() -> float:
    try:
        return load_lean_witnesses().get_float("derivedNeutronMass_MeV")
    except Exception:
        local = load_local_conditions()
        return float(local["local_neutron_mass_MeV_for_comparison"])


__all__ = [
    "ScaleWitness",
    "DEFAULT_SCALE_WITNESS",
    "load_local_conditions",
    "get_scale_witness_bundle",
    "local_cmb_temperature_K",
    "local_cmb_temperature_uncertainty_K",
    "local_earth_surface_g",
    "xi_g_for_witness",
    "derived_proton_mass_MeV",
    "derived_neutron_mass_MeV",
]
