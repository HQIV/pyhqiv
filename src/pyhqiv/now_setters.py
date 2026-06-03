"""
now_setters: canonical "now" setter (electron horizon).

Corrected HQIV philosophy enforced:
- The current discrete horizon cutoff θ_now **is** the electron horizon.
- The electron horizon shell index is the ONLY primary setter of "now".
- CMB/LSS/temperature witnesses are downstream outputs and must not set "now".

This module provides a single canonical setter:
  now_set_from_electron_horizon(m_electron)

It also keeps legacy temperature/CMB setter names as hard-deprecated
entrypoints that raise, to prevent accidental reintroduction of the old model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pyhqiv.auxiliary_field import phi_of_shell
from pyhqiv.lean_witnesses import load_lean_witnesses


class LegacyNowAnchorError(RuntimeError):
    pass


# Canonical global: "now" shell index is the electron horizon shell.
# Default is the Lean-exported reference shell.
m_now: int = load_lean_witnesses().get_int("m_now_electron_shell")


@dataclass(frozen=True)
class NowGeometry:
    """
    Minimal cached geometry at the current electron-horizon cutoff.
    """

    m_now: int
    phi_now: float


_now_geometry: NowGeometry | None = None
_active_slice: dict[str, float | int] | None = None


def rescale_geometry() -> NowGeometry:
    """
    Recompute cached present-epoch geometry derived from the electron horizon.
    """
    global _now_geometry
    _now_geometry = NowGeometry(m_now=m_now, phi_now=phi_of_shell(m_now))
    return _now_geometry


def now_set_from_electron_horizon(m_electron: int) -> NowGeometry:
    """
    Sets the current horizon cutoff to the electron horizon shell.
    This is the ONLY knob that defines 'now' in HQIV.
    All downstream geometry, lapse, φ(m), masses, couplings rescale from here.
    """
    global m_now
    # Accept mpmath.mpf (and other numeric types) but require an integer shell.
    try:
        as_int = int(m_electron)
    except Exception as e:  # noqa: BLE001
        raise ValueError("m_electron must be an integer-like shell index") from e
    if float(m_electron) != float(as_int) or as_int < 0:
        raise ValueError("m_electron must be a non-negative integer shell index")
    m_now = as_int
    return rescale_geometry()


def nowSetter(
    witness_value: float,
    slices: Iterable[dict[str, float | int]] | None = None,
) -> dict[str, float | int]:
    """
    Canonical active-slice selector for the computational engine.

    `slices` entries are dictionaries with keys:
      - `id` (int)
      - `lower` (float, inclusive)
      - `upper` (float, exclusive)
      - optional payload keys consumed by downstream engines
    """
    global _active_slice
    if slices is None:
        base = int(witness_value)
        now_set_from_electron_horizon(base)
        _active_slice = {"id": base, "lower": float(base), "upper": float(base + 1)}
        return _active_slice

    selected: dict[str, float | int] | None = None
    for entry in slices:
        lower = float(entry["lower"])
        upper = float(entry["upper"])
        if lower <= float(witness_value) < upper:
            selected = dict(entry)
            break
    if selected is None:
        raise ValueError("witness_value does not fall into any provided slice range")

    now_set_from_electron_horizon(int(selected["id"]))
    _active_slice = selected
    return selected


def active_slice() -> dict[str, float | int]:
    if _active_slice is None:
        return nowSetter(float(load_lean_witnesses().get_int("m_now_electron_shell")))
    return _active_slice


def now_geometry() -> NowGeometry:
    """
    Return cached geometry at the current "now" cutoff.
    """
    return rescale_geometry() if _now_geometry is None else _now_geometry


def now_set_from_temperature_witness(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise LegacyNowAnchorError(
        "Legacy anchor disabled: temperature/CMB must not define 'now'. "
        "Use now_set_from_electron_horizon(m_electron) instead."
    )


def cmb_effective_temperature_from_polarization(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise LegacyNowAnchorError(
        "Legacy anchor disabled: CMB polarization witness must not define 'now'."
    )


def now_set_from_cmb_polarization_witness(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise LegacyNowAnchorError(
        "Legacy anchor disabled: CMB polarization witness must not define 'now'. "
        "Use now_set_from_electron_horizon(m_electron) instead."
    )


__all__ = [
    "LegacyNowAnchorError",
    "NowGeometry",
    "active_slice",
    "m_now",
    "nowSetter",
    "now_geometry",
    "now_set_from_electron_horizon",
    "rescale_geometry",
]

