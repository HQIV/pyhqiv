"""
Galactic-scale HQIV metric knobs: varying Newton coupling and entanglement split.

Lean references:

- ``Hqiv.Geometry.HQVMetric`` (``G_eff``, ``gamma_HQIV``, lapse)
- ``Hqiv.Physics.SM_GR_Unification`` (auxiliary-field bridge into phenomenology)

Python implementation lives in :mod:`pyhqiv.metric` and :mod:`pyhqiv.state`.
"""

from __future__ import annotations

from pyhqiv.metric import g_eff, gamma_hqiv
from pyhqiv.state import HQIVState


def galactic_g_eff(phi_auxiliary: float) -> float:
    """``G_eff(φ) = G0 (H/H0)^α`` with ``H(φ)=φ`` in the homogeneous identification."""
    return g_eff(phi_auxiliary)


def galactic_gamma_hqiv() -> float:
    """``γ = 1 - α`` (entanglement-monogamy complement of the curvature exponent)."""
    return gamma_hqiv()


def galactic_metric_summary(state: HQIVState) -> dict[str, float]:
    """Delegate to :meth:`HQIVState.metric_summary` for a single call site in demos."""
    return state.metric_summary()


__all__ = ["galactic_g_eff", "galactic_gamma_hqiv", "galactic_metric_summary"]
