"""
Unified HQIV state orchestration (shell ladder, witnesses, metric slice, optional SO(8) carrier).

This module does not duplicate the derivation chain; it composes:

- ``Hqiv.Geometry.OctonionicLightCone`` → :mod:`pyhqiv.lightcone`
- ``Hqiv.Geometry.AuxiliaryField`` → :mod:`pyhqiv.auxiliary_field`
- ``Hqiv.Geometry.HQVMetric`` → :mod:`pyhqiv.metric`

Lean-backed anchors load through :mod:`pyhqiv.lean_witnesses` (no new hardcoded physics literals).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from pyhqiv import auxiliary_field as af
from pyhqiv import lightcone as lc
from pyhqiv import metric
from pyhqiv.carrier import So8Carrier
from pyhqiv.lean_witnesses import LeanWitnesses, load_lean_witnesses


@dataclass
class HQIVState:
    """
    Single object tying together a shell index, optional horizon for Ω_k ratios,
    auxiliary-field–driven metric inputs, Lean witnesses, and an optional octonionic carrier.
    """

    m: int
    horizon_n: int | None = None
    phi_newtonian: float = 0.0
    phi_auxiliary: float | None = None
    t: float = 0.0
    carrier: So8Carrier | None = None
    witness_path: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.m < 0:
            raise ValueError("shell index m must be non-negative")
        if self.horizon_n is not None and self.horizon_n < 0:
            raise ValueError("horizon_n must be non-negative when set")

    def phi_aux(self) -> float:
        """Auxiliary φ at this state; defaults to discrete shell value φ(m)."""
        if self.phi_auxiliary is not None:
            return float(self.phi_auxiliary)
        return af.phi_of_shell(self.m)

    def witnesses(self) -> LeanWitnesses:
        return load_lean_witnesses(self.witness_path)

    def curvature_summary(self) -> dict[str, float]:
        """
        Curvature ladder quantities at shell ``m`` (and optional horizon ratio).

        See :mod:`pyhqiv.lightcone` for Lean lemma names.
        """
        n = self.m + 1
        out: dict[str, float] = {
            "m": float(self.m),
            "alpha": lc.alpha(),
            "gamma_hqiv": metric.gamma_hqiv(),
            "shell_temperature": af.shell_temperature(self.m),
            "phi_auxiliary": self.phi_aux(),
            "shell_shape": lc.shell_shape(self.m),
            "curvature_integral_through_m": lc.curvature_integral(n),
            "omega_k_partial": lc.omega_k_partial(self.m),
        }
        if self.horizon_n is not None:
            out["omega_k_at_horizon"] = lc.omega_k_at_horizon(self.m, self.horizon_n)
        return out

    def metric_summary(self) -> dict[str, float]:
        """Lapse, metric component, and homogeneous G_eff(φ) at this state."""
        phi_a = self.phi_aux()
        snap = metric.build_metric_snapshot(self.phi_newtonian, phi_a, self.t)
        return {
            "phi_newtonian": float(self.phi_newtonian),
            "phi_auxiliary": phi_a,
            "t": float(self.t),
            "lapse": snap.lapse,
            "g_tt": snap.g_tt,
            "time_angle": snap.time_angle_value,
            "g_eff": metric.g_eff(phi_a),
        }

    def with_shell(self, m: int) -> HQIVState:
        """Copy with a new shell index (same metric knobs and carrier reference)."""
        if m < 0:
            raise ValueError("shell index m must be non-negative")
        return replace(self, m=m)

    @classmethod
    def from_snapshot(
        cls,
        *,
        m: int,
        horizon_n: int | None = None,
        phi_newtonian: float = 0.0,
        phi_auxiliary: float | None = None,
        t: float = 0.0,
        carrier: So8Carrier | None = None,
        witness_path: str | None = None,
    ) -> HQIVState:
        """
        Construct a state from explicit fields (same defaults as the dataclass).
        """
        return cls(
            m=m,
            horizon_n=horizon_n,
            phi_newtonian=phi_newtonian,
            phi_auxiliary=phi_auxiliary,
            t=t,
            carrier=carrier,
            witness_path=witness_path,
        )

    def as_dict(self) -> dict[str, Any]:
        """Round-trip friendly summary for logging and examples."""
        d: dict[str, Any] = {
            "m": self.m,
            "horizon_n": self.horizon_n,
            "phi_newtonian": self.phi_newtonian,
            "phi_auxiliary": self.phi_auxiliary,
            "t": self.t,
            "has_carrier": self.carrier is not None,
        }
        d["curvature"] = self.curvature_summary()
        d["metric"] = self.metric_summary()
        return d


__all__ = ["HQIVState"]
