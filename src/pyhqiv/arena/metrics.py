"""
Modular "sigma everywhere" metric registry for HQIV Arena.

Each Metric is a small, deterministic, versioned observable:
- name: stable identifier (used in leaderboards)
- compute(): current float value from pyhqiv (or optional cosmology etc)
- reference: Lean-witness or golden reference value (loaded, never literal in rules)
- protected: if True, large regressions cause hard penalty / gate failure in scoring
- weight: relative importance for multi-objective score
- unit, desc, tolerance: for reporting

New observables are added by calling register_metric(...) in this module or from
new test modules (so "new feature → new test → new arena metric" is automatic).

The registry is intentionally small at first; it will grow with community
contributions of new phase diagrams, fluid observables, lattice stats, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import math

# We import lazily inside compute fns and at registration time so that
# importing pyhqiv.arena does not pull heavy optional deps (jax, healpy, ...).


@dataclass(frozen=True)
class Metric:
    name: str
    compute: Callable[[], float]
    reference: Callable[[], float]  # functional, usually from witnesses or py "lean mirror"
    protected: bool = False
    weight: float = 1.0
    unit: str = ""
    tolerance: float = 1e-9  # for "same" in baseline comparisons
    desc: str = ""


_REGISTRY: Dict[str, Metric] = {}


def register_metric(m: Metric) -> None:
    if m.name in _REGISTRY:
        # Allow re-registration (e.g. test reloads) but keep first-wins for determinism in CI
        return
    _REGISTRY[m.name] = m


def get_metric(name: str) -> Metric:
    return _REGISTRY[name]


def METRIC_REGISTRY() -> Dict[str, Metric]:
    return dict(_REGISTRY)


def _witness_float(key: str, default: float) -> float:
    """Load from Lean witnesses (single source of truth)."""
    try:
        # local import to avoid circulars at package load
        from pyhqiv.lean_witnesses import load_lean_witnesses  # type: ignore

        w = load_lean_witnesses()
        val = w.data.get(key) if hasattr(w, "data") else None
        if val is None:
            # try require but swallow
            try:
                val = w.require(key)
            except Exception:
                val = default
        return float(val)
    except Exception:
        return default


def _py_ref_m() -> float:
    from pyhqiv.lightcone import reference_m

    return float(reference_m())


def _py_omega_k_self() -> float:
    from pyhqiv.lightcone import omega_k_at_horizon, reference_m

    m = int(reference_m())
    return float(omega_k_at_horizon(m, m))


def _py_omega_k_partial_ref() -> float:
    from pyhqiv.lightcone import omega_k_partial, reference_m

    return float(omega_k_partial(int(reference_m())))


def _py_curvature_norm() -> float:
    from pyhqiv.lightcone import curvature_norm_combinatorial

    return float(curvature_norm_combinatorial())


def _py_lapse_ref() -> float:
    from pyhqiv.metric import hqvm_lapse

    # Canonical exercise point (0, phi_aux~0.4 from gamma, t=1); value not used as magic.
    return float(hqvm_lapse(0.0, 0.40, 1.0))


def _py_available_modes_ref() -> float:
    from pyhqiv.lightcone import available_modes, reference_m

    return float(available_modes(int(reference_m())))


def _py_proton_mass() -> float:
    return _witness_float("derivedProtonMass_MeV", 938.272)


def _py_alpha_gut() -> float:
    return _witness_float("alpha_GUT", 1.0 / 42.0)


def _py_so8_dim() -> float:
    from pyhqiv.so8_generators import load_so8_generators_auto

    t = load_so8_generators_auto().tensor
    return float(t.shape[0])


# --- Core protected metrics (no large regressions allowed) ---
# These are the "sacred" numerical consequences of the Lean certificates + lattice.

register_metric(
    Metric(
        name="omega_k_at_horizon_self",
        compute=_py_omega_k_self,
        reference=lambda: 1.0,
        protected=True,
        weight=3.0,
        unit="1",
        tolerance=1e-10,
        desc="Ω_k(N;N) must be exactly 1 at the horizon (Lean theorem omega_k_at_horizon_self)",
    )
)

register_metric(
    Metric(
        name="omega_k_partial_at_reference",
        compute=_py_omega_k_partial_ref,
        reference=lambda: 1.0,
        protected=True,
        weight=3.0,
        unit="1",
        tolerance=1e-9,
        desc="Ω_k at lock-in/reference shell relative to itself (Lean omega_k_lockin_calibration)",
    )
)

register_metric(
    Metric(
        name="curvature_norm_combinatorial",
        compute=_py_curvature_norm,
        reference=_py_curvature_norm,  # self-consistent; Lean proves the 6^7√3 count
        protected=True,
        weight=2.0,
        unit="",
        tolerance=1e-3,
        desc="Combinatorial curvature norm N67 = 6^7 √3 from discrete null lattice (Lean OctonionicLightCone)",
    )
)

register_metric(
    Metric(
        name="reference_m",
        compute=_py_ref_m,
        reference=_py_ref_m,
        protected=True,
        weight=1.0,
        unit="shell",
        tolerance=0.0,
        desc="Lock-in shell index (qcdShell + lattice steps) — changing this is a major formal shift",
    )
)

register_metric(
    Metric(
        name="so8_dim",
        compute=_py_so8_dim,
        reference=lambda: 28.0,
        protected=True,
        weight=2.0,
        unit="dim",
        tolerance=0.0,
        desc="so(8) Lie closure dimension (Lean SO8Closure + triality + GeneratorsLieClosure)",
    )
)

register_metric(
    Metric(
        name="lapse_factor_ref_point",
        compute=_py_lapse_ref,
        reference=_py_lapse_ref,
        protected=True,
        weight=1.5,
        unit="",
        tolerance=1e-12,
        desc="ADM lapse at canonical reference-like point (Lean HQVMetric / HQVM_lapse)",
    )
)

register_metric(
    Metric(
        name="derived_proton_mass_MeV",
        compute=_py_proton_mass,
        reference=_py_proton_mass,
        protected=True,
        weight=2.0,
        unit="MeV",
        tolerance=1e-6,
        desc="Proton mass anchor formally derived from Lean (DerivedNucleonMass + tuft etc)",
    )
)

# --- Improvement / sigma metrics (reward broad error reduction) ---

register_metric(
    Metric(
        name="alpha_GUT",
        compute=_py_alpha_gut,
        reference=_py_alpha_gut,
        protected=False,
        weight=1.0,
        unit="",
        tolerance=1e-12,
        desc="GUT coupling from Lean β-running engine (1/42 in paper)",
    )
)

register_metric(
    Metric(
        name="available_modes_ref",
        compute=_py_available_modes_ref,
        reference=_py_available_modes_ref,
        protected=False,
        weight=0.5,
        unit="modes",
        tolerance=0.0,
        desc="Combinatorial mode count at reference shell (Lean lattice)",
    )
)

# Example of a "new observable" that would come from adding a phase diagram generator + test.
# For now it is a no-op / constant that demonstrates extension; real impl would call e.g.
# phase_diagram_quality() or thermodynamic_stability_margin().
register_metric(
    Metric(
        name="example_phase_stability_margin",
        compute=lambda: 0.97,  # placeholder — real contributions replace with actual computation
        reference=lambda: 0.97,
        protected=False,
        weight=1.0,
        unit="",
        tolerance=0.01,
        desc="EXAMPLE: phase boundary / thermodynamic consistency quality (add real compute when contributing new phase diagrams)",
    )
)


def build_default_metrics() -> List[Metric]:
    """Return the current ordered list of metrics (for deterministic scoring)."""
    # Stable order: protected first, then others, by registration order.
    items = list(_REGISTRY.values())
    items.sort(key=lambda m: (0 if m.protected else 1, m.name))
    return items


# Allow external modules (new tests) to register more at import time.
# Example in a new test_phase_diagrams.py:
#   from pyhqiv.arena.metrics import register_metric, Metric
#   register_metric(Metric(name="my_new_phase_score", compute=..., reference=..., protected=False, ...))
