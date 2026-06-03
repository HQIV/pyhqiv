"""
HQIV **lattice next shell** (Lean ``next_lattice_prime`` / ``next_prime_generator``).

This is **not** the smallest rational prime ``p > n``. It is the discrete shell ladder from:

* ``Hqiv.Physics.GlobalDetuning`` — ``effCorrected δ m = shellSurface(m) / (1 + c_rindler·m + δ)``
* ``Hqiv.Physics.OctonionicZeta`` — ``next_lattice_prime``: smallest ``m' > current_m`` with
  ``eff(m')/eff(current_m) ≥ threshold`` (default ``1.5`` in Lean)
* ``Hqiv.Physics.LatticeNextPrimeGenerator`` — ``decompose_to_fano_moduli`` then ``next_lattice_prime``
  on ``decompose_last_shell``

Input ``x`` is only used in the **greedy Fano-weighted** product along shells to pick ``last_m``;
the returned value is still a **shell index** ``m'``, not ``π(x)``.

Numerics reuse ``eff_corrected`` / ``rindler_den_with_delta`` from ``lepton_resonance_ladder``
(``γ = 2/5``, ``c_rindler = γ/2``).

**Experimental (Python-only):** ``eff_corrected_dynamic``, ``decompose_last_shell_dynamic``,
``next_lattice_prime_shell_dynamic``, ``hqiv_next_prime_generator_dynamic``, and
``hqiv_scaled_next_prime_stepwise`` allow per-shell δ(m), ``start_m ≠ 0``, and adaptive thresholds;
these are **not** the current Mathlib ``effCorrected`` / ``decompose_to_fano_moduli`` contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from pyhqiv.lepton_resonance_ladder import eff_corrected, rindler_den_with_delta, shell_surface


def delta_auxiliary_phi_per_shell(
    detuning_lambda: float, obs: float, phi: float, t: float, beta_cum: float
) -> float:
    """Lean ``GlobalDetuning.delta_auxiliary_phi_per_shell``: ``λ·obs + β_cum·φ·t``."""
    return float(detuning_lambda) * float(obs) + float(beta_cum) * float(phi) * float(t)


def fano_line_weight_for_shell(m: int) -> int:
    """Lean ``fanoLineWeight (fano_vertex_of_shell m)`` = ``(m % 7) % 3 + 1`` ∈ {1,2,3}."""
    return (m % 7) % 3 + 1


def eff_corrected_dynamic(delta_at_m: Callable[[int], float], m: int) -> float:
    """
    ``shellSurface(m) / rindlerDenWithDelta(δ(m), m)`` with **per-shell** δ(m).

    **Not Lean:** Mathlib ``effCorrected`` fixes one real δ for all m. Use for Python experiments only.
    """
    d = float(delta_at_m(m))
    return shell_surface(m) / rindler_den_with_delta(d, m)


def _require_rindler_den_dynamic(delta_at_m: Callable[[int], float], m: int, label: str) -> None:
    d = float(delta_at_m(m))
    if rindler_den_with_delta(d, m) <= 0.0:
        raise ValueError(f"RindlerDenDeltaPos failed at {label} m={m} with delta(m)={d}")


def decompose_last_shell(x: int, delta: float, fuel: int) -> int:
    """
    Lean ``decompose_last_shell``: last shell index in the greedy ``eff^l_f`` product trace, or ``0``
    if the trace is empty (e.g. ``x = 1`` stops immediately with ``acc = 1 ≥ target``).
    """
    if x <= 0:
        return 0
    if fuel < 0:
        raise ValueError("fuel must be nonnegative")
    target = float(x)
    m = 0
    acc = 1.0
    forward: list[int] = []
    for _ in range(fuel):
        if acc >= target:
            return forward[-1] if forward else 0
        lf = fano_line_weight_for_shell(m)
        eff = eff_corrected(delta, m)
        acc *= eff**lf
        forward.append(m)
        m += 1
    return forward[-1] if forward else 0


def decompose_last_shell_dynamic(
    x: int,
    delta_at_m: Callable[[int], float],
    fuel: int,
    *,
    start_m: int = 0,
) -> int:
    """
    Same greedy product as ``decompose_last_shell``, but δ may vary with m and the walk may start at
    ``start_m`` (default ``0``).

    **Not Lean:** decomposition always begins at ``m = 0`` in ``LatticeNextPrimeGenerator``.
    """
    if x <= 0:
        return 0
    if fuel < 0:
        raise ValueError("fuel must be nonnegative")
    if start_m < 0:
        raise ValueError("start_m must be nonnegative")
    target = float(x)
    m = start_m
    acc = 1.0
    forward: list[int] = []
    for _ in range(fuel):
        if acc >= target:
            return forward[-1] if forward else 0
        lf = fano_line_weight_for_shell(m)
        eff = eff_corrected_dynamic(delta_at_m, m)
        acc *= eff**lf
        forward.append(m)
        m += 1
    return forward[-1] if forward else 0


def _require_rindler_den_pos(delta: float, m: int, label: str) -> None:
    if rindler_den_with_delta(delta, m) <= 0.0:
        raise ValueError(f"RindlerDenDeltaPos failed at {label} m={m} with delta={delta}")


def next_lattice_prime_shell(
    current_m: int,
    delta: float,
    *,
    threshold: float = 1.5,
    max_shell: int = 10_000_000,
) -> int:
    """
    Lean ``next_lattice_prime``: minimal ``m' > current_m`` with
    ``effCorrected δ m' / effCorrected δ current_m ≥ threshold``.

    Requires ``threshold > 1`` and strictly positive Rindler denominator at ``current_m``
    (and at each candidate).
    """
    if threshold <= 1.0:
        raise ValueError("threshold must be > 1 (Lean: hth : 1 < threshold)")
    _require_rindler_den_pos(delta, current_m, "current_m")
    base = eff_corrected(delta, current_m)
    m_prime = current_m + 1
    while m_prime <= max_shell:
        _require_rindler_den_pos(delta, m_prime, "candidate")
        if eff_corrected(delta, m_prime) / base >= threshold - 1e-15:
            return m_prime
        m_prime += 1
    raise RuntimeError(f"next_lattice_prime_shell: no m' ≤ {max_shell} satisfies eff ratio ≥ {threshold}")


def next_lattice_prime_shell_dynamic(
    current_m: int,
    delta_at_m: Callable[[int], float],
    *,
    threshold: float = 1.5,
    max_shell: int = 10_000_000,
) -> int:
    """
    Like ``next_lattice_prime_shell`` but uses ``eff_corrected_dynamic`` so δ can depend on m.

    **Not Lean** (see ``eff_corrected_dynamic``).
    """
    if threshold <= 1.0:
        raise ValueError("threshold must be > 1 (Lean: hth : 1 < threshold)")
    _require_rindler_den_dynamic(delta_at_m, current_m, "current_m")
    base = eff_corrected_dynamic(delta_at_m, current_m)
    m_prime = current_m + 1
    while m_prime <= max_shell:
        _require_rindler_den_dynamic(delta_at_m, m_prime, "candidate")
        if eff_corrected_dynamic(delta_at_m, m_prime) / base >= threshold - 1e-15:
            return m_prime
        m_prime += 1
    raise RuntimeError(
        f"next_lattice_prime_shell_dynamic: no m' ≤ {max_shell} satisfies eff ratio ≥ {threshold}"
    )


@dataclass(frozen=True)
class HqivLatticePrimeParams:
    """Convenience bundle matching Lean's global detuning + rapidity slots."""

    detuning_lambda: float = 0.0
    obs: float = 0.0
    phi: float = 0.0
    t: float = 0.0
    beta_cum: float = 0.0
    fuel: int = 50_000
    threshold: float = 1.5
    max_shell: int = 10_000_000
    #: If set, ``delta`` ignores ``detuning_lambda`` / ``obs`` / ``phi`` / ``t`` / ``beta_cum``.
    fixed_delta: float | None = None

    @property
    def delta(self) -> float:
        if self.fixed_delta is not None:
            return float(self.fixed_delta)
        return delta_auxiliary_phi_per_shell(
            self.detuning_lambda, self.obs, self.phi, self.t, self.beta_cum
        )


def hqiv_next_prime_generator(x: int, params: HqivLatticePrimeParams | None = None) -> int:
    """
    Lean ``next_prime_generator x …``: ``next_lattice_prime (decompose_last_shell x)``.

    Returns an **integer shell index** ``m'``, not a rational prime.
    """
    p = params or HqivLatticePrimeParams()
    delta = p.delta
    if delta < 0.0:
        raise ValueError("delta_auxiliary_phi_per_shell must be ≥ 0 (Lean: 0 ≤ δ)")
    last_m = decompose_last_shell(x, delta, p.fuel)
    _require_rindler_den_pos(delta, last_m, "last_m")
    return next_lattice_prime_shell(
        last_m, delta, threshold=p.threshold, max_shell=p.max_shell
    )


def hqiv_next_prime_generator_dynamic(
    x: int,
    delta_at_m: Callable[[int], float],
    params: HqivLatticePrimeParams | None = None,
    *,
    decompose_start_m: int = 0,
) -> int:
    """
    ``next_lattice_prime`` after ``decompose_last_shell_dynamic`` with the same δ(m) law.

    **Not Lean.**
    """
    p = params or HqivLatticePrimeParams()
    last_m = decompose_last_shell_dynamic(
        x, delta_at_m, p.fuel, start_m=decompose_start_m
    )
    _require_rindler_den_dynamic(delta_at_m, last_m, "last_m")
    return next_lattice_prime_shell_dynamic(
        last_m, delta_at_m, threshold=p.threshold, max_shell=p.max_shell
    )


def hqiv_scaled_next_prime_stepwise(
    x: int,
    *,
    phi_base: float = 1.0,
    beta_cum: float = 0.05,
    base_threshold: float = 1.5,
    fuel: int = 500_000,
) -> int:
    """
    Experimental recipe (user sketch): linear ``φt(m) = phi_base * m`` folded into
    ``δ(m) = beta_cum * phi_base * m``, decomposition starting near ``√x``, adaptive threshold
    ``base_threshold + log2(log2(x+2)+1)``.

    **Not Lean.** Returns a shell index ``m'``. For ``x < 2`` returns ``2`` (convenience, not Mathlib).
    """
    if x < 2:
        return 2
    adaptive = base_threshold + math.log2(math.log2(x + 2) + 1.0)

    def delta_at_m(m: int) -> float:
        return float(beta_cum) * float(phi_base) * float(m)

    m_start = int(math.sqrt(x)) if x > 1 else 0
    params = HqivLatticePrimeParams(fuel=fuel, threshold=float(adaptive))
    return hqiv_next_prime_generator_dynamic(
        x, delta_at_m, params, decompose_start_m=m_start
    )


__all__ = [
    "HqivLatticePrimeParams",
    "delta_auxiliary_phi_per_shell",
    "decompose_last_shell",
    "decompose_last_shell_dynamic",
    "eff_corrected_dynamic",
    "fano_line_weight_for_shell",
    "hqiv_next_prime_generator",
    "hqiv_next_prime_generator_dynamic",
    "hqiv_scaled_next_prime_stepwise",
    "next_lattice_prime_shell",
    "next_lattice_prime_shell_dynamic",
]
