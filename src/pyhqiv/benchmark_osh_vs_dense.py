"""
Dense vs. sparse horizon-causal benchmark (HQIV / OSH paper, Sec.~Technical benchmark).

Mirrors ``HQIV_LEAN/scripts/benchmark_protein_osh_vs_dense.py`` so **pyhqiv** can
reproduce tables side-by-side with the manuscript text in
``HQIV_LEAN/paper/octonion_lightcone_to_oshoracle.tex``.

- **Dense path**: full ``O(n^2)`` additive-field accumulation each step on a dense
  amplitude vector (Python baseline).
- **Sparse path**: ``O(k^2)`` active-support additive field + expand / gate / flip–prune
  style update (OSH-style bookkeeping).

Lean touchpoints: ``Hqiv/ProteinResearch/AdditiveFieldAndTorque.lean``,
``Hqiv/QuantumComputing/OSHoracle.lean``. This harness is **computational** only.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Coord3 = Tuple[float, float, float]


def available_modes(m: int) -> float:
    return 4.0 * (m + 2) * (m + 1)


def phi_of_shell(m: int) -> float:
    return 2.0 * (m + 1)


def lattice_full_mode_energy(m: int) -> float:
    return available_modes(m) * (phi_of_shell(m) / 2.0)


def additive_field_kernel(i: int, j: int, shells: Sequence[int], pos: Sequence[Coord3]) -> float:
    dx = pos[i][0] - pos[j][0]
    dy = pos[i][1] - pos[j][1]
    dz = pos[i][2] - pos[j][2]
    r_proxy = abs(dx) + abs(dy) + abs(dz) + 1.0
    return lattice_full_mode_energy(shells[j]) / r_proxy


def should_update_torque(step: int, update_every: int = 10) -> bool:
    return True if update_every == 0 else (step % update_every == 0)


def init_problem(
    n_sites: int, active_fraction: float, seed: int
) -> Tuple[List[int], List[Coord3], List[float], List[float]]:
    rng = random.Random(seed)
    shells = [rng.randint(0, 12) for _ in range(n_sites)]
    pos = [(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(n_sites)]
    amp = [0.0] * n_sites
    n_active = max(1, int(n_sites * active_fraction))
    for idx in rng.sample(range(n_sites), n_active):
        amp[idx] = rng.uniform(0.1, 1.0)

    orientation = [rng.uniform(-1.0, 1.0) for _ in range(n_sites)]
    return shells, pos, amp, orientation


def dense_step(
    step: int,
    shells: Sequence[int],
    pos: Sequence[Coord3],
    orientation: Sequence[float],
    amp: List[float],
    torque_cache: List[float],
    update_every: int,
) -> None:
    n = len(amp)
    field = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += additive_field_kernel(i, j, shells, pos)
        field[i] = s

    if should_update_torque(step, update_every):
        for i in range(n):
            torque_cache[i] = orientation[i] * field[i]

    new_amp = [0.0] * n
    for i in range(n):
        expanded = amp[i] + amp[(i - 1) % n]
        new_amp[i] = expanded * (1.0 + 1e-6 * torque_cache[i])

    amp[:] = new_amp


def sparse_from_dense(amp: Sequence[float], eps: float = 1e-12) -> Dict[int, float]:
    return {i: a for i, a in enumerate(amp) if abs(a) > eps}


def sparse_step(
    step: int,
    shells: Sequence[int],
    pos: Sequence[Coord3],
    orientation: Sequence[float],
    sparse: Dict[int, float],
    torque_cache: Dict[int, float],
    n_basis: int,
    update_every: int,
) -> Dict[int, float]:
    active = list(sparse.keys())
    if not active:
        return {}

    field: Dict[int, float] = {}
    for i in active:
        s = 0.0
        for j in active:
            s += additive_field_kernel(i, j, shells, pos)
        field[i] = s

    if should_update_torque(step, update_every):
        for i in active:
            torque_cache[i] = orientation[i] * field[i]

    expanded: List[Tuple[int, float]] = []
    for i, a in sparse.items():
        expanded.append((i % n_basis, a))
        expanded.append(((i + 1) % n_basis, a))

    accum: Dict[int, float] = {}
    for i, a in expanded:
        accum[i] = accum.get(i, 0.0) + a

    after: Dict[int, float] = {}
    for i, a in accum.items():
        t = torque_cache.get(i, 0.0)
        after[i] = a * (1.0 + 1e-6 * t)

    before_idx = set(sparse.keys())
    after_idx = set(after.keys())
    flipped = (before_idx - after_idx) | (after_idx - before_idx)
    if not flipped:
        return after
    return {i: a for i, a in after.items() if i in flipped}


@dataclass
class BenchCase:
    n_sites: int
    steps: int
    active_fraction: float
    update_every: int
    runs: int


def run_case(case: BenchCase, seed: int) -> Dict[str, float]:
    dense_times_ms: List[float] = []
    sparse_times_ms: List[float] = []
    final_active_counts: List[int] = []

    for r in range(case.runs):
        shells, pos, amp0, orientation = init_problem(
            n_sites=case.n_sites,
            active_fraction=case.active_fraction,
            seed=seed + r,
        )

        amp_dense = amp0[:]
        dense_torque = [0.0] * case.n_sites
        t0 = time.perf_counter()
        for step in range(case.steps):
            dense_step(
                step=step,
                shells=shells,
                pos=pos,
                orientation=orientation,
                amp=amp_dense,
                torque_cache=dense_torque,
                update_every=case.update_every,
            )
        t1 = time.perf_counter()
        dense_times_ms.append((t1 - t0) * 1000.0)

        sparse = sparse_from_dense(amp0)
        sparse_torque: Dict[int, float] = {}
        s0 = time.perf_counter()
        for step in range(case.steps):
            sparse = sparse_step(
                step=step,
                shells=shells,
                pos=pos,
                orientation=orientation,
                sparse=sparse,
                torque_cache=sparse_torque,
                n_basis=case.n_sites,
                update_every=case.update_every,
            )
        s1 = time.perf_counter()
        sparse_times_ms.append((s1 - s0) * 1000.0)
        final_active_counts.append(len(sparse))

    dense_med = statistics.median(dense_times_ms)
    sparse_med = statistics.median(sparse_times_ms)
    speedup = dense_med / sparse_med if sparse_med > 0 else math.inf

    return {
        "n_sites": float(case.n_sites),
        "steps": float(case.steps),
        "active_fraction": case.active_fraction,
        "update_every": float(case.update_every),
        "runs": float(case.runs),
        "dense_median_ms": dense_med,
        "sparse_median_ms": sparse_med,
        "dense_vs_sparse_speedup": speedup,
        "final_active_median": statistics.median(final_active_counts),
    }


def paper_table_cases() -> List[BenchCase]:
    """The four rows published in octonion_lightcone_to_oshoracle.tex (median may differ slightly)."""
    return [
        BenchCase(n_sites=192, steps=24, active_fraction=0.05, update_every=10, runs=4),
        BenchCase(n_sites=192, steps=24, active_fraction=0.20, update_every=10, runs=4),
        BenchCase(n_sites=384, steps=20, active_fraction=0.05, update_every=10, runs=4),
        BenchCase(n_sites=384, steps=20, active_fraction=0.30, update_every=10, runs=4),
    ]


def run_paper_table(seed: int = 42) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for i, case in enumerate(paper_table_cases()):
        rows.append(run_case(case, seed=seed + i * 1000))
    return rows


def format_table_row(d: Dict[str, float]) -> str:
    return (
        f"{int(d['n_sites']):7d}  "
        f"{int(d['steps']):5d}  "
        f"{d['active_fraction']:.2f}   "
        f"{d['dense_median_ms']:8.2f}  "
        f"{d['sparse_median_ms']:9.2f}  "
        f"{d['dense_vs_sparse_speedup']:22.2f}  "
        f"{int(d['final_active_median']):16d}"
    )


__all__ = [
    "BenchCase",
    "additive_field_kernel",
    "available_modes",
    "dense_step",
    "format_table_row",
    "init_problem",
    "lattice_full_mode_energy",
    "paper_table_cases",
    "phi_of_shell",
    "run_case",
    "run_paper_table",
    "should_update_torque",
    "sparse_from_dense",
    "sparse_step",
]
