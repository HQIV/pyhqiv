"""
OSHoracle-style algorithm helpers built on the Python quantum simulator.

This module provides:
- Period finding support extraction (Shor-style, finite control register),
- Grover search utilities with an explicit marked-state oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from pyhqiv.now_setters import active_slice
from pyhqiv.quantum_simulator import Circuit, NoiseModel, QuantumState, run_circuit


def hidden_tag_of(x: int, period: int) -> int:
    if period <= 0:
        raise ValueError("period must be positive")
    return x % period


def period_support(control_qubits: int, period: int) -> List[int]:
    q = 1 << control_qubits
    if period <= 0:
        raise ValueError("period must be positive")
    step = q // period
    return [k * step for k in range(period)]


def period_probability_distribution(control_qubits: int, period: int) -> Dict[int, float]:
    support = period_support(control_qubits, period)
    if not support:
        return {}
    p = 1.0 / len(support)
    return {y: p for y in support}


@dataclass
class PeriodFindingResult:
    estimated_period: int
    support: List[int]
    probabilities: Dict[int, float]
    probability_errors: Dict[int, float]
    counts: Dict[int, int] | None = None


def estimate_period_from_peaks(peaks: Sequence[int], q: int) -> int:
    if q <= 0:
        raise ValueError("q must be positive")
    if not peaks:
        raise ValueError("peaks cannot be empty")
    peaks_sorted = sorted(set(int(p) for p in peaks))
    if len(peaks_sorted) == 1:
        return 1
    diffs = [b - a for a, b in zip(peaks_sorted, peaks_sorted[1:]) if b > a]
    if not diffs:
        return 1
    mean_gap = sum(diffs) / len(diffs)
    if mean_gap <= 0:
        return 1
    return max(1, int(round(q / mean_gap)))


def oshoracle_period_find(control_qubits: int, period: int) -> PeriodFindingResult:
    q = 1 << control_qubits
    support = period_support(control_qubits, period)
    probs = period_probability_distribution(control_qubits, period)
    est = estimate_period_from_peaks(support, q=q)
    return PeriodFindingResult(
        estimated_period=est,
        support=support,
        probabilities=probs,
        probability_errors={k: 0.0 for k in probs},
    )


def _marginal_control_probabilities(
    state: QuantumState, *, control_qubits: int, hidden_qubits: int
) -> np.ndarray:
    """
    Trace out hidden register by summing probabilities over hidden bit patterns.

    Assumes qubit ordering: control qubits are indices [0..control_qubits-1],
    hidden qubits are indices [control_qubits..control_qubits+hidden_qubits-1].
    """
    n_total = state.n_qubits
    if n_total != control_qubits + hidden_qubits:
        raise ValueError("state qubit count mismatch")
    qmask = (1 << control_qubits) - 1
    probs_control = np.zeros((1 << control_qubits,), dtype=float)
    for idx in range(state.amplitudes.size):
        control = idx & qmask
        probs_control[control] += float(abs(state.amplitudes[idx]) ** 2)
    return probs_control


def oshoracle_period_find_quantum(
    *,
    control_qubits: int,
    period: int,
    shots: int | None = None,
    seed: int | None = None,
    noise: NoiseModel | None = None,
) -> PeriodFindingResult:
    """
    Quantum simulation of OSHoracle-style period finding:
      1) Prepare uniform superposition over x in [0, 2^m)
      2) Apply hidden-tag encoding: |x>|0> -> |x>|x mod r>
      3) Apply inverse QFT on the control register
      4) Trace out hidden register (marginalize) to get Born probabilities on outputs y

    This is faithful statevector simulation for small toy periods.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    m = int(control_qubits)
    q = 1 << m
    context = active_slice()
    shift = int(context.get("reference_m", 0))

    hidden_qubits = max(1, int(math.ceil(math.log2(float(period)))))
    if period > (1 << hidden_qubits):
        raise ValueError("period too large for computed hidden register")

    n_total = m + hidden_qubits
    state = QuantumState(n_total)

    # Apply H on all control qubits to get uniform |x>.
    # Use direct single-qubit gate application to avoid building a circuit object.
    # (Qubit indices are little-endian in the statevector.)
    from pyhqiv.quantum_simulator import Gates as _Gates  # local import to avoid cycles

    for qb in range(m):
        state.apply_single_qubit_gate(_Gates.H, qb)

    # Hidden register starts in |0...0>; apply deterministic permutation:
    # for each x: amplitude at |x,0> moves to |x, x mod r>.
    r = int(period)
    amplitudes_old = state.amplitudes.copy()
    new = np.zeros_like(amplitudes_old)
    hidden_mask_zero = 0
    for x in range(q):
        # global index for hidden=0
        idx_in = x | (hidden_mask_zero << m)
        amp = amplitudes_old[idx_in]
        tag = hidden_tag_of(x + shift, r)
        idx_out = x | (tag << m)
        new[idx_out] += amp
    state.amplitudes = new

    if noise is not None:
        state.apply_noise(noise, seed=seed)

    # Inverse QFT on control qubits only.
    state.apply_qft_exact(qubits=list(range(m)), inverse=True)

    probs_control = _marginal_control_probabilities(
        state, control_qubits=m, hidden_qubits=hidden_qubits
    )

    counts: Dict[int, int] | None = None
    if shots is not None:
        rng = np.random.default_rng(seed)
        samples = rng.choice(np.arange(q), p=probs_control, size=int(shots))
        # Return empirical probabilities on support.
        counts = {}
        for s in samples.tolist():
            counts[s] = counts.get(s, 0) + 1
        # Convert counts to probabilities (support will be based on exact probs anyway).
        probs_out = {i: counts.get(i, 0) / float(shots) for i in range(q)}
        probs_err = {
            i: math.sqrt(max(0.0, probs_out[i] * (1.0 - probs_out[i])) / float(shots))
            for i in range(q)
        }
    else:
        probs_out = {i: float(p) for i, p in enumerate(probs_control)}
        probs_err = {i: 0.0 for i in range(q)}

    # Determine visible support peaks.
    # We use a tolerance relative to max probability.
    pmax = float(max(probs_out.values())) if probs_out else 0.0
    tol = pmax * 0.5 if pmax > 0 else 0.0
    support = [i for i in range(q) if float(probs_out[i]) >= tol and probs_out[i] > 0.0]
    est = estimate_period_from_peaks(support, q=q)

    # Normalize probabilities over computed support (prevents numeric noise issues).
    norm = sum(float(probs_out[i]) for i in support)
    if norm <= 0:
        probs_support = {i: 0.0 for i in support}
        err_support = {i: probs_err[i] for i in support}
    else:
        probs_support = {i: float(probs_out[i]) / norm for i in support}
        err_support = {i: probs_err[i] / norm for i in support}

    return PeriodFindingResult(
        estimated_period=est,
        support=support,
        probabilities=probs_support,
        probability_errors=err_support,
        counts=counts,
    )


OraclePredicate = Callable[[int], bool]


def _apply_oracle_phase_flip(state: QuantumState, oracle: OraclePredicate) -> None:
    for idx in range(state.amplitudes.size):
        if oracle(idx):
            state.amplitudes[idx] *= -1.0


def _apply_diffusion(state: QuantumState) -> None:
    avg = np.mean(state.amplitudes)
    state.amplitudes = 2.0 * avg - state.amplitudes


def optimal_grover_iterations(num_states: int, marked_count: int = 1) -> int:
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if marked_count <= 0 or marked_count > num_states:
        raise ValueError("marked_count must satisfy 1 <= marked_count <= num_states")
    theta = math.asin(math.sqrt(marked_count / num_states))
    return max(1, int(round((math.pi / (4.0 * theta)) - 0.5)))


@dataclass
class GroverResult:
    iterations: int
    probabilities: np.ndarray
    probability_errors: np.ndarray
    top_index: int
    top_probability: float
    top_probability_error: float


def oshoracle_grover_search(
    n_qubits: int,
    oracle: OraclePredicate,
    iterations: int | None = None,
    marked_count: int = 1,
) -> GroverResult:
    n = 1 << n_qubits
    init = QuantumState(n_qubits)
    had = Circuit(n_qubits)
    for q in range(n_qubits):
        had.h(q)
    state = run_circuit(had, initial_state=init).state

    k = optimal_grover_iterations(n, marked_count) if iterations is None else iterations
    if k < 0:
        raise ValueError("iterations must be non-negative")

    for _ in range(k):
        _apply_oracle_phase_flip(state, oracle)
        _apply_diffusion(state)
    probs = state.probabilities()
    probs_err = np.zeros_like(probs)
    top = int(np.argmax(probs))
    return GroverResult(
        iterations=k,
        probabilities=probs,
        probability_errors=probs_err,
        top_index=top,
        top_probability=float(probs[top]),
        top_probability_error=float(probs_err[top]),
    )


__all__ = [
    "GroverResult",
    "OraclePredicate",
    "PeriodFindingResult",
    "estimate_period_from_peaks",
    "hidden_tag_of",
    "optimal_grover_iterations",
    "oshoracle_grover_search",
    "oshoracle_period_find",
    "oshoracle_period_find_quantum",
    "period_probability_distribution",
    "period_support",
]

