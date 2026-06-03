"""
OSH-style sparse register + horizon-causal gate step, aligned with Lean
``Hqiv/QuantumComputing/OSHoracle.lean`` and the proved hooks in
``Hqiv/QuantumComputing/LatticeNextPrimeQCAlgorithm.lean``.

Uses **one complex amplitude per harmonic slot** (scalar surrogate for an octonion carrier per mode).
Norms match the Lean idea: ``discreteNormSq``-style sum of ``|z|^2`` over ``(L+1)^2`` slots;
``sparseNormSq`` sums ``|a|^2`` over **list entries** (duplicates are not merged before summing).

This is for **tests and demos**; it does not implement the classical
``LatticeNextPrimeGenerator`` / ``next_prime_generator`` pipeline.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Set, Tuple

SparseRegister = List[Tuple[int, complex]]
DenseState = List[complex]
SparseGate = Callable[[DenseState], DenseState]


def sparse_basis_card(L: int) -> int:
    return (L + 1) ** 2


def wrap_idx(L: int, i: int) -> int:
    return i % sparse_basis_card(L)


def decode_idx(L: int, i: int) -> int:
    """Flat harmonic index: wrapped natural into ``0 .. (L+1)^2 - 1`` (Lean ``decodeIdx`` surrogate)."""
    return wrap_idx(L, i)


def causal_expand_support(L: int, r: SparseRegister) -> SparseRegister:
    out: SparseRegister = []
    for i_raw, a in r:
        i = wrap_idx(L, i_raw)
        j = wrap_idx(L, i_raw + 1)
        out.append((i, a))
        out.append((j, a))
    return out


def dense_from_sparse(L: int, r: SparseRegister) -> DenseState:
    n = sparse_basis_card(L)
    acc = [0j] * n
    for i_raw, a in r:
        ij = decode_idx(L, i_raw)
        acc[ij] += a
    return acc


def apply_gate_sparse(L: int, gate: SparseGate, r: SparseRegister) -> SparseRegister:
    """
    Causal expand → dense lift → ``gate`` on dense vector → map amplitudes back on expanded support
    (same order as Lean ``applyGateSparse``).
    """
    expanded = causal_expand_support(L, r)
    dense = dense_from_sparse(L, expanded)
    evolved = gate(dense)
    return [(i_raw, evolved[decode_idx(L, i_raw)]) for i_raw, _ in expanded]


def discrete_norm_sq(v: Sequence[complex]) -> float:
    return float(sum(z.real * z.real + z.imag * z.imag for z in v))


def sparse_norm_sq(r: SparseRegister) -> float:
    return float(sum(z.real * z.real + z.imag * z.imag for _, z in r))


def detect_flipped_kets(before: SparseRegister, after: SparseRegister) -> List[int]:
    b_idx = [i for i, _ in before]
    a_idx = [i for i, _ in after]
    b_set: Set[int] = set(b_idx)
    a_set: Set[int] = set(a_idx)
    out = [i for i in b_idx if i not in a_set] + [i for i in a_idx if i not in b_set]
    return out


def prune_to_flipped(flipped: Sequence[int], r: SparseRegister) -> SparseRegister:
    flip_set = set(flipped)
    return [(i, a) for i, a in r if i in flip_set]


def phase_flip_gate(slot: int, n: int) -> SparseGate:
    """``HQIV`` ``phaseGate`` analogue: multiply one slot by ``-1`` (preserves discrete norm)."""

    def _g(v: DenseState) -> DenseState:
        if len(v) != n:
            raise ValueError("dense length mismatch")
        if not (0 <= slot < n):
            raise ValueError("slot out of range")
        return [(-z if i == slot else z) for i, z in enumerate(v)]

    return _g


def fano_line_cyclic_perm_indices() -> List[int]:
    """``finRotate 7``: image of ``i`` is ``(i + 1) % 7``."""
    return [(i + 1) % 7 for i in range(7)]


def fano_probability_mass_invariant(p: Sequence[float]) -> bool:
    """``Equiv.sum_comp`` surrogate: ``sum_i p[perm[i]] == sum_i p[i]`` for the 7-cycle."""
    if len(p) != 7:
        return False
    perm = fano_line_cyclic_perm_indices()
    s0 = sum(p)
    s1 = sum(p[perm[i]] for i in range(7))
    return abs(s0 - s1) < 1e-12


__all__ = [
    "DenseState",
    "SparseGate",
    "SparseRegister",
    "apply_gate_sparse",
    "causal_expand_support",
    "decode_idx",
    "dense_from_sparse",
    "detect_flipped_kets",
    "discrete_norm_sq",
    "fano_line_cyclic_perm_indices",
    "fano_probability_mass_invariant",
    "phase_flip_gate",
    "prune_to_flipped",
    "sparse_basis_card",
    "sparse_norm_sq",
    "wrap_idx",
]
