"""Tests for ``pyhqiv.oshoracle_lattice_qc`` (Lean OSHoracle / LatticeNextPrimeQCAlgorithm hooks)."""

from __future__ import annotations

import random

import pytest

from pyhqiv import oshoracle_lattice_qc as osh


def test_causal_expand_doubles_length() -> None:
    L = 2
    r: osh.SparseRegister = [(0, 1 + 0j), (3, 2j)]
    e = osh.causal_expand_support(L, r)
    assert len(e) == 2 * len(r)


def test_apply_gate_sparse_doubles_length() -> None:
    L = 1
    n = osh.sparse_basis_card(L)
    g = osh.phase_flip_gate(0, n)
    r: osh.SparseRegister = [(0, 1 + 0j)]
    out = osh.apply_gate_sparse(L, g, r)
    assert len(out) == 2 * len(r)


def test_phase_flip_preserves_dense_norm_on_expanded_lift() -> None:
    """Matches ``lattice_next_prime_oshoracle_preserves_dense_norm_sq`` (norm on dense before/after gate)."""
    L = 2
    n = osh.sparse_basis_card(L)
    g = osh.phase_flip_gate(1, n)
    r: osh.SparseRegister = [(0, 1 + 1j), (4, 0.5 - 0.25j)]
    expanded = osh.causal_expand_support(L, r)
    dense = osh.dense_from_sparse(L, expanded)
    n0 = osh.discrete_norm_sq(dense)
    n1 = osh.discrete_norm_sq(g(dense))
    assert n0 == pytest.approx(n1)


@pytest.mark.parametrize("L", [0, 1, 2, 3])
def test_random_unitary_phase_gates_preserve_norm(L: int) -> None:
    n = osh.sparse_basis_card(L)
    rng = random.Random(7)

    def rand_gate(v: osh.DenseState) -> osh.DenseState:
        out = list(v)
        for _ in range(3):
            k = rng.randrange(n)
            out[k] = -out[k]
        return out

    r: osh.SparseRegister = [(rng.randrange(n * 2), complex(rng.uniform(-1, 1), rng.uniform(-1, 1))) for _ in range(5)]
    expanded = osh.causal_expand_support(L, r)
    dense = osh.dense_from_sparse(L, expanded)
    n0 = osh.discrete_norm_sq(dense)
    n1 = osh.discrete_norm_sq(rand_gate(dense))
    assert n0 == pytest.approx(n1)


def test_prune_preserves_sparse_norm_when_all_indices_kept() -> None:
    """``lattice_next_prime_prune_preserves_sparse_norm_sq`` surrogate."""
    r: osh.SparseRegister = [(0, 1j), (1, 2 + 0j)]
    flipped = [0, 1, 2]
    pr = osh.prune_to_flipped(flipped, r)
    assert osh.sparse_norm_sq(pr) == pytest.approx(osh.sparse_norm_sq(r))


def test_fano_seven_cycle_preserves_mass() -> None:
    rng = random.Random(0)
    for _ in range(20):
        p = [rng.uniform(-2, 3) for _ in range(7)]
        assert osh.fano_probability_mass_invariant(p)


def test_pruned_support_length_bounded_by_doubled_input() -> None:
    """``practicalLittleO`` / ``horizonCausal_support_o_twoPow_practice`` style bound on list length."""
    L = 2
    n = osh.sparse_basis_card(L)
    g = osh.phase_flip_gate(0, n)
    r: osh.SparseRegister = [(0, 1 + 0j), (2, -0.5j)]
    after = osh.apply_gate_sparse(L, g, r)
    flipped = osh.detect_flipped_kets(r, after)
    pruned = osh.prune_to_flipped(flipped, after)
    assert len(pruned) <= len(after) == 2 * len(r)
