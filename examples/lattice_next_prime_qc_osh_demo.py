#!/usr/bin/env python3
"""
Run from repo root (with dev env or ``uv run``):

  uv run python examples/lattice_next_prime_qc_osh_demo.py

Demonstrates the Python mirror of Lean ``OSHoracle`` / ``LatticeNextPrimeQCAlgorithm`` sparse step:
causal expand, dense lift, norm-preserving phase gate, support doubling.
"""

from __future__ import annotations

from pyhqiv.oshoracle_lattice_qc import (
    apply_gate_sparse,
    causal_expand_support,
    dense_from_sparse,
    discrete_norm_sq,
    phase_flip_gate,
    sparse_basis_card,
)


def main() -> None:
    L = 2
    n = sparse_basis_card(L)
    gate = phase_flip_gate(0, n)
    r = [(0, 1 + 2j), (3, -0.5j)]
    expanded = causal_expand_support(L, r)
    dense = dense_from_sparse(L, expanded)
    after = apply_gate_sparse(L, gate, r)
    print("L =", L, "basis_dim =", n)
    print("register length:", len(r), "-> expanded:", len(expanded), "-> after gate:", len(after))
    print("||dense(expanded)||^2 before gate:", discrete_norm_sq(dense))
    print("||dense(expanded)||^2 after gate: ", discrete_norm_sq(gate(dense)))


if __name__ == "__main__":
    main()
