#!/usr/bin/env python3
"""
Print ``hqiv_next_prime_generator(p)`` for each classical prime ``p ≤ N`` (default ``N=100``),
alongside ``decompose_last_shell(p)`` and the ordinary next prime for comparison.

  uv run python examples/hqiv_lattice_prime_classical_primes_table.py
  uv run python examples/hqiv_lattice_prime_classical_primes_table.py --n 200
"""

from __future__ import annotations

import argparse

from pyhqiv.lattice_next_prime_hqiv import (
    HqivLatticePrimeParams,
    decompose_last_shell,
    hqiv_next_prime_generator,
)
from pyhqiv.next_integer_prime import next_prime as classical_next_prime


def primes_upto(n: int) -> list[int]:
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="upper bound for primes (inclusive)")
    ap.add_argument("--fuel", type=int, default=50_000, help="decomposition fuel (Lean fuel)")
    args = ap.parse_args()

    params = HqivLatticePrimeParams(fuel=args.fuel)
    delta = params.delta

    print(f"Primes p ≤ {args.n}  (δ = {delta}, threshold = {params.threshold}, fuel = {params.fuel})")
    print(f"{'p':>4}  {'last_m':>6}  {'HQIV m ':>8}  {'class next':>10}")
    print("-" * 40)
    for p in primes_upto(args.n):
        last_m = decompose_last_shell(p, delta, params.fuel)
        m_hqiv = hqiv_next_prime_generator(p, params)
        cnext = classical_next_prime(p)
        print(f"{p:4d}  {last_m:6d}  {m_hqiv:8d}  {cnext:10d}")


if __name__ == "__main__":
    main()
