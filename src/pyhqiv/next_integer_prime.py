"""
Classical **integer** next prime: smallest prime ``p`` with ``p > n``.

This is **not** the HQIV ``next_lattice_prime`` / ``next_prime_generator`` shell index from Lean
(``Hqiv.Physics.OctonionicZeta`` / ``LatticeNextPrimeGenerator``), which uses ``effCorrected`` ratios.
For that, use ``pyhqiv.lattice_next_prime_hqiv.hqiv_next_prime_generator``.

For inputs like ``5`` → ``7``, ``500`` → ``503``.
"""

from __future__ import annotations


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def next_prime(n: int) -> int:
    """
    Smallest prime strictly greater than ``n``.

    Examples: ``next_prime(5) == 7``, ``next_prime(500) == 503``.
    For ``n < 2``, returns ``2``.
    """
    if not isinstance(n, int):
        raise TypeError("n must be int")
    k = n + 1
    while not _is_prime(k):
        k += 1
    return k


__all__ = ["next_prime"]
