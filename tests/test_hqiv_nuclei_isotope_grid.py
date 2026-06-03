"""
Large-scale isotope checks: constructive ladder path and valley count invariants.

Runs without pytest (plain asserts); safe for CI once pytest collects this file.
"""

from __future__ import annotations

import math

from pyhqiv.hqiv_nuclei import (
    ladder_path_from_ZN,
    simulate_ladder_end,
    valley_count_from_A,
)
from pyhqiv.isotope_ladder import IsotopeLadderConfig, IsotopeState, nuclear_binding_energy_mev


def test_ladder_path_length_invariant_full_chart() -> None:
    """For every (Z,N) with 1 ≤ A = Z+N ≤ 118 (periodic table), path length = A-1."""
    for A in range(1, 119):
        for Z in range(1, A + 1):
            N = A - Z
            seed, steps = ladder_path_from_ZN(Z, N)
            assert seed == "proton"
            assert len(steps) == A - 1
            assert simulate_ladder_end("proton", steps) == (A, Z)


def test_neutron_matter_chain_Z0() -> None:
    """Pure-neutron chain Z=0: path length N-1 for N ≥ 2."""
    for N in range(2, 200):
        Z = 0
        A = N
        seed, steps = ladder_path_from_ZN(Z, N)
        assert seed == "neutron"
        assert len(steps) == N - 1 == A - 1
        assert simulate_ladder_end("neutron", steps) == (A, Z)


def test_valley_count_matches_two_A_minus_one() -> None:
    for A in range(1, 300):
        assert valley_count_from_A(A) == 2 * (A - 1)


def test_isotope_ladder_binding_finite_heavy_sample() -> None:
    """Network binding stays finite for a spread of heavy nuclides (explicit masses)."""
    cfg = IsotopeLadderConfig(
        shell_m=4,
        m_proton_mev=938.2720813,
        m_neutron_mev=939.5654133,
    )
    samples = [
        (1, 1),
        (26, 30),
        (82, 126),
        (92, 143),
        (118, 176),
    ]
    for Z, N in samples:
        b = nuclear_binding_energy_mev(IsotopeState(Z=Z, N=N, J=0.0), cfg)
        assert math.isfinite(b)
