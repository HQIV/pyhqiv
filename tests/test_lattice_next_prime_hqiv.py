"""HQIV lattice next shell (Lean ``next_lattice_prime`` / ``next_prime_generator`` port)."""

from __future__ import annotations

import pytest

from pyhqiv.lattice_next_prime_hqiv import (
    HqivLatticePrimeParams,
    decompose_last_shell,
    eff_corrected_dynamic,
    fano_line_weight_for_shell,
    hqiv_next_prime_generator,
    hqiv_scaled_next_prime_stepwise,
    next_lattice_prime_shell,
)
from pyhqiv.lepton_resonance_ladder import eff_corrected


def test_fano_line_weight_cycle() -> None:
    assert fano_line_weight_for_shell(0) == 1  # residue 0 -> weight 1
    assert fano_line_weight_for_shell(6) == 1  # residue 6 -> (6%3)+1 = 1
    assert fano_line_weight_for_shell(8) == 2  # residue 1 -> (1%3)+1 = 2


def test_decompose_x1_gives_last_shell_zero() -> None:
    assert decompose_last_shell(1, 0.0, fuel=100) == 0


def test_next_lattice_prime_m0_delta0_threshold_1p5_is_1() -> None:
    assert next_lattice_prime_shell(0, 0.0, threshold=1.5) == 1
    r = eff_corrected(0.0, 1) / eff_corrected(0.0, 0)
    assert r >= 1.5


def test_next_lattice_prime_requires_threshold_gt_one() -> None:
    with pytest.raises(ValueError, match="threshold"):
        next_lattice_prime_shell(0, 0.0, threshold=1.0)


def test_hqiv_generator_5_is_shell_not_classical_prime() -> None:
    m = hqiv_next_prime_generator(5, HqivLatticePrimeParams(fuel=500))
    assert isinstance(m, int)
    assert m > decompose_last_shell(5, 0.0, fuel=500)
    # Classical next prime after 5 is 7 — HQIV answer is a shell index, not 7 in general:
    assert m == 2


def test_eff_corrected_dynamic_matches_constant_delta() -> None:
    d = 0.03
    for m in range(0, 15):
        assert eff_corrected_dynamic(lambda _k: d, m) == pytest.approx(eff_corrected(d, m))


def test_hqiv_scaled_stepwise_varies_with_p() -> None:
    """Experimental path: m' depends on x (unlike plateau of Lean-like m' on many primes)."""
    a = hqiv_scaled_next_prime_stepwise(5, fuel=200_000)
    b = hqiv_scaled_next_prime_stepwise(97, fuel=200_000)
    assert a != b
    assert isinstance(a, int) and a >= 2


def test_hqiv_generator_500_is_integer_shell() -> None:
    m = hqiv_next_prime_generator(500, HqivLatticePrimeParams(fuel=50_000))
    assert m > 0
    last = decompose_last_shell(500, 0.0, fuel=50_000)
    assert m > last
    base = eff_corrected(0.0, last)
    assert eff_corrected(0.0, m) / base >= 1.5 - 1e-12
