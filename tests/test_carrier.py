"""Tests for :mod:`pyhqiv.carrier` (SO(8) ℝ⁸ carrier and Hamiltonian assembly)."""

from __future__ import annotations

import numpy as np
import pytest

from pyhqiv.carrier import So8Carrier, hamiltonian_from_so8_coeffs
from pyhqiv.so8_generators import So8Generators, lie_bracket, load_so8_generators


def test_so8_carrier_from_unit_axis_all_axes() -> None:
    g = load_so8_generators()
    for ax in range(8):
        c = So8Carrier.from_unit_axis(ax, generators=g)
        assert np.isclose(np.linalg.norm(c.psi), 1.0)
        assert int(np.argmax(np.abs(c.psi))) == ax


def test_so8_carrier_from_unit_axis_rejects_bad_axis() -> None:
    with pytest.raises(ValueError, match="0..7"):
        So8Carrier.from_unit_axis(-1)
    with pytest.raises(ValueError, match="0..7"):
        So8Carrier.from_unit_axis(8)


def test_so8_carrier_apply_generator_index_bounds() -> None:
    c = So8Carrier.from_unit_axis(0)
    with pytest.raises(IndexError):
        c.apply_generator(-1)
    with pytest.raises(IndexError):
        c.apply_generator(28)


def test_so8_carrier_normalize_zero_rejected() -> None:
    g = load_so8_generators()
    c = So8Carrier(psi=np.zeros(8), generators=g)
    with pytest.raises(ValueError, match="normalize"):
        c.normalize()


def test_so8_carrier_normalized_copy_unit_norm() -> None:
    c = So8Carrier.from_unit_axis(3)
    c.psi *= 5.0
    n = c.normalized_copy()
    assert np.isclose(np.linalg.norm(n.psi), 1.0)
    assert np.isclose(np.linalg.norm(c.psi), 5.0)


def test_so8_carrier_to_density_matrix_idempotent_rank_one() -> None:
    c = So8Carrier.from_unit_axis(1)
    rho = c.to_density_matrix()
    assert rho.shape == (8, 8)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.allclose(rho @ rho, rho, atol=1e-12)


def test_hamiltonian_from_so8_coeffs_is_antisymmetric() -> None:
    g = load_so8_generators()
    rng = np.random.default_rng(0)
    c = rng.normal(size=28)
    h = hamiltonian_from_so8_coeffs(c, g)
    assert h.shape == (8, 8)
    assert np.allclose(h + h.T, 0.0, atol=1e-12)


def test_hamiltonian_from_so8_coeffs_linear_in_coeffs() -> None:
    g = load_so8_generators()
    c1 = np.zeros(28)
    c1[0] = 1.0
    c2 = np.zeros(28)
    c2[0] = 2.0
    h1 = hamiltonian_from_so8_coeffs(c1, g)
    h2 = hamiltonian_from_so8_coeffs(c2, g)
    assert np.allclose(h2, 2.0 * h1)


def test_hamiltonian_wrong_coeff_size_raises() -> None:
    g = load_so8_generators()
    with pytest.raises(ValueError):
        hamiltonian_from_so8_coeffs(np.ones(10), g)


def test_so8_carrier_bad_generator_tensor_shape() -> None:
    bad = So8Generators(tensor=np.zeros((3, 8, 8)))
    with pytest.raises(ValueError, match="tensor shape"):
        So8Carrier(psi=np.ones(8) / np.sqrt(8), generators=bad)


def test_lie_bracket_on_lean_generators() -> None:
    g = load_so8_generators()
    g0, g1 = g.matrix(0), g.matrix(1)
    lb = lie_bracket(g0, g1)
    assert np.allclose(lb + lb.T, 0.0, atol=1e-10)
