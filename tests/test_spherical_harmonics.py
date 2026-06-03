"""Spherical harmonics: Laplace-on-S² eigenvalues and orthogonality (SciPy-backed)."""

from __future__ import annotations

import math

import numpy as np

from pyhqiv.spherical_harmonics import (
    enumerate_real_sh_indices,
    laplace_beltrami_eigenvalue_S2,
    real_basis_dimension,
    spherical_harmonic_Y,
    spherical_harmonic_Y_real,
    spherical_harmonic_real_basis_matrix,
    spherical_laplace_on_Y_mode,
)


def test_laplace_beltrami_eigenvalue_matches_angular_momentum() -> None:
    for ell in range(0, 8):
        assert laplace_beltrami_eigenvalue_S2(ell) == float(ell * (ell + 1))
        assert spherical_laplace_on_Y_mode(ell, 0).real == -ell * (ell + 1)


def test_Y00_is_constant() -> None:
    theta = np.linspace(0.1, math.pi - 0.1, 5)
    phi = np.linspace(0.0, 2.0 * math.pi, 7)
    TT, PP = np.meshgrid(theta, phi, indexing="ij")
    y = spherical_harmonic_Y(0, 0, TT, PP)
    expected = 1.0 / math.sqrt(4.0 * math.pi)
    assert np.allclose(np.abs(y), expected, rtol=1e-10)


def test_real_basis_count() -> None:
    assert real_basis_dimension(0) == 1
    assert real_basis_dimension(1) == 4
    assert real_basis_dimension(2) == 9
    assert len(enumerate_real_sh_indices(2)) == 9


def test_real_harmonics_orthogonality_product_quadrature() -> None:
    """Riemann sum in ``(cos θ, φ)``: ``∫ dΩ = ∫ dφ ∫ d(cos θ)``."""
    n = 400
    cos_t = np.linspace(-1.0, 1.0, n)
    phi = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    d_cos = cos_t[1] - cos_t[0]
    d_phi = phi[1] - phi[0]
    theta = np.arccos(cos_t)
    TT = theta[:, np.newaxis]
    PP = phi[np.newaxis, :]
    y00 = spherical_harmonic_Y_real(0, 0, TT, PP)
    y10 = spherical_harmonic_Y_real(1, 0, TT, PP)
    y1p = spherical_harmonic_Y_real(1, 1, TT, PP)
    inner_00 = float(np.sum(y00 * y00) * d_cos * d_phi)
    inner_01 = float(np.sum(y00 * y10) * d_cos * d_phi)
    inner_11 = float(np.sum(y10 * y10) * d_cos * d_phi)
    inner_cross = float(np.sum(y10 * y1p) * d_cos * d_phi)
    assert abs(inner_00 - 1.0) < 0.05
    assert abs(inner_01) < 0.05
    assert abs(inner_11 - 1.0) < 0.05
    assert abs(inner_cross) < 0.05


def test_roundtrip_indices_enumerate() -> None:
    pairs = enumerate_real_sh_indices(3)
    assert pairs[0] == (0, 0)
    assert pairs[1] == (1, -1)
    assert len(pairs) == real_basis_dimension(3)


def test_real_harmonic_is_real() -> None:
    t = 0.7
    p = 1.2
    for ell, m in [(2, 0), (3, -2), (4, 3)]:
        v = spherical_harmonic_Y_real(ell, m, t, p)
        assert abs(np.imag(v)) < 1e-14
