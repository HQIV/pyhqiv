"""
Spherical harmonics on S² (angular part of Laplace / angular momentum).

This is the standard mathematical layer used with:

- ``Hqiv.QuantumMechanics.Schrodinger`` (angular kinetic term ``L²/(2μ)`` on the sphere),
- multipole expansions (same ``ℓ`` quantum number as spherical Bessel ``j_ℓ`` in radial problems).

Conventions (aligned with current SciPy spherical harmonics):

- ``theta`` is **colatitude** (polar angle from +z) in radians, ``theta ∈ [0, π]``.
- ``phi`` is **azimuth** in radians, ``phi ∈ [0, 2π)``.
- Complex harmonics ``Y_ℓ^m`` satisfy ``-Δ_S² Y_ℓ^m = ℓ(ℓ+1) Y_ℓ^m`` and are orthonormal on the unit sphere.

SciPy ≥ 1.14 exposes ``sph_harm_y(n, m, theta, phi)``; older releases used
``sph_harm(m, n, phi, theta)``. This module dispatches between them.

No physical constants are fixed here — only angles and non-negative integers ``ℓ``, ``m``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import scipy.special as scipy_special


def _Y_lm_scipy(ell: int, m: int, theta: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray:
    """Complex ``Y_ℓ^m`` using SciPy (handles pre- and post-``sph_harm_y`` APIs)."""
    if hasattr(scipy_special, "sph_harm_y"):
        out = scipy_special.sph_harm_y(ell, m, theta, phi)
    else:
        out = scipy_special.sph_harm(m, ell, phi, theta)
    return np.asarray(out, dtype=np.complex128)


def _require_ell_m(ell: int, m: int) -> None:
    if ell < 0:
        raise ValueError("ell must be non-negative")
    if abs(m) > ell:
        raise ValueError("require |m| <= ell")


def laplace_beltrami_eigenvalue_S2(ell: int) -> float:
    """
    Eigenvalue of ``-Δ_S²`` on scalar spherical harmonics:

        -Δ_S² Y_ℓ^m = ℓ(ℓ+1) Y_ℓ^m.
    """
    if ell < 0:
        raise ValueError("ell must be non-negative")
    return float(ell * (ell + 1))


def real_basis_dimension(l_max: int) -> int:
    """
    Number of real orthonormal spherical harmonics through degree ``l_max`` (inclusive):

        Σ_{ℓ=0}^{l_max} (2ℓ+1) = (l_max+1)².
    """
    if l_max < 0:
        raise ValueError("l_max must be non-negative")
    return (l_max + 1) ** 2


def spherical_harmonic_Y(
    ell: int,
    m: int,
    theta: float | np.ndarray,
    phi: float | np.ndarray,
) -> complex | np.ndarray:
    """
    Complex orthonormal ``Y_ℓ^m(theta, phi)`` (Condon–Shortley phase, SciPy definition).

    ``theta`` colatitude, ``phi`` azimuth; broadcasts like ``numpy`` ufuncs.
    """
    _require_ell_m(ell, m)
    return _Y_lm_scipy(ell, m, theta, phi)


def spherical_harmonic_Y_real(
    ell: int,
    m: int,
    theta: float | np.ndarray,
    phi: float | np.ndarray,
) -> float | np.ndarray:
    """
    Real orthonormal basis on ``L²(S²)``, with ``m ∈ {-ℓ, …, ℓ}``.

    Uses the usual real combination of complex ``Y_ℓ^m`` (cosine / sine in ``phi``).
    """
    _require_ell_m(ell, m)
    if m == 0:
        return np.real(_Y_lm_scipy(ell, 0, theta, phi))
    if m > 0:
        return math.sqrt(2.0) * ((-1) ** m) * np.real(_Y_lm_scipy(ell, m, theta, phi))
    mp = -m
    return math.sqrt(2.0) * ((-1) ** mp) * np.imag(_Y_lm_scipy(ell, mp, theta, phi))


def angular_momentum_squared_eigenvalue(ell: int, mu: float = 1.0) -> float:
    """
    Eigenvalue of ``L²`` in units where ``L² Y_ℓ^m = ℏ² ℓ(ℓ+1) Y_ℓ^m``:

        ``<L²>`` proportional factor ``μ`` defaults to 1 (natural / abstract units).
    """
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    return float(mu) * laplace_beltrami_eigenvalue_S2(ell)


def spherical_laplace_on_Y_mode(ell: int, m: int) -> complex:
    """
    Action of ``Δ_S²`` on a normalized ``Y_ℓ^m`` mode (pointwise constant eigenvalue):

        ``Δ_S² Y_ℓ^m = -ℓ(ℓ+1) Y_ℓ^m``  ⇒  coefficient ``-ℓ(ℓ+1)`` multiplies the mode.
    """
    _require_ell_m(ell, m)
    return complex(-laplace_beltrami_eigenvalue_S2(ell), 0.0)


def enumerate_real_sh_indices(l_max: int) -> list[tuple[int, int]]:
    """
    Pairs ``(ℓ, m)`` in a stable order: increasing ``ℓ``, then ``m`` from ``-ℓ`` to ``ℓ``.
    """
    if l_max < 0:
        raise ValueError("l_max must be non-negative")
    out: list[tuple[int, int]] = []
    for ell in range(l_max + 1):
        for m in range(-ell, ell + 1):
            out.append((ell, m))
    return out


def spherical_harmonic_real_basis_matrix(
    l_max: int,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    layout: Literal["columns", "rows"] = "columns",
) -> np.ndarray:
    """
    Evaluate the real spherical-harmonic basis up to ``l_max`` on a grid ``(theta, phi)``.

    Parameters
    ----------
    theta, phi
        Broadcast-compatible arrays (same shape after broadcast).
    l_max
        Maximum ``ℓ``.
    layout
        If ``"columns"`` (default), shape ``(N, K)`` with ``K = (l_max+1)²`` columns
        and ``N`` flattened grid points (column ``j`` is the ``j``-th basis function).
        If ``"rows"``, shape ``(K, N)``.
    """
    theta_b = np.asarray(theta, dtype=np.float64)
    phi_b = np.asarray(phi, dtype=np.float64)
    b = np.broadcast_arrays(theta_b, phi_b)
    flat_theta = b[0].reshape(-1)
    flat_phi = b[1].reshape(-1)
    n_pts = flat_theta.shape[0]
    k = real_basis_dimension(l_max)
    mat = np.empty((n_pts, k), dtype=np.float64)
    col = 0
    for ell in range(l_max + 1):
        for m in range(-ell, ell + 1):
            mat[:, col] = np.asarray(
                spherical_harmonic_Y_real(ell, m, flat_theta, flat_phi),
                dtype=np.float64,
            ).reshape(-1)
            col += 1
    assert col == k
    if layout == "columns":
        return mat
    if layout == "rows":
        return mat.T
    raise ValueError("layout must be 'columns' or 'rows'")


__all__ = [
    "angular_momentum_squared_eigenvalue",
    "enumerate_real_sh_indices",
    "laplace_beltrami_eigenvalue_S2",
    "real_basis_dimension",
    "spherical_harmonic_Y",
    "spherical_harmonic_Y_real",
    "spherical_harmonic_real_basis_matrix",
    "spherical_laplace_on_Y_mode",
]
