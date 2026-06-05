"""
Discrete light-cone foundations for the fresh pyhqiv rebuild.

This module mirrors the derivation chain in:

- `HQIV_LEAN/Hqiv/Geometry/OctonionicLightCone.lean`

The implementation keeps only the light-cone combinatorics and the directly
derived curvature quantities needed downstream by the preliminary present-day
(`now`) module.
"""

from __future__ import annotations

import math
from fractions import Fraction

ALPHA_EXACT = Fraction(3, 5)


def _require_nonnegative_shell(m: int) -> None:
    if m < 0:
        raise ValueError("shell index must be non-negative")


def lattice_simplex_count(m: int) -> int:
    """
    Stars-and-bars numerator for x + y + z = m with x,y,z >= 0.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.latticeSimplexCount`
    """
    _require_nonnegative_shell(m)
    return (m + 2) * (m + 1)


def cumulative_lattice_simplex_count(n: int) -> int:
    """
    Closed-form cumulative simplex count up to shell n.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.cumLatticeSimplexCount_closed`
    """
    _require_nonnegative_shell(n)
    return ((n + 1) * (n + 2) * (n + 3)) // 3


def available_modes(m: int) -> float:
    """
    New available modes at shell m before differencing.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.available_modes`
    """
    return 4.0 * lattice_simplex_count(m)


def new_modes(m: int) -> float:
    """
    Incremental modes unlocked at shell m.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.new_modes`
    """
    _require_nonnegative_shell(m)
    if m == 0:
        return available_modes(0)
    return available_modes(m) - available_modes(m - 1)


def alpha() -> float:
    """
    HQIV varying-G / curvature exponent, derived as 3/5.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.alpha_eq_3_5`
    """
    return float(ALPHA_EXACT)


def qcd_shell() -> int:
    """
    First positive-curvature QCD transition shell.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.qcdShell`
    """
    return 1


def lattice_step_count() -> int:
    """
    Number of discrete lattice steps from QCD to lock-in.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.latticeStepCount`
    """
    return 3


def reference_m() -> int:
    """
    Lock-in / reference shell derived from the light-cone ladder.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.referenceM`
    """
    return qcd_shell() + lattice_step_count()


def curvature_density(x: float, alpha_value: float | None = None) -> float:
    """
    Continuous curvature-imprint density sampled on the shell ladder.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.curvatureDensity`
    """
    if x <= 0.0:
        raise ValueError("curvature density is defined only for x > 0")
    alpha_term = alpha() if alpha_value is None else alpha_value
    return (1.0 / x) * (1.0 + alpha_term * math.log(x))


def shell_shape(m: int, alpha_value: float | None = None) -> float:
    """
    Purely combinatorial shell-shape factor.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.shell_shape`
    """
    _require_nonnegative_shell(m)
    return curvature_density(float(m + 1), alpha_value=alpha_value)


def cube_axes() -> int:
    return 3


def signs_per_axis() -> int:
    return 2


def cube_directions() -> int:
    """
    Number of outward cubic lattice directions.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.cubeDirections`
    """
    return cube_axes() * signs_per_axis()


def octonion_imaginary_dim() -> int:
    """
    Number of imaginary octonion directions.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.octonionImaginaryDim`
    """
    return 7


def unit_cube_half_diagonal() -> float:
    """
    Euclidean half-diagonal of the unit cube.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.unitCubeHalfDiagonal`
    """
    return math.sqrt(3.0)


def curvature_norm_combinatorial() -> float:
    """
    First-principles combinatorial curvature norm 6^7 * sqrt(3).

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.curvature_norm_combinatorial`
    """
    return float(cube_directions() ** octonion_imaginary_dim()) * unit_cube_half_diagonal()


def delta_e(m: int, alpha_value: float | None = None) -> float:
    """
    Per-shell curvature imprint deltaE(m).

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.deltaE`
    """
    return curvature_norm_combinatorial() * shell_shape(m, alpha_value=alpha_value)


def curvature_integral(n: int, alpha_value: float | None = None) -> float:
    """
    Discrete shell integral over shells 0..n-1.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.curvature_integral`
    """
    _require_nonnegative_shell(n)
    return sum(abs(shell_shape(m, alpha_value=alpha_value)) for m in range(n))


def x_over_theta_from_horizons(n: int, N: int) -> float:
    """
    Continuous ``0 < x < θ`` term from Planck distances to shells ``n`` and ``N``.

    Radial scale to shell ``m`` is ``R_h = m + 1`` in Planck units, so
    ``x/θ = (n+1)/(N+1)``.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone` (distance-ratio factor in ``omega_k_at_horizon``).
    """
    _require_nonnegative_shell(n)
    _require_nonnegative_shell(N)
    if N <= 0:
        return 1.0
    return (float(n) + 1.0) / (float(N) + 1.0)


def omega_k_at_horizon(n: int, horizon: int, alpha_value: float | None = None) -> float:
    """
    Horizon-relative curvature ratio Omega_k(n; horizon).

    Combines the shell-shape integral ratio with the Planck-distance ratio
    ``x/θ`` from ``x_over_theta_from_horizons``, matching the Lean product form.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.omega_k_at_horizon`
    """
    _require_nonnegative_shell(n)
    _require_nonnegative_shell(horizon)
    denominator = curvature_integral(horizon, alpha_value=alpha_value)
    if denominator <= 0.0:
        return 1.0
    integral_ratio = curvature_integral(n, alpha_value=alpha_value) / denominator
    return integral_ratio * x_over_theta_from_horizons(n, horizon)


def omega_k_partial(n: int, alpha_value: float | None = None) -> float:
    """
    Curvature ratio relative to the derived reference shell.

    Lean reference:
    `Hqiv.Geometry.OctonionicLightCone.omega_k_partial`
    """
    return omega_k_at_horizon(n, reference_m(), alpha_value=alpha_value)


__all__ = [
    "ALPHA_EXACT",
    "alpha",
    "available_modes",
    "cube_axes",
    "cube_directions",
    "curvature_density",
    "curvature_integral",
    "curvature_norm_combinatorial",
    "cumulative_lattice_simplex_count",
    "delta_e",
    "lattice_simplex_count",
    "lattice_step_count",
    "new_modes",
    "octonion_imaginary_dim",
    "omega_k_at_horizon",
    "omega_k_partial",
    "qcd_shell",
    "x_over_theta_from_horizons",
    "reference_m",
    "shell_shape",
    "signs_per_axis",
    "unit_cube_half_diagonal",
]
