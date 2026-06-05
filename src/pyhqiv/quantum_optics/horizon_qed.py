"""
Horizon QED / quantum optics layer mirroring `Hqiv.QuantumOptics.HorizonQED`.

Uses the same shell ladder and simplex counts as `pyhqiv.lightcone` and
`pyhqiv.auxiliary_field` (Lean: `OctonionicLightCone`, `AuxiliaryField`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pyhqiv.auxiliary_field import phi_of_shell, shell_temperature, t_pl_natural
from pyhqiv.lightcone import lattice_simplex_count


def shell_spatial_mode_count(m: int) -> int:
    """
    Lean `Hqiv.QuantumOptics.shellSpatialModeCount` = `latticeSimplexCount`.
    """
    return lattice_simplex_count(m)


def vacuum_zero_point_natural(m_uv: int, m_ir: int) -> float:
    """
    Exact match to attached paper script (finite_mode_kirchhoff/scripts/kirchhoff_finite_mode.py):
    vacuum_zero_point(m_uv, m_ir) = sum( 0.5 * lattice_simplex_count(m) * (1/(m+1)) for m=muv to mir )
    This is the finite vacuum zero-point "energy density" in natural units (omega_nat = 1/(m+1)).
    Used for CC problem solution: sum only up to causal now, no infinite or Planck cutoff disaster.
    """
    if m_uv < 0 or m_ir < m_uv:
        raise ValueError("m_uv, m_ir must satisfy 0 <= m_uv <= m_ir")
    return sum(
        0.5 * lattice_simplex_count(m) * (1.0 / (m + 1))
        for m in range(m_uv, m_ir + 1)
    )


def dimensionless_omega_shell(m: int) -> float:
    r"""
    \(\tilde\omega_m = T(m)/T_{\mathrm{Pl}}\).

    Lean `Hqiv.QuantumOptics.dimensionlessOmegaShell` / `dimensionlessOmegaShell_eq`.
    """
    return shell_temperature(m) / t_pl_natural()


def omega_shell_si(m: int, k_b: float, hbar: float) -> float:
    r"""Lean `omegaShellSI`: \(\omega_m = k_B T(m)/\hbar\)."""
    return k_b * shell_temperature(m) / hbar


def zero_point_energy_shell_si(m: int, k_b: float, hbar: float) -> float:
    r"""
    \(\tfrac12 \hbar \omega_m\) with \(\omega_m = k_B T/\hbar\) ⇒ \(\tfrac12 k_B T(m)\).

    Lean `zeroPointEnergyShellSI` / `zeroPointEnergyShellSI_eq`.
    """
    return 0.5 * k_b * shell_temperature(m)


def truncated_vacuum_zero_point_si(cap_m: int, k_b: float) -> float:
    r"""
    \(\sum_{m=0}^{M-1} N_m \tfrac12 k_B T(m)\).

    Lean `truncatedVacuumZeroPointSI`.
    """
    if cap_m < 0:
        raise ValueError("cap_m must be non-negative")
    total = 0.0
    for m in range(cap_m):
        n_m = shell_spatial_mode_count(m)
        total += n_m * 0.5 * k_b * shell_temperature(m)
    return total


def field_quantization_prefactor_si(
    m: int,
    *,
    k_b: float,
    hbar: float,
    epsilon_0: float,
    volume: float,
) -> float:
    r"""
    \(\sqrt{\hbar \omega_m / (2 \varepsilon_0 V)}\).

    Lean `fieldQuantizationPrefactorSI`.
    """
    if volume <= 0.0 or epsilon_0 <= 0.0:
        raise ValueError("volume and epsilon_0 must be positive")
    omega = omega_shell_si(m, k_b, hbar)
    return math.sqrt(hbar * omega / (2.0 * epsilon_0 * volume))


# Pauli matrices in the computational basis (complex, 2×2).
# Lean `sigmaPlus`, `sigmaMinus`, `sigmaZ`.

sigma_plus: np.ndarray[Any, Any] = np.array(
    [[0.0 + 0.0j, 1.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128
)
sigma_minus: np.ndarray[Any, Any] = np.array(
    [[0.0 + 0.0j, 0.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128
)
sigma_z: np.ndarray[Any, Any] = np.array(
    [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=np.complex128
)


def commutator(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return a @ b - b @ a


def commutator_sigma_plus_minus() -> np.ndarray[Any, Any]:
    """Lean `commutator_sigma_plus_sigmaMinus`."""
    return commutator(sigma_plus, sigma_minus)


def commutator_sigma_z_plus() -> np.ndarray[Any, Any]:
    """Lean `commutator_sigmaZ_sigmaPlus`."""
    return commutator(sigma_z, sigma_plus)


def commutator_sigma_z_minus() -> np.ndarray[Any, Any]:
    """Lean `commutator_sigmaZ_sigmaMinus`."""
    return commutator(sigma_z, sigma_minus)


def jc_coupling_tag(m: int) -> float:
    r"""
    Coupling tag \(g_{\mathrm{tag}}(m) = \sqrt{T(m)\,\varphi(m)} = \sqrt{2}\) in natural units.

    Lean `jcCouplingTag` / `jcCouplingTag_eq_sqrt_two`.
    """
    return math.sqrt(shell_temperature(m) * phi_of_shell(m))


def rabi_angular_frequency(g: float) -> float:
    r"""Lean `rabiAngularFrequency`: \(\Omega = 2g\)."""
    return 2.0 * g


def lindblad_scalar_rate(gamma: float) -> float:
    """Lean `lindbladScalarRate` (identity on the rate parameter)."""
    return gamma
