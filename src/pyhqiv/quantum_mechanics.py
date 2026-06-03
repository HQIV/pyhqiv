"""
Quantum-mechanics emergence layer for the fresh pyhqiv rebuild.

This ports the computable “skeleton” content from:
- `HQIV_LEAN/Hqiv/QuantumMechanics/Schrodinger.lean`

**Birefringence channel (measurement / redshift closure):** Lean’s finite Born layer defines
``birefringenceRedshiftN`` and collapse-energy balance with the auxiliary field — see
``Hqiv/QuantumMechanics/BornMeasurementFinite.lean``,
``Hqiv/QuantumMechanics/AuxFieldBellDelayedChoice.lean`` (same formula as
``Hqiv.QM.birefringenceRedshift``). :func:`birefringence_redshift` and :func:`redshifted_energy_n`
mirror those definitions. The self-clock angle β at ``now`` (radians) aligns with
``pyhqiv.surface_wave_self_clock.cosmic_birefringence_rad_at_now`` when using the packaged witness.

We currently implement:
1. Lapse factor and lapse-corrected Hamiltonian scaling.
2. Time-dependent Schrödinger residual check utilities for generic operators.
3. Shell-resolved hydrogenic expected ground energy using O-Maxwell-derived
   shell Coulomb strength.

We intentionally do NOT yet implement the full continuum Laplacian wiring
that in Lean is a placeholder, so this layer focuses on the emergence logic
and consistency checks rather than full spectral theorems.

Angular / ``S²`` factors (orthonormal ``Y_ℓ^m``, ``-Δ_S²`` eigenvalues ``ℓ(ℓ+1)``)
live in ``pyhqiv.spherical_harmonics`` for multipole and partial-wave bookkeeping.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
import math
from typing import Callable

from pyhqiv.metric import hqvm_lapse
from pyhqiv.forces import hbar_si
from pyhqiv.omaxwell_couplings import coulomb_strength_shell, expected_ground_energy_at_shell


Wavefunction = Callable[[tuple[float, float, float]], complex]
Operator = Callable[[Wavefunction], Wavefunction]  # H: ψ ↦ Hψ


def lapse_factor(phi_newtonian: float, phi_auxiliary: float, t: float) -> float:
    """
    Lean reference:
    `lapseFactor Φ φ t = HQVM_lapse Φ φ t`.
    """
    return hqvm_lapse(phi_newtonian, phi_auxiliary, t)


def lapse_corrected_hamiltonian(
    H: Operator,
    *,
    phi_newtonian: float,
    phi_auxiliary: float,
    t: float,
) -> Operator:
    """
    Lean reference:
    `lapseCorrectedHamiltonian Φ φ t = lapseFactor(Φ, φ, t) * hqivHamiltonian`.
    """
    N = lapse_factor(phi_newtonian, phi_auxiliary, t)

    def Hcorr(psi: Wavefunction) -> Wavefunction:
        Hpsi = H(psi)
        return lambda x: N * Hpsi(x)

    return Hcorr


def satisfies_time_dependent_schrodinger_residual(
    H: Operator,
    psi_of_t: Callable[[float], Wavefunction],
    t: float,
    x: tuple[float, float, float],
    *,
    hbar: float | None = None,
    dt: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Numeric residual check for:
      i ħ ∂ψ/∂t = Hψ
    using a central finite difference for ∂ψ/∂t.
    """
    if hbar is None:
        hbar = hbar_si()

    psi_plus = psi_of_t(t + dt)
    psi_minus = psi_of_t(t - dt)
    dpsi_dt_x = (psi_plus(x) - psi_minus(x)) / (2.0 * dt)
    lhs = 1j * hbar * dpsi_dt_x
    rhs = H(psi_of_t(t))(x)
    return abs(lhs - rhs) <= atol


def satisfies_lapse_corrected_schrodinger_residual(
    H: Operator,
    psi_of_t: Callable[[float], Wavefunction],
    t: float,
    x: tuple[float, float, float],
    *,
    phi_newtonian: float,
    phi_auxiliary: float,
    hbar: float | None = None,
    dt: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Numeric residual check for:
      i ħ ∂ψ/∂t = N(t) * Hψ
    where N(t) = HQVM_lapse(Φ, φ, t).
    """
    if hbar is None:
        hbar = hbar_si()

    psi_plus = psi_of_t(t + dt)
    psi_minus = psi_of_t(t - dt)
    dpsi_dt_x = (psi_plus(x) - psi_minus(x)) / (2.0 * dt)
    lhs = 1j * hbar * dpsi_dt_x

    N = lapse_factor(phi_newtonian, phi_auxiliary, t)
    rhs = N * H(psi_of_t(t))(x)
    return abs(lhs - rhs) <= atol


def birefringence_redshift(beta_rad: float, kappa_beta: float) -> float:
    """
    Lean ``birefringenceRedshiftN`` / ``BornMeasurementFinite.birefringenceRedshiftN``:

    ``z_β = exp(β_rad / κ_β) - 1``.

    ``β_rad`` is the cosmological polarization rotation; ``κ_β`` is the HQIV normalization
    (must be supplied by a proved witness / paper slot — not hardcoded here).
    """
    if kappa_beta == 0.0:
        raise ValueError("kappa_beta must be nonzero")
    return math.exp(beta_rad / kappa_beta) - 1.0


def redshifted_energy_n(e_post: float, z: float) -> float:
    """
    Lean ``redshiftedEnergyN Epost z = Epost / (1 + z)`` (Born finite-state bookkeeping).
    """
    return float(e_post) / (1.0 + float(z))


def redshifted_energy_birefringence_balance(
    e_post: float,
    beta_rad: float,
    kappa_beta: float,
) -> float:
    """
    Lean ``redshiftedEnergyN_birefringence_balance``:

    ``redshiftedEnergyN Epost z_β · exp(β/κ) = Epost`` with ``z_β = birefringence_redshift(β, κ)``.
    Returns the left-hand side; should equal ``e_post`` (up to float tolerance).
    """
    z = birefringence_redshift(beta_rad, kappa_beta)
    return redshifted_energy_n(e_post, z) * math.exp(beta_rad / kappa_beta)


def coulomb_potential(Z: int, alpha: float, x: tuple[float, float, float]) -> float:
    """
    Coulomb potential V(r) = - Z * alpha / r in natural units.
    """
    r = math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    if r <= 0.0:
        return 0.0
    return -float(Z) * alpha / r


def shell_coulomb_strength(m: int, c: float = 1.0) -> float:
    """
    Convenience wrapper around O-Maxwell couplings.
    """
    return coulomb_strength_shell(m, c=c)


def expected_ground_energy_shell(m: int, Z: int, mu: float, c: float = 1.0) -> float:
    """
    Shell-resolved ground energy proxy from BoundStates.lean.
    """
    return expected_ground_energy_at_shell(m, Z=Z, mu=mu, c=c)


__all__ = [
    "Operator",
    "Wavefunction",
    "birefringence_redshift",
    "coulomb_potential",
    "expected_ground_energy_shell",
    "expected_ground_energy_at_shell",
    "lapse_corrected_hamiltonian",
    "lapse_factor",
    "redshifted_energy_birefringence_balance",
    "redshifted_energy_n",
    "satisfies_lapse_corrected_schrodinger_residual",
    "satisfies_time_dependent_schrodinger_residual",
    "shell_coulomb_strength",
]

