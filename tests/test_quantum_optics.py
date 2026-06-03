"""Numerical checks against `Hqiv.QuantumOptics.HorizonQED` (Lean)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyhqiv.quantum_optics.horizon_qed import (
    commutator_sigma_plus_minus,
    commutator_sigma_z_minus,
    commutator_sigma_z_plus,
    dimensionless_omega_shell,
    field_quantization_prefactor_si,
    jc_coupling_tag,
    omega_shell_si,
    rabi_angular_frequency,
    sigma_minus,
    sigma_plus,
    sigma_z,
    truncated_vacuum_zero_point_si,
    zero_point_energy_shell_si,
)


def test_dimensionless_omega_matches_closed_form() -> None:
    for m in range(8):
        expected = 1.0 / (m + 1)
        assert dimensionless_omega_shell(m) == pytest.approx(expected)


def test_zero_point_identity() -> None:
    k_b, hbar = 1.38e-23, 1.054571817e-34
    m = 3
    zpe = zero_point_energy_shell_si(m, k_b, hbar)
    assert zpe == pytest.approx(0.5 * k_b * (1.0 / (m + 1)))


def test_jc_coupling_tag_is_sqrt_two() -> None:
    for m in range(5):
        assert jc_coupling_tag(m) == pytest.approx(math.sqrt(2.0))


def test_pauli_commutators_numeric() -> None:
    tol = 1e-14
    assert np.allclose(commutator_sigma_plus_minus(), sigma_z, atol=tol)
    assert np.allclose(commutator_sigma_z_plus(), 2.0 * sigma_plus, atol=tol)
    assert np.allclose(commutator_sigma_z_minus(), -2.0 * sigma_minus, atol=tol)


def test_rabi_frequency() -> None:
    g = jc_coupling_tag(0)
    assert rabi_angular_frequency(g) == pytest.approx(2.0 * math.sqrt(2.0))


def test_field_prefactor_positive() -> None:
    m = 2
    e0 = 8.8541878128e-12
    v = 1e-3
    pref = field_quantization_prefactor_si(
        m, k_b=1.38e-23, hbar=1.054571817e-34, epsilon_0=e0, volume=v
    )
    assert pref > 0.0


def test_truncated_vacuum_sum_finite() -> None:
    s = truncated_vacuum_zero_point_si(10, k_b=1.0)
    assert s > 0.0
    assert math.isfinite(s)


def test_omega_shell_si() -> None:
    m = 4
    k_b, hbar = 1.0, 2.0
    assert omega_shell_si(m, k_b, hbar) == pytest.approx(k_b / (hbar * (m + 1)))
