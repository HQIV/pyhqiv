"""Tests for axiom E = mc² + ħc/Δx with harmonic Δx from Laplace PDE."""

from __future__ import annotations

import math

import numpy as np

from pyhqiv.forces import c_si, hbar_si
from pyhqiv.hqiv_continuum_minimum_energy import (
    analytic_laplace_delta_x_1d,
    continuum_total_energy_2d,
    hbar_c_si,
    information_energy_j,
    information_energy_line_integral_1d,
    laplace_delta_x_1d,
    laplace_delta_x_2d_rectangle,
    minimize_two_body_separation_1d,
    rest_energy_j,
    two_body_line_information_energy_analytic,
)


def test_information_plus_rest_matches_axiom_si() -> None:
    m = 1.0e-30  # kg
    dx = 1.0e-15
    e = rest_energy_j(m) + information_energy_j(dx)
    assert e > rest_energy_j(m)
    assert math.isclose(
        information_energy_j(dx),
        hbar_si() * c_si() / dx,
        rel_tol=1e-12,
        abs_tol=0.0,
    )


def test_1d_laplace_delta_x_matches_linear_analytic() -> None:
    L = 2.0e-15
    d0, d1 = 0.9e-15, 1.1e-15
    s, u = laplace_delta_x_1d(L, d0, d1, 201)
    ua = analytic_laplace_delta_x_1d(s, L, d0, d1)
    np.testing.assert_allclose(u, ua, rtol=1e-8, atol=1e-22)


def test_line_integral_matches_analytic_formula() -> None:
    L = 1.4e-15
    d0, d1 = 0.85e-15, 0.95e-15
    s, u = laplace_delta_x_1d(L, d0, d1, 2001)
    num = information_energy_line_integral_1d(s, u)
    ana = two_body_line_information_energy_analytic(L, d0, d1)
    assert math.isclose(num, ana, rel_tol=1e-4, abs_tol=0.0)


def test_two_body_minimum_at_contact_scale() -> None:
    """For unequal δ_L, δ_R, ∫ ħc/Δx ds grows linearly in L ⇒ minimum at L_min."""
    m = 1.67e-27
    d0, d1 = 0.8e-15, 1.0e-15
    l_min = d0 + d1
    res = minimize_two_body_separation_1d(m, m, d0, d1, l_max_m=20e-15)
    assert math.isclose(res.separation_m, l_min, rel_tol=1e-6, abs_tol=1e-18)
    assert res.total_energy_j == res.rest_energy_j + res.information_energy_j


def test_2d_harmonic_linear_in_x() -> None:
    w, hgt = 3e-15, 2e-15

    def bc(x: float, y: float) -> float:
        if y <= 0.0 or y >= hgt:
            return 1.0e-15 + (2.0e-15 - 1.0e-15) * (x / w)
        if x <= 0.0:
            return 1.0e-15
        return 2.0e-15

    xg, yg, u = laplace_delta_x_2d_rectangle(w, hgt, 41, 31, bc)
    u_ex = 1.0e-15 + (2.0e-15 - 1.0e-15) * (xg / w)
    # interior should match u_ex(x,y)=u_ex(x) to high accuracy
    np.testing.assert_allclose(u, u_ex, rtol=2e-3, atol=1e-18)


def test_2d_continuum_information_energy_positive() -> None:
    w, hgt = 1e-15, 1e-15

    def bc(_x: float, _y: float) -> float:
        return 1.0e-15

    _xg, _yg, u = laplace_delta_x_2d_rectangle(w, hgt, 21, 21, bc)
    dx_cell = w / 20.0
    dy_cell = hgt / 20.0
    e = continuum_total_energy_2d(u, dx_cell, dy_cell, mass_density_kg_m3=None)
    # uniform Δx → ∫ ħc/Δx dA = ħc/Δx * area
    assert e > 0.0
    expected = (hbar_c_si() / 1e-15) * w * hgt
    assert math.isclose(e, expected, rel_tol=0.05, abs_tol=0.0)
