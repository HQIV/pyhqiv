"""
Continuum minimum-energy construction from the HQIV axiom (no bak reuse).

**Axiom (per locality):**  E = m c² + ħc/Δx   with Δx > 0 a physical resolution /
horizon spacing (Δx ≤ Θ_local in the full theory).

**PDE step (void between sources):** In a charge- / mass-free region we take **Δx**
to be **harmonic**, i.e. it solves

    ∇²(Δx) = 0

with **Dirichlet** data on nucleon (or quark) boundaries. That is the unique field
that **minimizes** the Dirichlet energy ∫|∇(Δx)|² dV among all admissible Δx with
the same boundary values — the standard elliptic variational principle.

**Energy evaluation:** On that solution,

    E_contribution = ∫_Ω ( ρ c² + ħc / max(Δx, Δx_floor) ) dV ,

with ρ point-like or cell-averaged masses at boundaries / sources. For vacuum
segments, only the **ħc/Δx** term integrates along the bond.

**Two-body minimum:** For two nucleons separated by distance L, E_total(L) =
m₁c² + m₂c² + ∫₀^L ħc/Δx(s) ds with Δx linear (1D Laplace). Minimize over
L ≥ L_min (default L_min = Δx_left + Δx_right as a geometric contact scale).

Lean-facing references: same axiom as ``Hqiv.Geometry.AuxiliaryField`` / horizon
documentation; PDE layer matches ``docs/binding_energy_walkthrough.md`` §6.5.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pyhqiv.forces import c_si, hbar_si


def hbar_c_si() -> float:
    """ħc in J·m."""
    return hbar_si() * c_si()


def rest_energy_j(mass_kg: float) -> float:
    """m c² in joules."""
    c = c_si()
    return float(mass_kg) * c * c


def information_energy_j(delta_x_m: float, hbar_times_c: float | None = None) -> float:
    """Single-point ħc/Δx in joules (Δx > 0)."""
    hc = hbar_c_si() if hbar_times_c is None else float(hbar_times_c)
    dx = max(float(delta_x_m), 1e-50)
    return hc / dx


def laplace_delta_x_1d(
    length_m: float,
    delta_left_m: float,
    delta_right_m: float,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve d²Δx/ds² = 0 on [0, length] with Δx(0)=δ_L, Δx(length)=δ_R.

    Interior: second-order centered FD, uniform grid. Returns (s, delta_x).

    Notes
    -----
    The analytic solution is linear; this routine exists to validate the discrete
    Laplacian and to mirror the 2D discrete solver interface.
    """
    if length_m <= 0.0:
        raise ValueError("length_m must be positive")
    if n_nodes < 3:
        raise ValueError("n_nodes must be at least 3 for interior Laplace solve")
    L = float(length_m)
    d0 = float(delta_left_m)
    d1 = float(delta_right_m)
    n = int(n_nodes)
    h = L / float(n - 1)
    inv_h2 = 1.0 / (h * h)
    # (u[i-1] - 2u[i] + u[i+1]) / h^2 = 0  on interior
    main = (-2.0 * inv_h2) * np.ones(n - 2)
    upper = np.ones(n - 3) * inv_h2
    lower = np.ones(n - 3) * inv_h2
    A = sparse.diags([lower, main, upper], [-1, 0, 1], format="csr")
    b = np.zeros(n - 2)
    b[0] = -d0 * inv_h2
    b[-1] = -d1 * inv_h2
    u_int = spsolve(A, b)
    u = np.empty(n)
    u[0] = d0
    u[1:-1] = u_int
    u[-1] = d1
    s = np.linspace(0.0, L, n)
    return s, u


def information_energy_line_integral_1d(
    s_m: np.ndarray,
    delta_x_m: np.ndarray,
    hbar_times_c: float | None = None,
    delta_x_floor_m: float = 1e-18,
) -> float:
    """
    ∫ ħc / max(Δx(s), floor) ds along the 1D bond (trapezoid rule).
    """
    hc = hbar_c_si() if hbar_times_c is None else float(hbar_times_c)
    dx = np.maximum(np.asarray(delta_x_m, dtype=float), float(delta_x_floor_m))
    integrand = hc / dx
    s = np.asarray(s_m, dtype=float)
    # NumPy 2.0+: trapezoid; older: trapz
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(integrand, s))
    return float(np.trapz(integrand, s))


def analytic_laplace_delta_x_1d(s_m: np.ndarray, L: float, d0: float, d1: float) -> np.ndarray:
    """Linear harmonic on [0, L]."""
    t = np.asarray(s_m, dtype=float) / L
    return d0 + (d1 - d0) * t


def two_body_line_information_energy_analytic(
    length_m: float,
    delta_left_m: float,
    delta_right_m: float,
    hbar_times_c: float | None = None,
) -> float:
    """
    Closed form ∫₀^L ħc/Δx(s) ds for linear Δx(s) = δ_L + (δ_R-δ_L) s/L.

    E = (ħc L / (δ_R - δ_L)) ln(δ_R/δ_L)  when δ_L ≠ δ_R;
    E = ħc L / δ_L  when δ_L = δ_R.
    """
    hc = hbar_c_si() if hbar_times_c is None else float(hbar_times_c)
    L = float(length_m)
    a = float(delta_left_m)
    b = float(delta_right_m)
    if L <= 0.0:
        raise ValueError("length_m must be positive")
    if a <= 0.0 or b <= 0.0:
        raise ValueError("boundary Δx values must be positive")
    if abs(b - a) < 1e-30 * max(abs(a), abs(b), 1.0):
        return hc * L / a
    return hc * L / (b - a) * np.log(b / a)


@dataclass(frozen=True)
class TwoBodyMinimum1D:
    """Result of minimizing total axiom energy over separation L ≥ L_min."""

    separation_m: float
    total_energy_j: float
    rest_energy_j: float
    information_energy_j: float
    delta_left_m: float
    delta_right_m: float


def minimize_two_body_separation_1d(
    mass1_kg: float,
    mass2_kg: float,
    delta_left_m: float,
    delta_right_m: float,
    *,
    l_min_m: float | None = None,
    l_max_m: float = 50e-15,
    hbar_times_c: float | None = None,
) -> TwoBodyMinimum1D:
    """
    Minimize  E(L) = m₁c² + m₂c² + ∫₀^L ħc/Δx(s) ds

    with Δx harmonic (linear) between boundaries δ_L, δ_R at separation L.

    Default L_min = δ_L + δ_R (geometric contact of the two horizon scales).
    """
    m1 = float(mass1_kg)
    m2 = float(mass2_kg)
    d0 = float(delta_left_m)
    d1 = float(delta_right_m)
    if d0 <= 0.0 or d1 <= 0.0:
        raise ValueError("delta_left_m and delta_right_m must be positive")
    l_lo = d0 + d1 if l_min_m is None else float(l_min_m)
    if l_lo <= 0.0:
        raise ValueError("l_min_m must be positive")
    l_hi = float(l_max_m)
    if l_hi <= l_lo:
        raise ValueError("l_max_m must exceed l_min")

    e_rest = rest_energy_j(m1) + rest_energy_j(m2)
    hc = hbar_c_si() if hbar_times_c is None else float(hbar_times_c)

    # E_info(L) = (ħc L / (δ_R-δ_L)) ln(δ_R/δ_L) for δ_L≠δ_R — linear in L.
    # Minimum of a linear function on [L_min, L_max] is at an endpoint.
    if abs(d1 - d0) < 1e-30 * max(abs(d0), abs(d1), 1.0):
        slope = hc / d0  # E_info = ħc L / δ
        L_opt = l_lo if slope >= 0.0 else l_hi
    else:
        slope = hc / (d1 - d0) * math.log(d1 / d0)
        L_opt = l_lo if slope >= 0.0 else l_hi
    L_opt = float(max(l_lo, min(l_hi, L_opt)))
    e_info = two_body_line_information_energy_analytic(L_opt, d0, d1, hbar_times_c)
    return TwoBodyMinimum1D(
        separation_m=L_opt,
        total_energy_j=float(e_rest + e_info),
        rest_energy_j=e_rest,
        information_energy_j=float(e_info),
        delta_left_m=d0,
        delta_right_m=d1,
    )


def laplace_delta_x_2d_rectangle(
    width_m: float,
    height_m: float,
    nx: int,
    ny: int,
    boundary_delta_x: Callable[[float, float], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve ∇²(Δx)=0 on a rectangle [0,width]×[0,height] with Dirichlet Δx on the boundary.

    boundary_delta_x(x, y) is evaluated on the four edges (corners use x,y on boundary).

    Returns (x_grid, y_grid, delta_x) each shaped (ny, nx).
    """
    if nx < 3 or ny < 3:
        raise ValueError("nx and ny must be at least 3")
    w = float(width_m)
    hgt = float(height_m)
    if w <= 0.0 or hgt <= 0.0:
        raise ValueError("width and height must be positive")
    dx = w / float(nx - 1)
    dy = hgt / float(ny - 1)
    ax = 1.0 / (dx * dx)
    ay = 1.0 / (dy * dy)
    x1d = np.linspace(0.0, w, nx)
    y1d = np.linspace(0.0, hgt, ny)
    xg, yg = np.meshgrid(x1d, y1d, indexing="xy")

    u_full = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            if j == 0:
                u_full[j, i] = boundary_delta_x(x1d[i], 0.0)
            elif j == ny - 1:
                u_full[j, i] = boundary_delta_x(x1d[i], hgt)
            elif i == 0:
                u_full[j, i] = boundary_delta_x(0.0, y1d[j])
            elif i == nx - 1:
                u_full[j, i] = boundary_delta_x(w, y1d[j])

    def idx(i: int, j: int) -> int:
        return (j - 1) * (nx - 2) + (i - 1)

    n_unknown = (nx - 2) * (ny - 2)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(n_unknown)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = idx(i, j)
            center = -2.0 * (ax + ay)
            rows.append(k)
            cols.append(k)
            data.append(center)
            for ni, nj, coeff in (
                (i - 1, j, ax),
                (i + 1, j, ax),
                (i, j - 1, ay),
                (i, j + 1, ay),
            ):
                if 1 <= ni <= nx - 2 and 1 <= nj <= ny - 2:
                    rows.append(k)
                    cols.append(idx(ni, nj))
                    data.append(coeff)
                else:
                    rhs[k] -= coeff * u_full[nj, ni]

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_unknown, n_unknown))
    u_int = spsolve(A, rhs)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u_full[j, i] = u_int[idx(i, j)]
    return xg, yg, u_full


def continuum_total_energy_2d(
    delta_x_field_m: np.ndarray,
    cell_dx_m: float,
    cell_dy_m: float,
    mass_density_kg_m3: np.ndarray | None = None,
    *,
    hbar_times_c: float | None = None,
    delta_x_floor_m: float = 1e-18,
) -> float:
    """
    ∫ ( ρ c² + ħc/max(Δx,floor) ) dA on a uniform rectangular mesh.

    ``delta_x_field_m`` is nodal values on a tensor grid (ny, nx), one node per
    corner. Each **cell** uses the arithmetic mean of its four corners for Δx;
    there are (ny−1)(nx−1) cells of area ``cell_dx_m * cell_dy_m``.
    """
    c = c_si()
    hc = hbar_c_si() if hbar_times_c is None else float(hbar_times_c)
    u = np.asarray(delta_x_field_m, dtype=float)
    if u.ndim != 2:
        raise ValueError("delta_x_field_m must be 2D (ny, nx)")
    ny, nx = u.shape
    if ny < 2 or nx < 2:
        raise ValueError("need at least 2×2 nodes for cell integration")
    area = float(cell_dx_m) * float(cell_dy_m)
    u_cell = 0.25 * (u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:])
    dx = np.maximum(u_cell, float(delta_x_floor_m))
    info = hc / dx
    if mass_density_kg_m3 is None:
        rho_part = 0.0
    else:
        rho = np.asarray(mass_density_kg_m3, dtype=float)
        if rho.shape != u_cell.shape:
            raise ValueError("mass_density_kg_m3 must match cell shape (ny-1, nx-1)")
        rho_part = float(np.sum(rho * (c**2) * area))
    return rho_part + float(np.sum(info * area))


__all__ = [
    "TwoBodyMinimum1D",
    "analytic_laplace_delta_x_1d",
    "continuum_total_energy_2d",
    "hbar_c_si",
    "information_energy_j",
    "information_energy_line_integral_1d",
    "laplace_delta_x_1d",
    "laplace_delta_x_2d_rectangle",
    "minimize_two_body_separation_1d",
    "rest_energy_j",
    "two_body_line_information_energy_analytic",
]
