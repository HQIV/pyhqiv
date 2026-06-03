"""
Modified Navier–Stokes (HQIV fluid): f(a_loc, φ), g_vac, ν_eddy.

Canonical implementation; full roadmap at repository root: ``AGENTS/FLUID_OMAXWELL_ROADMAP.md``.
Uses ``gamma_hqiv()`` from ``pyhqiv.metric`` (Lean: ``Hqiv.gamma_HQIV``).

The single-source axiom is E_tot = m c² + ħ c/Δx with Δx ≤ Θ_local(x), giving
φ(x) = 2c²/Θ_local(x) and the lapse compression f(a_loc, φ) = a_loc/(a_loc + φ/6).
This module implements that f in the momentum equation (modified inertia), the
vacuum source g_vac = -γ ∇(φ δ̇θ′)/6, and eddy viscosity ν_eddy = γ Θ_local |δ̇θ′| ℓ_coh² C.
Laminar limit |a| ≫ φ/6 → f→1, g_vac→0 → standard Navier–Stokes (when other terms match).

F2 — O-Maxwell / plasma ↔ these inputs (hypothesis map; same as roadmap §F2 and
``Hqiv.Physics.HQIVFluidClosureScaffold``). Nothing below is a proved equality across sectors.

+----------------+------------------------------------------+---------------------------+
| Fluid argument | Python hooks (package)                   | Status / gap              |
+================+==========================================+===========================+
| **γ**          | ``gamma_hqiv()``                         | matched to Lean 2/5       |
+----------------+------------------------------------------+---------------------------+
| **φ**          | ``auxiliary_field.phi_of_shell``, etc.   | identify with continuum φ |
+----------------+------------------------------------------+---------------------------+
| **∇φ**         | ``modified_maxwell.grad_phi`` (stub)     | use chart gradients when  |
|                |                                          | wired; map 3 spatial comp.|
+----------------+------------------------------------------+---------------------------+
| **δ̇θ′** / rate | ``modified_maxwell.delta_theta_prime``   | Lean/Python δθ′ is from   |
|                |                                          | E′ (tipping), not ∂ₜ; gap |
+----------------+------------------------------------------+---------------------------+
| **∇δ̇θ′**       | (not in ``modified_maxwell``)            | supply from model         |
+----------------+------------------------------------------+---------------------------+
| **Θ_local**    | ``lightcone.x_over_theta_from_horizons``,| pick horizon proxy        |
|                | ``compton_horizon_bridge``, etc.         |                           |
+----------------+------------------------------------------+---------------------------+
| **ℓ_coh**      | narratively: Debye scale / plasma        | hypothesis vs Debye-style |
|                |                                          | scales (see Lean scaffold) |
+----------------+------------------------------------------+---------------------------+
| **Plasma J**   | ``modified_maxwell.current_o`` (stub);   | no J→τ / ν theorem yet    |
|                | see ``SchematicPlasmaCurrent`` in Lean     |                           |
+----------------+------------------------------------------+---------------------------+
| **C**          | ``coherence_factor`` in ``eddy_viscosity`` | phenomenological        |
+----------------+------------------------------------------+---------------------------+

**F3:** ``PlasmaFluidClosureHypothesis`` — scalar ``ν_total = ν_mol + ν_eddy`` with ``ν_eddy`` from
``eddy_viscosity``; call ``.holds()`` to check (Lean: ``PlasmaFluidClosureAssumptions``).

**F4:** coefficient-level toward classical NS is documented in Lean ``CoefficientsTowardClassicalNS``;
not a PDE theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from pyhqiv.metric import gamma_hqiv


def f_inertia(
    a_loc: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    f_min: float = 0.01,
) -> Union[float, np.ndarray]:
    """
    Modified inertia factor f(a_loc, φ) = a_loc / (a_loc + φ/6). Paper particle action.

    In momentum equation: ρ f Dv/Dt = rhs ⇒ Dv/Dt = rhs / (ρ f).
    Laminar limit |a| ≫ φ/6 ⇒ f → 1.

    Parameters
    ----------
    a_loc : float or array
        Magnitude of local acceleration |a| (or scale).
    phi : float or array
        Auxiliary field φ = 2c²/Θ_local.
    f_min : float
        Floor for f (default 0.01) to avoid division by zero.

    Returns
    -------
    float or array
        f ∈ [f_min, 1].
    """
    a = np.asarray(a_loc, dtype=float)
    p = np.asarray(phi, dtype=float)
    denom = np.maximum(a + p / 6.0, 1e-30)
    f = a / denom
    return np.maximum(np.minimum(f, 1.0), f_min)


def g_vac_vector(
    phi: Union[float, np.ndarray],
    dot_delta_theta: Union[float, np.ndarray],
    grad_phi: np.ndarray,
    grad_dot_delta_theta: np.ndarray,
    gamma: Optional[float] = None,
) -> np.ndarray:
    """
    Vacuum source g_vac = -γ ∇(φ δ̇θ′) / 6 (per unit mass) for momentum equation.

    ∇(φ δ̇θ′) = φ ∇δ̇θ′ + δ̇θ′ ∇φ. So g_vac = -γ/6 * (φ * grad_dot_delta_theta + dot_delta_theta * grad_phi).

    Parameters
    ----------
    phi : float or array
        φ at point(s).
    dot_delta_theta : float or array
        δ̇θ′ at point(s).
    grad_phi : array
        ∇φ, shape (..., 3) or (3,).
    grad_dot_delta_theta : array
        ∇δ̇θ′, shape (..., 3) or (3,).
    gamma : float, optional
        Monogamy coefficient; default ``gamma_hqiv()`` (= 2/5).

    Returns
    -------
    array
        g_vac vector, same shape as grad_phi.
    """
    if gamma is None:
        gamma = gamma_hqiv()
    phi = np.asarray(phi, dtype=float)
    dot = np.asarray(dot_delta_theta, dtype=float)
    g_phi = np.asarray(grad_phi, dtype=float)
    g_dot = np.asarray(grad_dot_delta_theta, dtype=float)
    term = phi * g_dot + dot * g_phi
    return (-gamma / 6.0) * term


def eddy_viscosity(
    Theta_local: Union[float, np.ndarray],
    dot_delta_theta: Union[float, np.ndarray],
    l_coh: Union[float, np.ndarray],
    coherence_factor: float = 1.0,
    gamma: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    HQIV eddy viscosity ν_eddy = γ Θ_local |δ̇θ′| ℓ_coh² C.

    τ_total = τ_mol + ρ ν_eddy S. High coherence (plasma): C≈1; entropic turbulence: C smaller.

    Parameters
    ----------
    Theta_local : float or array
        Local causal-horizon radius (Θ), same units as ℓ_coh.
    dot_delta_theta : float or array
        |δ̇θ′| (phase-lift clock, ≈ H in homogeneous limit).
    l_coh : float or array
        Coherence length (e.g. Debye length, or integral scale).
    coherence_factor : float
        C ∈ [0, 1]; default 1.0.
    gamma : float, optional
        Default ``gamma_hqiv()``.

    Returns
    -------
    float or array
        ν_eddy (same units as Θ * (1/s) * length²).
    """
    if gamma is None:
        gamma = gamma_hqiv()
    Theta = np.asarray(Theta_local, dtype=float)
    dot = np.asarray(dot_delta_theta, dtype=float)
    lc = np.asarray(l_coh, dtype=float)
    return gamma * Theta * np.abs(dot) * (lc**2) * coherence_factor


def modified_momentum_rhs(
    grad_p: np.ndarray,
    div_tau_mol: np.ndarray,
    g_ext: np.ndarray,
    g_vac: np.ndarray,
    rho: Union[float, np.ndarray] = 1.0,
) -> np.ndarray:
    """
    RHS of modified momentum (before dividing by ρ f): -∇p/ρ + ∇·τ/ρ + g_ext + g_vac.

    Then a_modified = this_rhs / f (with f = f_inertia(|a|, φ)).
    """
    rho = np.asarray(rho, dtype=float)
    if rho.shape != grad_p.shape and np.ndim(rho) < np.ndim(grad_p):
        rho = np.broadcast_to(rho, grad_p.shape)
    return -grad_p / np.maximum(rho, 1e-30) + div_tau_mol / np.maximum(rho, 1e-30) + g_ext + g_vac


@dataclass(frozen=True)
class PlasmaFluidClosureHypothesis:
    """
    F3 scalar bookkeeping: ``ν_total = ν_mol + ν_eddy`` with ``ν_eddy`` given by ``eddy_viscosity``.

    Mirrors ``PlasmaFluidClosureAssumptions`` in ``Hqiv.Physics.HQIVFluidClosureScaffold``.
    Does **not** assert derivation from kinetic theory or O-Maxwell.
    """

    nu_mol: float
    nu_eddy: float
    nu_total: float
    theta_local: float
    dot_delta_theta: float
    l_coh: float
    coherence: float
    gamma: Optional[float] = None

    def holds(self, tol: float = 1e-9) -> bool:
        """Return True if split + HQIV eddy match and ``coherence`` ∈ [0, 1]."""
        g = self.gamma if self.gamma is not None else gamma_hqiv()
        nu_h = float(
            g * self.theta_local * abs(self.dot_delta_theta) * (self.l_coh**2) * self.coherence
        )
        split_ok = abs(self.nu_total - (self.nu_mol + self.nu_eddy)) <= tol
        eddy_ok = abs(self.nu_eddy - nu_h) <= tol
        c = self.coherence
        c_ok = -tol <= c <= 1.0 + tol
        return split_ok and eddy_ok and c_ok
