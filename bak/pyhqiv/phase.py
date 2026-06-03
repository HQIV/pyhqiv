"""
HQIV phase lift: δθ′(E′), ˙δθ′, homogeneous limit ˙δθ′≈H, ADM lapse compression,
and modified Maxwell lift terms (γ(φ/c²)(˙δθ′/c)). Paper Sec. 2 & 5.

Ω_k (spatial curvature from shell integral, paper Sec. curvature) is owned here so the
O-lifted Maxwell / phase-horizon layer applies it universally. Use curvature_factor()
for effective horizon Θ_eff = Θ_local × curvature_factor(); default omega_k from lattice.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

import numpy as np

from pyhqiv.constants import C_SI, GAMMA, LAPSE_COMPRESSION_PAPER

# Cached default phase lift for universal curvature (used by utils, fields, etc.)
_default_phase_lift: Optional["HQIVPhaseLift"] = None


def default_phase_lift() -> "HQIVPhaseLift":
    """Default phase lift (O-lifted Maxwell layer). Carries dynamic Ω_k; use for universal curvature."""
    global _default_phase_lift
    if _default_phase_lift is None:
        _default_phase_lift = HQIVPhaseLift()
    return _default_phase_lift


def delta_theta_prime(E_prime: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Phase-horizon angle δθ′(E′) = arctan(E′)×(π/2).
    E′ is normalized energy in [0, 1] (or same units as scale); paper Sec. 2.
    """
    return np.arctan(np.asarray(E_prime, dtype=float)) * (math.pi / 2.0)


def delta_theta_prime_dot_homogeneous(H: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Homogeneous limit: ˙δθ′ ≈ H (natural units). In SI: ˙δθ′ has units 1/s.
    """
    return np.asarray(H, dtype=float)


def adm_lapse_compression_factor(
    phi_over_c2: Union[float, np.ndarray],
    delta_theta_dot_over_c: Union[float, np.ndarray],
    gamma: float = GAMMA,
) -> Union[float, np.ndarray]:
    """
    Effective lapse factor from γ(φ/c²)(˙δθ′/c) term.
    Homogeneous: φ = cH, ˙δθ′ = H ⇒ γ H²/c²; compression factor ≈ 3.96 (paper).
    """
    phi = np.asarray(phi_over_c2, dtype=float)
    dtdc = np.asarray(delta_theta_dot_over_c, dtype=float)
    return 1.0 + gamma * phi * dtdc


def apparent_age_from_wall_clock(
    age_wall_yr: Union[float, np.ndarray],
    lapse_compression: float = LAPSE_COMPRESSION_PAPER,
) -> Union[float, np.ndarray]:
    """Apparent age (local chronometers) from wall-clock age and lapse compression."""
    return np.asarray(age_wall_yr, dtype=float) / lapse_compression


class HQIVPhaseLift:
    """
    Phase-horizon lift: δθ′(E′), ˙δθ′ = u^μ ∇_μ δθ′, homogeneous ˙δθ′≈H,
    ADM lapse compression, and modified Maxwell terms. Owns Ω_k so curvature
    is universally applied (same 0 < x < θ as shell integral; paper Sec. curvature).
    """

    def __init__(
        self,
        gamma: float = GAMMA,
        c_si: float = C_SI,
        omega_k: Optional[float] = None,
        lattice: Optional[Any] = None,
    ) -> None:
        self.gamma = gamma
        self.c_si = c_si
        if omega_k is not None:
            self._omega_k = float(omega_k)
        elif lattice is not None and hasattr(lattice, "omega_k_true"):
            self._omega_k = float(lattice.omega_k_true())
        else:
            from pyhqiv.lattice import DiscreteNullLattice
            self._omega_k = DiscreteNullLattice().omega_k_true()

    @property
    def omega_k(self) -> float:
        """Spatial curvature (dynamic from lattice by default). Same as shell integral."""
        return self._omega_k

    def curvature_factor(self) -> float:
        """Factor for effective horizon: Θ_eff = Θ_local × curvature_factor(). Paper: 1 + Ω_k."""
        return 1.0 + self._omega_k

    def delta_theta_prime(self, E_prime: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """δθ′(E′) = arctan(E′)×(π/2)."""
        return delta_theta_prime(E_prime)

    def delta_theta_prime_dot(
        self,
        u_mu: Optional[np.ndarray] = None,
        grad_delta_theta: Optional[np.ndarray] = None,
        H_homogeneous: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """
        ˙δθ′ = u^μ ∇_μ δθ′. If grad_delta_theta and u_mu not provided, use homogeneous limit ˙δθ′ ≈ H.
        """
        if H_homogeneous is not None:
            return delta_theta_prime_dot_homogeneous(H_homogeneous)
        if u_mu is not None and grad_delta_theta is not None:
            return np.dot(np.asarray(u_mu).ravel(), np.asarray(grad_delta_theta).ravel())
        raise ValueError("Provide either H_homogeneous or (u_mu, grad_delta_theta)")

    def lapse_compression(
        self,
        phi_local: Union[float, np.ndarray],
        delta_theta_dot: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Lapse factor from γ(φ/c²)(˙δθ′/c). phi_local in (m/s²) or natural; delta_theta_dot in 1/s."""
        phi_over_c2 = np.asarray(phi_local, dtype=float) / (self.c_si**2)
        dtdc = np.asarray(delta_theta_dot, dtype=float) / self.c_si
        return adm_lapse_compression_factor(phi_over_c2, dtdc, gamma=self.gamma)

    def maxwell_lift_coefficient(
        self,
        phi_over_c2: Union[float, np.ndarray],
        delta_theta_dot_over_c: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Coefficient γ(φ/c²)(˙δθ′/c) for modified Maxwell D/Dt = ∂/∂t′ + ˙δθ′ ∂/∂δθ′."""
        return self.gamma * np.asarray(phi_over_c2) * np.asarray(delta_theta_dot_over_c)
