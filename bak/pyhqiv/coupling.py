"""
HQIV effective coupling from the single axiom (Lean-aligned).

1/α_eff(φ) = (1/α_GUT) × (1 + c·α·log(φ+1)), with α_GUT = 1/42 and α = 3/5
(SM_GR_Unification.lean, Schrodinger.lean). No free parameters; use these in the
engine instead of any external fine-structure constant. Reserve CODATA 1/α_EM
for the interaction layer (comparison only).
"""

from __future__ import annotations

from typing import Union

import numpy as np

from pyhqiv.constants import ALPHA

# Derived from lattice: cubeDirections × octonionImaginaryDim = 6×7 (OctonionicLightCone, SM_GR_Unification)
ALPHA_GUT: float = 1.0 / 42.0

# "Now" hypersurface: shell index at discrete-to-continuous transition (lattice.M_TRANS).
# In natural units φ(m) = 2(m+1); at "now" φ_now = 2(M_TRANS+1). Lean nowCondition uses φ=1
# in their normalization; pyhqiv "now" is set by T_CMB and m_trans (M_TRANS).
try:
    from pyhqiv.constants import M_TRANS
except ImportError:
    M_TRANS = 500  # fallback if constants not available


def phi_of_shell_natural(m: Union[int, float]) -> float:
    """
    Auxiliary field φ at shell m in natural units (Lean: φ(m) = 2(m+1)).

    T(m) = T_Pl/(m+1) with T_Pl=1 ⇒ φ(m) = 2/T(m) = 2(m+1).
    """
    return 2.0 * (float(m) + 1.0)


def one_over_alpha_eff(
    phi: Union[float, np.ndarray],
    c: float = 1.0,
    alpha: float = ALPHA,
) -> Union[float, np.ndarray]:
    """
    Effective inverse fine-structure 1/α_eff(φ) = (1/α_GUT) × (1 + c·α·log(φ+1)).

    φ is in natural units (dimensionless scale from lattice). For φ+1 ≤ 0 returns
    (1/α_GUT) so the result remains positive and finite.
    """
    is_scalar = np.isscalar(phi) or (isinstance(phi, np.ndarray) and phi.ndim == 0)
    phi_arr = np.asarray(phi, dtype=float)
    log_arg = np.maximum(phi_arr + 1.0, 1e-300)
    ln_term = 1.0 + c * alpha * np.log(log_arg)
    out = (1.0 / ALPHA_GUT) * ln_term
    return float(out) if is_scalar else out


def alpha_eff_shell(
    m: Union[int, float],
    c: float = 1.0,
    alpha: float = ALPHA,
) -> float:
    """
    Effective fine-structure α_eff at shell m: 1 / one_over_alpha_eff(φ(m)).

    φ(m) = 2(m+1) in natural units. Use this in the engine for Coulomb/binding
    coupling at shell m instead of a fixed 1/α_EM.
    """
    phi = phi_of_shell_natural(m)
    inv = one_over_alpha_eff(phi, c=c, alpha=alpha)
    return 1.0 / max(float(inv), 1e-30)


def coulomb_strength_shell(
    m: Union[int, float],
    c: float = 1.0,
) -> float:
    """
    Coulomb strength at shell m (natural units). Same as α_eff(m) in Lean
    (coulombStrengthShell m = alphaEffShell m).
    """
    return alpha_eff_shell(m, c=c)


def alpha_eff_at_now(c: float = 1.0) -> float:
    """
    Effective fine-structure at the "now" hypersurface (shell M_TRANS).
    Use for observer-centric coupling when no explicit shell is available.
    """
    return alpha_eff_shell(M_TRANS, c=c)
