"""
Executable SM layer aligned with ``Hqiv.Algebra.SMEmbedding.lean``.

Provides:
- Exact ``hyperchargeEigenvalue : Fin 8 → ℚ`` (as floats) from the Lean table.
- ``chargeFromY``-style electric charge ``Q = T₃ + Y/2`` on a component.
- Doublet / branching bookkeeping (dimensions) matching the Lean docstrings.
- Optional **8-component state-vector evolution** on ℝ^8 ℂ^8 using:

  * SU(2)ₗ generators ``SMEmbedding.su2_L_gen_{1,2,3}`` implemented as
    ``-[L(e₁),L(e₂)]``, ``-[L(e₁),L(e₃)]``, ``-[T₁,T₂]`` from ``HQVM.matrices.OctonionHQIVAlgebra``
    (same construction as ``g2_comm_12``, ``g2_comm_13`` in ``GeneratorsFromAxioms.lean``).
  * U(1) hypercharge: diagonal evolution in the **SM line basis** indexed by ``Fin 8``
    (same index convention as ``hyperchargeEigenvalue``).

If ``HQIV`` / ``HQVM`` is not on ``PYTHONPATH``, SU(2) matrices are unavailable and
the module still exposes quantum numbers for mass hooks and diagnostics.

Lean reference: ``HQIV_LEAN/Hqiv/Algebra/SMEmbedding.lean``,
``Hqiv.Physics.SM_GR_Unification`` (label → hypercharge index).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from scipy.linalg import expm

from pyhqiv.so8_generators import lie_bracket

# Same string union as ``sm_embedding.SMLabel`` (avoid import cycle).
SMLabelSim = Literal[
    "electron",
    "muon",
    "tau",
    "up",
    "down",
    "strange",
    "charm",
    "bottom",
    "top",
    "nu_e",
    "nu_mu",
    "nu_tau",
]

# --- SMEmbedding.lean hyperchargeEigenvalue (Y/2 for Q = T₃ + Y/2) -----------------
# if i = 0 then 1/6 else if i = 1 then 1/6 else if i = 2 then -2/3 else if i = 3 then 1/3
# else if i = 4 then -1/2 else if i = 5 then -1/2 else if i = 6 then 1 else 0

HYPERCHARGE_Y2: Tuple[float, ...] = (
    1.0 / 6.0,
    1.0 / 6.0,
    -2.0 / 3.0,
    1.0 / 3.0,
    -1.0 / 2.0,
    -1.0 / 2.0,
    1.0,
    0.0,
)


def hypercharge_eigenvalue(i: int) -> float:
    """Lean ``Hqiv.Algebra.hyperchargeEigenvalue`` (rational Y/2)."""
    if i < 0 or i > 7:
        raise ValueError("hypercharge index must be in 0..7 (Fin 8)")
    return HYPERCHARGE_Y2[i]


def charge_from_y(i: int, t3: float) -> float:
    """Lean ``chargeFromY``: ``Q = T₃ + Y/2`` with ``Y/2 = hyperchargeEigenvalue i``."""
    return t3 + hypercharge_eigenvalue(i)


def doublet_component_index(k: int, c: int) -> int:
    """Lean ``doubletComponent``: ``2 * k + c`` for ``k : Fin 4``, ``c : Fin 2``."""
    if k < 0 or k > 3 or c < 0 or c > 1:
        raise ValueError("k in 0..3, c in 0..1")
    return 2 * k + c


def branching_sector_dims() -> dict[str, int]:
    """Dimensions of branching summands (6+3+3+2+1+1 = 16)."""
    return {
        "QuarkDoubletL": 6,
        "ConjUR": 3,
        "ConjDR": 3,
        "LeptonDoubletL": 2,
        "ER": 1,
        "NuR": 1,
    }


def sm_hypercharge_y2_for_label(label: SMLabelSim) -> float:
    """
    Lean ``SM_GR_Unification.smHyperchargeWeight`` maps each label to ``hyperchargeEigenvalue j``.

    Indices: 6 (e_R), 2 (u_R), 3 (d_R), 7 (ν_R).
    """
    if label in ("electron", "muon", "tau"):
        return hypercharge_eigenvalue(6)
    if label in ("up", "charm", "top"):
        return hypercharge_eigenvalue(2)
    if label in ("down", "strange", "bottom"):
        return hypercharge_eigenvalue(3)
    if label in ("nu_e", "nu_mu", "nu_tau"):
        return hypercharge_eigenvalue(7)
    raise ValueError(f"unknown SM label: {label}")


def hypercharge_generator_diagonal_sm_basis() -> np.ndarray:
    """
    Diagonal ``Y/2`` operator in the SM line basis (``Fin 8`` indexing from Lean).

    Used for U(1) phase evolution ``exp(-i θ Y)`` in that basis (not the full
    ``phaseLiftDelta`` matrix unless the state is expressed in Y eigenbasis).
    """
    return np.diag(np.array(HYPERCHARGE_Y2, dtype=np.float64))


def _try_load_octonion_algebra():
    """Return ``OctonionHQIVAlgebra`` instance or ``None``."""
    roots = [
        Path(os.environ.get("HQIV_ROOT", "")),
        Path(__file__).resolve().parents[2].parent / "HQIV",
        Path.home() / "Repos" / "HQIV",
    ]
    for r in roots:
        if r and (r / "HQVM" / "matrices.py").is_file():
            import sys

            if str(r) not in sys.path:
                sys.path.insert(0, str(r))
            try:
                from HQVM.matrices import OctonionHQIVAlgebra  # type: ignore[import-untyped]

                return OctonionHQIVAlgebra(verbose=False)
            except Exception:
                continue
    return None


@dataclass(frozen=True)
class Su2Generators:
    """SU(2)ₗ generators on 8s (real antisymmetric 8×8)."""

    t1: np.ndarray
    t2: np.ndarray
    t3: np.ndarray


def su2_l_generators_from_octonion() -> Su2Generators:
    """
    Match ``SMEmbedding.su2_L_gen_1/2/3``:

    ``T₁ = g2Generator 0 = [L(e₁),L(e₂)]``, ``T₂ = g2Generator 1 = [L(e₁),L(e₃)]``,
    ``T₃ = -[T₁,T₂]``.
    """
    alg = _try_load_octonion_algebra()
    if alg is None:
        raise RuntimeError(
            "Could not import HQVM.matrices.OctonionHQIVAlgebra. "
            "Set HQIV_ROOT to a checkout containing HQVM/ or add HQIV to PYTHONPATH."
        )
    # alg.L[k] = L(e_{k+1}) for k=0..6
    l1, l2, l3 = alg.L[0], alg.L[1], alg.L[2]
    t1 = lie_bracket(l1, l2)
    t2 = lie_bracket(l1, l3)
    t3 = -lie_bracket(t1, t2)
    return Su2Generators(t1=t1.astype(np.float64), t2=t2.astype(np.float64), t3=t3.astype(np.float64))


def verify_su2_lie_algebra(su2: Su2Generators, *, tol: float = 1e-9) -> bool:
    """Check ``[T₁,T₂] ≈ -T₃`` (Lean ``su2_bracket_12``)."""
    br = lie_bracket(su2.t1, su2.t2)
    return bool(np.linalg.norm(br + su2.t3) < tol)


def evolve_su2(psi: np.ndarray, su2: Su2Generators, theta: Tuple[float, float, float]) -> np.ndarray:
    """Apply ``U = exp(-i (θ₁ T₁ + θ₂ T₂ + θ₃ T₃))`` to state ``psi`` (length 8)."""
    h = theta[0] * su2.t1 + theta[1] * su2.t2 + theta[2] * su2.t3
    u = expm(-1j * h)
    psi = np.asarray(psi, dtype=np.complex128).reshape(8)
    return u @ psi


def evolve_u1_hypercharge_sm_basis(psi: np.ndarray, theta: float) -> np.ndarray:
    """``exp(-i θ Y_diag)`` with ``Y_diag`` the Lean ``hyperchargeEigenvalue`` list."""
    y = hypercharge_generator_diagonal_sm_basis()
    u = expm(-1j * theta * y)
    psi = np.asarray(psi, dtype=np.complex128).reshape(8)
    return u @ psi


def random_normalized_state_8(seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    z /= np.linalg.norm(z)
    return z.astype(np.complex128)


__all__ = [
    "HYPERCHARGE_Y2",
    "SMLabelSim",
    "Su2Generators",
    "branching_sector_dims",
    "charge_from_y",
    "doublet_component_index",
    "evolve_su2",
    "evolve_u1_hypercharge_sm_basis",
    "hypercharge_eigenvalue",
    "hypercharge_generator_diagonal_sm_basis",
    "random_normalized_state_8",
    "sm_hypercharge_y2_for_label",
    "su2_l_generators_from_octonion",
    "verify_su2_lie_algebra",
]
