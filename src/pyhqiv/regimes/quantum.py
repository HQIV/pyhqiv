"""
Matrix evolution on the Lean so(8) carrier and Born weights from a real amplitude.

Formal sparse simulation bookkeeping is proved in:

- ``Hqiv.QuantumComputing.OSHoracle`` (expand → gate → flip-detect → prune)
- ``Hqiv.QuantumComputing.DigitalGates``

This module exposes a **dense** ``expm`` stepping helper for the ℝ⁸ carrier, suitable
for small experiments and cross-checks against :mod:`pyhqiv.ins_dsf_simulator` / QC demos.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from pyhqiv.carrier import So8Carrier, hamiltonian_from_so8_coeffs
from pyhqiv.lepton_resonance_coherence import LeptonShellCoherence, lepton_shell_coherence
from pyhqiv.so8_generators import So8Generators


def evolve_so8_vector_expm(
    psi: np.ndarray,
    coeffs: np.ndarray,
    dt: float,
    generators: So8Generators,
) -> np.ndarray:
    """
    One step ``ψ ← exp(dt · Σ_k c_k G_k) ψ`` with antisymmetric generators ``G_k``.

    For real ``ψ`` and real ``c_k``, the exponential is orthogonal (ℓ² norm preserved).
    """
    h = hamiltonian_from_so8_coeffs(coeffs, generators)
    return expm(h * float(dt)) @ np.asarray(psi, dtype=np.float64).reshape(8)


def evolve_so8_carrier_expm(
    carrier: So8Carrier,
    coeffs: np.ndarray,
    dt: float,
) -> So8Carrier:
    """Like :func:`evolve_so8_vector_expm` but returns a new :class:`~pyhqiv.carrier.So8Carrier`."""
    psi_new = evolve_so8_vector_expm(carrier.psi, coeffs, dt, carrier.generators)
    return So8Carrier(psi=psi_new, generators=carrier.generators)


def born_probs_from_real_state(psi: np.ndarray) -> np.ndarray:
    """
    Born-style probabilities from a **real** amplitude on ``ℝ⁸``:

    ``p_i = ψ_i² / Σ_j ψ_j²``.
    """
    v = np.asarray(psi, dtype=np.float64).reshape(8)
    p = v * v
    s = float(p.sum())
    if s <= 0.0:
        raise ValueError("psi must have positive ℓ² norm for Born weights")
    return p / s


def quantum_lepton_coherence_snapshot(
    *,
    t_cmb_natural: float | None = None,
) -> LeptonShellCoherence:
    """
    Observer / CMB shell coherence bundle (auxiliary φ and phase lifts on lepton shells).

    Delegates to :func:`pyhqiv.lepton_resonance_coherence.lepton_shell_coherence`.
    """
    if t_cmb_natural is None:
        return lepton_shell_coherence()
    return lepton_shell_coherence(t_cmb_natural=t_cmb_natural)


__all__ = [
    "born_probs_from_real_state",
    "evolve_so8_carrier_expm",
    "evolve_so8_vector_expm",
    "quantum_lepton_coherence_snapshot",
]
