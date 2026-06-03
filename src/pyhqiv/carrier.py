"""
ℝ⁸ octonion carrier with certified so(8) action.

Generator matrices come from :mod:`pyhqiv.so8_generators` (Lean ``Hqiv/Generators.lean``
or packaged ``so8_generators.json``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyhqiv.so8_generators import So8Generators, lie_bracket, load_so8_generators_auto


@dataclass
class So8Carrier:
    """
    Real 8-vector acted on by the 28-dimensional so(8) basis.

    ``generators`` defaults to :func:`~pyhqiv.so8_generators.load_so8_generators_auto`.
    """

    psi: np.ndarray
    generators: So8Generators = field(default_factory=load_so8_generators_auto, repr=False)

    def __post_init__(self) -> None:
        self.psi = np.asarray(self.psi, dtype=np.float64).reshape(8)
        if self.generators.tensor.shape != (28, 8, 8):
            raise ValueError("generators must have tensor shape (28, 8, 8)")

    @classmethod
    def from_unit_axis(cls, axis: int, *, generators: So8Generators | None = None) -> So8Carrier:
        """``e_axis`` with unit norm; ``axis`` in ``0 .. 7``."""
        if axis < 0 or axis > 7:
            raise ValueError("axis must be in 0..7")
        v = np.zeros(8, dtype=np.float64)
        v[axis] = 1.0
        g = generators if generators is not None else load_so8_generators_auto()
        return cls(psi=v, generators=g)

    def apply_generator(self, k: int) -> np.ndarray:
        """Return ``G_k @ psi`` (column action)."""
        if k < 0 or k >= 28:
            raise IndexError(f"generator index must be in 0..27, got {k}")
        return self.generators.matrix(k) @ self.psi

    def normalize(self) -> None:
        n = float(np.linalg.norm(self.psi))
        if n == 0.0:
            raise ValueError("cannot normalize zero vector")
        self.psi /= n

    def normalized_copy(self) -> So8Carrier:
        out = So8Carrier(psi=self.psi.copy(), generators=self.generators)
        out.normalize()
        return out

    def to_density_matrix(self) -> np.ndarray:
        """Rank-one projector ``|ψ⟩⟨ψ|`` in the real 8×8 matrix model."""
        p = self.psi.reshape(8, 1)
        return p @ p.T


def hamiltonian_from_so8_coeffs(coeffs: np.ndarray, generators: So8Generators) -> np.ndarray:
    """
    ``H = Σ_k c_k G_k`` with ``G_k`` the Lean so(8) basis (real antisymmetric 8×8).
    """
    c = np.asarray(coeffs, dtype=np.float64).reshape(28)
    return np.tensordot(c, generators.tensor, axes=(0, 0))


__all__ = ["So8Carrier", "hamiltonian_from_so8_coeffs"]
