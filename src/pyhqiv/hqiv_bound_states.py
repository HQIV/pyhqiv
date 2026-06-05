"""
8×8 network binding hierarchy (structural sums over so(8) generators).

Lean source:
  `HQIV_LEAN/Hqiv/Physics/BoundStates.lean`

Implements carrier dimensions, ``NetworkWeight``, ``bindingCouplingAtShell``,
``E_bind_from_network``, QCD/nuclear aliases, atomic shell magnitudes, and
composite mass formulas **exactly as definitional combinations** in Lean:
no SEMF, no PDG binding tables — only witness masses where Lean uses
``m_proton_MeV_central`` / SM layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

from pyhqiv.hqiv_schrodinger_shell import alpha_eff_shell

# Lean: ``Hqiv.Physics.so8Dim`` / ``Fin 28``
SO8_DIM: int = 28

NetworkWeightLike = Union[Sequence[float], "NetworkWeight28"]


@dataclass(frozen=True)
class NetworkWeight28:
    """
    Lean: ``NetworkWeight := So8Index → ℝ`` with ``So8Index = Fin 28``.

    One coefficient per so(8) generator; in a full 8×8 build these come from
    traces / matrix elements (Lean comment in BoundStates).
    """

    w: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.w) != SO8_DIM:
            raise ValueError(f"NetworkWeight28 requires {SO8_DIM} weights, got {len(self.w)}")

    def __getitem__(self, k: int) -> float:
        if k < 0 or k >= SO8_DIM:
            raise IndexError(f"generator index k must be in 0..{SO8_DIM - 1}")
        return self.w[k]

    @staticmethod
    def uniform(value_per_generator: float = 1.0) -> NetworkWeight28:
        """All generators carry the same weight (neutral placeholder)."""
        v = float(value_per_generator)
        return NetworkWeight28(tuple(v for _ in range(SO8_DIM)))

    @staticmethod
    def from_sequence(seq: Sequence[float]) -> NetworkWeight28:
        return NetworkWeight28(tuple(float(x) for x in seq))


def binding_coupling_at_shell(m: int, k: int, c: float = 1.0) -> float:
    """
    Lean: ``bindingCouplingAtShell m k c = alphaEffAtShell m c`` (independent of ``k`` here).

    In a richer port, k-dependent couplings would split EM vs strong; the abstract
    Lean layer uses the same φ(m)-driven scale for every generator index.
    """
    if k < 0 or k >= SO8_DIM:
        raise IndexError(f"k must be in 0..{SO8_DIM - 1}")
    return alpha_eff_shell(m, c)


def e_bind_from_network(m: int, weights: NetworkWeightLike, c: float = 1.0) -> float:
    """
    Lean: ``E_bind_from_network m w c = ∑_{k : So8Index}, w k * bindingCouplingAtShell m k c``.
    """
    if isinstance(weights, NetworkWeight28):
        w = weights
    else:
        w = NetworkWeight28.from_sequence(weights)
    return sum(w[k] * binding_coupling_at_shell(m, k, c) for k in range(SO8_DIM))


def e_bind_qcd_from_network(m: int, weights: NetworkWeightLike, c: float = 1.0) -> float:
    """Lean: ``E_bind_QCD_from_network`` — same sum as nuclear at this abstract layer."""
    return e_bind_from_network(m, weights, c)


def e_bind_nuclear_from_network(m: int, weights: NetworkWeightLike, c: float = 1.0) -> float:
    """Lean: ``E_bind_nuclear_from_network``."""
    return e_bind_from_network(m, weights, c)


def expected_ground_energy_at_shell(m: int, Z: int, mu: float, c: float = 1.0) -> float:
    """Lean: ``expectedGroundEnergyAtShell m Z μ c = - μ Z² (α_eff)² / 2``."""
    if Z < 0:
        raise ValueError("Z must be non-negative")
    a = alpha_eff_shell(m, c)
    z = float(Z)
    muf = float(mu)
    return -muf * z * z * a * a / 2.0


def e_bind_atomic_shell_magnitude(m: int, Z: int, mu: float, c: float = 1.0) -> float:
    """Lean: ``E_bind_atomic_shell_magnitude m Z μ c = μ Z² (α_eff)² / 2``."""
    if Z < 0:
        raise ValueError("Z must be non-negative")
    a = alpha_eff_shell(m, c)
    z = float(Z)
    muf = float(mu)
    return muf * z * z * a * a / 2.0


def e_bind_atomic(m: int, Z: int, mu: float, c: float = 1.0) -> float:
    """Lean: ``E_bind_atomic`` re-export (magnitude form)."""
    return e_bind_atomic_shell_magnitude(m, Z, mu, c)


def m_nucleon_from_network(
    m: int,
    m_constituent: float,
    weights: NetworkWeightLike,
    c: float = 1.0,
) -> float:
    """Lean: ``M_nucleon_from_network m M_constituent w c = M_constituent - E_bind_QCD``."""
    return float(m_constituent) - e_bind_qcd_from_network(m, weights, c)


def m_nucleus_from_network(
    m: int,
    A: int,
    Z: int,
    m_nucleon_avg: float,
    weights: NetworkWeightLike,
    c: float = 1.0,
) -> float:
    """
    Lean: ``M_nucleus_from_network m A Z M_nucleon_avg w c = A * M_nucleon_avg - E_bind_nuclear``.

    Note: ``Z`` appears in Lean's type signature for book-keeping; the formula uses ``A`` and
    the nuclear binding sum only.
    """
    if A < 0 or Z < 0:
        raise ValueError("A and Z must be non-negative")
    return float(A) * float(m_nucleon_avg) - e_bind_nuclear_from_network(m, weights, c)


def m_atom_from_network(
    m: int,
    Z: int,
    mu: float,
    m_nucleus: float,
    m_e: float,
    c: float = 1.0,
) -> float:
    """Lean: ``M_atom_from_network m Z μ M_nucleus m_e c = M_nucleus + Z*m_e - E_bind_atomic``."""
    if Z < 0:
        raise ValueError("Z must be non-negative")
    z = float(Z)
    return float(m_nucleus) + z * float(m_e) - e_bind_atomic(m, Z, mu, c)


@dataclass(frozen=True)
class NuclearConfigurationLean:
    """
    Lean: ``Hqiv.Physics.Configuration`` in ``NuclearAndAtomicSpectra.lean``
    (proton displacement, neutron spin misalignment, shell promotion).
    """

    proton_displacement: float = 0.0
    neutron_spin_misalignment: float = 0.0
    shell_promotion: int = 0


@dataclass(frozen=True)
class NuclidePN:
    """Minimal isotope identifier: proton number ``P`` (Z) and neutron number ``N``."""

    P: int
    N: int

    @property
    def A(self) -> int:
        return self.P + self.N


def network_weight_for_nuclide(
    m: int,
    nuclide: NuclidePN,
    *,
    modes_scale: float | None = None,
) -> NetworkWeight28:
    """
    Construct a **first-principles placeholder** weight vector from light-cone mode count.

    Lean does not fix ``w`` numerically in BoundStates; it must come from the 8×8 state.
    Here each of the 28 generators receives an equal share of ``available_modes(m) / 28``
    so that ``∑_k w_k = available_modes(m)``, tying the network to
    ``Hqiv.Geometry.OctonionicLightCone.available_modes`` (single-source combinatorics).

    This is **not** a uniqueness claim — it is a deterministic, axiom-linked default until
    matrix elements replace it.
    """
    from pyhqiv.lightcone import available_modes

    mm = float(available_modes(m)) if modes_scale is None else float(modes_scale)
    per = mm / float(SO8_DIM)
    return NetworkWeight28.uniform(per)


__all__ = [
    "SO8_DIM",
    "NetworkWeight28",
    "NuclearConfigurationLean",
    "NuclidePN",
    "binding_coupling_at_shell",
    "e_bind_atomic",
    "e_bind_atomic_shell_magnitude",
    "e_bind_from_network",
    "e_bind_nuclear_from_network",
    "e_bind_qcd_from_network",
    "expected_ground_energy_at_shell",
    "m_atom_from_network",
    "m_nucleon_from_network",
    "m_nucleus_from_network",
    "network_weight_for_nuclide",
]
