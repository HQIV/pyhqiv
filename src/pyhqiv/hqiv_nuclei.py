"""
HQIV nuclei: Casimir surfaces, Fresnel caustics, and the **constructive isotope ladder**.

Lean source (single module):
  ``HQIV_LEAN/Hqiv/Physics/HQIVNuclei.lean``

This file is the computable Python mirror: same identifiers and formulas where they
evaluate to floats (no Mathlib proofs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pyhqiv.auxiliary_field import phi_of_shell
from pyhqiv.hqiv_nuclear_spectra import R_m, half_life_from_width, modes
from pyhqiv.lightcone import available_modes
from pyhqiv.metric import gamma_hqiv
from pyhqiv.sm_gr_unification import alpha_EM_at_MZ
from pyhqiv.spin_statistics import hbar_MeV_s, resonance_half_life
from pyhqiv.lean_witnesses import load_lean_witnesses


# --- SphericalHarmonicsBridge (cumulative S² degeneracy) ---------------------------------


def spherical_harmonic_cumulative_count(m: int) -> float:
    """
    ``(m+1)²`` — cumulative spherical-harmonic degeneracy with cutoff ``L = m``.

    Lean: ``Hqiv.sphericalHarmonicCumulativeCount``
    """
    if m < 0:
        raise ValueError("m must be non-negative")
    return float(m + 1) ** 2


# --- Casimir surface energy (§2 HQIVNuclei) ----------------------------------------------


def available_modes_nat(m: int) -> int:
    """``4 * latticeSimplexCount m`` as ℕ; cast equals ``available_modes m`` in ℝ."""
    from pyhqiv.lightcone import lattice_simplex_count

    return 4 * lattice_simplex_count(m)


def omega_casimir(m: int) -> float:
    """``φ(m)`` — HQIV frequency unit for Casimir modes. Lean: ``omegaCasimir``."""
    return float(phi_of_shell(m))


def casimir_energy_surface(m: int) -> float:
    """
    Full lattice zero-point sum: ``available_modes · φ/2`` (natural units, ℏ = 1).

    Lean: ``CasimirEnergySurface`` / theorem ``casimir_energy_full_mode_sum``.
    """
    return float(available_modes(m)) * omega_casimir(m) / 2.0


# --- Fresnel / vacuum density (§3) -------------------------------------------------------


def vacuum_mode_density(m: int) -> float:
    """``available_modes / R_m``. Lean: ``vacuumModeDensity``."""
    return float(available_modes(m)) / R_m(m)


@dataclass(frozen=True)
class CausticSurface:
    """Lean ``CausticSurface``: radius + curvature proxy."""

    radius: float
    curvature: float


def fresnel_caustic(m: int) -> CausticSurface:
    """Lean ``fresnelCaustic``: ``radius = R_m``, ``curvature = vacuumModeDensity``."""
    rm = R_m(m)
    return CausticSurface(radius=rm, curvature=vacuum_mode_density(m))


def spherical_fresnel_envelope(m: int) -> CausticSurface:
    """
    Lean ``sphericalFresnelEnvelope``: ``radius = R_m``, ``curvature = (m+1)² / R_m``.

    Uses ``spherical_harmonic_cumulative_count`` for ``H.cumulativeCount``.
    """
    rm = R_m(m)
    h = spherical_harmonic_cumulative_count(m)
    return CausticSurface(radius=rm, curvature=h / rm)


# --- Valley potentials (§4) --------------------------------------------------------------


def caustic_overlap(c1: CausticSurface, c2: CausticSurface) -> float:
    """Lean ``causticOverlap``: product of radii."""
    return c1.radius * c2.radius


def valley_potential_pair(m: int) -> float:
    """
    Lean ``valleyPotential``: ``- R_m²`` for identical shells (theorem ``valleyPotential_neg_overlap``).
    """
    rm = R_m(m)
    return -(rm * rm)


def valley_potential_em(m: int, z_eff: float, r: float) -> float:
    """
    Lean ``valleyPotentialEM``: ``valleyPotential + α_EM Z_eff / r``.
    """
    if r == 0.0:
        raise ValueError("r must be non-zero")
    return valley_potential_pair(m) + alpha_EM_at_MZ() * float(z_eff) / float(r)


# --- Toroidal step (§5) -----------------------------------------------------------------


def new_modes_succ_identity(m: int) -> tuple[float, float]:
    """
    Lean ``toroidal_ring_closure``: ``new_modes (m+1) = 8 * (m+2)``.

    Returns ``(lhs, rhs)`` for numeric check.
    """
    from pyhqiv.lightcone import new_modes

    lhs = new_modes(m + 1)
    rhs = 8.0 * float(m + 2)
    return lhs, rhs


# --- Constructive isotope ladder (§6) ----------------------------------------------------

LadderSeed = Literal["proton", "neutron"]
BindStep = Literal["bind_proton", "bind_neutron"]


class IsospinLabel(str, Enum):
    """Lean ``IsospinLabel``."""

    PROTON = "proton"
    NEUTRON = "neutron"


def valley_count_from_A(A: int) -> int:
    """
    Number of toroidal valleys along **any** construction path of mass number ``A``.

    Lean: ``valleyCount`` satisfies ``valleyCount n = 2 * (A - 1)`` for ``A ≥ 1``
    (each ``bindProton`` / ``bindNeutron`` adds ``+2``; base proton/neutron is ``0``).
    """
    if A < 1:
        raise ValueError("mass number A must be ≥ 1")
    return 2 * (A - 1)


def _apply_step(A: int, Z: int, step: BindStep) -> tuple[int, int]:
    if step == "bind_neutron":
        return A + 1, Z
    return A + 1, Z + 1


def simulate_ladder_end(seed: LadderSeed, steps: list[BindStep]) -> tuple[int, int]:
    """Apply ``steps`` from ``proton`` (1,1) or ``neutron`` (1,0); return final ``(A, Z)``."""
    if seed == "proton":
        A, Z0 = 1, 1
    else:
        A, Z0 = 1, 0
    for s in steps:
        A, Z0 = _apply_step(A, Z0, s)
    return A, Z0


def ladder_path_from_ZN(Z: int, N: int) -> tuple[LadderSeed, list[BindStep]]:
    """
    Construct **one** shortest ladder path to nuclide ``(Z, N)``.

    Mirrors Lean ``IsotopeLadder``: ``proton : (1,1)``, ``neutron : (1,0)``,
    ``bindProton`` / ``bindNeutron`` edges.

    **Construction (same length as any shortest path, ``A-1`` binds):**

    - ``Z ≥ 1``: start from ``proton``, then ``N`` ``bind_neutron`` and ``Z-1`` ``bind_proton``.
    - ``Z = 0``, ``N ≥ 2``: start from ``neutron``, then ``N-1`` ``bind_neutron``.

    Other orderings (e.g. Lean's ``helium4`` = n–p–n) are also valid; this canonical order
    is O(A) and suitable for large-isotope scans.
    """
    if Z < 0 or N < 0:
        raise ValueError("Z and N must be non-negative")
    A = Z + N
    if A < 1:
        raise ValueError("empty nucleus")
    if Z == 0 and N == 1:
        return ("neutron", [])
    if Z >= 1:
        steps = ["bind_neutron"] * N + ["bind_proton"] * (Z - 1)
        end = simulate_ladder_end("proton", steps)
        if end != (A, Z):
            raise RuntimeError(f"ladder simulation bug: got {end}, want ({A},{Z})")
        return ("proton", steps)
    # Z == 0, N >= 2
    steps = ["bind_neutron"] * (N - 1)
    end = simulate_ladder_end("neutron", steps)
    if end != (A, Z):
        raise RuntimeError(f"ladder simulation bug: got {end}, want ({A},{Z})")
    return ("neutron", steps)


def ladder_path_helium4_lean() -> tuple[LadderSeed, list[BindStep]]:
    """
    Lean-named chain: ``helium4 = bindNeutron helium3``, ``helium3 = bindProton deuteron``,
    ``deuteron = bindNeutron proton``.
    """
    return (
        "proton",
        ["bind_neutron", "bind_proton", "bind_neutron"],
    )


def ladder_path_deuteron_lean() -> tuple[LadderSeed, list[BindStep]]:
    """Lean ``deuteron = bindNeutron proton``."""
    return ("proton", ["bind_neutron"])


# --- Deuteron binding scale + spectra anchor (§6) -----------------------------------------


def deuteron_binding_scale(m: int) -> float:
    """
    ``γ · modes / R_m`` at shell ``m`` (same factor as ``V_nuclear`` horizon term).

    Lean: ``deuteronBindingScale``; theorem ``deuteron_binding_scale_eq``.
    """
    return gamma_hqiv() * modes(m) / R_m(m)


SPECTRA_DEUTERON_BINDING_MEV: float = load_lean_witnesses().get_float("spectraDeuteronBinding_MeV")
"""Lean ``spectraDeuteronBinding_MeV`` — spectroscopic anchor (MeV). Loaded from witnesses (value lives in json)."""


# --- Decay width / half-life (§7) --------------------------------------------------------


def decay_width_per_s(delta_E_mev: float) -> float:
    """Lean ``decayWidth_per_s``: ``ΔE / ħ`` with ``ħ`` in MeV·s."""
    h = hbar_MeV_s()
    if h == 0.0:
        raise ZeroDivisionError("hbar_MeV_s is zero")
    return float(delta_E_mev) / h


def half_life_from_width_seconds(width_per_s: float) -> float:
    """Lean ``half_life_from_width`` re-export (seconds)."""
    return half_life_from_width(width_per_s)


def spin_statistics_half_life_matches_resonance(delta_E_mev: float) -> bool:
    """
    Numeric check of Lean ``spin_statistics_determines_half_life``:
    ``half_life_from_width(ΔE/ħ) ≈ resonance_half_life(ΔE)``.
    """
    if delta_E_mev <= 0.0:
        return False
    w = decay_width_per_s(delta_E_mev)
    t1 = half_life_from_width(w)
    t2 = resonance_half_life(delta_E_mev)
    return math.isclose(t1, t2, rel_tol=1e-9, abs_tol=0.0)


@dataclass(frozen=True)
class Nucleus:
    """Lean ``Nucleus``: mass number, proton number, and a ladder path."""

    A: int
    Z: int
    seed: LadderSeed
    steps: tuple[BindStep, ...]


def make_nucleus(Z: int, N: int) -> Nucleus:
    """Pack ``ladder_path_from_ZN`` into ``Nucleus``."""
    seed, steps = ladder_path_from_ZN(Z, N)
    return Nucleus(A=Z + N, Z=Z, seed=seed, steps=tuple(steps))


def odd_nucleon_count_predicate(A: int, Z: int) -> bool:
    """Lean ``oddNucleonCount``: ``A`` odd and ``Z`` odd."""
    return (A % 2 == 1) and (Z % 2 == 1)


def odd_odd_width(A: int, Z: int) -> float:
    """Lean ``oddOddWidth``: ``1`` if odd-odd, else ``0``."""
    return 1.0 if odd_nucleon_count_predicate(A, Z) else 0.0


def isotope_ladder_stability_valley_bound(A: int) -> bool:
    """
    Lean ``isotope_ladder_stability_le_sixteen``: ``valleyCount ≤ 30`` when ``A ≤ 16``.

    Here ``valleyCount = 2*(A-1)``.
    """
    if A > 16:
        return True
    return valley_count_from_A(A) <= 30


__all__ = [
    "BindStep",
    "CausticSurface",
    "IsospinLabel",
    "LadderSeed",
    "Nucleus",
    "SPECTRA_DEUTERON_BINDING_MEV",
    "available_modes_nat",
    "caustic_overlap",
    "casimir_energy_surface",
    "decay_width_per_s",
    "deuteron_binding_scale",
    "fresnel_caustic",
    "half_life_from_width_seconds",
    "isotope_ladder_stability_valley_bound",
    "ladder_path_deuteron_lean",
    "ladder_path_from_ZN",
    "ladder_path_helium4_lean",
    "make_nucleus",
    "new_modes_succ_identity",
    "odd_nucleon_count_predicate",
    "odd_odd_width",
    "omega_casimir",
    "spherical_fresnel_envelope",
    "spherical_harmonic_cumulative_count",
    "simulate_ladder_end",
    "spin_statistics_half_life_matches_resonance",
    "valley_count_from_A",
    "valley_potential_em",
    "valley_potential_pair",
    "vacuum_mode_density",
]
