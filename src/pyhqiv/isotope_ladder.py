"""
Isotope ladder: (Z, N), angular momentum, network binding, decay topology, scales.

This packages the **structural** pieces already in pyhqiv:

- ``BoundStates`` / ``NuclearAndAtomicSpectra`` — ``m_nucleus_from_network``, ``e_bind_nuclear``,
  ``R_m``, ``modes``, ``V_nuclear``-style horizon scale.
- ``spherical_harmonics`` — ``L²`` eigenvalues ``ℓ(ℓ+1)`` for single-particle angular factors.
- ``spin_statistics`` — ``τ = ħ/ΔE`` / half-life from level width when only a width is known.
- ``hqiv_nuclear_spectra`` — beta width scaffold ``G_F² m_e⁵ ℳ²``, ``half_life_from_width``.

**Nucleon masses** come from Lean witnesses (``m_proton_MeV`` / ``derivedProtonMass_MeV``, etc.)
when present, or you pass ``m_proton_mev`` / ``m_neutron_mev`` explicitly.

Rotational excitation uses a **rigid-index proxy** (no SEMF): energy
``∝ J(J+1) / (A · R_m²)`` with an explicit ``scale_mev`` — not a fitted nuclear level scheme.

Geometric cross sections use ``π r²`` with ``r = r₀ A^{1/3}`` (``r₀`` supplied; default 1.2 fm
is documented as a geometric placeholder, not a derived HQIV constant).

Lean-facing: ``Hqiv.Physics.BoundStates``, ``Hqiv.Physics.NuclearAndAtomicSpectra``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from pyhqiv.hqiv_bound_states import (
    NetworkWeight28,
    NuclidePN,
    e_bind_nuclear_from_network,
    m_nucleus_from_network,
    network_weight_for_nuclide,
)
from pyhqiv.hqiv_nuclear_spectra import R_m, half_life_from_width
from pyhqiv.lean_witnesses import LeanWitnessError, load_lean_witnesses
from pyhqiv.lightcone import reference_m
from pyhqiv.scale_witness import load_local_conditions as _load_local
from pyhqiv.spherical_harmonics import laplace_beltrami_eigenvalue_S2
from pyhqiv.spin_statistics import resonance_half_life

# 1 barn = 100 fm² (since 1 barn = 10⁻²⁴ cm², 1 fm² = 10⁻²⁶ cm²).
FM2_PER_BARN: float = 100.0


class DecayMode(str, Enum):
    """Top-level decay topology (book-keeping only — selection rules not enforced)."""

    ALPHA = "alpha"  # ⁴He cluster
    BETA_MINUS = "beta_minus"
    BETA_PLUS = "beta_plus"  # includes electron capture as same (Z,N) endpoint
    NEUTRON_EMISSION = "neutron"
    PROTON_EMISSION = "proton"
    GAMMA = "gamma"  # same (Z,N), different J/parity


@dataclass(frozen=True)
class IsotopeState:
    """Nuclide plus total angular momentum (and optional parity)."""

    Z: int
    N: int
    J: float = 0.0
    parity: int | None = None  # +1 or -1 if set

    @property
    def A(self) -> int:
        return self.Z + self.N

    def nuclide_pn(self) -> NuclidePN:
        return NuclidePN(P=self.Z, N=self.N)


@dataclass
class IsotopeLadderConfig:
    """
    Ladder parameters: shell index ``m`` for network weights/couplings, O-Maxwell ``c``,
    and optional mass overrides.
    """

    shell_m: int = field(default_factory=reference_m)
    c: float = 1.0
    m_proton_mev: float | None = None
    m_neutron_mev: float | None = None
    #: Scale for :func:`rotational_excitation_mev` (MeV); structural, not from PDG levels.
    rotational_scale_mev: float = 1.0
    #: Geometric radius parameter r₀ (fm) for :func:`nuclear_radius_fm`.
    r0_fm: float = 1.2


@dataclass(frozen=True)
class DecayStep:
    """One allowed decay vertex with Q-value from network masses."""

    parent: IsotopeState
    daughter: IsotopeState
    mode: DecayMode
    q_value_mev: float


def _resolve_nucleon_masses_mev(
    m_proton_mev: float | None,
    m_neutron_mev: float | None,
) -> tuple[float, float]:
    if m_proton_mev is not None and m_neutron_mev is not None:
        return float(m_proton_mev), float(m_neutron_mev)
    w = load_lean_witnesses().data
    mp_keys = ("m_proton_MeV", "derivedProtonMass_MeV")
    mn_keys = ("m_neutron_MeV", "derivedNeutronMass_MeV")
    mp = next((float(w[k]) for k in mp_keys if k in w), None)
    mn = next((float(w[k]) for k in mn_keys if k in w), None)
    if mp is None or mn is None:
        raise LeanWitnessError(
            "Nucleon masses not in witnesses.json and not passed explicitly. "
            "Provide m_proton_mev and m_neutron_mev on IsotopeLadderConfig or add "
            "m_proton_MeV / m_neutron_MeV (or derived*) keys to the Lean export."
        )
    return mp, mn


def network_weight(config: IsotopeLadderConfig, state: IsotopeState) -> NetworkWeight28:
    """``NetworkWeight28`` from light-cone mode count at ``config.shell_m``."""
    return network_weight_for_nuclide(config.shell_m, state.nuclide_pn())


def nucleus_mass_mev(
    state: IsotopeState,
    config: IsotopeLadderConfig,
    *,
    weights: NetworkWeight28 | None = None,
) -> float:
    """
    Ground-state nuclear mass from ``M_nucleus_from_network`` (Lean).

    Does **not** add rotational energy; use :func:`state_mass_mev_with_rotation` for J ≠ 0.
    """
    mp, mn = _resolve_nucleon_masses_mev(config.m_proton_mev, config.m_neutron_mev)
    A = state.A
    if A <= 0:
        raise ValueError("mass number A must be positive")
    z, n = state.Z, state.N
    m_avg = (z * mp + n * mn) / float(A)
    w = weights if weights is not None else network_weight(config, state)
    return m_nucleus_from_network(config.shell_m, A, z, m_avg, w, config.c)


def nuclear_binding_energy_mev(
    state: IsotopeState,
    config: IsotopeLadderConfig,
    *,
    weights: NetworkWeight28 | None = None,
) -> float:
    """``B = Z m_p + N m_n - M_nucleus`` (MeV)."""
    mp, mn = _resolve_nucleon_masses_mev(config.m_proton_mev, config.m_neutron_mev)
    z, n = state.Z, state.N
    m_nuc = nucleus_mass_mev(state, config, weights=weights)
    return float(z * mp + n * mn - m_nuc)


def e_bind_network_only(
    state: IsotopeState,
    config: IsotopeLadderConfig,
    *,
    weights: NetworkWeight28 | None = None,
) -> float:
    """Direct Lean sum ``E_bind_nuclear_from_network`` (same units as coupling ladder)."""
    w = weights if weights is not None else network_weight(config, state)
    return e_bind_nuclear_from_network(config.shell_m, w, config.c)


def rotational_excitation_mev(
    state: IsotopeState,
    config: IsotopeLadderConfig,
) -> float:
    """
    Rigid-index proxy for rotational / collective enhancement on top of the ground mass:

        E_rot ∝ J(J+1) / (2 A R_m²) · rotational_scale_mev.

    ``R_m`` is the nuclear-spectrum horizon radius index :func:`hqiv_nuclear_spectra.R_m`.
    This is a **scaffold** — not a fitted rotational band.
    """
    j = float(state.J)
    if j == 0.0:
        return 0.0
    a = max(state.A, 1)
    rm = R_m(config.shell_m)
    return config.rotational_scale_mev * j * (j + 1.0) / (2.0 * float(a) * rm * rm)


def laplace_angular_factor(ell: int) -> float:
    """
    ``ℓ(ℓ+1)`` factor (same as ``-Δ_S²`` eigenvalue on spin-``ℓ`` scalars).

    Use ``ell = int(round(J))`` only when you intentionally identify J with a partial wave index.
    """
    return laplace_beltrami_eigenvalue_S2(ell)


def state_mass_mev_with_rotation(
    state: IsotopeState,
    config: IsotopeLadderConfig,
    *,
    weights: NetworkWeight28 | None = None,
) -> float:
    """``M_nucleus + E_rot(J)``."""
    return nucleus_mass_mev(state, config, weights=weights) + rotational_excitation_mev(
        state, config
    )


def nuclear_radius_fm(A: int, r0_fm: float) -> float:
    """``r = r₀ A^{1/3}`` (fm). ``r0_fm`` is user-supplied geometric input."""
    if A <= 0:
        raise ValueError("A must be positive")
    return float(r0_fm) * float(A) ** (1.0 / 3.0)


def cross_section_geometric_barns(r_fm: float) -> float:
    """``σ = π r²`` in barns (1 barn = 100 fm²)."""
    r = float(r_fm)
    if r <= 0.0:
        raise ValueError("radius must be positive")
    area_fm2 = math.pi * r * r
    return area_fm2 / FM2_PER_BARN


def cross_section_geometric_isotope_barns(state: IsotopeState, r0_fm: float) -> float:
    """Geometric ``π r²`` with ``r = r₀ A^{1/3}``."""
    return cross_section_geometric_barns(nuclear_radius_fm(state.A, r0_fm))


def _alpha_like_mass_mev(config: IsotopeLadderConfig) -> float:
    st = IsotopeState(Z=2, N=2, J=0.0)
    return nucleus_mass_mev(st, config)


def enumerate_decay_steps(
    parent: IsotopeState,
    config: IsotopeLadderConfig,
    *,
    weights_parent: NetworkWeight28 | None = None,
) -> list[DecayStep]:
    """
    Enumerate simple one-body / cluster decays and compute Q from network masses.

    Masses use the **same** ``config.shell_m`` and regenerated daughter weights.
    Alpha uses the **network** mass of (Z,N)=(2,2) as the emitted cluster mass.

    Q-values use **ground-state** nuclear masses (``J = 0``) for parent/daughters so they
    match usual mass-table bookkeeping; rotational energy is a separate layer.
    """
    mp, mn = _resolve_nucleon_masses_mev(config.m_proton_mev, config.m_neutron_mev)
    z, n = parent.Z, parent.N
    g_parent = IsotopeState(Z=z, N=n, J=0.0, parity=parent.parity)
    m_parent = nucleus_mass_mev(g_parent, config, weights=weights_parent)
    steps: list[DecayStep] = []

    def daughter_mass_ground(st: IsotopeState) -> float:
        g = IsotopeState(Z=st.Z, N=st.N, J=0.0, parity=st.parity)
        return nucleus_mass_mev(g, config, weights=None)

    w = load_lean_witnesses().data
    me = float(w["m_electron_MeV"]) if "m_electron_MeV" in w else float(_load_local()["local_proton_mass_MeV_for_comparison"] * 0.0005446)  # approx m_e / m_p ratio; value from data

    # Beta minus: (Z,N) → (Z+1, N-1) + e⁻ + ν̄
    if n >= 1:
        d = IsotopeState(Z=z + 1, N=n - 1, J=parent.J, parity=parent.parity)
        if d.Z > 0 and d.A > 0:
            m_d = daughter_mass_ground(d)
            q_bm = m_parent - m_d - me
            steps.append(DecayStep(parent, d, DecayMode.BETA_MINUS, q_value_mev=q_bm))

    # Beta plus / EC endpoint mass balance uses two electron masses for β⁺
    if z >= 1 and n >= 0:
        d = IsotopeState(Z=z - 1, N=n + 1, J=parent.J, parity=parent.parity)
        if d.A > 0:
            m_d = daughter_mass_ground(d)
            q_bp = m_parent - m_d - 2.0 * me
            steps.append(DecayStep(parent, d, DecayMode.BETA_PLUS, q_value_mev=q_bp))

    # Neutron / proton emission
    if n >= 1:
        d = IsotopeState(Z=z, N=n - 1, J=parent.J, parity=parent.parity)
        m_d = daughter_mass_ground(d)
        q = m_parent - m_d - mn
        steps.append(DecayStep(parent, d, DecayMode.NEUTRON_EMISSION, q_value_mev=q))
    if z >= 1:
        d = IsotopeState(Z=z - 1, N=n, J=parent.J, parity=parent.parity)
        m_d = daughter_mass_ground(d)
        q = m_parent - m_d - mp
        steps.append(DecayStep(parent, d, DecayMode.PROTON_EMISSION, q_value_mev=q))

    # Alpha
    if z >= 2 and n >= 2:
        d = IsotopeState(Z=z - 2, N=n - 2, J=parent.J, parity=parent.parity)
        m_d = daughter_mass_ground(d)
        m_alpha = _alpha_like_mass_mev(config)
        q = m_parent - m_d - m_alpha
        steps.append(DecayStep(parent, d, DecayMode.ALPHA, q_value_mev=q))

    return steps


def exoergic_decay_steps(steps: Iterable[DecayStep]) -> list[DecayStep]:
    """Keep only vertices with ``Q > 0`` (common filter for physical channels)."""
    return [s for s in steps if s.q_value_mev > 0.0]


def gamma_half_life_proxy_s(delta_E_mev: float) -> float:
    """
    Order-of-magnitude electromagnetic width proxy: ``τ = ħ/ΔE``, ``t_{1/2} = ln(2) τ``.

    Uses :func:`pyhqiv.spin_statistics.resonance_half_life` (MeV and seconds).
    """
    return resonance_half_life(abs(float(delta_E_mev)))


def beta_half_life_s(
    g_fermi: float,
    m_e: float,
    matrix_element: float,
) -> float:
    """``t_{1/2} = ln(2)/Γ`` with ``Γ = G_F² m_e⁵ ℳ²`` (same units throughout)."""
    from pyhqiv.hqiv_nuclear_spectra import beta_decay_rate_with_gf

    width = beta_decay_rate_with_gf(g_fermi, m_e, matrix_element)
    return half_life_from_width(width)


def horizon_potential_scale(config: IsotopeLadderConfig) -> float:
    """
    Same as :func:`pyhqiv.hqiv_nuclei.deuteron_binding_scale` at ``config.shell_m``
    (horizon piece of :func:`hqiv_nuclear_spectra.V_nuclear`).
    """
    from pyhqiv.hqiv_nuclei import deuteron_binding_scale

    return deuteron_binding_scale(config.shell_m)


def iter_isobar_chain_same_A(A: int, z_min: int = 1) -> Iterable[IsotopeState]:
    """Yield ``(Z, N)`` with ``Z + N = A`` for chart-walking."""
    for z in range(z_min, A):
        n = A - z
        if n >= 0:
            yield IsotopeState(Z=z, N=n, J=0.0)


__all__ = [
    "DecayMode",
    "DecayStep",
    "FM2_PER_BARN",
    "IsotopeLadderConfig",
    "IsotopeState",
    "beta_half_life_s",
    "cross_section_geometric_barns",
    "cross_section_geometric_isotope_barns",
    "e_bind_network_only",
    "enumerate_decay_steps",
    "exoergic_decay_steps",
    "gamma_half_life_proxy_s",
    "horizon_potential_scale",
    "iter_isobar_chain_same_A",
    "laplace_angular_factor",
    "network_weight",
    "nuclear_binding_energy_mev",
    "nuclear_radius_fm",
    "nucleus_mass_mev",
    "rotational_excitation_mev",
    "state_mass_mev_with_rotation",
]
