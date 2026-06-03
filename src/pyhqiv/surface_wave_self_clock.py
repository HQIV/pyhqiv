"""
Surface-wave **self-clock** and Mexican-hat effective potential (Lean mirrors).

* ``Hqiv.Physics.SurfaceWaveSelfClock`` — ``comptonAngularFrequency``, ``selfClockPhase``,
  ``mexicanHatVeff``
* ``Hqiv.Physics.GlobalDetuning`` — ``deltaGlobal`` from ``fromLapseScalars``
* ``Hqiv.Physics.LeptonGenerationLockin`` — lock-in shell chain parallel to quarks

**Compton frequency in Lean** (this module): ``ω_m = (m+1)`` in natural units
(``T m = 1/(m+1)``), *not* ``2(m+1)`` from ``AuxiliaryField.phi_of_shell``.

PDG mass checks belong in ``lepton_resonance_ladder`` (``ChargedLeptonResonance`` shells:
``reference_m``, ``81``, ``16336``).

**Lock-in shells** ``(reference_m, 81, 16336)`` (``LeptonGenerationLockin``) use
``geometricResonanceStep``; mass ratios follow ``lepton_resonance_ladder`` (``k_{τµ}``, ``k_{µe}``),
not the raw surface ratio ``eff(τ)/eff(µ)`` alone.

**Cumulative rapidity (Lean):** ``SurfaceWaveSelfClock.selfClock_rapidity_update`` proves
``selfClockPhase m (η + φ·Δt) = selfClockPhase m η + φ·Δt`` — i.e. **rapidity accumulation** is
additive on the phase ledger. The **cosmic-epoch** contribution is modeled as a fixed **0.3°**
rotation (polarization / birefringence at ``now``), stored in radians as the **base** ``η`` via
:func:`self_clock_cumulative_rapidity_cosmic_now`; further lapse increments **stack** as ``φ·Δt``
on top, unchanged by that identity. Override angle with witness ``cosmic_birefringence_deg_now``
when Lean exports a measured β. Redshift bookkeeping for β also appears in
``BornMeasurementFinite`` / ``AuxFieldBellDelayedChoice`` (``birefringenceRedshiftN``).

**τ birth at lock-in:** :func:`self_clock_phase` with ``η = 0`` stays Lean-identical for the τ birth
line (quarter-turn baseline).
"""

from __future__ import annotations

import math

from pyhqiv.lepton_resonance_ladder import eff_corrected, effective_surface_at_shell
from pyhqiv.lean_witnesses import load_lean_witnesses
from pyhqiv.lightcone import reference_m

COSMIC_BIREFRINGENCE_DEG_NOW_WITNESS_KEY = "cosmic_birefringence_deg_now"
# Default self-clock angle at observer epoch (degrees); witness may override (e.g. ACT-class β).
SELF_CLOCK_DEG_AT_NOW_DEFAULT = 0.3
_DEFAULT_COSMIC_BIREFRINGENCE_DEG_NOW = SELF_CLOCK_DEG_AT_NOW_DEFAULT


def compton_angular_frequency_natural(m: int) -> float:
    """Lean ``comptonAngularFrequency m`` = ``(m+1)`` in natural units."""
    return float(m + 1)


def compton_quarter_turn_at_T_lockin() -> float:
    """Lean ``compton_quarter_turn_at_T_lockin`` = ``(m_lockin+1) * (π/2)``."""
    ml = reference_m()
    return compton_angular_frequency_natural(ml) * (math.pi / 2.0)


def cosmic_birefringence_deg_at_now() -> float:
    """
    Cosmic birefringence angle β (degrees) at the observer epoch, for the self-clock ledger.

    Merged Lean witness key ``cosmic_birefringence_deg_now`` overrides; otherwise **0.3** (napkin /
    CMB-order anchor; compare ACT DR6 ~0.2°–0.3° class results).
    """
    data = load_lean_witnesses().data
    if COSMIC_BIREFRINGENCE_DEG_NOW_WITNESS_KEY in data:
        return float(data[COSMIC_BIREFRINGENCE_DEG_NOW_WITNESS_KEY])
    return float(_DEFAULT_COSMIC_BIREFRINGENCE_DEG_NOW)


def cosmic_birefringence_rad_at_now() -> float:
    """β in radians (``radians(cosmic_birefringence_deg_at_now())``)."""
    return math.radians(cosmic_birefringence_deg_at_now())


def self_clock_cumulative_rapidity_cosmic_now() -> float:
    """
    Base ``cumulativeRapidity`` (radians) at cosmic ``now``: **0.3°** by default (self-clock / β).

    Same slot as in Lean ``selfClockPhase``; ``selfClock_rapidity_update`` then adds ``φ·Δt`` on top.
    """
    return cosmic_birefringence_rad_at_now()


def self_clock_phase_with_cosmic_now(m: int, cumulative_rapidity: float = 0.0) -> float:
    """
    Self-clock phase including the cosmic-now birefringence ledger term:

    ``self_clock_phase(m, cumulative_rapidity + self_clock_cumulative_rapidity_cosmic_now())``.
    """
    return self_clock_phase(
        m,
        float(cumulative_rapidity) + self_clock_cumulative_rapidity_cosmic_now(),
    )


def self_clock_phase(m: int, cumulative_rapidity: float) -> float:
    """
    Lean ``selfClockPhase m cumulativeRapidity`` =
    ``comptonAngularFrequency m * (π/2) + cumulativeRapidity``.
    """
    return compton_angular_frequency_natural(m) * (math.pi / 2.0) + float(cumulative_rapidity)


def self_clock_rapidity_update_additivity(
    m: int,
    cumulative_rapidity: float,
    phi: float,
    delta_t: float,
) -> bool:
    """
    Lean ``Hqiv.Physics.SurfaceWaveSelfClock.selfClock_rapidity_update``:

    ``selfClockPhase m (η + φ·Δt) = selfClockPhase m η + φ·Δt``.
    """
    left = self_clock_phase(m, cumulative_rapidity + phi * delta_t)
    right = self_clock_phase(m, cumulative_rapidity) + phi * delta_t
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def delta_global_from_lapse_scalars(lam: float, phi_newtonian: float, phi_aux: float, t: float) -> float:
    """``deltaGlobal (fromLapseScalars lam Φ φ t)`` = ``λ * (Φ + φ·t)``."""
    return lam * (phi_newtonian + phi_aux * t)


def mexican_hat_veff(
    lam: float,
    phi_newtonian: float,
    phi_aux: float,
    t: float,
    m: int,
) -> float:
    """
    Lean ``mexicanHatVeff lam Φ φ t m`` = ``1 / effCorrected(δ_global, m)`` with
    ``δ_global = λ·(Φ + φ·t)``.
    """
    d = delta_global_from_lapse_scalars(lam, phi_newtonian, phi_aux, t)
    e = eff_corrected(d, m)
    return 1.0 / e


def lepton_heavy_vertex_shell() -> int:
    """Lean ``leptonHeavyVertexShell`` = ``referenceM``."""
    return reference_m()


def lepton_muon_shell() -> int:
    """Lean ``leptonMuonShell`` (placeholder)."""
    return 81


def lepton_electron_shell() -> int:
    """Lean ``leptonElectronShell`` (placeholder)."""
    return 16336


def geometric_resonance_step_lepton_tau_mu() -> float:
    """``geometricResonanceStep leptonMuonShell leptonHeavyVertexShell`` (= ``k_{τµ}``)."""
    m_from = lepton_muon_shell()
    m_to = lepton_heavy_vertex_shell()
    return effective_surface_at_shell(m_from) / effective_surface_at_shell(m_to)


def geometric_resonance_step_lepton_mu_e() -> float:
    """``geometricResonanceStep leptonElectronShell leptonMuonShell`` (= ``k_{µe}``)."""
    m_from = lepton_electron_shell()
    m_to = lepton_muon_shell()
    return effective_surface_at_shell(m_from) / effective_surface_at_shell(m_to)


def lepton_tau_birth_phase_matches_quarter_turn_at_lockin() -> bool:
    """Lean ``lepton_tau_birth_at_lockin``."""
    m = lepton_heavy_vertex_shell()
    return math.isclose(
        self_clock_phase(m, 0.0),
        compton_quarter_turn_at_T_lockin(),
        rel_tol=0.0,
        abs_tol=1e-12,
)


__all__ = [
    "COSMIC_BIREFRINGENCE_DEG_NOW_WITNESS_KEY",
    "SELF_CLOCK_DEG_AT_NOW_DEFAULT",
    "compton_angular_frequency_natural",
    "compton_quarter_turn_at_T_lockin",
    "cosmic_birefringence_deg_at_now",
    "cosmic_birefringence_rad_at_now",
    "delta_global_from_lapse_scalars",
    "geometric_resonance_step_lepton_mu_e",
    "geometric_resonance_step_lepton_tau_mu",
    "lepton_electron_shell",
    "lepton_heavy_vertex_shell",
    "lepton_muon_shell",
    "lepton_tau_birth_phase_matches_quarter_turn_at_lockin",
    "mexican_hat_veff",
    "self_clock_cumulative_rapidity_cosmic_now",
    "self_clock_phase",
    "self_clock_phase_with_cosmic_now",
    "self_clock_rapidity_update_additivity",
]
