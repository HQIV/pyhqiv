"""
Dynamic Rindler detuning for charged leptons (auxiliary-field / self-clock bridge).

**Authoritative masses and ``k`` factors** remain ``Hqiv.Physics.ChargedLeptonResonance`` with
``δ = 0`` (``lepton_resonance_ladder``). This module is an **optional exploratory layer**: same
``GlobalDetuning.effCorrected`` machinery as Lean, but with an extra additive detuning tied to
``AuxiliaryField.phi_of_shell`` differences between shells. It is **not** a substitute for a proved
lemma in Lean — do not tune real constants in Python to fit PDG.

**Default scale** ``s``: ``α/3`` with ``α`` from ``OctonionicLightCone`` (single coefficient, no
ad-hoc rational combinations). If Lean later proves a numerical scale, export it as
``DYNAMIC_RINDLER_PHI_SCALE_WITNESS_KEY`` in the witness JSON; :func:`dynamic_rindler_phi_scale` picks that up
when present (see ``pyhqiv.lean_witnesses`` merge).

Lean context: ``SurfaceWaveSelfClock``, ``GlobalDetuning``, ``AuxiliaryField``, ``ConservedContentMassBridge``.
"""

from __future__ import annotations

from pyhqiv.auxiliary_field import phi_of_shell
from pyhqiv.lean_witnesses import load_lean_witnesses
from pyhqiv.lepton_resonance_ladder import (
    C_RINDLER_SHARED,
    M_E,
    M_MU,
    M_TAU,
    PDG_ELECTRON_MEV,
    PDG_MUON_MEV,
    M_TAU_GEV_PDG_ANCHOR,
    eff_corrected,
)
from pyhqiv.lightcone import alpha

# Lean export key (optional): when present in merged witnesses JSON, overrides default ``α/3``.
DYNAMIC_RINDLER_PHI_SCALE_WITNESS_KEY = "dynamic_rindler_phi_scale"


def dynamic_rindler_phi_scale() -> float:
    """
    Coefficient ``s`` in ``δ_aux = s · c_rindler · Δφ_aux``.

    If the merged Lean witness bundle defines ``dynamic_rindler_phi_scale``, that value is used
    (proved/exported in Lean). Otherwise the default is **``α/3``** only — one hypothesis, not a
    fitted blend of constants.
    """
    data = load_lean_witnesses().data
    if DYNAMIC_RINDLER_PHI_SCALE_WITNESS_KEY in data:
        return float(data[DYNAMIC_RINDLER_PHI_SCALE_WITNESS_KEY])
    return float(alpha()) / 3.0


def delta_auxiliary_phi_per_shell(*, scale: float | None = None) -> tuple[float, float, float]:
    """
    Extra detuning ``(δ_τ, δ_µ, δ_e)`` for shells ``(M_TAU, M_MU, M_E)``.

    τ is the reference: ``δ_τ = 0``. Middle and outer shells pick up
    ``s · c_rindler · (φ_aux(m) − φ_aux(m_τ))``.
    """
    s = dynamic_rindler_phi_scale() if scale is None else float(scale)
    p_t = phi_of_shell(M_TAU)
    p_m = phi_of_shell(M_MU)
    p_e = phi_of_shell(M_E)
    c = C_RINDLER_SHARED
    return (
        0.0,
        s * c * (p_m - p_t),
        s * c * (p_e - p_t),
    )


def resonance_k_tau_mu_dynamic_rindler(*, scale: float | None = None) -> float:
    """``k_{τµ}`` with per-shell ``δ`` from :func:`delta_auxiliary_phi_per_shell`."""
    d_tau, d_mu, _d_e = delta_auxiliary_phi_per_shell(scale=scale)
    return eff_corrected(d_mu, M_MU) / eff_corrected(d_tau, M_TAU)


def resonance_k_mu_e_dynamic_rindler(*, scale: float | None = None) -> float:
    """``k_{µe}`` with per-shell ``δ`` from :func:`delta_auxiliary_phi_per_shell`."""
    _d_tau, d_mu, d_e = delta_auxiliary_phi_per_shell(scale=scale)
    return eff_corrected(d_e, M_E) / eff_corrected(d_mu, M_MU)


def lepton_masses_gev_from_tau_anchor_dynamic_rindler(
    m_tau_gev: float = M_TAU_GEV_PDG_ANCHOR,
    *,
    scale: float | None = None,
) -> tuple[float, float, float]:
    """``(m_e, m_µ, m_τ)`` GeV with τ anchored and dynamic-Rindler ``k`` factors."""
    k_tm = resonance_k_tau_mu_dynamic_rindler(scale=scale)
    k_me = resonance_k_mu_e_dynamic_rindler(scale=scale)
    m_mu = m_tau_gev / k_tm
    m_e = m_mu / k_me
    return (m_e, m_mu, m_tau_gev)


def lepton_resonance_relative_errors_vs_pdg_dynamic_rindler(
    *,
    scale: float | None = None,
) -> dict[str, float]:
    """τ anchor: relative |pred − PDG| / PDG for µ and e after dynamic-Rindler refinement."""
    m_e_gev, m_mu_gev, _ = lepton_masses_gev_from_tau_anchor_dynamic_rindler(scale=scale)
    return {
        "rel_err_muon": abs(m_mu_gev * 1000.0 - PDG_MUON_MEV) / PDG_MUON_MEV,
        "rel_err_electron": abs(m_e_gev * 1000.0 - PDG_ELECTRON_MEV) / PDG_ELECTRON_MEV,
    }


def dynamic_rindler_pdg_improvement_report(*, scale: float | None = None) -> str:
    """Before/after relative errors (bare δ=0 vs auxiliary φ shift)."""
    from pyhqiv.lepton_resonance_ladder import lepton_resonance_relative_errors_vs_pdg

    bare = lepton_resonance_relative_errors_vs_pdg()
    dyn = lepton_resonance_relative_errors_vs_pdg_dynamic_rindler(scale=scale)
    if scale is None:
        s = dynamic_rindler_phi_scale()
        src = (
            f"witness {DYNAMIC_RINDLER_PHI_SCALE_WITNESS_KEY!r}"
            if DYNAMIC_RINDLER_PHI_SCALE_WITNESS_KEY in load_lean_witnesses().data
            else f"default α/3 (α = {alpha():.17g})"
        )
    else:
        s = float(scale)
        src = "explicit scale= argument"
    lines = [
        "Dynamic Rindler (auxiliary φ_aux shell offsets, δ_aux = s·c_rindler·Δφ_aux)",
        f"  s = {s:.17g}  ({src})",
        "",
        "τ anchor vs PDG (relative):",
        f"  bare:   rel_err_µ = {bare['rel_err_muon']:.6g}  rel_err_e = {bare['rel_err_electron']:.6g}",
        f"  dynamic: rel_err_µ = {dyn['rel_err_muon']:.6g}  rel_err_e = {dyn['rel_err_electron']:.6g}",
    ]
    return "\n".join(lines)


__all__ = [
    "DYNAMIC_RINDLER_PHI_SCALE_WITNESS_KEY",
    "delta_auxiliary_phi_per_shell",
    "dynamic_rindler_phi_scale",
    "dynamic_rindler_pdg_improvement_report",
    "lepton_masses_gev_from_tau_anchor_dynamic_rindler",
    "lepton_resonance_relative_errors_vs_pdg_dynamic_rindler",
    "resonance_k_mu_e_dynamic_rindler",
    "resonance_k_tau_mu_dynamic_rindler",
]
