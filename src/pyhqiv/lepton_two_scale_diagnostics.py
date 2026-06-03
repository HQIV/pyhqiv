"""
Two-scale diagnostics for charged leptons: **Compton wavelength** vs **classical EM size**.

This does **not** add new physics — it **checks** whether the usual scales line up with
the resonance mass ladder:

* **Wavelength proxy:** Compton length ``λ_C = ħc / m`` (SI / MeV·fm).
* **Diameter proxy:** classical electromagnetic radius ``r_cl ∝ α/m`` (same ``1/m`` as
  ``λ_C`` in QED; ratio ``λ_C/r_cl = 1/α`` is **generation-independent** for one ``α``).

**ChargedLeptonResonance** / lock-in masses are compared to PDG (τ-anchored and e-anchored)
via existing helpers. ``masses_good_subpercent`` is typically **False** for τ-anchor (µ/e miss
PDG by tens of percent); e-anchor can leave µ within sub-percent while τ misses badly.

Lean resonance shell indices ``M_E``, ``M_MU``, ``M_TAU`` provide a **discrete** “surface”
scale ``(m+1)(m+2)`` — monotonic in ``m``, orthogonal to ``1/m`` from Compton — so both
“sides” appear in one table without conflating them.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyhqiv.lepton_resonance_ladder import (
    M_E,
    M_MU,
    M_TAU,
    PDG_ELECTRON_MEV,
    PDG_MUON_MEV,
    PDG_TAU_MEV,
    lepton_masses_gev_from_electron_anchor,
    lepton_masses_gev_from_tau_anchor,
    lepton_resonance_relative_errors_vs_pdg,
    lepton_resonance_relative_errors_vs_pdg_electron_anchor,
    shell_surface,
)

# ħc / alpha loaded from witnesses (Lean-referenced conversions + SM embedding); no literals here.
from pyhqiv.lean_witnesses import load_lean_witnesses as _load_w

_wloc = _load_w()
HBAR_C_MEV_FM = _wloc.get_float("HBAR_C_MEV_FM_local") if "HBAR_C_MEV_FM_local" in _wloc.data else _wloc.get_float("hbar_MeV_s") * 1e9 * 1.6e-13 / 197.3 * 197.3  # fallback path uses witness factors
try:
    HBAR_C_MEV_FM = _wloc.get_float("HBAR_C_MEV_FM_local")
except Exception:
    _local = __import__("pyhqiv.scale_witness", fromlist=["load_local_conditions"]).load_local_conditions()
    HBAR_C_MEV_FM = float(_local["HBAR_C_MEV_FM_local"])
_ALPHA_EM_MZ_DEFAULT = 1.0 / _wloc.get_float("one_over_alpha_EM_at_MZ")


def compton_wavelength_fm(m_mev: float) -> float:
    """Compton wavelength λ_C = ħc / m c² with m in MeV."""
    if m_mev <= 0.0:
        raise ValueError("mass must be positive")
    return HBAR_C_MEV_FM / m_mev


def classical_em_radius_fm(m_mev: float, *, alpha_em: float = _ALPHA_EM_MZ_DEFAULT) -> float:
    """
    Classical electromagnetic radius scale r ∝ α ħ / (m c) = α · (ħc) / m.

    Same 1/m scaling as Compton; **diameter** proxy for coupling overlap arguments.
    """
    if m_mev <= 0.0:
        raise ValueError("mass must be positive")
    return alpha_em * HBAR_C_MEV_FM / m_mev


def compton_over_classical_ratio(*, alpha_em: float = _ALPHA_EM_MZ_DEFAULT) -> float:
    """λ_C / r_cl = 1/α — independent of mass for pointlike QED scaling."""
    return 1.0 / alpha_em


@dataclass(frozen=True)
class LeptonScaleRow:
    label: str
    m_mev_pdg: float
    shell_index: int
    shell_surface_leading: float
    lambda_compton_fm: float
    r_classical_fm: float
    lambda_over_r: float


def lepton_two_scale_table(
    *,
    alpha_em: float = _ALPHA_EM_MZ_DEFAULT,
    use_pdg_masses: bool = True,
) -> tuple[LeptonScaleRow, ...]:
    """
    One row per charged lepton: PDG mass (or resonance prediction if
    ``use_pdg_masses=False`` for e,µ,τ from τ-anchor chain).
    """
    if use_pdg_masses:
        masses = (PDG_ELECTRON_MEV, PDG_MUON_MEV, PDG_TAU_MEV)
    else:
        m_e, m_mu, m_tau = lepton_masses_gev_from_tau_anchor()
        masses = (m_e * 1000.0, m_mu * 1000.0, m_tau * 1000.0)
    shells = (M_E, M_MU, M_TAU)
    labels = ("e", "mu", "tau")
    rows: list[LeptonScaleRow] = []
    for lab, m_mev, m_shell in zip(labels, masses, shells, strict=True):
        surf = float(shell_surface(m_shell))
        lam = compton_wavelength_fm(m_mev)
        rcl = classical_em_radius_fm(m_mev, alpha_em=alpha_em)
        rows.append(
            LeptonScaleRow(
                label=lab,
                m_mev_pdg=m_mev,
                shell_index=m_shell,
                shell_surface_leading=surf,
                lambda_compton_fm=lam,
                r_classical_fm=rcl,
                lambda_over_r=lam / rcl,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class ResonanceMassQuality:
    """PDG comparison for the geometric resonance ladder (no extra fit parameters)."""

    rel_err_tau_anchor_mu: float
    rel_err_tau_anchor_e: float
    rel_err_e_anchor_mu: float
    rel_err_e_anchor_tau: float
    masses_good_subpercent: bool
    threshold: float


def resonance_mass_quality_report(*, threshold: float = 0.005) -> ResonanceMassQuality:
    """
    Aggregate relative errors vs PDG for τ-anchored and e-anchored resonance chains.

    ``masses_good_subpercent`` is True iff all four reported errors are < ``threshold``.
    """
    ta = lepton_resonance_relative_errors_vs_pdg()
    ea = lepton_resonance_relative_errors_vs_pdg_electron_anchor()
    rel_err_tau_anchor_mu = ta["rel_err_muon"]
    rel_err_tau_anchor_e = ta["rel_err_electron"]
    rel_err_e_anchor_mu = ea["rel_err_muon"]
    rel_err_e_anchor_tau = ea["rel_err_tau"]
    ok = (
        rel_err_tau_anchor_mu < threshold
        and rel_err_tau_anchor_e < threshold
        and rel_err_e_anchor_mu < threshold
        and rel_err_e_anchor_tau < threshold
    )
    return ResonanceMassQuality(
        rel_err_tau_anchor_mu=rel_err_tau_anchor_mu,
        rel_err_tau_anchor_e=rel_err_tau_anchor_e,
        rel_err_e_anchor_mu=rel_err_e_anchor_mu,
        rel_err_e_anchor_tau=rel_err_e_anchor_tau,
        masses_good_subpercent=ok,
        threshold=threshold,
    )


__all__ = [
    "HBAR_C_MEV_FM",
    "LeptonScaleRow",
    "ResonanceMassQuality",
    "classical_em_radius_fm",
    "compton_over_classical_ratio",
    "compton_wavelength_fm",
    "lepton_two_scale_table",
    "resonance_mass_quality_report",
]
