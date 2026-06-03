"""
Charged-lepton resonance masses — **Lean-aligned** with:

* ``Hqiv.Physics.ChargedLeptonResonance`` — ``effectiveSurface m (m : ℝ)`` agrees with
  ``effCorrected 0 m``; ratios ``k_{τµ} = eff(m_µ)/eff(m_τ)``, ``k_{µe} = eff(m_e)/eff(m_µ)``
* ``Hqiv.Physics.GlobalDetuning`` — ``rindlerDenWithDelta``, ``effCorrected``
* ``Hqiv.Physics.LeptonResonanceGlobalDetuning`` — obstruction numerics (same ``m_*``)

**Shells:** τ at ``referenceM``, µ at ``81``, e at ``16336`` (``LeptonGenerationLockin``).
The abandoned τ-highest-shell model lives only in Lean ``archive/``, not here.

**Single source of truth:** ``eff_corrected(δ, m) = shellSurface(m) / (1 + c_rindler·m + δ)`` with
``c_rindler = gamma_HQIV/2``, ``γ = 2/5``. Local ladder uses ``δ = 0``.

Mass flow: ``m_µ = m_τ / k_{τµ}``, ``m_e = m_µ / k_{µe}`` (τ anchor); e anchor multiplies up.
PDG residuals are **large** for µ/e with this geometry — see ``lepton_resonance_check_report()``.

See ``lepton_resonance_coherence`` for φ at shells (observer split only).
"""

from __future__ import annotations

from pyhqiv.lean_witnesses import load_lean_witnesses
from pyhqiv.lightcone import reference_m
from pyhqiv.metric import gamma_hqiv

# --- Lean: FanoResonance / SM_GR_Unification: gamma_HQIV = 2/5 (loaded, not literal 0.4)
GAMMA_HQIV = gamma_hqiv()

# Lean: c_rindler_shared = gamma_HQIV / 2  (coefficient of shell index in denominator)
C_RINDLER_SHARED = GAMMA_HQIV / 2.0

# Shell indices — ``LeptonGenerationLockin`` / ``ChargedLeptonResonance``
# (M_TAU is reference lock-in; M_MU / M_E are specific rungs from Lean ChargedLeptonResonance)
M_TAU: int = reference_m()
M_MU = 81
M_E = 16_336

# Lean resonance parameters loaded from witnesses (no literals in .py)
_w = load_lean_witnesses()
M_TAU_GEV_PDG_ANCHOR = _w.get_float("m_tau_from_resonance")  # resonance readout (comparison anchor in some paths)
REF_K_TAU_MU = _w.get_float("resonance_k_tau_mu")
REF_K_MU_E = _w.get_float("resonance_k_mu_e")

# Note: PDG centrals live exclusively in tests/data/*_with_errors.py (with source error bars).
# They are never imported or referenced from src for computation.


def shell_surface(m: int) -> float:
    """``shellSurface m = (m+1)(m+2)``."""
    return float((m + 1) * (m + 2))


def rindler_detuning_shared(x: float) -> float:
    """``rindlerDetuningShared(x) = 1 + c_rindler_shared * x``."""
    return 1.0 + C_RINDLER_SHARED * float(x)


def rindler_den_with_delta(delta: float, m: int) -> float:
    """``GlobalDetuning.rindlerDenWithDelta δ m = 1 + c_rindler_shared·m + δ``."""
    return 1.0 + C_RINDLER_SHARED * float(m) + float(delta)


def eff_corrected(delta: float, m: int) -> float:
    """``GlobalDetuning.effCorrected δ m``."""
    return shell_surface(m) / rindler_den_with_delta(delta, m)


def effective_surface_at_shell(m: int) -> float:
    """``effectiveSurface m m`` — same as ``eff_corrected(0, m)``."""
    return eff_corrected(0.0, m)


def _resonance_k_tau_mu_value() -> float:
    w = load_lean_witnesses().data
    if "resonance_k_tau_mu" in w:
        return float(w["resonance_k_tau_mu"])
    return eff_corrected(0.0, M_MU) / eff_corrected(0.0, M_TAU)


def _resonance_k_mu_e_value() -> float:
    w = load_lean_witnesses().data
    if "resonance_k_mu_e" in w:
        return float(w["resonance_k_mu_e"])
    return eff_corrected(0.0, M_E) / eff_corrected(0.0, M_MU)


def resonance_k_tau_mu() -> float:
    """``resonance_k_tau_mu`` = ``effectiveSurface m_µ / effectiveSurface m_τ`` (Lean)."""
    return _resonance_k_tau_mu_value()


def resonance_k_mu_e() -> float:
    """``resonance_k_mu_e`` = ``effectiveSurface m_e / effectiveSurface m_µ``."""
    return _resonance_k_mu_e_value()


def resonance_k_tau_mu_corrected(delta: float) -> float:
    return eff_corrected(delta, M_MU) / eff_corrected(delta, M_TAU)


def resonance_k_mu_e_corrected(delta: float) -> float:
    return eff_corrected(delta, M_E) / eff_corrected(delta, M_MU)


# --- Obstruction (LeptonResonanceGlobalDetuning): PDG ratio targets as r1 = mµ/mτ, r2 = me/mµ


def _S_tau() -> float:
    return float(shell_surface(M_TAU))


def _S_mu() -> float:
    return float(shell_surface(M_MU))


def _S_e() -> float:
    return float(shell_surface(M_E))


def pdg_mass_ratio_mu_over_tau() -> float:
    """Target m_µ / m_τ from PDG centrals."""
    return PDG_MUON_MEV / PDG_TAU_MEV


def pdg_mass_ratio_e_over_mu() -> float:
    """Target m_e / m_µ from PDG centrals."""
    return PDG_ELECTRON_MEV / PDG_MUON_MEV


def delta_num_tau_mu(r1: float) -> float:
    """Lean ``δNumTauMu`` with ``r1 = mµ/mτ``."""
    St, Smu = _S_tau(), _S_mu()
    tr, mur = float(M_TAU), float(M_MU)
    return r1 * Smu * (1.0 + C_RINDLER_SHARED * tr) - St * (1.0 + C_RINDLER_SHARED * mur)


def delta_num_mu_e(r2: float) -> float:
    """Lean ``δNumMuE`` with ``r2 = m_e/m_µ``."""
    Smu, Se = _S_mu(), _S_e()
    mur, er = float(M_MU), float(M_E)
    return r2 * Se * (1.0 + C_RINDLER_SHARED * mur) - Smu * (1.0 + C_RINDLER_SHARED * er)


def delta_den_tau_mu(r1: float) -> float:
    return _S_tau() - r1 * _S_mu()


def delta_den_mu_e(r2: float) -> float:
    return _S_mu() - r2 * _S_e()


def single_delta_compat_residual(r1: float, r2: float) -> float:
    """
    Lean ``singleDeltaCompatResidual r₁ r₂``. If both corrected ratios could match ``r1``, ``r2``
    with one ``δ``, this is zero. For generic PDG targets it is **nonzero**.
    """
    return delta_num_tau_mu(r1) * delta_den_mu_e(r2) - delta_num_mu_e(r2) * delta_den_tau_mu(r1)


def pdg_single_delta_compat_residual() -> float:
    """Compat residual using PDG mass ratios (not k ratios from bare ladder)."""
    return single_delta_compat_residual(pdg_mass_ratio_mu_over_tau(), pdg_mass_ratio_e_over_mu())


# --- Mass predictions (anchor τ or e)


def lepton_resonance_masses_gev_triple() -> tuple[float, float, float]:
    """
    ``(m_e, m_µ, m_τ)`` in GeV consistent with ``ChargedLeptonResonance``.

    If ``witnesses.json`` carries ``m_tau_from_resonance`` and ``m_mu_from_resonance`` (GeV),
    those are used with ``m_e = m_µ / k_{µe}`` unless ``m_e_from_resonance`` is also supplied.
    Otherwise the τ-anchor chain uses ``M_TAU_GEV_PDG_ANCHOR`` and computed ``k`` factors
    (optionally overridden by ``resonance_k_*`` witness keys).
    """
    w = load_lean_witnesses().data
    k_tm = _resonance_k_tau_mu_value()
    k_me = _resonance_k_mu_e_value()
    if "m_tau_from_resonance" in w and "m_mu_from_resonance" in w:
        m_tau_gev = float(w["m_tau_from_resonance"])
        m_mu_gev = float(w["m_mu_from_resonance"])
        if "m_e_from_resonance" in w:
            m_e_gev = float(w["m_e_from_resonance"])
        else:
            m_e_gev = m_mu_gev / k_me
        return (m_e_gev, m_mu_gev, m_tau_gev)
    return lepton_masses_gev_from_tau_anchor()


def lepton_masses_gev_from_tau_anchor(m_tau_gev: float = M_TAU_GEV_PDG_ANCHOR) -> tuple[float, float, float]:
    """``(m_e, m_µ, m_τ)`` GeV with τ anchored."""
    k_tm = resonance_k_tau_mu()
    k_me = resonance_k_mu_e()
    m_mu = m_tau_gev / k_tm
    m_e = m_mu / k_me
    return (m_e, m_mu, m_tau_gev)


def lepton_masses_gev_from_electron_anchor(m_e_gev: float | None = None) -> tuple[float, float, float]:
    """``(m_e, m_µ, m_τ)`` GeV with e anchored."""
    if m_e_gev is None:
        m_e_gev = PDG_ELECTRON_MEV / 1000.0
    k_tm = resonance_k_tau_mu()
    k_me = resonance_k_mu_e()
    m_mu = m_e_gev * k_me
    m_tau = m_mu * k_tm
    return (m_e_gev, m_mu, m_tau)


def relative_sigma_tau_from_electron_sigma(sigma_m_e_over_m_e: float) -> float:
    return float(sigma_m_e_over_m_e)


def tau_mass_mev_uncertainty_from_electron_sigma_mev(sigma_m_e_mev: float | None = None) -> float:
    if sigma_m_e_mev is None:
        sigma_m_e_mev = PDG_ELECTRON_MEV_1SIGMA
    _, _, m_tau_gev = lepton_masses_gev_from_electron_anchor()
    m_tau_mev = m_tau_gev * 1000.0
    return float(sigma_m_e_mev * (m_tau_mev / PDG_ELECTRON_MEV))


def lepton_resonance_relative_errors_vs_pdg() -> dict[str, float]:
    """τ anchor: relative |pred − PDG| / PDG for µ and e."""
    m_e_gev, m_mu_gev, _ = lepton_masses_gev_from_tau_anchor()
    return {
        "rel_err_muon": abs(m_mu_gev * 1000.0 - PDG_MUON_MEV) / PDG_MUON_MEV,
        "rel_err_electron": abs(m_e_gev * 1000.0 - PDG_ELECTRON_MEV) / PDG_ELECTRON_MEV,
    }


def lepton_resonance_relative_errors_vs_pdg_electron_anchor() -> dict[str, float]:
    """e anchor: relative errors for µ and τ."""
    _, m_mu_gev, m_tau_gev = lepton_masses_gev_from_electron_anchor()
    return {
        "rel_err_muon": abs(m_mu_gev * 1000.0 - PDG_MUON_MEV) / PDG_MUON_MEV,
        "rel_err_tau": abs(m_tau_gev * 1000.0 - PDG_TAU_MEV) / PDG_TAU_MEV,
    }


def lepton_resonance_check_report() -> str:
    """Human-readable mass check vs PDG (τ- and e-anchored) and obstruction."""
    k1, k2 = resonance_k_tau_mu(), resonance_k_mu_e()
    me_t, mmu_t, mt_t = lepton_masses_gev_from_tau_anchor()
    me_e, mmu_e, mt_e = lepton_masses_gev_from_electron_anchor()
    err_t = lepton_resonance_relative_errors_vs_pdg()
    err_e = lepton_resonance_relative_errors_vs_pdg_electron_anchor()
    compat = pdg_single_delta_compat_residual()
    lines = [
        "Charged-lepton resonance (ChargedLeptonResonance / lock-in shells, δ=0)",
        f"  k_tau_mu = {k1:.17g}  (ref {REF_K_TAU_MU})",
        f"  k_mu_e   = {k2:.17g}  (ref {REF_K_MU_E})",
        "",
        "τ anchor (PDG τ central):",
        f"  m_µ pred = {mmu_t*1000:.6f} MeV  PDG {PDG_MUON_MEV:.7f}  rel_err = {err_t['rel_err_muon']:.6g}",
        f"  m_e pred = {me_t*1000:.6f} MeV  PDG {PDG_ELECTRON_MEV:.8f}  rel_err = {err_t['rel_err_electron']:.6g}",
        "",
        "e anchor (PDG e central):",
        f"  m_µ pred = {mmu_e*1000:.6f} MeV  rel_err = {err_e['rel_err_muon']:.6g}",
        f"  m_τ pred = {mt_e*1000:.6f} MeV  PDG {PDG_TAU_MEV:.5f}  rel_err = {err_e['rel_err_tau']:.6g}",
        "",
        "Single-δ obstruction (PDG mass ratios): compat_residual = "
        f"{compat:.6g}  (nonzero ⇒ no one δ fixes both ratios exactly)",
    ]
    return "\n".join(lines)


__all__ = [
    "C_RINDLER_SHARED",
    "GAMMA_HQIV",
    "M_E",
    "M_MU",
    "M_TAU",
    "M_TAU_GEV_PDG_ANCHOR",
    "PDG_ELECTRON_MEV",
    "PDG_ELECTRON_MEV_1SIGMA",
    "PDG_MUON_MEV",
    "PDG_TAU_MEV",
    "PDG_TAU_MEV_1SIGMA",
    "REF_K_MU_E",
    "REF_K_TAU_MU",
    "delta_den_mu_e",
    "delta_den_tau_mu",
    "delta_num_mu_e",
    "delta_num_tau_mu",
    "eff_corrected",
    "effective_surface_at_shell",
    "lepton_masses_gev_from_electron_anchor",
    "lepton_masses_gev_from_tau_anchor",
    "lepton_resonance_masses_gev_triple",
    "lepton_resonance_check_report",
    "lepton_resonance_relative_errors_vs_pdg",
    "lepton_resonance_relative_errors_vs_pdg_electron_anchor",
    "pdg_mass_ratio_e_over_mu",
    "pdg_mass_ratio_mu_over_tau",
    "pdg_single_delta_compat_residual",
    "relative_sigma_tau_from_electron_sigma",
    "resonance_k_mu_e",
    "resonance_k_mu_e_corrected",
    "resonance_k_tau_mu",
    "resonance_k_tau_mu_corrected",
    "rindler_den_with_delta",
    "rindler_detuning_shared",
    "shell_surface",
    "single_delta_compat_residual",
    "tau_mass_mev_uncertainty_from_electron_sigma_mev",
]
