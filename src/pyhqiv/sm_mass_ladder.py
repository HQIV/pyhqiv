"""
Geometric SM mass ladder (corrected "now" philosophy).

This mirrors the core scalar functional from:
  `HQIV_LEAN/Hqiv/Physics/SM_GR_Unification.lean`

**Lepton mass ratios** from ``ChargedLeptonResonance`` / lock-in shells are evaluated in
``pyhqiv.lepton_resonance_ladder`` (PDG comparison is diagnostic, not a sub-percent fit).
``HarmonicLadderMass.lean`` ties the φ-ladder to ``alphaEffAtShell`` and hydrogenic binding,
not to these fermion masses.

Corrected framework philosophy (axiomatic for pyhqiv):
- The current discrete horizon cutoff θ_now **is** the electron horizon.
- The electron horizon shell index is **small** (near `referenceM`), not derived
  from CMB temperature.
- All numeric anchors are loaded from a Lean-export artifact (`witnesses.json`).

Key definitions (Lean form, evaluated at the electron-horizon base shell):
  smMassShellReal(label) = m_base + Δ_triality(label)
  smMassFromGeometry(label) =
      m_electron_natural *
      (smSectorMultiplicity(label) *
       shellMassGeometryFactor(smMassShellReal(label)) /
       electronGeometryFactor)
  electronGeometryFactor = shellMassGeometryFactor(smMassShellReal(electron))
"""

from __future__ import annotations

from typing import Dict

from pyhqiv.auxiliary_field import shell_mass_geometry_factor
from pyhqiv.lean_witnesses import load_lean_witnesses
from pyhqiv.now_setters import m_now
from pyhqiv.sm_embedding import (
    SMLabel,
    sm_generation_index,
    sm_sector_multiplicity,
)
from pyhqiv.sm_gr_unification import m_electron_natural


def natural_mass_to_eV(m_natural: float) -> float:
    """
    Convert a mass measured in the ladder's Planck-normalized natural units
    into eV.

    If `M_Pl_natural = 1`, then `m_natural * M_Pl_eV = m_eV`.
    """
    w = load_lean_witnesses()
    return m_natural * w.get_float("M_Pl_GeV") * w.get_float("GEV_TO_EV")


def sm_mass_shell_real(label: SMLabel, *, m_base: float | None = None) -> float:
    """
    Generation-shifted shell index on the real ladder, evaluated near a small
    electron-horizon base shell.
    """
    base = float(m_now if m_base is None else m_base)
    gen = sm_generation_index(label)
    # Triality generation tag is already the integer {0,1,2}.
    return base + float(gen)


def electron_geometry_factor(*, m_base: float | None = None) -> float:
    """shellMassGeometryFactor(smMassShellReal(electron))."""
    return shell_mass_geometry_factor(sm_mass_shell_real("electron", m_base=m_base))


def sm_mass_from_geometry(label: SMLabel, *, m_base: float | None = None) -> float:
    """
    Lean: smMassFromGeometry (label : SMMassLabel) : ℝ.
    Evaluated at small shell separations around the electron horizon.

    When a Lean export includes ``m_*_from_resonance`` (GeV), that label uses
    ``m_GeV / M_Pl`` **if the key is present**. Missing keys fall back to the
    geometric shell construction. The electron usually stays on the geometric
    path unless ``m_e_from_resonance`` is explicitly exported.
    """
    # Resonance ladder outputs are loaded from Lean-exported witnesses.
    # No hard-coded per-particle mass literals are used here.
    resonance_key_by_label = {
        "electron": "m_e_from_resonance",
        "muon": "m_mu_from_resonance",
        "tau": "m_tau_from_resonance",
        "up": "m_up_from_resonance",
        "down": "m_down_from_resonance",
        "strange": "m_strange_from_resonance",
        "charm": "m_charm_from_resonance",
        "bottom": "m_bottom_from_resonance",
        "top": "m_top_from_resonance",
    }
    if label in resonance_key_by_label:
        w = load_lean_witnesses()
        key = resonance_key_by_label[label]
        if key in w.data:
            m_pl = w.get_float("M_Pl_GeV")
            return w.get_float(key) / m_pl

    m0 = m_electron_natural()
    e_geom = electron_geometry_factor(m_base=m_base)
    shell_geom = shell_mass_geometry_factor(sm_mass_shell_real(label, m_base=m_base))
    sector = sm_sector_multiplicity(label)

    # Charged leptons: match Lean's normalized charged-lepton functional:
    #   m(label) = m_e * ((sector + hypercharge) * shell_geom / e_geom)
    #                               / (sector+hypercharge at electron).
    return m0 * (sector * shell_geom / e_geom)


def sm_mass_from_geometry_eV(label: SMLabel) -> float:
    """`sm_mass_from_geometry(label)` converted to eV."""
    return natural_mass_to_eV(sm_mass_from_geometry(label))


def sm_masses_from_geometry_eV_at_default_now() -> Dict[str, float]:
    """Compute all SM labels at the canonical electron-horizon "now", in eV."""
    labels = [
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
    return {lab: sm_mass_from_geometry_eV(lab) for lab in labels}


def sm_masses_from_geometry_at_default_now() -> Dict[str, float]:
    """Compute all SM labels at the canonical electron-horizon "now"."""
    labels = [
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
    return {lab: sm_mass_from_geometry(lab) for lab in labels}


__all__ = [
    "natural_mass_to_eV",
    "sm_mass_shell_real",
    "sm_mass_from_geometry",
    "electron_geometry_factor",
    "sm_mass_from_geometry_eV",
    "sm_masses_from_geometry_at_default_now",
    "sm_masses_from_geometry_eV_at_default_now",
]

