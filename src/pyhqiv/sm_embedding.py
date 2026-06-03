"""
SM embedding data (minimal port).

This module ports the *data* needed by `SM_GR_Unification.lean`:
- generation offsets (rep8V / rep8SPlus / rep8SMinus => 0 / 1 / 2)
- sector multiplicities (cards of ER / ConjUR / NuR => 1 / 3 / 1)
- hypercharge weights (hyperchargeEigenvalue indices => rational Y/2 values)

We do not yet port the full SO(8)/triality matrices; only the values that feed
the geometric mass functional:
  smMassFromGeometry(label) = m_electron_natural * (smSectorMultiplicity + smHyperchargeWeight)
                              * shellMassGeometryFactor(smMassShellReal(label))
                              / electronGeometryFactor
"""

from __future__ import annotations

from typing import Literal

from pyhqiv.lean_witnesses import load_lean_witnesses
from pyhqiv.sm_embedding_sim import sm_hypercharge_y2_for_label

SMLabel = Literal[
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


def sm_generation_index(label: SMLabel) -> int:
    """
    Integer triality generation index in {0,1,2}.

    Lean reference:
      `Hqiv.Algebra.So8RepIndex = Fin 3` with:
        rep8V = 0, rep8SPlus = 1, rep8SMinus = 2
    and `SM_GR_Unification.smGenerationIndex` mapping labels to these reps.
    """
    if label in ("electron", "up", "down", "nu_e"):
        return 0
    if label in ("muon", "strange", "charm", "nu_mu"):
        return 1
    if label in ("tau", "bottom", "top", "nu_tau"):
        return 2
    raise ValueError(f"unknown SM label: {label}")


def sm_generation_offset(label: SMLabel) -> float:
    """
    Lean: smGenerationOffset = smGenerationIndex(label).val

    Triality tags (Lean):
      rep8V = 0, rep8SPlus = 1, rep8SMinus = 2

    Mapping (Lean in SM_GR_Unification.lean):
      electron/up/down/nu_e     -> rep8V  -> 0
      muon/strange/charm/nu_mu -> rep8S+ -> 1
      tau/bottom/top/nu_tau    -> rep8S- -> 2
    """
    return float(sm_generation_index(label))


def sm_sector_multiplicity(label: SMLabel) -> float:
    """
    Lean: smSectorMultiplicity extracted as Fintype.card of carrier types:
      charged leptons: card ER = 1
      quarks:          card ConjUR = 3
      neutrinos:      card NuR = 1
    """
    w = load_lean_witnesses()
    if label in ("electron", "muon", "tau"):
        return w.get_float("lepton_sector_multiplicity")
    if label in ("up", "down", "strange", "charm", "bottom", "top"):
        return w.get_float("quark_sector_multiplicity")
    if label in ("nu_e", "nu_mu", "nu_tau"):
        return w.get_float("neutrino_sector_multiplicity")
    raise ValueError(f"unknown SM label: {label}")


def sm_hypercharge_weight(label: SMLabel) -> float:
    """
    Lean: ``smHyperchargeWeight`` = ``Hqiv.Algebra.hyperchargeEigenvalue`` at the
    index fixed in ``SM_GR_Unification.lean`` (via ``sm_hypercharge_y2_for_label``).
    """
    return sm_hypercharge_y2_for_label(label)


__all__ = [
    "SMLabel",
    "sm_generation_index",
    "sm_generation_offset",
    "sm_hypercharge_weight",
    "sm_sector_multiplicity",
]

