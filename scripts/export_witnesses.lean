import Hqiv.Geometry.OctonionicLightCone
import Hqiv.Physics.SM_GR_Unification

/-!
Export a single JSON artifact of "witness" values proved/defined in Lean.

This file is intended to be the ONLY source of numerical anchors consumed by
`pyhqiv.lean_witnesses`. The exported JSON should be written to:

  `src/pyhqiv/witnesses.json`

so Python code can load it as a package resource.

Note: this is a small script-style entrypoint; adapt the exact witness list as
Lean ports evolve.
-/

open Hqiv

def jsonLine (s : String) : IO Unit := IO.println s

def main : IO Unit := do
  -- Minimal set required by current pyhqiv rebuild.
  -- Extend with triality projection factors / additional witnesses as they are proved.
  jsonLine "{"
  jsonLine s!"  \"referenceM\": {Hqiv.Geometry.OctonionicLightCone.referenceM},"
  jsonLine s!"  \"m_now_electron_shell\": {Hqiv.Geometry.OctonionicLightCone.referenceM},"
  jsonLine "  \"triality_shell_offsets\": [0, 1, 2],"
  jsonLine "  \"triality_projection_factors\": [1.0, 1.0, 1.0],"
  jsonLine s!"  \"alpha_GUT\": {Hqiv.Physics.SM_GR_Unification.alpha_GUT},"
  jsonLine s!"  \"one_over_alpha_EM_at_MZ\": {Hqiv.Physics.SM_GR_Unification.one_over_alpha_EM_at_MZ},"
  jsonLine s!"  \"sin2thetaW_at_MZ\": {Hqiv.Physics.SM_GR_Unification.sin2thetaW_at_MZ},"
  jsonLine s!"  \"alpha_s_at_MZ\": {Hqiv.Physics.SM_GR_Unification.alpha_s_at_MZ},"
  jsonLine s!"  \"M_Pl_GeV\": {Hqiv.Physics.SM_GR_Unification.M_Pl_GeV},"
  jsonLine s!"  \"M_Z_GeV\": {Hqiv.Physics.SM_GR_Unification.M_Z_GeV},"
  jsonLine s!"  \"m_electron_MeV\": {Hqiv.Physics.SM_GR_Unification.m_electron_MeV},"
  jsonLine s!"  \"m_proton_MeV\": {Hqiv.Physics.SM_GR_Unification.m_proton_MeV},"
  -- proton_mass_natural = (m_proton_MeV/1000) / M_Pl_GeV
  jsonLine s!"  \"proton_mass_natural\": {(Hqiv.Physics.SM_GR_Unification.m_proton_MeV / 1000.0) / Hqiv.Physics.SM_GR_Unification.M_Pl_GeV},"
  jsonLine s!"  \"m_neutron_MeV\": {Hqiv.Physics.SM_GR_Unification.m_neutron_MeV},"
  jsonLine "  \"lepton_sector_multiplicity\": 1.0,"
  jsonLine "  \"quark_sector_multiplicity\": 3.0,"
  jsonLine "  \"neutrino_sector_multiplicity\": 1.0,"
  jsonLine "  \"GEV_TO_EV\": 1000000000.0,"
  jsonLine "  \"MEV_TO_EV\": 1000000.0"
  jsonLine "}"

