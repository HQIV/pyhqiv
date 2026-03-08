"""
Reference particle masses (MeV) for tests. PDG / CODATA; pyhqiv constants and
SUBATOMIC_PDG_MEV must match these so mass-related tests pass.

Quarks: PDG current (MS / pole where applicable). Hadrons: flavor_content string
(u,d,s,c,b,t) → rest mass MeV. See PDG Particle Summaries, CODATA.
"""

from typing import Dict

# -----------------------------------------------------------------------------
# Quark masses (MeV). PDG-style; light at ~2 GeV, heavy at own mass.
# Package: pyhqiv.constants M_U_MEV_QCD ... M_T_MEV_QCD
# -----------------------------------------------------------------------------
QUARK_MASSES_MEV: Dict[str, float] = {
    "u": 2.2,       # up, PDG order ~2.16
    "d": 4.7,       # down, PDG ~4.67
    "s": 95.0,      # strange, PDG ~93.5
    "c": 1270.0,    # charm, PDG ~1273
    "b": 4180.0,    # bottom, PDG ~4183
    "t": 172_000.0, # top, PDG ~172.5e3
}

# -----------------------------------------------------------------------------
# Hadron masses (MeV): flavor_content -> rest mass.
# Package: pyhqiv.subatomic.SUBATOMIC_PDG_MEV. confined_energy_mev() returns
# these when flavor is in registry and epoch="now".
# -----------------------------------------------------------------------------
HADRON_MASSES_MEV: Dict[str, float] = {
    "uud": 938.272,   # proton
    "udd": 939.565,   # neutron
    "uuu": 1232.0,    # Δ++
    "ddd": 1232.0,    # Δ-
    "uus": 1189.37,   # Σ+
    "uds": 1115.683,  # Λ
    "dds": 1197.45,   # Σ-
    "uss": 1314.86,   # Ξ0
    "dss": 1321.71,   # Ξ-
    "udc": 2286.46,   # Λc+
    "uuc": 2452.9,    # Σc++
    "ddc": 2453.98,   # Σc0
    "usc": 2467.9,    # Ξc+
    "dsc": 2470.88,   # Ξc0
    "ssc": 2695.2,    # Ωc0
    "udb": 5619.60,   # Λb0
    "uudcc": 4311.9,  # Pc+ pentaquark
}

# Human-readable labels for hadrons (optional, for docs/assert messages)
HADRON_LABELS: Dict[str, str] = {
    "uud": "proton",
    "udd": "neutron",
    "uuu": "Δ++",
    "ddd": "Δ-",
    "uus": "Σ+",
    "uds": "Λ",
    "dds": "Σ-",
    "uss": "Ξ0",
    "dss": "Ξ-",
    "udc": "Λc+",
    "uuc": "Σc++",
    "ddc": "Σc0",
    "usc": "Ξc+",
    "dsc": "Ξc0",
    "ssc": "Ωc0",
    "udb": "Λb0",
    "uudcc": "Pc+",
}
