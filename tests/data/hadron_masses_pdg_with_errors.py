"""
Published hadron masses with uncertainty bands for test validation.

Notes:
- Hadron masses are PDG quantities (not CODATA constants).
- Values are in MeV and keyed by flavor-content strings used by pyhqiv tests.
- Uncertainties here are practical test error bars for the published states in
  this repository's hadron table.
"""

from typing import Dict, Tuple

# flavor_content -> (central_mass_MeV, abs_uncertainty_MeV)
HADRON_MASSES_WITH_ERRORS_MEV: Dict[str, Tuple[float, float]] = {
    "uud": (938.272, 0.001),     # proton
    "udd": (939.565, 0.001),     # neutron
    "uuu": (1232.0, 2.0),        # Delta++
    "ddd": (1232.0, 2.0),        # Delta-
    "uus": (1189.37, 0.07),      # Sigma+
    "uds": (1115.683, 0.006),    # Lambda
    "dds": (1197.45, 0.03),      # Sigma-
    "uss": (1314.86, 0.20),      # Xi0
    "dss": (1321.71, 0.07),      # Xi-
    "udc": (2286.46, 0.14),      # Lambda_c+
    "uuc": (2452.9, 0.4),        # Sigma_c++
    "ddc": (2453.98, 0.16),      # Sigma_c0
    "usc": (2467.9, 0.4),        # Xi_c+
    "dsc": (2470.88, 0.31),      # Xi_c0
    "ssc": (2695.2, 1.7),        # Omega_c0
    "udb": (5619.60, 0.17),      # Lambda_b0
    "uudcc": (4311.9, 0.7),      # Pc+
}


__all__ = ["HADRON_MASSES_WITH_ERRORS_MEV"]
