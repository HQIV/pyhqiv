"""
Regime façades: galactic (G_eff, lapse), horizon / Compton bridge, quantum (SO(8) evolution).

Each submodule cites the relevant ``HQIV_LEAN`` entry points in its docstring.
"""

from pyhqiv.regimes.blackhole import (
    blackhole_compton_quarter_period_s,
    blackhole_horizon_quarter_angle_rad,
    blackhole_reference_shell_m,
)
from pyhqiv.regimes.galactic import galactic_gamma_hqiv, galactic_g_eff, galactic_metric_summary
from pyhqiv.regimes.quantum import (
    born_probs_from_real_state,
    evolve_so8_carrier_expm,
    evolve_so8_vector_expm,
    quantum_lepton_coherence_snapshot,
)

__all__ = [
    "blackhole_compton_quarter_period_s",
    "blackhole_horizon_quarter_angle_rad",
    "blackhole_reference_shell_m",
    "born_probs_from_real_state",
    "evolve_so8_carrier_expm",
    "evolve_so8_vector_expm",
    "galactic_gamma_hqiv",
    "galactic_g_eff",
    "galactic_metric_summary",
    "quantum_lepton_coherence_snapshot",
]
