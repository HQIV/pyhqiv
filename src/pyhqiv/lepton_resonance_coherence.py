"""
Non-resonance **observer** context (same split as Lean): CMB temperature → real shell index;
``m_now`` from witnesses for ``sm_mass_geometry_factor`` — see ``sm_mass_ladder``.

**Resonance masses and ``GlobalDetuning`` math are only in** ``lepton_resonance_ladder``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyhqiv.lepton_resonance_ladder import M_E, M_MU, M_TAU
from pyhqiv.auxiliary_field import phi_of_shell
from pyhqiv.now import shell_index_for_temperature
from pyhqiv.now_setters import m_now

from pyhqiv.scale_witness import load_local_conditions as _lc
T_CMB_NATURAL_DEFAULT = float(_lc()["T_CMB_NATURAL_DEFAULT"])


@dataclass(frozen=True)
class LeptonShellCoherence:
    m_tau: int
    m_mu: int
    m_e: int
    phi_tau: float
    phi_mu: float
    phi_e: float
    phase_lift_tau: float
    phase_lift_mu: float
    phase_lift_e: float
    m_now_electron_horizon: int
    observer_shell_cmb: float


def observer_shell_index_from_t_cmb(t_cmb_natural: float = T_CMB_NATURAL_DEFAULT) -> float:
    return shell_index_for_temperature(t_cmb_natural)


def resonance_shell_temperature(m: int) -> float:
    return 1.0 / float(m + 1)


def lepton_shell_coherence(
    *,
    t_cmb_natural: float = T_CMB_NATURAL_DEFAULT,
) -> LeptonShellCoherence:
    pt = phi_of_shell(M_TAU)
    pm = phi_of_shell(M_MU)
    pe = phi_of_shell(M_E)
    return LeptonShellCoherence(
        m_tau=M_TAU,
        m_mu=M_MU,
        m_e=M_E,
        phi_tau=pt,
        phi_mu=pm,
        phi_e=pe,
        phase_lift_tau=pt / 6.0,
        phase_lift_mu=pm / 6.0,
        phase_lift_e=pe / 6.0,
        m_now_electron_horizon=int(m_now),
        observer_shell_cmb=observer_shell_index_from_t_cmb(t_cmb_natural),
    )


__all__ = [
    "T_CMB_NATURAL_DEFAULT",
    "LeptonShellCoherence",
    "lepton_shell_coherence",
    "observer_shell_index_from_t_cmb",
    "resonance_shell_temperature",
]
