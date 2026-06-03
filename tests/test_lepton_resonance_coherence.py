"""Observer shell snapshot only (resonance math lives in test_lepton_resonance)."""

from pyhqiv.lepton_resonance_coherence import (
    T_CMB_NATURAL_DEFAULT,
    lepton_shell_coherence,
    observer_shell_index_from_t_cmb,
)
from pyhqiv.lepton_resonance_ladder import M_E, M_MU, M_TAU


def test_coherence_indices_match_ladder() -> None:
    c = lepton_shell_coherence()
    assert c.m_tau == M_TAU and c.m_mu == M_MU and c.m_e == M_E


def test_observer_shell_huge_for_default_cmb() -> None:
    s = observer_shell_index_from_t_cmb(T_CMB_NATURAL_DEFAULT)
    assert s > 1e30
