from pyhqiv.sm_gr_unification import (
    M_Pl_natural,
    M_Z_natural,
    alpha_EM_at_MZ,
    alpha_gut,
    alpha_s_at_MZ,
    m_electron_natural,
    m_neutron_MeV_central,
    m_proton_MeV_central,
    one_over_alpha_EM_at_MZ,
    sin2thetaW_at_MZ,
    sm_constants_at_now,
)


def test_sm_constants_match_lean_witnesses() -> None:
    c = sm_constants_at_now()
    assert c.alpha_GUT == alpha_gut()
    assert c.one_over_alpha_EM == one_over_alpha_EM_at_MZ()
    assert c.alpha_EM == alpha_EM_at_MZ()
    assert c.sin2thetaW == sin2thetaW_at_MZ()
    assert c.alpha_s == alpha_s_at_MZ()
    assert c.M_Pl == M_Pl_natural()
    assert c.M_Z == M_Z_natural()
    assert c.m_electron_natural == m_electron_natural()
    assert c.m_proton_MeV == m_proton_MeV_central()
    assert c.m_neutron_MeV == m_neutron_MeV_central()


def test_basic_numeric_values() -> None:
    assert alpha_gut() == 1.0 / 42.0
    assert one_over_alpha_EM_at_MZ() == 127.9
    assert alpha_EM_at_MZ() == 1.0 / 127.9
    assert sin2thetaW_at_MZ() == 0.23122
    assert alpha_s_at_MZ() == 0.1180

