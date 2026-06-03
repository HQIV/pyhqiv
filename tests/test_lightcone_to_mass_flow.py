"""
End-to-end regression: light cone → QCD shell temperature → electron now → ADM lapse → masses.

This mirrors the conceptual order in Lean:

- ``Hqiv.Geometry.OctonionicLightCone`` (``qcdShell``, ``referenceM``, ladder)
- ``Hqiv.Geometry.AuxiliaryField`` (``T``, ``phi_of_shell``)
- ``Hqiv.Geometry.HQVMetric`` (``HQVM_lapse``)
- ``Hqiv.Physics.SM_GR_Unification`` / mass ladder (electron anchor)

``now_setters`` implements the corrected policy: the discrete cutoff is the **electron
horizon shell** (witness ``m_now_electron_shell``), not a CMB temperature anchor.
The electron mass enters as the Lean **witness** used by ``m_electron_natural`` and
as the normalization check for ``sm_mass_from_geometry_eV("electron")``.
"""

from __future__ import annotations

import math

from pyhqiv.auxiliary_field import phi_of_shell, shell_temperature
from pyhqiv.lean_witnesses import load_lean_witnesses
from pyhqiv.lightcone import (
    lattice_step_count,
    omega_k_at_horizon,
    qcd_shell,
    reference_m,
    x_over_theta_from_horizons,
)
from pyhqiv.metric import hqvm_lapse
from pyhqiv.now import build_preliminary_now
from pyhqiv import now_setters
from pyhqiv.sm_gr_unification import m_electron_natural
from pyhqiv.sm_mass_ladder import natural_mass_to_eV, sm_mass_from_geometry_eV


def _witness_has_electron_resonance_mass() -> bool:
    """Packaged Lean export may include ``m_e_from_resonance`` (GeV) for the ladder fast path."""
    return "m_e_from_resonance" in load_lean_witnesses().data


def test_x_over_theta_from_horizons_ratio() -> None:
    """Lean Planck-distance ratio (n+1)/(N+1); matches OctonionicLightCone earmark."""
    assert x_over_theta_from_horizons(0, 500) > 0.0
    assert x_over_theta_from_horizons(0, 500) <= 1.0
    assert abs(x_over_theta_from_horizons(500, 500) - 1.0) < 1e-10
    assert x_over_theta_from_horizons(100, 500) < 1.0
    assert x_over_theta_from_horizons(100, 500) > 0.0


def test_omega_k_at_reference_shell_is_one() -> None:
    """At n = horizon, integral ratio and x/θ are both 1 ⇒ Ω_k = 1."""
    ref = reference_m()
    assert abs(omega_k_at_horizon(ref, ref) - 1.0) < 1e-10


def test_lightcone_qcd_shell_lock_in_reference_m() -> None:
    """QCD shell + discrete step count matches ``referenceM`` (Lean ``referenceM``)."""
    m_qcd = qcd_shell()
    steps = lattice_step_count()
    assert reference_m() == m_qcd + steps


def test_t_qcd_from_lightcone_shell() -> None:
    """
    QCD transition temperature in natural Planck units: ``T(m) = T_Pl/(m+1)`` at ``m = qcdShell``.

    Lean: ``Hqiv.Geometry.AuxiliaryField.T`` on the QCD shell.
    """
    m_qcd = qcd_shell()
    t_qcd = shell_temperature(m_qcd)
    assert t_qcd > 0.0 and math.isfinite(t_qcd)
    # m_qcd = 1 ⇒ T = 1/2 in T_Pl = 1 units
    assert math.isclose(t_qcd, 1.0 / float(m_qcd + 1), rel_tol=0.0, abs_tol=1e-15)


def test_phi_qcd_consistent_with_t_qcd() -> None:
    """φ = 2/T on the QCD shell (``phi_of_shell``)."""
    m_qcd = qcd_shell()
    t_qcd = shell_temperature(m_qcd)
    phi_qcd = phi_of_shell(m_qcd)
    assert math.isclose(phi_qcd, 2.0 / t_qcd, rel_tol=1e-12, abs_tol=0.0)


def test_preliminary_now_bundles_qcd_and_reference_curvature() -> None:
    """``build_preliminary_now`` carries QCD shell and curvature integral at ``reference_m``."""
    bundle = build_preliminary_now()
    assert bundle.qcd_transition_shell == qcd_shell()
    assert bundle.reference_shell == reference_m()
    assert bundle.reference_curvature_integral > 0.0


def test_now_setter_electron_shell_phi_matches_auxiliary_field() -> None:
    """
    ``now_set_from_electron_horizon`` sets the global cutoff; ``phi_now`` agrees with
    ``phi_of_shell`` at that index (electron mass sets *which shell* via witness, not T_CMB).
    """
    w = load_lean_witnesses()
    m_e = w.get_int("m_now_electron_shell")
    old_m = now_setters.m_now
    try:
        geom = now_setters.now_set_from_electron_horizon(m_e)
        assert geom.m_now == m_e
        assert math.isclose(geom.phi_now, phi_of_shell(m_e), rel_tol=1e-12, abs_tol=0.0)
    finally:
        now_setters.m_now = old_m
        now_setters.rescale_geometry()


def test_adm_lapse_at_electron_now_uses_auxiliary_phi() -> None:
    """
    ADM lapse ``N = 1 + Φ_N + φ t`` (Lean ``HQVM_lapse``): at t=0 and Φ_N=0, N=1;
    with φ = φ_now from the electron shell, N grows linearly in t.
    """
    w = load_lean_witnesses()
    m_e = w.get_int("m_now_electron_shell")
    old_m = now_setters.m_now
    try:
        phi_aux = now_setters.now_set_from_electron_horizon(m_e).phi_now
        n0 = hqvm_lapse(0.0, phi_aux, 0.0)
        assert math.isclose(n0, 1.0, rel_tol=0.0, abs_tol=1e-15)
        t = 0.01
        nt = hqvm_lapse(0.0, phi_aux, t)
        assert math.isclose(nt, 1.0 + phi_aux * t, rel_tol=1e-12, abs_tol=0.0)
        assert nt > 1.0
    finally:
        now_setters.m_now = old_m
        now_setters.rescale_geometry()


def test_masses_from_geometry_after_electron_now_anchor() -> None:
    """
    After anchoring ``now`` to the electron shell witness:

    - The **electron anchor** is always ``m_electron_natural`` from SM–GR witnesses
      (Planck-normalized); converted to eV it must match ``m_electron_MeV``.
    - If ``m_e_from_resonance`` is present in ``witnesses.json``, ``sm_mass_from_geometry_eV``
      uses that resonance output (current ``sm_mass_ladder`` fast path for charged fermions).
    """
    w = load_lean_witnesses()
    m_e = w.get_int("m_now_electron_shell")
    old_m = now_setters.m_now
    try:
        now_setters.now_set_from_electron_horizon(m_e)

        anchor_eV = natural_mass_to_eV(m_electron_natural())
        expected_eV = w.get_float("m_electron_MeV") * w.get_float("MEV_TO_EV")
        assert abs(anchor_eV - expected_eV) < 1e-6

        if _witness_has_electron_resonance_mass():
            got_eV = sm_mass_from_geometry_eV("electron")
            assert abs(got_eV - expected_eV) < 1e-6
    finally:
        now_setters.m_now = old_m
        now_setters.rescale_geometry()


def test_full_chain_lightcone_tqcd_now_lapse_electron_mass() -> None:
    """
    Single narrative test: combinatorial QCD shell → T_qcd → (witness) electron shell
    → ADM lapse at t=0 → electron mass anchor in eV (witness-normalized natural units).
    """
    w = load_lean_witnesses()
    m_qcd = qcd_shell()
    t_qcd = shell_temperature(m_qcd)
    assert t_qcd == 1.0 / float(m_qcd + 1)

    m_e = w.get_int("m_now_electron_shell")
    old_m = now_setters.m_now
    try:
        phi_aux = now_setters.now_set_from_electron_horizon(m_e).phi_now
        assert phi_aux == phi_of_shell(m_e)

        n = hqvm_lapse(0.0, phi_aux, 0.0)
        assert n == 1.0

        expected_eV = w.get_float("m_electron_MeV") * w.get_float("MEV_TO_EV")
        assert abs(natural_mass_to_eV(m_electron_natural()) - expected_eV) < 1e-6
        if _witness_has_electron_resonance_mass():
            assert (
                abs(sm_mass_from_geometry_eV("electron") - expected_eV) < 1e-6
            )
    finally:
        now_setters.m_now = old_m
        now_setters.rescale_geometry()
