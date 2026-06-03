"""
Nuclear-related tests for the Lean-witness pyhqiv rebuild.

The full ``pyhqiv.nuclear`` module (HorizonNetwork binding, decay chains, algebraic
fusion) still depends on the extended stack: ``constants``, ``fluid``,
``hqiv_scalings``, ``horizon_network``, ``subatomic``/algebra, etc. That stack is
not part of the slim package; imports are optional below.

This file always tests SM–GR / witness-level nucleon anchors that *are* shipped
(``witnesses.json`` via ``sm_gr_unification``). When the extended stack exists,
the legacy nuclear dynamics tests run as well.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_WITNESS = REPO_ROOT / "data" / "hqiv_witnesses.json"


def _nucleon_mass_keys(w) -> tuple[str, str] | None:
    """Return (proton_key, neutron_key) for whichever schema the JSON uses."""
    d = w.data
    if "derivedProtonMass_MeV" in d and "derivedNeutronMass_MeV" in d:
        return ("derivedProtonMass_MeV", "derivedNeutronMass_MeV")
    if "m_proton_MeV" in d and "m_neutron_MeV" in d:
        return ("m_proton_MeV", "m_neutron_MeV")
    return None


def test_packaged_witnesses_nucleon_schema() -> None:
    """
    Bundled ``witnesses.json`` should carry nucleon masses under one of two schemes:

    - Lean-derived export: ``derivedProtonMass_MeV`` / ``derivedNeutronMass_MeV``
      (what ``sm_gr_unification.m_proton_MeV_central`` expects today), or
    - Legacy anchor keys: ``m_proton_MeV`` / ``m_neutron_MeV``.
    """
    from pyhqiv.lean_witnesses import load_lean_witnesses

    w = load_lean_witnesses()
    keys = _nucleon_mass_keys(w)
    if keys is None:
        pytest.skip(
            "Packaged witnesses.json has no nucleon mass keys; add derived* or m_proton_MeV / m_neutron_MeV"
        )
    pk, nk = keys
    mp = w.get_float(pk)
    mn = w.get_float(nk)
    assert np.isfinite(mp) and mp > 100.0
    assert np.isfinite(mn) and mn > mp


def test_sm_gr_nucleon_masses_consistent_with_witness_loader() -> None:
    """``m_*_MeV_central()`` matches the same keys read from the witness file."""
    from pyhqiv.lean_witnesses import load_lean_witnesses
    from pyhqiv.sm_gr_unification import m_neutron_MeV_central, m_proton_MeV_central

    w = load_lean_witnesses()
    keys = _nucleon_mass_keys(w)
    if keys is None:
        pytest.skip("No nucleon keys in witnesses.json — cannot test sm_gr nucleon accessors")
    pk, nk = keys
    assert m_proton_MeV_central() == w.get_float(pk)
    assert m_neutron_MeV_central() == w.get_float(nk)


def test_neutron_heavier_than_proton() -> None:
    from pyhqiv.lean_witnesses import load_lean_witnesses
    from pyhqiv.sm_gr_unification import m_neutron_MeV_central, m_proton_MeV_central

    if _nucleon_mass_keys(load_lean_witnesses()) is None:
        pytest.skip("No nucleon keys in witnesses.json")
    assert m_neutron_MeV_central() > m_proton_MeV_central()


@pytest.mark.skipif(not DATA_WITNESS.is_file(), reason="data/hqiv_witnesses.json not present")
def test_data_witness_derived_nucleon_keys_when_present() -> None:
    """
    Pure-derived nucleon keys from the data export (if checked in).

    ``derived_nucleon_report`` reads the default packaged JSON; this test
    validates the *data* artifact used by ``test_hqiv_witness_json_derived_payload_only``.
    """
    data = json.loads(DATA_WITNESS.read_text(encoding="utf-8"))
    for key in ("derivedProtonMass_MeV", "derivedNeutronMass_MeV", "derivedDeltaM_MeV"):
        assert key in data
        v = float(data[key])
        assert np.isfinite(v) and v > 0.0
    assert float(data["derivedNeutronMass_MeV"]) > float(data["derivedProtonMass_MeV"])


def _nuclear_stack_available() -> bool:
    try:
        import pyhqiv.constants  # noqa: F401
        import pyhqiv.fluid  # noqa: F401
        import pyhqiv.hqiv_scalings  # noqa: F401
        import pyhqiv.horizon_network  # noqa: F401
        import pyhqiv.nuclear  # noqa: F401
    except ImportError:
        return False
    return True


extended = pytest.mark.skipif(
    not _nuclear_stack_available(),
    reason="extended nuclear stack (constants, horizon_network, …) not installed",
)


@extended
def test_nuclide_from_symbol() -> None:
    from pyhqiv.nuclear import nuclide_from_symbol

    assert nuclide_from_symbol("H") == (1, 0)
    assert nuclide_from_symbol("He") == (2, 2)
    assert nuclide_from_symbol("C") == (6, 6)
    assert nuclide_from_symbol("C", N=8) == (6, 8)


@extended
def test_theta_nuclear_stable_unstable() -> None:
    from pyhqiv.nuclear import theta_nuclear_stable_m, theta_nuclear_unstable_m

    theta_s = theta_nuclear_stable_m(6, 8)
    theta_u = theta_nuclear_unstable_m(6, 8)
    assert theta_s > 0 and theta_u > 0
    assert theta_u <= theta_s + 1e-25


@extended
def test_delta_E_info_mev() -> None:
    from pyhqiv.nuclear import delta_E_info_mev

    theta_s = 1.2e-15
    theta_u = 1.15e-15
    de = delta_E_info_mev(theta_u, theta_s)
    assert de > 0 and np.isfinite(de)


@extended
def test_nuclear_config_c14() -> None:
    from pyhqiv.nuclear import NuclearConfig

    c14 = NuclearConfig(6, 8, "C14")
    assert c14.A == 14
    snaps = c14.allowed_snaps()
    if snaps:
        daughter, _de, mode = snaps[0]
        if mode == "β-":
            assert daughter.P == 7 and daughter.N == 7


@extended
def test_decay_channel_reports_sum_matches_total_rate() -> None:
    from pyhqiv.nuclear import NuclearConfig

    n = NuclearConfig(0, 1)
    ch = n.decay_channel_reports()
    lam_sum = sum(c.decay_rate_per_s for c in ch)
    lam_tot = n.decay_rate_per_s()
    assert ch
    np.testing.assert_allclose(lam_sum, lam_tot, rtol=1e-9)
    br = sum(c.branching_ratio for c in ch)
    np.testing.assert_allclose(br, 1.0, rtol=1e-9)


@extended
def test_nuclear_system_report_he4() -> None:
    from pyhqiv.nuclear import NuclearConfig, nuclear_system_report

    rep = nuclear_system_report(2, 2, name="He-4")
    assert rep.composition.protons == 2 and rep.composition.neutrons == 2
    assert rep.structure.mass_number == 4
    assert rep.structure.geometry in (
        "tetrahedron_candidate",
        "two_body_segment",
        "many_body_minimized",
    )
    assert rep.binding_energy_mev == NuclearConfig(2, 2).binding_energy_mev
    assert np.isfinite(rep.binding_energy_mev_algebraic)
    modes = {c.mode for c in rep.decay_channels}
    assert "β-" not in modes and "β+" not in modes


@extended
def test_nuclear_system_report_extra_baryons_rejected() -> None:
    from pyhqiv.nuclear import nuclear_system_report

    with pytest.raises(ValueError, match="extra_baryons"):
        nuclear_system_report(1, 1, extra_baryons={"Lambda": 1})


@extended
def test_free_neutron_prefers_beta_minus() -> None:
    from pyhqiv.nuclear import NuclearConfig

    neutron = NuclearConfig(0, 1, "n")
    snaps = neutron.allowed_snaps()
    assert any(
        mode == "β-" and daughter.P == 1 and daughter.N == 0
        for daughter, _, mode in snaps
    )


@extended
def test_tritium_beta_minus_to_he3() -> None:
    from pyhqiv.nuclear import NuclearConfig

    tritium = NuclearConfig(1, 2, "H3")
    snaps = tritium.allowed_snaps()
    assert any(
        mode == "β-" and daughter.P == 2 and daughter.N == 1
        for daughter, _, mode in snaps
    )


@extended
def test_he4_has_no_beta_channel() -> None:
    from pyhqiv.nuclear import NuclearConfig

    he4 = NuclearConfig(2, 2, "He4")
    modes = {mode for _, _, mode in he4.allowed_snaps()}
    assert "β-" not in modes
    assert "β+" not in modes


@extended
def test_half_life_nuclide_hqiv() -> None:
    from pyhqiv.nuclear import half_life_nuclide_hqiv

    t2 = half_life_nuclide_hqiv(6, 8)
    if t2 is not None:
        assert t2 > 0 and np.isfinite(t2)


@extended
def test_decay_chain() -> None:
    from pyhqiv.nuclear import decay_chain

    chain = decay_chain(6, 8, max_steps=5)
    assert chain[0] == (6, 8)
    assert len(chain) >= 1


@extended
def test_decay_chain_nuclide_hqiv() -> None:
    from pyhqiv.nuclear import decay_chain_nuclide_hqiv

    t_half, mode, dP, dN, chain2 = decay_chain_nuclide_hqiv(6, 8)
    assert chain2[0] == (6, 8)
    if len(chain2) >= 2 and mode == "β-":
        assert dP == 7 and dN == 7


@extended
def test_phase_lift_gives_nonzero_trace() -> None:
    from pyhqiv.algebra import OctonionHQIVAlgebra
    from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants
    from pyhqiv.nuclear import build_nucleon_matrix_with_phase

    L = get_hqiv_nuclear_constants(2.725)["LATTICE_BASE_M"]
    Mp = build_nucleon_matrix_with_phase(True, L)
    Mn = build_nucleon_matrix_with_phase(False, L)
    alg = OctonionHQIVAlgebra(verbose=False)
    D = alg.Delta
    assert abs(np.trace(Mp @ D)) > 1e-10
    assert abs(np.trace(Mn @ D)) > 1e-10


@extended
def test_binding_energy_mev_algebraic_returns_float() -> None:
    from pyhqiv.nuclear import binding_energy_mev_algebraic

    b = binding_energy_mev_algebraic(2, 2)
    assert isinstance(b, (int, float))
    assert np.isfinite(b)


@extended
def test_stable_nuclide_zero_decay_rate() -> None:
    from pyhqiv.nuclear import NuclearConfig

    n14 = NuclearConfig(7, 7, "N14")
    rate = n14.decay_rate_per_s()
    t_half = n14.half_life_s()
    assert rate == 0.0 or (t_half is not None and t_half > 3.15e7)


@extended
def test_binding_gold_standard_2h_4he() -> None:
    from pyhqiv.nuclear import BindingResult, binding_energy_mev_functional

    for P, N in [(1, 1), (2, 2)]:
        res = binding_energy_mev_functional(P, N)
        assert isinstance(res, BindingResult)
        assert res.positions_m.shape[0] == P + N
        assert np.isfinite(res.theta_p_m) and res.theta_p_m > 0
        assert np.isfinite(res.theta_n_m) and res.theta_n_m > 0
        assert np.isfinite(res.B_mev) and np.isfinite(res.E_free_mev) and np.isfinite(res.E_bound_mev)
        np.testing.assert_allclose(res.B_mev, res.E_free_mev - res.E_bound_mev, rtol=1e-9)
        if res.B_mev > 0:
            assert res.E_bound_mev < res.E_free_mev


@extended
def test_binding_2h_4he_nucleon_level() -> None:
    from pyhqiv.nuclear import binding_energy_mev_nucleon_level

    for P, N in [(1, 1), (2, 2)]:
        B, E_free, E_bound = binding_energy_mev_nucleon_level(P, N)
        assert E_bound < E_free
        assert B > 0 and np.isfinite(B)


@extended
def test_binding_2h_4he_quark_level() -> None:
    from pyhqiv.nuclear import binding_energy_mev_quark_level

    for P, N in [(1, 1), (2, 2)]:
        B, E_free, E_bound = binding_energy_mev_quark_level(P, N)
        assert E_bound < E_free
        assert B > 0 and np.isfinite(B)
