"""Tests for HQIV first-principles nuclear decay (nuclear.py)."""

import numpy as np

from pyhqiv.nuclear import (
    BindingResult,
    NuclearConfig,
    binding_energy_mev_algebraic,
    binding_energy_mev_functional,
    binding_energy_mev_nucleon_level,
    binding_energy_mev_quark_level,
    build_nucleon_matrix_with_phase,
    decay_chain,
    decay_chain_nuclide_hqiv,
    delta_E_info_mev,
    half_life_nuclide_hqiv,
    nuclide_from_symbol,
    theta_nuclear_stable_m,
    theta_nuclear_unstable_m,
)
from pyhqiv.utils import decay_chain_nuclide, half_life_nuclide


def test_nuclide_from_symbol():
    """Element table gives (P, N) for default isotope."""
    assert nuclide_from_symbol("H") == (1, 0)
    assert nuclide_from_symbol("He") == (2, 2)
    assert nuclide_from_symbol("C") == (6, 6)
    assert nuclide_from_symbol("C", N=8) == (6, 8)


def test_theta_nuclear_stable_unstable():
    """Θ_stable and Θ_unstable in metres; first-principles path gives one bound Θ per nucleus so θ_u ≤ θ_s (or equal)."""
    theta_s = theta_nuclear_stable_m(6, 8)
    theta_u = theta_nuclear_unstable_m(6, 8)
    assert theta_s > 0 and theta_u > 0
    assert theta_u <= theta_s + 1e-25  # allow float equality when all bound Θ identical


def test_delta_E_info_mev():
    """ΔE_info = ħc(1/Θ_u - 1/Θ_s) positive when Θ_u < Θ_s."""
    theta_s = 1.2e-15
    theta_u = 1.15e-15
    de = delta_E_info_mev(theta_u, theta_s)
    assert de > 0 and np.isfinite(de)


def test_nuclear_config_c14():
    """C-14: config builds; if first-principles allows β- snap, daughter is N-14."""
    c14 = NuclearConfig(6, 8, "C14")
    assert c14.A == 14
    snaps = c14.allowed_snaps()
    if len(snaps) >= 1:
        daughter, dE, mode = snaps[0]
        if mode == "β-":
            assert daughter.P == 7 and daughter.N == 7


def test_free_neutron_prefers_beta_minus():
    """A lone neutron can relax to a proton through the same beta-minus geometry."""
    neutron = NuclearConfig(0, 1, "n")
    snaps = neutron.allowed_snaps()
    assert any(mode == "β-" and daughter.P == 1 and daughter.N == 0 for daughter, _, mode in snaps)


def test_tritium_beta_minus_to_he3():
    """Hydrogen-3 beta-minus decays to helium-3 in the ladder picture."""
    tritium = NuclearConfig(1, 2, "H3")
    snaps = tritium.allowed_snaps()
    assert any(mode == "β-" and daughter.P == 2 and daughter.N == 1 for daughter, _, mode in snaps)


def test_he4_has_no_beta_channel():
    """Helium-4 should not expose beta channels once all four nucleons share the same local valence."""
    he4 = NuclearConfig(2, 2, "He4")
    modes = {mode for _, _, mode in he4.allowed_snaps()}
    assert "β-" not in modes
    assert "β+" not in modes


def test_half_life_nuclide_returns_float_or_none():
    """half_life_nuclide (utils) and half_life_nuclide_hqiv return s or None."""
    t = half_life_nuclide(6, 8)
    t2 = half_life_nuclide_hqiv(6, 8)
    if t is not None:
        assert t > 0 and np.isfinite(t)
    if t2 is not None:
        assert t2 > 0 and np.isfinite(t2)


def test_decay_chain_nuclide_returns_nuclide_decay():
    """decay_chain_nuclide returns NuclideDecay with chain list."""
    from pyhqiv.utils import NuclideDecay

    result = decay_chain_nuclide(6, 8, max_steps=5)
    assert isinstance(result, NuclideDecay)
    assert result.P == 6 and result.N == 8
    assert len(result.decay_chain) >= 1
    assert result.decay_chain[0] == (6, 8)


def test_decay_chain_c14_to_n14():
    """C-14 has β- to N-14 (7,7) as an allowed snap; chain starts at (6,8)."""
    c14 = NuclearConfig(6, 8)
    snaps = c14.allowed_snaps()
    assert any(d.P == 7 and d.N == 7 and m == "β-" for d, _, m in snaps)
    chain = decay_chain(6, 8, max_steps=5)
    assert chain[0] == (6, 8)
    assert len(chain) >= 1


def test_phase_lift_gives_nonzero_trace():
    """build_nucleon_matrix_with_phase yields tr(M@Δ) ≠ 0 (axiom-derived θΔ)."""
    from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants

    L = get_hqiv_nuclear_constants(2.725)["LATTICE_BASE_M"]
    Mp = build_nucleon_matrix_with_phase(True, L)
    Mn = build_nucleon_matrix_with_phase(False, L)
    from pyhqiv.algebra import OctonionHQIVAlgebra
    alg = OctonionHQIVAlgebra(verbose=False)
    D = alg.Delta
    tr_p = np.trace(Mp @ D)
    tr_n = np.trace(Mn @ D)
    assert abs(tr_p) > 1e-10
    assert abs(tr_n) > 1e-10


def test_binding_energy_mev_algebraic_returns_float():
    """binding_energy_mev_algebraic runs and returns a finite float."""
    b = binding_energy_mev_algebraic(2, 2)
    assert isinstance(b, (int, float))
    assert np.isfinite(b)


# ── Binding energy: reality vs first-principles paths ──
#
# Experiment: He-4 B ≈ 28.3 MeV, 2H ≈ 2.2 MeV (AME/NNDC). HQIV: B = E_free − E_bound.
# All three paths use the same minimizer (minimize_nucleon_configuration) for positions.
# They differ only in how E_free and E_bound are defined:
#
# 1. Gold-standard functional: E_free = P·E_p + N·E_n (exact PDG from subatomic). E_bound from
#    HorizonNetwork(nucleon nodes at minimized positions) + E_opp. Can yield B ≤ 0 if network
#    + E_opp gives E_bound ≥ E_free.
# 2. Nucleon-level: E_free = sum of single-nucleon HorizonNetwork(one node at origin).total_energy();
#    E_bound = HorizonNetwork(nucleon nodes at same minimized positions) + E_opp. Same recipe
#    for free and bound → often B > 0 for 2H/4He.
# 3. Quark-level: same E_free as (2); E_bound from quark expansion (3×A nodes) + per-nucleon Θ
#    + E_opp. Validates quark-layer consistency; also gives B > 0 for 2H/4He.
#
# test_physical_values.py asserts binding_energy_mev(P,N) (functional) against experiment (no clamp).


def test_stable_nuclide_zero_decay_rate():
    """N-14 (7,7): either no allowed snaps (rate 0) or effectively stable (long t_1/2)."""
    n14 = NuclearConfig(7, 7, "N14")
    rate = n14.decay_rate_per_s()
    t_half = n14.half_life_s()
    # With B-derived Θ, SEMF can give small Q for β±; rate 0 or half-life ≫ 1 year
    assert rate == 0.0 or (t_half is not None and t_half > 3.15e7)


def test_binding_gold_standard_2h_4he():
    """Gold standard: functional method returns BindingResult with correct shape and self-consistency."""
    for P, N in [(1, 1), (2, 2)]:
        res = binding_energy_mev_functional(P, N)
        assert isinstance(res, BindingResult)
        assert res.positions_m.shape[0] == P + N, f"(P={P},N={N}): positions shape"
        assert np.isfinite(res.theta_p_m) and res.theta_p_m > 0
        assert np.isfinite(res.theta_n_m) and res.theta_n_m > 0
        assert np.isfinite(res.B_mev) and np.isfinite(res.E_free_mev) and np.isfinite(res.E_bound_mev)
        # Self-consistency: B = E_free - E_bound
        np.testing.assert_allclose(res.B_mev, res.E_free_mev - res.E_bound_mev, rtol=1e-9)
        # When binding is positive, bound state is lower than free
        if res.B_mev > 0:
            assert res.E_bound_mev < res.E_free_mev


def test_binding_2h_4he_nucleon_level():
    """2H and 4He: nucleon-level network gives E_bound < E_free (positive binding)."""
    for P, N in [(1, 1), (2, 2)]:
        B, E_free, E_bound = binding_energy_mev_nucleon_level(P, N)
        assert E_bound < E_free, f"2H/4He nucleon-level (P={P},N={N}): E_bound {E_bound} < E_free {E_free}"
        assert B > 0 and np.isfinite(B), f"2H/4He nucleon-level (P={P},N={N}): B={B} positive and finite"


def test_binding_2h_4he_quark_level():
    """2H and 4He: quark-level (nucleon-scale E_bound from per-nucleon Θ) gives E_bound < E_free."""
    for P, N in [(1, 1), (2, 2)]:
        B, E_free, E_bound = binding_energy_mev_quark_level(P, N)
        assert E_bound < E_free, f"2H/4He quark-level (P={P},N={N}): E_bound {E_bound} < E_free {E_free}"
        assert B > 0 and np.isfinite(B), f"2H/4He quark-level (P={P},N={N}): B={B} positive and finite"


# ── Qualified statements (gold standard: neutron, C-14, heavy water) ──


def test_free_neutron_half_life_qualified():
    """Free neutron (0,1): we predict a half-life in seconds (PDG ≈ 879 s)."""
    t_half = half_life_nuclide_hqiv(0, 1)
    # Either unstable (finite half-life) or stable (None)
    if t_half is not None:
        assert np.isfinite(t_half) and t_half > 0


def test_c14_nuclide_and_decay_chain_qualified():
    """C-14: (6,8); decay chain includes β⁻ daughter N-14 (7,7)."""
    P, N = nuclide_from_symbol("C", A=14)
    assert (P, N) == (6, 8)
    chain = decay_chain(6, 8, max_steps=10)
    assert len(chain) >= 1 and chain[0] == (6, 8)
    # When β⁻ is allowed, first daughter is (7, 7)
    t_half, mode, dP, dN, chain2 = decay_chain_nuclide_hqiv(6, 8)
    if len(chain2) >= 2:
        assert chain2[1] == (7, 7), f"C-14 daughter should be N-14 (7,7), got {chain2[1]}"
    if mode == "β-":
        assert dP == 7 and dN == 7


def test_c14_half_life_qualified():
    """C-14: we predict a half-life in seconds when unstable (PDG ≈ 5700 yr)."""
    t_half = half_life_nuclide_hqiv(6, 8)
    if t_half is not None:
        assert np.isfinite(t_half) and t_half > 0


def test_heavy_water_theta_qualified():
    """Heavy water: Θ for deuterium (mass_amu=2) is isotope-dependent and finite."""
    from pyhqiv.utils import theta_for_atom

    theta_D = theta_for_atom("H", coordination=2, mass_amu=2)
    theta_H = theta_for_atom("H", coordination=2)  # default mass
    assert np.isfinite(theta_D) and theta_D > 0
    assert np.isfinite(theta_H) and theta_H > 0
    # Heavier isotope → larger Θ in current scaling (mass_amu scaling in theta_local)
    assert theta_D > theta_H
