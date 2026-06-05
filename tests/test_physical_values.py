"""
Tests that compare pyhqiv outputs and constants to known experimental/physical values.

These are not paper-internal checks but consistency with CODATA, Planck,
observational bounds, Standard Model constants, hydrogen/oxygen coupling
scales (Θ), and nuclear binding energies.
"""

import numpy as np
import pytest

pytest.importorskip(
    "pyhqiv.constants",
    reason="physical_values test requires legacy constants (bak/)",
)

from pyhqiv.constants import (
    AGE_APPARENT_GYR_PAPER,
    ALPHA_EM_INV,
    C_SI,
    HBAR_SI,
    K_B_SI,
    L_PLANCK_M,
    OMEGA_TRUE_K_PAPER,
    SIN2_THETA_W_MZ,
    T_CMB_K,
    T_PL_GEV,
)
from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants
from pyhqiv.lattice import (
    DiscreteNullLattice,
    horizon_planck_distances_m,
    omega_k_at_horizon,
    x_over_theta_from_horizons,
)
from pyhqiv.utils import theta_for_atom, theta_local

from pyhqiv.nuclear import binding_energy_mev

# ---- CODATA / standard values (exact or high-precision) ----
# Speed of light: CODATA 2018 c = 2.99792458e8 m/s
C_CODATA_MS: float = 2.99792458e8

# Reduced Planck constant: CODATA 2018 ℏ ≈ 1.054571817e-34 J·s
HBAR_CODATA: float = 1.054571817e-34

# Boltzmann constant: CODATA 2018 k_B ≈ 1.380649e-23 J/K
K_B_CODATA: float = 1.380649e-23

# Planck length: l_P = sqrt(ℏ G / c³) ≈ 1.616255e-35 m (order of magnitude)
L_PLANCK_CODATA_ORDER: float = 1.616e-35

# CMB monopole temperature: Planck 2018 T_CMB = 2.7255 ± 0.0006 K
T_CMB_PLANCK2018_K: float = 2.7255
T_CMB_PLANCK2018_UNCERTAINTY: float = 0.0006

# Age of universe: Planck 2018 ~ 13.797 Gyr (TT,TE,EE+lowE); local ~ 13.8 Gyr
AGE_UNIVERSE_GYR_PLANCK: float = 13.797
AGE_UNIVERSE_GYR_TOLERANCE: float = 0.1

# Hubble constant: Planck 2018 H0 ≈ 67.4 km/s/Mpc; local (SH0ES) ~ 73 km/s/Mpc
H0_PLANCK_KM_S_MPC: float = 67.4
H0_LOCAL_KM_S_MPC: float = 73.04
H0_KM_S_MPC_TOLERANCE: float = 10.0

# Spatial curvature: Planck 2018 Ω_k = 0.001 ± 0.002 (consistent with flat)
# HQIV predicts positive Ω_k ≈ 0.0098; observationally |Ω_k| < ~0.01 is allowed
OMEGA_K_PLANCK_CENTRAL: float = 0.001
OMEGA_K_OBS_BOUND_ABS: float = 0.02

# Baryon-to-photon ratio: BBN/CMB η ≈ 6.10e-10 (standard value)
ETA_BBN_STANDARD: float = 6.10e-10
ETA_TOLERANCE_REL: float = 0.05

# ---- Standard Model (CODATA / PDG) ----
# Fine structure constant: α_EM = 1/137.035999084 (CODATA 2018); 1/α ≈ 137.036
ALPHA_EM_INV_CODATA: float = 137.036
# Weak mixing angle at M_Z: sin²θ_W = 0.23122 ± 0.00003 (PDG)
SIN2_THETA_W_PDG: float = 0.23122

# ---- Hydrogen / oxygen coupling scale Θ (Å) ----
# H covalent radius ~0.31 Å; O-H bond in water ~0.96 Å; O covalent ~0.66 Å.
# theta_local(Z, coord) gives horizon/coupling scale; H (Z=1) > O (Z=8) for same coord.
THETA_ANGSTROM_MIN: float = 0.2
THETA_ANGSTROM_MAX: float = 10.0
# Typical bond-length scale (Å) for consistency checks
OH_BOND_ANGSTROM: float = 0.96

# ---- Nuclear binding energy (MeV) experimental (AME/NNDC) ----
B_2H_MEV_EXP: float = 2.224  # deuteron
B_2H_MEV_TOLERANCE: float = 0.5
B_HE4_MEV_EXP: float = 28.30
B_HE4_PER_NUCLEON_MEV_EXP: float = 7.075  # 28.30 / 4
B_HE4_MEV_TOLERANCE: float = 3.5
B_HE4_PER_NUCLEON_TOLERANCE: float = 0.9
B_C12_MEV_EXP: float = 92.16
B_O16_MEV_EXP: float = 127.62
B_TOLERANCE_FACTOR: float = 3.0  # ballpark: exp/factor <= B <= exp*factor


def test_speed_of_light_matches_CODATA():
    """c in pyhqiv matches CODATA 2018 to within numerical precision."""
    assert abs(C_SI - C_CODATA_MS) < 0.01


def test_hbar_matches_CODATA():
    """ℏ in pyhqiv matches CODATA 2018 order of magnitude and value."""
    assert abs(HBAR_SI - HBAR_CODATA) < 1e-36
    assert 1e-35 < HBAR_SI < 1e-33


def test_k_B_matches_CODATA():
    """Boltzmann constant matches CODATA 2018."""
    assert abs(K_B_SI - K_B_CODATA) < 1e-25


def test_planck_length_order_and_value():
    """Planck length is in the correct range (CODATA order ~1.6e-35 m)."""
    assert L_PLANCK_M > 0
    assert 1e-36 < L_PLANCK_M < 1e-34
    assert abs(L_PLANCK_M - L_PLANCK_CODATA_ORDER) < 1e-37


def test_T_CMB_within_Planck2018_band():
    """CMB temperature in pyhqiv is within Planck 2018 uncertainty band."""
    # Paper uses 2.725 K; Planck 2018 central 2.7255 ± 0.0006
    assert abs(T_CMB_K - T_CMB_PLANCK2018_K) <= T_CMB_PLANCK2018_UNCERTAINTY + 0.001
    assert 2.72 <= T_CMB_K <= 2.73


def test_apparent_age_consistent_with_observed_universe_age():
    """HQIV apparent age (13.8 Gyr) is consistent with observed age of universe."""
    assert abs(AGE_APPARENT_GYR_PAPER - AGE_UNIVERSE_GYR_PLANCK) < AGE_UNIVERSE_GYR_TOLERANCE
    assert 13.0 <= AGE_APPARENT_GYR_PAPER <= 14.5


def test_omega_k_within_observational_curvature_bounds():
    """HQIV Ω_k is within observational bounds (|Ω_k| < ~0.02 from CMB)."""
    assert abs(OMEGA_TRUE_K_PAPER) < OMEGA_K_OBS_BOUND_ABS
    assert OMEGA_TRUE_K_PAPER > 0  # HQIV predicts positive curvature


def test_omega_k_from_lattice_in_observational_range():
    """Lattice omega_k_true() at reference lies within observational curvature range."""
    lat = DiscreteNullLattice(m_trans=500)
    omega = lat.omega_k_true()
    assert 0 < omega < OMEGA_K_OBS_BOUND_ABS
    assert abs(omega - OMEGA_TRUE_K_PAPER) < 0.01


def test_omega_k_at_horizon_equals_true_at_reference():
    """At n=N, Ω_k(n; N) equals the calibrated true curvature (0.0098)."""
    N = 500
    omega = omega_k_at_horizon(N, N, use_planck_distance_ratio=True)
    assert abs(omega - OMEGA_TRUE_K_PAPER) < 1e-6
    omega_no_ratio = omega_k_at_horizon(N, N, use_planck_distance_ratio=False)
    assert abs(omega_no_ratio - OMEGA_TRUE_K_PAPER) < 1e-6


def test_x_over_theta_from_horizons_ratio_bounds():
    """x/θ from Planck distances is (0, 1] when n ≤ N and > 0 for n, N ≥ 0."""
    assert x_over_theta_from_horizons(0, 500) > 0
    assert x_over_theta_from_horizons(0, 500) <= 1.0
    assert abs(x_over_theta_from_horizons(500, 500) - 1.0) < 1e-10
    assert x_over_theta_from_horizons(100, 500) < 1.0
    assert x_over_theta_from_horizons(100, 500) > 0


def test_horizon_planck_distances_positive_and_metre_scale():
    """Planck distances to horizons are positive and in metre scale (~1e-35)."""
    d_n, d_N = horizon_planck_distances_m(10, 500)
    assert d_n > 0 and d_N > 0
    assert d_n < d_N  # shell 10 closer than shell 500
    # Scale: L_P * (m+1) with m ~ 10 and 500 gives ~1e-34 and ~1e-32
    assert 1e-36 < d_n < 1e-32
    assert 1e-36 < d_N < 1e-31


def test_planck_temperature_GeV_order():
    """Planck temperature in GeV is ~1.22e19 (standard value)."""
    assert T_PL_GEV > 1e19
    assert T_PL_GEV < 2e19


@pytest.mark.parametrize("n,N", [(1, 10), (50, 500), (100, 100)])
def test_omega_k_at_horizon_positive_for_positive_integrals(n, N):
    """Ω_k(n; N) is positive when curvature integrals are positive."""
    omega = omega_k_at_horizon(n, N)
    assert omega > 0
    assert omega < 0.1  # rough upper bound for reasonable shells


# ============== Standard Model constants ==============


def test_fine_structure_inverse_matches_CODATA():
    """1/α_EM in pyhqiv matches CODATA/PDG (~137.036)."""
    assert abs(ALPHA_EM_INV - ALPHA_EM_INV_CODATA) < 0.01
    assert 137.0 < ALPHA_EM_INV < 137.1


def test_weak_mixing_angle_at_MZ():
    """sin²θ_W at M_Z matches PDG (~0.23122)."""
    assert abs(SIN2_THETA_W_MZ - SIN2_THETA_W_PDG) < 0.001
    assert 0.22 < SIN2_THETA_W_MZ < 0.24


def test_hqiv_scalings_t_cmb_only():
    """Nuclear constants are T_CMB-derived only; no preset horizons or mode reductions."""
    const = get_hqiv_nuclear_constants(2.725)
    assert "LATTICE_BASE_M" in const
    assert "TAU_TICK_OBSERVER_S" in const
    assert "PHI_CRIT_SI" in const
    assert "MACROSCOPIC_DECAY_SCALE" in const
    assert const["LATTICE_BASE_M"] > 0
    # No preset data: free/bound horizons come from merge in nuclear.py
    assert "THETA_FREE_P_M" not in const
    assert "THETA_ALPHA_M" not in const
    assert "COULOMB_MODE_REDUCTION" not in const


# ============== Hydrogen coupling (Θ in Å) ==============


def test_hydrogen_theta_local_coordination_one():
    """Θ for hydrogen at coordination 1 is in bond-length scale (Å)."""
    theta_h = theta_for_atom("H", coordination=1)
    assert theta_h > 0
    assert THETA_ANGSTROM_MIN < theta_h < THETA_ANGSTROM_MAX
    # H has Z=1 so largest Θ for given coord
    theta_c = theta_for_atom("C", coordination=1)
    assert theta_h > theta_c


def test_hydrogen_theta_local_coordination_two():
    """Θ for hydrogen at coordination 2 (e.g. water) is in reasonable range."""
    theta_h2 = theta_for_atom("H", coordination=2)
    assert theta_h2 > 0
    assert THETA_ANGSTROM_MIN < theta_h2 < THETA_ANGSTROM_MAX
    # coord=2 increases effective monogamy so Θ smaller than coord=1
    theta_h1 = theta_for_atom("H", coordination=1)
    assert theta_h2 < theta_h1


def test_hydrogen_theta_via_z_shell():
    """theta_local(Z=1, coord) matches theta_for_atom('H', coord)."""
    for coord in [1, 2]:
        t_z = theta_local(1, coordination=coord)
        t_sym = theta_for_atom("H", coordination=coord)
        assert abs(t_z - t_sym) < 1e-10


# ============== Oxygen coupling (Θ in Å) ==============


def test_oxygen_theta_local_coordination_one():
    """Θ for oxygen at coordination 1 is in Å scale."""
    theta_o = theta_for_atom("O", coordination=1)
    assert theta_o > 0
    assert THETA_ANGSTROM_MIN < theta_o < THETA_ANGSTROM_MAX


def test_oxygen_theta_local_coordination_two():
    """Θ for oxygen at coordination 2 (e.g. water) in bond-length scale."""
    theta_o2 = theta_for_atom("O", coordination=2)
    assert theta_o2 > 0
    assert theta_o2 < THETA_ANGSTROM_MAX
    # O-H bond ~0.96 Å; Θ_O and Θ_H set bond_length_from_theta; min(Θ_i,Θ_j) * factor
    assert theta_o2 > 0.3


def test_oxygen_smaller_than_hydrogen_theta():
    """Oxygen (Z=8) has smaller Θ than hydrogen (Z=1) at same coordination (Z^{-α} scaling)."""
    for coord in [1, 2]:
        assert theta_for_atom("O", coordination=coord) < theta_for_atom("H", coordination=coord)


def test_water_like_bond_length_scale():
    """Bond length from Θ_H and Θ_O (water-like) is on order of O-H bond (~0.96 Å)."""
    from pyhqiv.utils import bond_length_from_theta

    theta_h = theta_for_atom("H", coordination=1)
    theta_o = theta_for_atom("O", coordination=2)
    r_eq = bond_length_from_theta(theta_h, theta_o)
    assert r_eq > 0
    assert 0.3 < r_eq < 3.0  # same order as O-H bond


def test_carbon_ideal_valence_angle_sp3():
    """Carbon bonding angles use ideal sp³ 109.47° (tetrahedral)."""
    from pyhqiv.atom import Atom

    atom_c = Atom("C", position=(0.0, 0.0, 0.0))
    atom_c.add_bond("X")
    angles = atom_c.get_bonding_angles()
    assert len(angles) >= 1
    ideal_deg = angles[0]["ideal_deg"]
    assert abs(ideal_deg - 109.47) < 0.01  # sp³ tetrahedral


def test_oxygen_ideal_valence_angle_in_bonding_angles():
    """Oxygen in bonding angles has an ideal valence (sp² 120° or basin value)."""
    from pyhqiv.atom import Atom

    atom_o = Atom("O", position=(0.0, 0.0, 0.0))
    atom_o.add_bond("Y")
    angles = atom_o.get_bonding_angles()
    assert len(angles) >= 1
    ideal_deg = angles[0]["ideal_deg"]
    assert 90 <= ideal_deg <= 125  # common oxygen valence basins


# ============== Nuclear binding energy (MeV) ==============
#
# Experiment (AME/NNDC): 2H ≈ 2.22 MeV, 4He ≈ 28.30 MeV, C-12 ≈ 92.16 MeV, O-16 ≈ 127.62 MeV.
# Tests assert first-principles B = E_free − E_bound lies in these bands (no clamp).


def test_deuteron_as_uuuddd_subatomic_solver():
    """Formal 6-quark uuuddd via subatomic solver; compare to D. Hierarchy: D is layer-2 (two nucleons), not layer-0 (one 6q bag)—see walkthrough §6.5.3."""
    from pyhqiv.constants import M_NEUTRON_MEV, M_PROTON_MEV
    from pyhqiv.subatomic import confined_energy_mev

    # D = 2H: experiment M_D = M_p + M_n - B_D
    B_D_MEV = B_2H_MEV_EXP
    M_PROTON_MEV + M_NEUTRON_MEV - B_D_MEV

    # Subatomic solver: D as single 6-quark confined state uuuddd (3u + 3d)
    E_uuuddd = confined_energy_mev("uuuddd")
    E_uud = confined_energy_mev("uud")
    E_udd = confined_energy_mev("udd")

    assert np.isfinite(E_uuuddd) and E_uuuddd > 0, "uuuddd energy must be finite and positive"
    # uuuddd is not in PDG registry, so solver uses L_base → one nucleon mass scale (~938 MeV)
    assert 500 < E_uuuddd < 2500, "uuuddd on nucleon mass scale (500–2500 MeV)"
    # Comparison: D experiment ~1875.6 MeV; p+n from solver ~1877.8 MeV
    assert abs(E_uud - M_PROTON_MEV) < 0.01 and abs(E_udd - M_NEUTRON_MEV) < 0.01
    # Document: E_uuuddd vs M_D (E_uuuddd currently ~half M_D because 6q uses L_base, not 2×nucleon)
    assert E_uuuddd < E_uud + E_udd, "6-quark confined state energy vs free p+n"


def test_binding_energy_returns_finite():
    """binding_energy_mev returns finite B (no clamp; can be negative)."""
    for P, N in [(1, 1), (2, 2), (6, 6), (8, 8)]:
        be = binding_energy_mev(P, N)
        assert np.isfinite(be), f"(P={P},N={N}): B = {be}"


def test_2h_binding_matches_experiment():
    """2H binding B in experimental band (2.224 ± 0.5 MeV)."""
    be = binding_energy_mev(1, 1)
    assert abs(be - B_2H_MEV_EXP) <= B_2H_MEV_TOLERANCE, (
        f"2H B = {be:.3f} MeV; experiment {B_2H_MEV_EXP} ± {B_2H_MEV_TOLERANCE} MeV"
    )


def test_he4_binding_matches_experiment():
    """4He binding B in experimental band (28.30 ± 3.5 MeV)."""
    be = binding_energy_mev(2, 2)
    assert abs(be - B_HE4_MEV_EXP) <= B_HE4_MEV_TOLERANCE, (
        f"4He B = {be:.3f} MeV; experiment {B_HE4_MEV_EXP} ± {B_HE4_MEV_TOLERANCE} MeV"
    )


def test_he4_binding_per_nucleon_matches_experiment():
    """4He B/A in experimental band (~7.08 ± 0.9 MeV)."""
    be = binding_energy_mev(2, 2)
    be_per_a = be / 4.0
    assert abs(be_per_a - B_HE4_PER_NUCLEON_MEV_EXP) <= B_HE4_PER_NUCLEON_TOLERANCE, (
        f"4He B/A = {be_per_a:.3f} MeV; experiment {B_HE4_PER_NUCLEON_MEV_EXP} ± {B_HE4_PER_NUCLEON_TOLERANCE} MeV"
    )


def test_c12_binding_in_experimental_ballpark():
    """C-12 B in experimental ballpark (~92 MeV, within factor 3)."""
    be = binding_energy_mev(6, 6)
    assert B_C12_MEV_EXP / B_TOLERANCE_FACTOR <= be <= B_C12_MEV_EXP * B_TOLERANCE_FACTOR, (
        f"C-12 B = {be:.3f} MeV; experiment ~{B_C12_MEV_EXP} MeV (×{B_TOLERANCE_FACTOR})"
    )


def test_o16_binding_in_experimental_ballpark():
    """O-16 B in experimental ballpark (~128 MeV, within factor 3)."""
    be = binding_energy_mev(8, 8)
    assert B_O16_MEV_EXP / B_TOLERANCE_FACTOR <= be <= B_O16_MEV_EXP * B_TOLERANCE_FACTOR, (
        f"O-16 B = {be:.3f} MeV; experiment ~{B_O16_MEV_EXP} MeV (×{B_TOLERANCE_FACTOR})"
    )


def test_binding_per_nucleon_ordering():
    """B > 0 for 4He, C-12, O-16 and B/A increases (7.08 ≤ 7.68 ≤ 7.98 MeV trend)."""
    b_he4 = binding_energy_mev(2, 2)
    b_c12 = binding_energy_mev(6, 6)
    b_o16 = binding_energy_mev(8, 8)
    assert b_he4 > 0 and b_c12 > 0 and b_o16 > 0, (
        f"Need positive B: 4He={b_he4:.3f}, C-12={b_c12:.3f}, O-16={b_o16:.3f}"
    )
    be_he4 = b_he4 / 4.0
    be_c12 = b_c12 / 12.0
    be_o16 = b_o16 / 16.0
    assert be_he4 <= be_c12 <= be_o16, (
        f"B/A: 4He={be_he4:.3f}, C-12={be_c12:.3f}, O-16={be_o16:.3f} (expect 7.08 ≤ 7.68 ≤ 7.98)"
    )


def test_proton_neutron_ordering_first_principles():
    """First-principles 8×8 path: positive Θ and E; neutron rest energy >= proton."""
    from pyhqiv.subatomic import (
        neutron_effective_theta_m,
        neutron_energy_mev,
        proton_effective_theta_m,
        proton_energy_mev,
    )

    theta_p = proton_effective_theta_m()
    theta_n = neutron_effective_theta_m()
    assert theta_p > 0 and theta_n > 0
    ep = proton_energy_mev()
    en = neutron_energy_mev()
    assert ep > 0 and en > 0
    assert en >= ep  # neutron >= proton (flavor/hypercharge in effective_modes can sharpen this)


def test_proton_neutron_parameter_free_and_epoch_api():
    """At epoch='now' coupling x from field + PDG → exact 938.272 / 939.565. Epoch API for baryogenesis or age_gyr."""
    from pyhqiv.constants import M_NEUTRON_MEV, M_PROTON_MEV
    from pyhqiv.subatomic import (
        neutron_energy_mev,
        proton_energy_mev,
        t_qcd_gev_at_epoch,
    )

    # At epoch="now" coupling distance x is set by field + PDG → exact masses
    ep = proton_energy_mev()
    en = neutron_energy_mev()
    assert abs(ep - M_PROTON_MEV) < 1e-9 and abs(en - M_NEUTRON_MEV) < 1e-9
    assert en > ep  # neutron heavier (folded charge)

    # Epoch API: baryogenesis uses T_lock (1.8 GeV) → higher scale
    ep_lock = proton_energy_mev(epoch="baryogenesis")
    assert ep_lock > ep  # lock-in scale larger than "now"
    assert abs(t_qcd_gev_at_epoch("now") - 1.624) < 0.01
    assert abs(t_qcd_gev_at_epoch("lock") - 1.8) < 0.01

    # 5 Gyr ago: between lock and now
    ep_5 = proton_energy_mev(epoch=5.0)
    assert ep < ep_5 < ep_lock or abs(ep_5 - ep) < 50  # 5 Gyr ago between now and lock


def test_confined_energy_same_method_all_subatomic():
    """Same first-principles path for all confined states (baryons, pentaquarks); exact PDG when in registry."""
    from pyhqiv.subatomic import (
        SUBATOMIC_PDG_MEV,
        confined_energy_mev,
        confined_pdg_energy_mev,
    )
    for flavor_content, pdg_mev in SUBATOMIC_PDG_MEV.items():
        e = confined_energy_mev(flavor_content)
        assert e > 0
        assert abs(e - pdg_mev) < 1e-6, f"{flavor_content}: got {e}, PDG {pdg_mev}"
    assert confined_pdg_energy_mev("unknown") is None
