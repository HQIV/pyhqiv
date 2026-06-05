import math

from pyhqiv.action import (
    friedmann_from_action_equivalence_holds,
    s_hqvm_grav,
    s_hqvm_grav_zero_holds,
)
from pyhqiv.auxiliary_field import phi_of_shell, shell_temperature
from pyhqiv.conservations import (
    phase_conservation_numeric,
)
from pyhqiv.forces import (
    ForceSector,
    UnitSystem,
    ValueInUnits,
    emergent_maxwell_inhomogeneous_o_metric,
    emergent_maxwell_inhomogeneous_o_si,
    in_metric,
    in_si,
    o_component_to_sector,
    time_axis,
)
from pyhqiv.gr_from_maxwell import (
    hqvm_friedmann_power_residual,
    o_maxwell_determines_hqvm_gr_homogeneous_equivalence,
)
from pyhqiv.lightcone import (
    alpha,
    available_modes,
    curvature_norm_combinatorial,
    lattice_simplex_count,
    new_modes,
    reference_m,
    shell_shape,
)
from pyhqiv.metric import (
    build_metric_snapshot,
    g_eff,
    gamma_hqiv,
    h0,
    hqvm_friedmann_holds,
    hqvm_friedmann_residual,
    hqvm_g_tt,
    hqvm_lapse,
    hqvm_spatial_coeff,
    three_minus_gamma,
    time_angle,
    two_pi,
)
from pyhqiv.modified_maxwell import (
    charge_conservation_o,
    classic_maxwell_inhomogeneous,
    delta_theta_prime,
    emergent_maxwell_inhomogeneous_h,
    emergent_maxwell_inhomogeneous_o,
    horizon_quarter_period,
    in_h,
    spatial_indices,
)
from pyhqiv.now import (
    age_ratio_at_now,
    build_preliminary_now,
    h0_ref,
    now_condition,
    shell_index_for_temperature,
    temperature_from_shell_index,
)
from pyhqiv.spin_statistics import (
    SpinClass,
    exchange_phase_identical,
    resonance_half_life,
    resonance_lifetime,
    two_pi_phase,
)


def test_lightcone_mode_formulas() -> None:
    assert lattice_simplex_count(0) == 2
    assert lattice_simplex_count(2) == 12
    assert available_modes(0) == 8.0
    assert new_modes(1) == 16.0


def test_alpha_and_curvature_norm() -> None:
    assert alpha() == 3.0 / 5.0
    assert math.isclose(curvature_norm_combinatorial(), (6**7) * math.sqrt(3.0))


def test_auxiliary_field_ladder() -> None:
    assert shell_temperature(0) == 1.0
    assert shell_temperature(4) == 0.2
    assert phi_of_shell(4) == 10.0


def test_shell_shape_formula() -> None:
    expected = (1.0 / 5.0) * (1.0 + alpha() * math.log(5.0))
    assert math.isclose(shell_shape(4), expected)


def test_now_temperature_inverse() -> None:
    temperature = 0.2
    shell_real = shell_index_for_temperature(temperature)
    assert shell_real == 4.0
    assert temperature_from_shell_index(shell_real) == temperature


def test_preliminary_now_bundle_without_witness() -> None:
    now_bundle = build_preliminary_now()
    assert now_bundle.natural_now_phi == h0_ref()
    assert now_bundle.reference_shell == reference_m()
    assert now_bundle.witness is None
    assert now_condition(now_bundle.natural_now_phi)


def test_preliminary_now_bundle_with_witness() -> None:
    now_bundle = build_preliminary_now(temperature_natural=0.2)
    assert now_bundle.witness is not None
    assert now_bundle.witness.shell_index_real == 4.0
    assert now_bundle.witness.lower_shell == 4
    assert now_bundle.witness.lower_shell_phi == 10.0


def test_homogeneous_age_ratio_at_now() -> None:
    assert age_ratio_at_now(4.0) == 3.0


def test_metric_and_lapse_definitions() -> None:
    lapse = hqvm_lapse(0.2, 1.5, 4.0)
    assert lapse == 7.2
    assert time_angle(1.5, 4.0) == 6.0
    assert hqvm_g_tt(lapse) == -(7.2**2)
    assert math.isclose(hqvm_spatial_coeff(3.0, 0.2), 9.0 * 0.6)
    assert math.isclose(two_pi(), 2.0 * math.pi)


def test_metric_gamma_and_geff() -> None:
    assert gamma_hqiv() == 2.0 / 5.0
    assert three_minus_gamma() == 13.0 / 5.0
    assert h0() == h0_ref() == 1.0
    assert g_eff(1.0) == 1.0


def test_metric_snapshot() -> None:
    snapshot = build_metric_snapshot(0.0, 1.0, 2.0)
    assert snapshot.lapse == 3.0
    assert snapshot.time_angle_value == 2.0
    assert snapshot.g_tt == -9.0


def test_homogeneous_friedmann_helpers() -> None:
    phi = 2.0
    rhs_density = three_minus_gamma() * (phi**2) / (8.0 * math.pi * g_eff(phi))
    assert hqvm_friedmann_holds(phi, rhs_density, 0.0)
    assert math.isclose(hqvm_friedmann_residual(phi, rhs_density, 0.0), 0.0, abs_tol=1e-12)


def test_modified_maxwell_scaffold() -> None:
    assert in_h(0)
    assert in_h(3)
    assert not in_h(4)
    assert spatial_indices() == (1, 2, 3)
    assert emergent_maxwell_inhomogeneous_o(0, 0) == 0.0
    assert emergent_maxwell_inhomogeneous_h(1) == 0.0
    assert classic_maxwell_inhomogeneous(2) == 0.0
    assert charge_conservation_o(0, 0) == 0.0


def test_delta_theta_prime_properties() -> None:
    assert horizon_quarter_period() == math.pi / 2.0
    assert delta_theta_prime(0.0) == 0.0
    assert math.isclose(delta_theta_prime(-2.0), -delta_theta_prime(2.0))


def test_force_sector_selector() -> None:
    assert o_component_to_sector(0) is ForceSector.EM
    assert o_component_to_sector(1) is ForceSector.WEAK
    assert o_component_to_sector(3) is ForceSector.WEAK
    assert o_component_to_sector(4) is ForceSector.STRONG
    assert o_component_to_sector(7) is ForceSector.STRONG
    assert time_axis() == 0


def test_force_unit_tags_and_equation_bridge() -> None:
    metric_value = in_metric(2.5)
    si_value = in_si(7.5)
    assert metric_value == ValueInUnits(system=UnitSystem.METRIC, value=2.5)
    assert si_value == ValueInUnits(system=UnitSystem.SI, value=7.5)
    assert metric_value.to_real() == 2.5
    assert si_value.to_real() == 7.5
    assert emergent_maxwell_inhomogeneous_o_metric(0, 0) == emergent_maxwell_inhomogeneous_o_si(0, 0)


def test_conservations_phase_periodicity() -> None:
    # Choose phi=1 so the interval becomes exactly [0, 2π] and equality checks are robust.
    cond0, cond_period, cond_interval = phase_conservation_numeric(phi_auxiliary=1.0, t=math.pi)
    assert cond0 is True
    assert cond_period is True
    assert cond_interval is True


def test_action_s_hqvm_grav_matches_friedmann() -> None:
    phi = 1.0
    gamma = gamma_hqiv()
    three_minus_gamma_val = 3.0 - gamma
    rho_total = three_minus_gamma_val * (phi**2) / (8.0 * math.pi * g_eff(phi))
    s_val = s_hqvm_grav(phi, rho_total, 0.0)
    assert s_hqvm_grav_zero_holds(phi, rho_total, 0.0)
    assert abs(s_val) <= 1e-12

    assert hqvm_friedmann_residual(phi, rho_total, 0.0) == 0.0 or abs(
        hqvm_friedmann_residual(phi, rho_total, 0.0)
    ) <= 1e-12

    assert friedmann_from_action_equivalence_holds(phi, rho_total, 0.0, atol=1e-12)


def test_gr_from_maxwell_power_residual() -> None:
    phi = 1.0
    rho_total = (13.0 / 5.0) * (phi**2) / (8.0 * math.pi * (phi**alpha()))
    assert abs(hqvm_friedmann_power_residual(phi, rho_total, 0.0)) <= 1e-12
    assert o_maxwell_determines_hqvm_gr_homogeneous_equivalence(phi, rho_total, 0.0, atol=1e-12)


def test_spin_statistics_sign_rule() -> None:
    assert two_pi_phase(SpinClass.INTEGER) == 1.0 + 0.0j
    assert two_pi_phase(SpinClass.HALF_INTEGER) == -1.0 + 0.0j
    assert exchange_phase_identical(SpinClass.INTEGER) == 1.0 + 0.0j
    assert exchange_phase_identical(SpinClass.HALF_INTEGER) == -1.0 + 0.0j


def test_resonance_lifetime_and_half_life() -> None:
    # Pick a simple delta_E to get deterministic numbers.
    delta_E = 1.0  # MeV
    tau = resonance_lifetime(delta_E)
    assert math.isclose(tau, 6.582119569e-22, rel_tol=0.0, abs_tol=1e-30)
    half = resonance_half_life(delta_E)
    assert math.isclose(half, math.log(2.0) * 6.582119569e-22, rel_tol=0.0, abs_tol=1e-30)
