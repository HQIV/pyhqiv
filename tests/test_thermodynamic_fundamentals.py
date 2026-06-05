
from pyhqiv.lightcone import lattice_simplex_count
from pyhqiv.thermodynamic_fundamentals import (
    clausius_equality_proxy_power_form,
    clausius_residual_proxy,
    entropy_increment_per_shell,
    equilibrium_rho_total_for_phi,
    horizon_entropy_counting,
    local_equilibrium_proxy,
    second_law_arrow_holds,
    temperature_at_shell,
)


def test_second_law_arrow_strictly_positive() -> None:
    assert second_law_arrow_holds(30)
    for m in range(1, 10):
        assert entropy_increment_per_shell(m) > 0.0


def test_entropy_increment_matches_lattice() -> None:
    for m in [1, 2, 3, 5, 10, 20]:
        assert entropy_increment_per_shell(m) == float(lattice_simplex_count(m))


def test_horizon_entropy_counting_monotone() -> None:
    for m in range(0, 10):
        assert horizon_entropy_counting(m + 1) >= horizon_entropy_counting(m)


def test_temperature_ladder() -> None:
    assert temperature_at_shell(0) == 1.0
    assert temperature_at_shell(4) == 0.2


def test_local_equilibrium_proxy_and_clausius_residual() -> None:
    phi = 1.0
    rho_total_eq = equilibrium_rho_total_for_phi(phi)
    rho_m = rho_total_eq
    rho_r = 0.0
    assert local_equilibrium_proxy(phi, rho_m, rho_r, atol=1e-10)

    res = clausius_residual_proxy(phi, rho_m, rho_r)
    assert res <= 1e-10
    res_power = clausius_equality_proxy_power_form(phi, rho_m, rho_r)
    assert res_power <= 1e-10

    # Non-equilibrium: perturb density
    eps = 0.05
    rho_total_noneq = rho_total_eq * (1.0 + eps)
    rho_m_noneq = rho_total_noneq
    assert not local_equilibrium_proxy(phi, rho_m_noneq, rho_r, atol=1e-10)
    assert clausius_residual_proxy(phi, rho_m_noneq, rho_r) > 1e-8

