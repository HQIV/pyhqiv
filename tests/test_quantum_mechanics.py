import cmath
import math

from pyhqiv.auxiliary_field import phi_of_shell
from pyhqiv.lightcone import alpha, cube_directions, octonion_imaginary_dim
from pyhqiv.metric import hqvm_lapse
from pyhqiv.omaxwell_couplings import (
    alpha_eff_at_shell,
    alpha_gut,
    coulomb_strength_shell,
    one_over_alpha_eff,
)
from pyhqiv.quantum_mechanics import (
    Wavefunction,
    birefringence_redshift,
    redshifted_energy_birefringence_balance,
    satisfies_lapse_corrected_schrodinger_residual,
    satisfies_time_dependent_schrodinger_residual,
)
from pyhqiv.surface_wave_self_clock import cosmic_birefringence_rad_at_now


def test_omaxwell_alpha_gut_derived() -> None:
    assert alpha_gut() == 1.0 / float(cube_directions() * octonion_imaginary_dim())
    assert alpha_gut() == 1.0 / 42.0


def test_one_over_alpha_eff_formula() -> None:
    phi = phi_of_shell(4)
    c = 1.0
    expected = (1.0 / alpha_gut()) * (1.0 + c * alpha() * math.log(phi + 1.0))
    assert one_over_alpha_eff(phi, c=c) == expected


def test_alpha_eff_at_shell_matches_coulomb_strength() -> None:
    m = 5
    assert alpha_eff_at_shell(m) == coulomb_strength_shell(m)


def test_lapse_corrected_schrodinger_residual_constant_lapse() -> None:
    """
    For constant lapse N = 1 + Phi (set phi_auxiliary=0), with Hψ = E ψ:
      i ħ dψ/dt = N E ψ
    """
    Phi = 0.3
    phi_aux = 0.0
    t0 = 0.25
    x = (1.0, 0.0, 0.0)
    E = 2.0
    # Use natural-units normalization for numeric stability of finite differences.
    # With SI hbar the phase oscillates at ~1e34 Hz and overwhelms float precision.
    hbar = 1.0
    N = hqvm_lapse(Phi, phi_aux, t0)

    def phi_x(_: tuple[float, float, float]) -> complex:
        return 1.0 + 0.0j

    def H(psi: Wavefunction) -> Wavefunction:
        return lambda xx: E * psi(xx)

    def psi_of_t(t: float) -> Wavefunction:
        phase = cmath.exp(-1j * N * E * t / hbar)
        return lambda xx: phase * phi_x(xx)

    assert satisfies_lapse_corrected_schrodinger_residual(
        H,
        psi_of_t,
        t=t0,
        x=x,
        phi_newtonian=Phi,
        phi_auxiliary=phi_aux,
        hbar=hbar,
        dt=1e-6,
        atol=1e-5,
    )


def test_time_dependent_schrodinger_residual_unit_lapse() -> None:
    Phi = 0.0
    phi_aux = 0.0
    t0 = 0.1
    x = (0.5, 0.0, 0.0)
    E = 3.0
    hbar = 1.0
    N = hqvm_lapse(Phi, phi_aux, t0)
    assert N == 1.0

    def phi_x(_: tuple[float, float, float]) -> complex:
        return 2.0 + 0.0j

    def H(psi: Wavefunction) -> Wavefunction:
        return lambda xx: E * psi(xx)

    def psi_of_t(t: float) -> Wavefunction:
        phase = cmath.exp(-1j * N * E * t / hbar)
        return lambda xx: phase * phi_x(xx)

    assert satisfies_time_dependent_schrodinger_residual(
        H,
        psi_of_t,
        t=t0,
        x=x,
        hbar=hbar,
        dt=1e-6,
        atol=1e-5,
    )


def test_birefringence_redshift_zero_when_beta_zero() -> None:
    assert birefringence_redshift(0.0, 1.0) == 0.0


def test_redshifted_energy_birefringence_balance_matches_lean() -> None:
    """``BornMeasurementFinite.redshiftedEnergyN_birefringence_balance`` identity."""
    e_post = 2.5
    beta_rad = 0.02
    kappa = 1.0
    lhs = redshifted_energy_birefringence_balance(e_post, beta_rad, kappa)
    assert math.isclose(lhs, e_post, rel_tol=0.0, abs_tol=1e-12)


def test_self_clock_beta_rad_feeds_birefringence_channel() -> None:
    """β from ``now`` witness is in radians for ``birefringence_redshift``."""
    beta = cosmic_birefringence_rad_at_now()
    z = birefringence_redshift(beta, kappa_beta=1.0)
    assert z > 0.0 and math.isfinite(z)

