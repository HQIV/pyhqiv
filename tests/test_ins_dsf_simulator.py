"""Tests for arXiv:2603.15608-style INS / DSF exact diagonalisation workflow."""

import math

import numpy as np

from pyhqiv.ins_dsf_simulator import (
    benchmark_ins_xxz_dense_vs_trotter,
    center_site_index,
    expectation_pauli,
    ground_state_vector,
    kick_unitary_on_site,
    rgf_to_spectral_mirror_avg,
    simulate_xxz_rgf_center_kick,
    simulate_xxz_rgf_center_kick_trotter,
    spectrum_metrics_mse_wasserstein,
    xxz_hamiltonian_open_chain,
)


def test_center_site_index_matches_paper_convention() -> None:
    assert center_site_index(50) == 24
    assert center_site_index(49) == 24


def test_xxz_ground_state_energy_two_site_xx_limit() -> None:
    # ε=0 gives XX-only; two-site ground energy is -J/2 per bond × 2J prefactor structure
    n = 2
    J = 1.0
    H = xxz_hamiltonian_open_chain(n, J=J, epsilon=0.0)
    e0 = float(np.linalg.eigvalsh(H)[0])
    # H = (J/2)(XX+YY); eigenvalues of XX+YY on two qubits: bond singlet sector
    assert math.isclose(e0, -J, rel_tol=0.0, abs_tol=1e-10)


def test_kick_preserves_norm_and_rgf_is_real() -> None:
    n = 4
    H = xxz_hamiltonian_open_chain(n, J=1.0, epsilon=1.0)
    gs = ground_state_vector(H)
    jc = center_site_index(n)
    uk = kick_unitary_on_site(n, jc, "x")
    psi = uk @ gs
    assert math.isclose(float(np.vdot(psi, psi).real), 1.0, rel_tol=0.0, abs_tol=1e-10)
    for j in range(n):
        v = expectation_pauli(psi, n, j, "z")
        assert abs(v - float(v)) < 1e-14


def test_simulate_small_chain_finite_spectrum() -> None:
    res = simulate_xxz_rgf_center_kick(
        n_sites=5,
        J=1.0,
        epsilon=0.8,
        delta_t=0.2,
        n_time_steps=4,
        beta="y",
        alpha="z",
    )
    assert res.rgf.shape == (5, 5)
    assert np.all(np.isfinite(res.rgf))
    assert res.spectral_intensity.shape == res.rgf.shape
    assert np.all(res.spectral_intensity >= 0.0)
    assert res.evolution_method == "dense_expm"


def test_trotter_method_tag_and_dense_vs_trotter_benchmark() -> None:
    tr = simulate_xxz_rgf_center_kick_trotter(
        n_sites=4,
        J=0.7,
        epsilon=0.9,
        delta_t=0.04,
        n_time_steps=6,
        beta="x",
        alpha="z",
    )
    assert tr.evolution_method == "trotter_first_order"
    bench = benchmark_ins_xxz_dense_vs_trotter(
        n_sites=4,
        J=0.7,
        epsilon=0.9,
        delta_t=0.04,
        n_time_steps=6,
        beta="x",
        alpha="z",
    )
    assert bench.dense.evolution_method == "dense_expm"
    assert bench.trotter.evolution_method == "trotter_first_order"
    # Small dt: first-order Trotter should stay close to expm on this grid
    assert bench.rgf_rmse < 0.08


def test_spectrum_metrics_self_zero() -> None:
    a = np.random.RandomState(3).rand(4, 5)
    mse, w1 = spectrum_metrics_mse_wasserstein(a, a)
    assert math.isclose(mse, 0.0, abs_tol=1e-15)
    assert math.isclose(w1, 0.0, abs_tol=1e-15)


def test_rgf_to_spectral_shape() -> None:
    rgf = np.ones((3, 2))
    c, i = rgf_to_spectral_mirror_avg(rgf, delta_t=0.5)
    assert c.shape == rgf.shape
    assert i.shape == rgf.shape
