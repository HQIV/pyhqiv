"""
Exact (small-N) simulator for the INS / dynamical-structure-factor workflow in
arXiv:2603.15608 (Lee et al., quantum simulation benchmarked against neutron
scattering).

Formal proofs and the octonion / light-cone / OSHoracle narrative live in the
**HQIV_LEAN** repository (e.g. ``~/Repos/HQIV_LEAN``):

- TeX companion: ``paper/octonion_lightcone_to_oshoracle.tex`` (lists Lean entry
  points in its header).
- Representative Lean modules: ``Hqiv/Geometry/OctonionicLightCone.lean``,
  ``Hqiv/QuantumComputing/OSHoracle.lean``, ``Hqiv/QuantumComputing/OctonionicFT.lean``,
  ``Hqiv/Algebra/OctonionBasics.lean``.

That manuscript is **not** the neutron-scattering benchmark paper; it supplies the
HQIV discrete-null and quantum-simulation *formal* stack. This module implements
the **INS DSF spin-chain protocol** from Lee et al. in two classical backends
(**dense** ``expm`` vs **first-order Trotter**) for side-by-side benchmarking.
The manuscript’s **dense vs sparse protein / mean-field** timing table is mirrored
by ``pyhqiv.benchmark_osh_vs_dense`` (same workloads as
``HQIV_LEAN/scripts/benchmark_protein_osh_vs_dense.py``).

Implementation notes
--------------------
- Uses open-boundary 1D spin-1/2 chains with Pauli algebra (exact diagonalisation
  and dense ``expm`` time evolution). Feasible for ``n_sites`` roughly ≤ 14 on
  a typical workstation.
- Reproduces the **center-site kick** protocol and retarded-Green’s-function
  reduction in the paper’s supplementary Eqs. (S2)–(S4): local rotation
  :math:`U_{j_c} = (1/\\sqrt{2})(1 - i\\sigma^\\beta_{j_c})`, evolution
  :math:`U(t)=e^{-iHt}`, then :math:`\\langle\\psi(t)|\\sigma^\\alpha_j|\\psi(t)\\rangle`.
- The emergent **light-cone** patterns in the spatio-temporal RGF (ballistic /
  superdiffusive spreading in Lee et al.) sit alongside the **discrete**
  light-cone combinatorics mirrored in ``pyhqiv.lightcone`` (Lean:
  ``Hqiv.Geometry.OctonionicLightCone``). This module does **not** embed
  octonion multiplication—it implements standard spin-chain physics for the INS
  benchmark.

Hamiltonian conventions match the paper’s Eq. (3) for the KCuF\\ :sub:`3` chain::

    H = 2J \\sum_i [ S_i^X S_{i+1}^X + S_i^Y S_{i+1}^Y + \\epsilon S_i^Z S_{i+1}^Z ]

with spin-1/2 operators :math:`S^a = \\sigma^a / 2`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import expm
from scipy.stats import wasserstein_distance

PauliAxis = Literal["x", "y", "z"]

I2 = np.eye(2, dtype=np.complex128)
SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SY = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _pauli_1(a: PauliAxis) -> np.ndarray:
    if a == "x":
        return SX
    if a == "y":
        return SY
    return SZ


def _kron_stack(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def pauli_product_on_sites(n_sites: int, site: int, axis: PauliAxis) -> np.ndarray:
    """Full Hilbert-space Pauli :math:`\\sigma^a` on ``site`` (qubit 0 = LSB)."""
    if n_sites <= 1:
        raise ValueError("n_sites must be ≥ 2 for a chain observable")
    if site < 0 or site >= n_sites:
        raise ValueError(f"site {site} out of range for n_sites={n_sites}")
    mats: list[np.ndarray] = []
    p = _pauli_1(axis)
    for q in range(n_sites):
        mats.append(p if q == site else I2)
    return _kron_stack(mats)


def xxz_hamiltonian_open_chain(
    n_sites: int, *, J: float, epsilon: float
) -> np.ndarray:
    """
    Paper Eq. (3), open chain, spin-1/2.

    Returns Hermitian matrix in Pauli-:math:`\\sigma` units with the same energy
    scale as Eq. (3) (via :math:`S^a=\\sigma^a/2`).
    """
    if n_sites < 2:
        raise ValueError("need at least two spins")
    h = np.zeros((2**n_sites, 2**n_sites), dtype=np.complex128)
    pref = 0.5 * J
    for j in range(n_sites - 1):
        sx_j = pauli_product_on_sites(n_sites, j, "x")
        sx_jp = pauli_product_on_sites(n_sites, j + 1, "x")
        sy_j = pauli_product_on_sites(n_sites, j, "y")
        sy_jp = pauli_product_on_sites(n_sites, j + 1, "y")
        sz_j = pauli_product_on_sites(n_sites, j, "z")
        sz_jp = pauli_product_on_sites(n_sites, j + 1, "z")
        h += pref * (sx_j @ sx_jp + sy_j @ sy_jp + epsilon * sz_j @ sz_jp)
    return h


def xxz_nnn_hamiltonian_open_chain(
    n_sites: int,
    *,
    J: float,
    J_prime: float,
    epsilon: float,
    epsilon_prime: float,
) -> np.ndarray:
    """
    Paper Eq. (4): NN ZZ + transverse + ferromagnetic NNN (minus sign in front
    of the NNN block as written in the paper).
    """
    if n_sites < 3:
        raise ValueError("NNN requires at least three spins")
    h = np.zeros((2**n_sites, 2**n_sites), dtype=np.complex128)
    pref_nn = 0.5 * J
    for j in range(n_sites - 1):
        h += pref_nn * (
            pauli_product_on_sites(n_sites, j, "z") @ pauli_product_on_sites(n_sites, j + 1, "z")
            + epsilon
            * (
                pauli_product_on_sites(n_sites, j, "x") @ pauli_product_on_sites(n_sites, j + 1, "x")
                + pauli_product_on_sites(n_sites, j, "y")
                @ pauli_product_on_sites(n_sites, j + 1, "y")
            )
        )
    pref_nnn = -0.5 * J_prime
    for j in range(n_sites - 2):
        h += pref_nnn * (
            pauli_product_on_sites(n_sites, j, "z") @ pauli_product_on_sites(n_sites, j + 2, "z")
            + epsilon_prime
            * (
                pauli_product_on_sites(n_sites, j, "x") @ pauli_product_on_sites(n_sites, j + 2, "x")
                + pauli_product_on_sites(n_sites, j, "y")
                @ pauli_product_on_sites(n_sites, j + 2, "y")
            )
        )
    return h


def ground_state_vector(H: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    psi = evecs[:, 0].astype(np.complex128, copy=True)
    psi /= math.sqrt(float(np.vdot(psi, psi).real))
    return psi


def center_site_index(n_sites: int) -> int:
    """Paper’s central column index for the center-site measurement reduction."""
    return (n_sites - 1) // 2


def kick_unitary_on_site(n_sites: int, site: int, beta: PauliAxis) -> np.ndarray:
    """Paper Eq. (S2): ``(1/√2)(𝟙 − i σ^β)`` on one site, identity elsewhere."""
    p = _pauli_1(beta)
    u1 = (I2 - 1j * p) / math.sqrt(2.0)
    mats: list[np.ndarray] = []
    for q in range(n_sites):
        mats.append(u1 if q == site else I2)
    return _kron_stack(mats)


def _hilbert_bit_index(*, n_sites: int, site: int) -> int:
    """
    Map chain site index (same as ``pauli_product_on_sites``) to computational
    bitmask position.

    ``pauli_product_on_sites`` stacks ``kron(mats[0], mats[1], …)`` with site ``0``
    as the **outer** (slow) tensor factor, i.e. **MSB** of the little-endian index.
    """
    if site < 0 or site >= n_sites:
        raise ValueError("site out of range")
    return n_sites - 1 - site


def expectation_pauli(
    psi: np.ndarray, n_sites: int, site: int, axis: PauliAxis
) -> float:
    return expectation_pauli_statevector(psi, n_sites, site, axis)


def expectation_pauli_statevector(
    psi: np.ndarray, n_sites: int, site: int, axis: PauliAxis
) -> float:
    """``⟨ψ|σ^a_site|ψ⟩`` without constructing the full ``2^n×`` Pauli matrix."""
    dim = psi.size
    if dim != (1 << n_sites):
        raise ValueError("psi size mismatch")
    m = 1 << _hilbert_bit_index(n_sites=n_sites, site=site)
    if axis == "z":
        s = 0j
        for i in range(dim):
            sign = -1.0 if (i & m) else 1.0
            s += sign * (abs(psi[i]) ** 2)
        return float(s.real)
    if axis == "x":
        s = 0j
        for i in range(dim):
            j = i ^ m
            s += np.conj(psi[i]) * psi[j]
        return float(s.real)
    s = 0j
    for i in range(dim):
        j = i ^ m
        if (i & m) == 0:
            s += (-1j) * np.conj(psi[i]) * psi[j]
        else:
            s += (1j) * np.conj(psi[i]) * psi[j]
    return float(s.real)


@dataclass
class INSXxzResult:
    """Time-domain RGF and derived spectrum after Eq. (S4) style 2D transform."""

    times: np.ndarray  # shape (n_times,)
    rgf: np.ndarray  # shape (n_sites, n_times), real
    q_bins: np.ndarray  # FFT bin indices 0..n_sites-1  →  q_l = 2π l / n_sites
    omega_bins: np.ndarray  # m = 0..n_times-1  →  ω_m = 2π m / (n_times * Δt)
    spectral_complex: np.ndarray  # shape (n_sites, n_times), mirror-averaged
    spectral_intensity: np.ndarray  # non-negative magnitude proxy for plots / metrics
    evolution_method: str  # "dense_expm" | "trotter_first_order"


def apply_two_qubit_unitary_state(
    psi: np.ndarray, n_sites: int, qa: int, qb: int, u4: np.ndarray
) -> np.ndarray:
    """
    Apply ``u4`` on sites ``qa`` and ``qb`` (same site labelling as ``pauli_product_on_sites``).

    ``u4`` is in ``kron(P_qa, P_{qb})`` order: row/column block index ``ba*2 + bb`` for
    bits ``(ba, bb)`` on sites ``(qa, qb)``.

    Used for first-order Trotter factors on NN bonds ``(j, j+1)``.
    """
    if u4.shape != (4, 4):
        raise ValueError("u4 must be 4×4")
    if qa == qb:
        raise ValueError("qa and qb must differ")
    pa = _hilbert_bit_index(n_sites=n_sites, site=qa)
    pb = _hilbert_bit_index(n_sites=n_sites, site=qb)
    other_pos = [r for r in range(n_sites) if r not in (pa, pb)]
    out = np.zeros_like(psi, dtype=np.complex128)
    n_rest = len(other_pos)
    for mask_rest in range(1 << n_rest):
        idx_base = 0
        for t, r in enumerate(other_pos):
            if (mask_rest >> t) & 1:
                idx_base |= 1 << r
        four_idxs: list[int] = []
        for ba in (0, 1):
            for bb in (0, 1):
                four_idxs.append(idx_base | (ba << pa) | (bb << pb))
        vec4 = np.array([psi[i] for i in four_idxs], dtype=np.complex128)
        new4 = u4 @ vec4
        for i, v in zip(four_idxs, new4):
            out[i] = v
    return out


def _xxz_trotter_one_step_apply(
    psi: np.ndarray, n_sites: int, J: float, epsilon: float, dt: float
) -> np.ndarray:
    """First-order Lie–Trotter step for one interval ``dt`` (same NN decomposition as Eq. (S6) spirit)."""
    out = psi.copy()
    for j in range(n_sites - 1):
        u4_xx = expm(-1j * (J / 2.0) * dt * np.kron(SX, SX))
        u4_yy = expm(-1j * (J / 2.0) * dt * np.kron(SY, SY))
        u4_zz = expm(-1j * (J / 2.0) * epsilon * dt * np.kron(SZ, SZ))
        u4 = u4_zz @ u4_yy @ u4_xx
        out = apply_two_qubit_unitary_state(out, n_sites, j, j + 1, u4)
    return out


def simulate_xxz_rgf_center_kick(
    *,
    n_sites: int,
    J: float,
    epsilon: float,
    delta_t: float,
    n_time_steps: int,
    beta: PauliAxis,
    alpha: PauliAxis,
    jc: int | None = None,
    H: np.ndarray | None = None,
    ground_state: np.ndarray | None = None,
) -> INSXxzResult:
    """
    Exact simulation of the protocol in Fig. 1A / Eqs. (S2)–(S3).

    Time samples are ``t_k = k * delta_t`` for ``k = 0 … n_time_steps``.
    """
    if n_time_steps < 0:
        raise ValueError("n_time_steps must be non-negative")
    if delta_t <= 0:
        raise ValueError("delta_t must be positive")
    if H is None:
        H_local = xxz_hamiltonian_open_chain(n_sites, J=J, epsilon=epsilon)
    else:
        H_local = H
    jc_i = center_site_index(n_sites) if jc is None else jc
    psi_gs = ground_state_vector(H_local) if ground_state is None else ground_state.copy()
    if psi_gs.shape[0] != 2**n_sites:
        raise ValueError("ground_state dimension mismatch")
    psi_gs /= math.sqrt(float(np.vdot(psi_gs, psi_gs).real))

    u_kick = kick_unitary_on_site(n_sites, site=jc_i, beta=beta)
    psi0 = u_kick @ psi_gs

    n_times = n_time_steps + 1
    times = np.arange(n_times, dtype=float) * delta_t
    rgf = np.zeros((n_sites, n_times), dtype=float)
    # One step unitary U(Δt) = exp(-i H Δt); ψ_k = U^k ψ_0 matches exp(-i H t_k) ψ_0.
    u_dt = expm(-1j * H_local * delta_t)
    psi = psi0.copy()
    for k in range(n_times):
        if k > 0:
            psi = u_dt @ psi
        for j in range(n_sites):
            rgf[j, k] = expectation_pauli(psi, n_sites, j, alpha)

    spec_c, spec_i = rgf_to_spectral_mirror_avg(rgf, delta_t=delta_t)
    q_bins = np.arange(n_sites, dtype=float) * (2.0 * math.pi / max(n_sites, 1))
    omega_bins = np.arange(n_times, dtype=float) * (2.0 * math.pi / (n_times * delta_t))

    return INSXxzResult(
        times=times,
        rgf=rgf,
        q_bins=q_bins,
        omega_bins=omega_bins,
        spectral_complex=spec_c,
        spectral_intensity=spec_i,
        evolution_method="dense_expm",
    )


def simulate_xxz_rgf_center_kick_trotter(
    *,
    n_sites: int,
    J: float,
    epsilon: float,
    delta_t: float,
    n_time_steps: int,
    beta: PauliAxis,
    alpha: PauliAxis,
    jc: int | None = None,
    H: np.ndarray | None = None,
    ground_state: np.ndarray | None = None,
) -> INSXxzResult:
    """
    Same INS protocol as ``simulate_xxz_rgf_center_kick`` but time evolution uses
    **first-order Trotter** steps of length ``delta_t`` (circuit-style decomposition),
    comparable to near-term gate models (cf. arXiv:2603.15608 Trotter discussion).

    This is the **defined algorithm** classical simulation path; ``dense_expm`` is the
    exact-in-``dt`` baseline for the same kick + measurement grid.
    """
    if n_time_steps < 0:
        raise ValueError("n_time_steps must be non-negative")
    if delta_t <= 0:
        raise ValueError("delta_t must be positive")
    if n_sites < 2:
        raise ValueError("need at least two spins")
    jc_i = center_site_index(n_sites) if jc is None else jc
    if H is None:
        H_local = xxz_hamiltonian_open_chain(n_sites, J=J, epsilon=epsilon)
    else:
        H_local = H
    psi_gs = ground_state_vector(H_local) if ground_state is None else ground_state.copy()
    psi_gs /= math.sqrt(float(np.vdot(psi_gs, psi_gs).real))

    u_kick = kick_unitary_on_site(n_sites, site=jc_i, beta=beta)
    psi0 = u_kick @ psi_gs

    n_times = n_time_steps + 1
    times = np.arange(n_times, dtype=float) * delta_t
    rgf = np.zeros((n_sites, n_times), dtype=float)
    psi = psi0.copy()
    for k in range(n_times):
        if k > 0:
            psi = _xxz_trotter_one_step_apply(psi, n_sites, J, epsilon, delta_t)
        for j in range(n_sites):
            rgf[j, k] = expectation_pauli(psi, n_sites, j, alpha)

    spec_c, spec_i = rgf_to_spectral_mirror_avg(rgf, delta_t=delta_t)
    q_bins = np.arange(n_sites, dtype=float) * (2.0 * math.pi / max(n_sites, 1))
    omega_bins = np.arange(n_times, dtype=float) * (2.0 * math.pi / (n_times * delta_t))

    return INSXxzResult(
        times=times,
        rgf=rgf,
        q_bins=q_bins,
        omega_bins=omega_bins,
        spectral_complex=spec_c,
        spectral_intensity=spec_i,
        evolution_method="trotter_first_order",
    )


@dataclass
class INSXxzDenseVsTrotter:
    """Side-by-side INS benchmark: exact ``expm`` vs first-order Trotter (same parameters)."""

    dense: INSXxzResult
    trotter: INSXxzResult
    rgf_max_abs_diff: float
    rgf_rmse: float
    spectral_intensity_mse: float
    spectral_wasserstein: float


def benchmark_ins_xxz_dense_vs_trotter(
    *,
    n_sites: int,
    J: float,
    epsilon: float,
    delta_t: float,
    n_time_steps: int,
    beta: PauliAxis,
    alpha: PauliAxis,
    jc: int | None = None,
) -> INSXxzDenseVsTrotter:
    """
    Run **both** evolution methods on identical kick + observable grid and report
    agreement metrics (for manuscript / figure panels).
    """
    kw = dict(
        n_sites=n_sites,
        J=J,
        epsilon=epsilon,
        delta_t=delta_t,
        n_time_steps=n_time_steps,
        beta=beta,
        alpha=alpha,
        jc=jc,
    )
    d_res = simulate_xxz_rgf_center_kick(**kw)
    t_res = simulate_xxz_rgf_center_kick_trotter(**kw)
    diff = d_res.rgf - t_res.rgf
    rgf_max_abs = float(np.max(np.abs(diff)))
    rgf_rmse = float(math.sqrt(float(np.mean(diff**2))))
    smse, sw = spectrum_metrics_mse_wasserstein(
        d_res.spectral_intensity, t_res.spectral_intensity
    )
    return INSXxzDenseVsTrotter(
        dense=d_res,
        trotter=t_res,
        rgf_max_abs_diff=rgf_max_abs,
        rgf_rmse=rgf_rmse,
        spectral_intensity_mse=smse,
        spectral_wasserstein=sw,
    )


def rgf_to_spectral_mirror_avg(
    rgf: np.ndarray, *, delta_t: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discrete analogue of Eq. (S4) via 2D FFT along space (``j``) and time (``k``),
    then average with the ``(q, ω) → (-q, -ω)`` mirror as in the 50-qubit discussion.
    """
    if delta_t <= 0:
        raise ValueError("delta_t must be positive")
    n_sites, n_times = rgf.shape
    # Spatial FFT: Σ_j e^{-i q j} G(j,k) with q_l = 2π l / n
    g_q = np.fft.fft(rgf, axis=0)
    # Temporal sum Σ_k e^{+i ω t_k} G_q(k) with t_k = k Δt, ω_m = 2π m / (n_times Δt)
    spec = np.fft.ifft(g_q, axis=1) * float(n_times)

    spec_mirror = 0.5 * (spec + np.flip(np.flip(spec, axis=0), axis=1))
    intensity = np.abs(spec_mirror)
    return spec_mirror, intensity


def spectrum_metrics_mse_wasserstein(
    a: np.ndarray, b: np.ndarray, *, clip_nonnegative: bool = True
) -> tuple[float, float]:
    """
    Paper Fig. 3 style global image metrics on **flattened**, normalised spectra.

    Returns (MSE, Wasserstein-1 distance). SSIM is omitted to avoid an extra
    dependency; call ``skimage.metrics.structural_similarity`` in user code if needed.
    """
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("spectra must have the same shape before flattening")
    if clip_nonnegative:
        x = np.maximum(x, 0.0)
        y = np.maximum(y, 0.0)
    sx = x.sum()
    sy = y.sum()
    if sx > 0:
        x = x / sx
    if sy > 0:
        y = y / sy
    mse = float(np.mean((x - y) ** 2))
    # Wasserstein on 1D empirical measures supported on pixel indices (paper-style)
    idx = np.arange(x.size, dtype=float)
    w1 = float(wasserstein_distance(idx, idx, u_weights=x, v_weights=y))
    return mse, w1


__all__ = [
    "INSXxzResult",
    "INSXxzDenseVsTrotter",
    "apply_two_qubit_unitary_state",
    "benchmark_ins_xxz_dense_vs_trotter",
    "center_site_index",
    "expectation_pauli",
    "ground_state_vector",
    "kick_unitary_on_site",
    "pauli_product_on_sites",
    "rgf_to_spectral_mirror_avg",
    "simulate_xxz_rgf_center_kick",
    "simulate_xxz_rgf_center_kick_trotter",
    "spectrum_metrics_mse_wasserstein",
    "xxz_hamiltonian_open_chain",
    "xxz_nnn_hamiltonian_open_chain",
]
