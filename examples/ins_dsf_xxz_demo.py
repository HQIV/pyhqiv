#!/usr/bin/env python3
"""
Minimal demo: exact small-chain INS / DSF workflow after arXiv:2603.15608.

Requires: ``pip install numpy scipy`` (or ``pip install -e .`` from repo root).

This is a **classical exact-diagonalisation** reference, not IBM hardware noise.
For 50 qubits use MPS or hardware as in the paper.
"""

from __future__ import annotations

# ruff: noqa: T201

def main() -> None:
    import numpy as np

    from pyhqiv.ins_dsf_simulator import benchmark_ins_xxz_dense_vs_trotter

    # KCuF3-like: isotropic Heisenberg (epsilon=1), small chain for laptop ED
    n = 10
    b = benchmark_ins_xxz_dense_vs_trotter(
        n_sites=n,
        J=1.0,
        epsilon=1.0,
        delta_t=0.3,
        n_time_steps=24,
        beta="x",
        alpha="z",
    )
    res = b.dense

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; dense/Trotter RGF RMSE", b.rgf_rmse)
        print("spectral MSE", b.spectral_intensity_mse)
        return

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    tgrid, jgrid = np.meshgrid(res.times, np.arange(n))
    ax[0, 0].pcolormesh(
        tgrid,
        jgrid,
        res.rgf,
        shading="auto",
        cmap="RdBu_r",
    )
    ax[0, 0].set_xlabel("t")
    ax[0, 0].set_ylabel("site j")
    ax[0, 0].set_title("Dense expm: RGF ⟨σ^z_j⟩")

    ax[0, 1].pcolormesh(
        tgrid,
        jgrid,
        b.trotter.rgf,
        shading="auto",
        cmap="RdBu_r",
    )
    ax[0, 1].set_xlabel("t")
    ax[0, 1].set_ylabel("site j")
    ax[0, 1].set_title("Trotter: same observable grid")

    qg, wg = np.meshgrid(res.q_bins, res.omega_bins, indexing="ij")
    ax[1, 0].pcolormesh(
        wg,
        qg,
        res.spectral_intensity,
        shading="auto",
        cmap="magma",
    )
    ax[1, 0].set_xlabel("ω bin (FFT convention)")
    ax[1, 0].set_ylabel("q bin (2π l / L)")
    ax[1, 0].set_title("Dense |S| proxy")

    ax[1, 1].pcolormesh(
        wg,
        qg,
        b.trotter.spectral_intensity,
        shading="auto",
        cmap="magma",
    )
    ax[1, 1].set_xlabel("ω bin (FFT convention)")
    ax[1, 1].set_ylabel("q bin (2π l / L)")
    ax[1, 1].set_title(f"Trotter |S| (RGF RMSE={b.rgf_rmse:.4g})")
    fig.tight_layout()
    out = "ins_dsf_xxz_demo.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
