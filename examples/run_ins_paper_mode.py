#!/usr/bin/env python3
"""
Run INS / DSF simulation in **arXiv:2603.15608 paper protocol** (Fig. 2 captions).

Compares **dense expm** (exact-in-Δt reference) vs **first-order Trotter** on the
same grid. Experimental INS curves are not bundled here—export ``.npz`` for
overlay in plotting tools or future χ² / SSIM against published data.

Example:
  PYTHONPATH=src python3 examples/run_ins_paper_mode.py --mode kcuf3 --n-sites 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="INS paper-mode spin-chain sim (Lee et al.)")
    parser.add_argument("--mode", choices=("kcuf3", "xx"), default="kcuf3")
    parser.add_argument(
        "--n-sites",
        type=int,
        default=10,
        help="Chain length (paper uses 50 on hardware; dense expm is practical for n≲11).",
    )
    parser.add_argument(
        "--dense-max-qubits",
        type=int,
        default=11,
        help="Run dense+Trotter benchmark only if n_sites <= this (Hilbert dim 2^n).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional path to write .npz (rgf_dense, rgf_trotter, times, q_bins, omega_bins, ...).",
    )
    args = parser.parse_args()

    repo_src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(repo_src))

    from pyhqiv.ins_dsf_simulator import (
        benchmark_ins_xxz_dense_vs_trotter,
        simulate_xxz_rgf_center_kick_trotter,
    )
    from pyhqiv.ins_paper_presets import preset_for_mode

    preset = preset_for_mode(args.mode)
    print("=== INS paper preset ===")
    print(preset.description)
    print(
        f"  ε={preset.epsilon}  Δt={preset.delta_t}  N_t={preset.n_time_steps}  "
        f"J={preset.J}  kick={preset.beta}  observe=σ^{preset.alpha}"
    )
    print(f"  n_sites={args.n_sites} (paper hardware/MPS: 50)")
    print()

    if args.n_sites > args.dense_max_qubits:
        print(
            f"n_sites > {args.dense_max_qubits}: Trotter-only (skip dense expm for cost).",
        )
        tr = simulate_xxz_rgf_center_kick_trotter(
            n_sites=args.n_sites,
            J=preset.J,
            epsilon=preset.epsilon,
            delta_t=preset.delta_t,
            n_time_steps=preset.n_time_steps,
            beta=preset.beta,
            alpha=preset.alpha,
        )
        summary = {
            "mode": preset.mode,
            "n_sites": args.n_sites,
            "evolution": "trotter_first_order_only",
            "rgf_shape": list(tr.rgf.shape),
            "rgf_max_abs": float(abs(tr.rgf).max()),
            "spectral_intensity_max": float(tr.spectral_intensity.max()),
            "t_total": float(tr.times[-1]),
        }
        print(json.dumps(summary, indent=2))
        if args.out:
            import numpy as np

            np.savez(
                args.out,
                times=tr.times,
                rgf_trotter=tr.rgf,
                q_bins=tr.q_bins,
                omega_bins=tr.omega_bins,
                spectral_intensity_trotter=tr.spectral_intensity,
                preset_mode=np.array([preset.mode]),
            )
            print("wrote", args.out)
        return 0

    bench = benchmark_ins_xxz_dense_vs_trotter(
        n_sites=args.n_sites,
        J=preset.J,
        epsilon=preset.epsilon,
        delta_t=preset.delta_t,
        n_time_steps=preset.n_time_steps,
        beta=preset.beta,
        alpha=preset.alpha,
    )
    summary = {
        "mode": preset.mode,
        "n_sites": args.n_sites,
        "t_total": float(bench.dense.times[-1]),
        "rgf_rmse_dense_vs_trotter": bench.rgf_rmse,
        "rgf_max_abs_diff": bench.rgf_max_abs_diff,
        "spectral_intensity_mse": bench.spectral_intensity_mse,
        "spectral_wasserstein": bench.spectral_wasserstein,
        "rgf_max_abs_dense": float(abs(bench.dense.rgf).max()),
        "rgf_max_abs_trotter": float(abs(bench.trotter.rgf).max()),
        "spectral_max_dense": float(bench.dense.spectral_intensity.max()),
        "spectral_max_trotter": float(bench.trotter.spectral_intensity.max()),
    }
    print("=== Benchmark (same protocol, two classical backends) ===")
    print(json.dumps(summary, indent=2))
    print()
    print(
        "Note: INS experiment curves in the paper are normalized and measured at finite T, L; "
        "this run uses open BCs and small L for exact reference. Scale J (energy) to align ω axis "
        "when comparing shapes.",
    )

    if args.out:
        import numpy as np

        np.savez(
            args.out,
            times=bench.dense.times,
            rgf_dense=bench.dense.rgf,
            rgf_trotter=bench.trotter.rgf,
            q_bins=bench.dense.q_bins,
            omega_bins=bench.dense.omega_bins,
            spectral_intensity_dense=bench.dense.spectral_intensity,
            spectral_intensity_trotter=bench.trotter.spectral_intensity,
            preset_mode=np.array([preset.mode]),
        )
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
