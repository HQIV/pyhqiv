"""
Timing / protocol presets aligned with arXiv:2603.15608 (Lee et al.) Figure 2 captions.

The hardware experiment uses **50 qubits** and MPS reference; dense ``expm`` here is
limited to modest ``n_sites``. Use ``n_sites`` as large as your machine allows and
interpret finite-size effects versus the published INS panels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PaperMode = Literal["kcuf3", "xx"]


@dataclass(frozen=True)
class INSPaperPreset:
    """Published discrete-time protocol (simulation / MPS / hardware)."""

    mode: PaperMode
    description: str
    epsilon: float
    delta_t: float
    n_time_steps: int
    J: float
    beta: Literal["x", "y", "z"]
    alpha: Literal["x", "y", "z"]


def preset_kcuf3_fig2() -> INSPaperPreset:
    """Fig. 2D,E — isotropic 1D XXZ (Heisenberg) point, KCuF\\ :sub:`3` chain; 20 Trotter intervals."""
    return INSPaperPreset(
        mode="kcuf3",
        description="KCuF3 class: ε=1, Δt=0.6, N_t=20 (Fig. 2 caption, D/E).",
        epsilon=1.0,
        delta_t=0.6,
        n_time_steps=20,
        J=1.0,
        beta="x",
        alpha="z",
    )


def preset_xx_fig2() -> INSPaperPreset:
    """Fig. 2B,C — XX / ballistic limit; 30 intervals."""
    return INSPaperPreset(
        mode="xx",
        description="XX model: ε=0, Δt=0.6, N_t=30 (Fig. 2 caption, B/C).",
        epsilon=0.0,
        delta_t=0.6,
        n_time_steps=30,
        J=1.0,
        beta="x",
        alpha="z",
    )


def preset_for_mode(mode: PaperMode) -> INSPaperPreset:
    if mode == "kcuf3":
        return preset_kcuf3_fig2()
    return preset_xx_fig2()


__all__ = [
    "INSPaperPreset",
    "PaperMode",
    "preset_for_mode",
    "preset_kcuf3_fig2",
    "preset_xx_fig2",
]
