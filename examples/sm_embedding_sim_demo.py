#!/usr/bin/env python3
"""
Demo: SM quantum numbers from ``SMEmbedding.lean`` + optional U(1)/SU(2) moves on ℂ⁸.

Requires ``scipy``. For SU(2)ₗ matrices, set ``PYTHONPATH`` to include a checkout
that contains ``HQVM/matrices.py`` (e.g. ``~/Repos/HQIV`` next to ``hqvmpy``).

  PYTHONPATH=src:~/Repos/HQIV python3 examples/sm_embedding_sim_demo.py
"""

from __future__ import annotations


def main() -> None:
    import numpy as np

    from pyhqiv.sm_embedding_sim import (
        HYPERCHARGE_Y2,
        evolve_u1_hypercharge_sm_basis,
        random_normalized_state_8,
        sm_hypercharge_y2_for_label,
        su2_l_generators_from_octonion,
        verify_su2_lie_algebra,
    )

    print("Y/2 eigenvalues (Fin 8), SMEmbedding.hyperchargeEigenvalue:")
    for i, y in enumerate(HYPERCHARGE_Y2):
        print(f"  i={i}: Y/2 = {y}")
    print("electron Y/2 weight:", sm_hypercharge_y2_for_label("electron"))

    psi = random_normalized_state_8(seed=42)
    psi2 = evolve_u1_hypercharge_sm_basis(psi, 0.15)
    print("U(1) evolution: |psi| before/after", np.linalg.norm(psi), np.linalg.norm(psi2))

    try:
        su2 = su2_l_generators_from_octonion()
    except RuntimeError as e:
        print("SU(2) generators:", e)
        return
    ok = verify_su2_lie_algebra(su2)
    print("[T1,T2] ≈ -T3 (su2_bracket_12):", ok)


if __name__ == "__main__":
    main()
