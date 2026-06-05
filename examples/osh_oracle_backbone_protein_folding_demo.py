"""
OSHoracle sparse minimization demo on a synthetic heavy-atom backbone.

This mirrors the PROtien API:
  - horizon_physics.proteins.osh_oracle_backbone.minimize_backbone_with_osh_oracle

Run (from this checkout):
  ./.venv/bin/python examples/osh_oracle_backbone_protein_folding_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _ensure_protein_folder_on_path() -> None:
    # hqvmpy/examples/... -> hqvmpy -> Repos -> PROtien
    repo_root = Path(__file__).resolve().parents[2]
    protien_src = repo_root / "PROtien" / "src"
    if str(protien_src) not in sys.path:
        sys.path.insert(0, str(protien_src))


def main() -> None:
    _ensure_protein_folder_on_path()

    from horizon_physics.proteins.casp_submission import _place_full_backbone
    from horizon_physics.proteins.osh_oracle_backbone import minimize_backbone_with_osh_oracle

    # --- synthetic starting point ---
    seq = "ACAG"  # small so the demo runs quickly
    n_res = len(seq)

    rng = np.random.default_rng(0)
    # Simple Cα straight-ish chain with small perturbations.
    ca = np.stack(
        [
            np.array([3.8 * i, 0.15 * float(rng.normal()), 0.15 * float(rng.normal())], dtype=float)
            for i in range(n_res)
        ],
        axis=0,
    )

    # Convert Cα positions -> (N, CA, C, O) for each residue.
    bb_atoms = _place_full_backbone(ca, seq)
    pos_init = np.array([xyz for _, xyz in bb_atoms], dtype=float)

    print("=== OSHoracle Backbone Folding Demo (synthetic) ===")
    print(f"Sequence: {seq}  | residues: {n_res}  | atoms: {pos_init.shape[0]}")

    # --- run sparse OSHoracle minimization ---
    pos_final, info = minimize_backbone_with_osh_oracle(
        pos_init,
        sequence=seq,
        n_iter=10,
        step_size=0.02,
        ansatz_depth=2,
        use_energy_reservoir=False,  # keep acceptance rule simple for the demo
        strict_descent_budget_mode=True,
        random_seed=0,
    )

    # --- report end state ---
    d = np.linalg.norm(pos_final - pos_init) / np.sqrt(pos_init.size)
    print(f"iterations_executed: {info.iterations_executed}/{info.iterations}")
    print(f"accepted_steps:      {info.accepted_steps}")
    print(f"final_energy_ev:     {info.final_energy_ev:.6e}")
    print(f"last_flipped_count:  {info.last_flipped_count}")
    print(f"avg_flipped_count:   {info.avg_flipped_count:.6f}")
    print(f"RMSD(backbone atoms): {d:.6e} (grid units)")

    # Print coordinates of first residue only (N, CA, C, O).
    atoms_per_res = 4
    first_res_atoms = pos_final[:atoms_per_res]
    print("Final coordinates (residue 0 atoms N/CA/C/O):")
    for k in range(atoms_per_res):
        print(f"  atom{k}: {first_res_atoms[k].tolist()}")


if __name__ == "__main__":
    main()

