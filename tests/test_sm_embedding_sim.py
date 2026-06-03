"""SMEmbedding.lean-aligned quantum numbers and optional HQVM SU(2) dynamics."""

import math

import numpy as np

from pyhqiv.sm_embedding import sm_hypercharge_weight
from pyhqiv.sm_embedding_sim import (
    HYPERCHARGE_Y2,
    branching_sector_dims,
    charge_from_y,
    evolve_u1_hypercharge_sm_basis,
    hypercharge_eigenvalue,
    random_normalized_state_8,
    sm_hypercharge_y2_for_label,
    verify_su2_lie_algebra,
)


def test_hypercharge_table_matches_lean() -> None:
    assert HYPERCHARGE_Y2[0] == HYPERCHARGE_Y2[1] == 1 / 6
    assert HYPERCHARGE_Y2[2] == -2 / 3
    assert HYPERCHARGE_Y2[3] == 1 / 3
    assert HYPERCHARGE_Y2[4] == HYPERCHARGE_Y2[5] == -1 / 2
    assert HYPERCHARGE_Y2[6] == 1
    assert HYPERCHARGE_Y2[7] == 0


def test_nu_r_charge_zero() -> None:
    assert charge_from_y(7, 0.0) == 0.0


def test_branching_sum_16() -> None:
    d = branching_sector_dims()
    assert sum(d.values()) == 16


def test_sm_hypercharge_consistent_with_sm_embedding() -> None:
    for lab in (
        "electron",
        "muon",
        "tau",
        "up",
        "down",
        "top",
        "nu_e",
    ):
        assert math.isclose(
            sm_hypercharge_weight(lab),  # type: ignore[arg-type]
            sm_hypercharge_y2_for_label(lab),  # type: ignore[arg-type]
        )


def test_u1_evolution_unitary() -> None:
    psi = random_normalized_state_8(seed=1)
    out = evolve_u1_hypercharge_sm_basis(psi, 0.3)
    assert math.isclose(float(np.linalg.norm(out)), 1.0, rel_tol=0.0, abs_tol=1e-9)


def test_su2_generators_satisfy_bracket_if_hqvm() -> None:
    try:
        from pyhqiv.sm_embedding_sim import su2_l_generators_from_octonion
    except Exception:
        return
    try:
        g = su2_l_generators_from_octonion()
    except RuntimeError:
        return
    assert verify_su2_lie_algebra(g)


def test_hypercharge_index_helpers() -> None:
    assert hypercharge_eigenvalue(6) == 1.0
    assert hypercharge_eigenvalue(7) == 0.0
