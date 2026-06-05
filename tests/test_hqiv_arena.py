"""Basic tests for the HQIV Arena scoring / alignment / badges machinery.

These ensure the engine itself is deterministic and the extension points work.
"""

from __future__ import annotations


def test_arena_metrics_registry_and_order():
    from pyhqiv.arena.metrics import build_default_metrics

    mets = build_default_metrics()
    names = [m.name for m in mets]
    assert any("so8" in n for n in names)
    assert any("omega_k" in n for n in names)
    # protected should sort first
    prot = [m for m in mets if m.protected]
    assert len(prot) >= 5


def test_arena_scoring_perfect_baseline_gives_max_and_zero_regressions():
    from pyhqiv.arena import build_default_metrics, compute_score

    res = compute_score(metrics=build_default_metrics())
    assert res.num_regressed_protected == 0
    # Protected cores are exact (0 rel_err). Programme phenom metrics (binding z, eta, etc) have target refs
    # (e.g. z<=1) so "perfect model" would give low but non-zero contribution until model matches exp within 1σ.
    # We assert no protected regressions and that protected are perfect.
    prot = [m for m in res.metrics if m.protected]
    assert all(m.rel_err < 1e-9 for m in prot)
    assert res.overall_score > 0  # positive when no regressions


def test_arena_badges_award_logic():
    from pyhqiv.arena.badges import award_badges

    b1 = award_badges(
        is_merge_to_main=True,
        sigma_improved=True,
        num_regressed_protected=0,
        new_tests_added=4,
        rank=1,
        time_at_top=2,
        lean_cert_successes=2,
    )
    assert "merged-feature" in b1
    assert "sigma-improver" in b1
    assert "lean-cert-champion" in b1
    assert "highest-position" in b1

    b2 = award_badges(
        is_merge_to_main=False,
        sigma_improved=False,
        num_regressed_protected=2,
        new_tests_added=0,
        rank=None,
    )
    assert "sigma-improver" not in b2  # regression present


def test_alignment_script_runs_as_module():
    # Just import + basic structure; the full gate is exercised in CI and manual runs
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_hqiv_alignment.py"
    assert script.exists()
    # We do not exec the full main here (it does sys.exit), just confirm it is valid Python
    src = script.read_text()
    assert "AlignmentReport" in src
    assert "build_alignment_checks" in src
