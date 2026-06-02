#!/usr/bin/env python3
"""
HQIV Arena score computation entrypoint (used by CI).

Runs the default metrics, compares to optional baseline, writes results.json,
and prints a compact human summary + badge preview.

Intended to be called after Stages 1-3 have passed.

Example (in CI):
  python -m pyhqiv.arena  --baseline arena/baseline.json --out arena/results.json --git-sha $GITHUB_SHA
or
  python scripts/arena/compute_score.py --out results.arena.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make src importable when run from repo root (CI does "pip install -e .")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from pyhqiv.arena import build_default_metrics, compute_score, serialize_score, award_badges  # type: ignore
except Exception:
    # Fallback for direct script runs without install
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from pyhqiv.arena import build_default_metrics, compute_score, serialize_score, award_badges  # type: ignore


def _get_env(k: str, default: str | None = None) -> str | None:
    return os.environ.get(k, default)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="arena_results.json", help="Path for results JSON")
    p.add_argument("--baseline", type=str, default=None, help="Previous results.json for deltas (usually from main)")
    p.add_argument("--git-sha", type=str, default=None)
    p.add_argument("--git-ref", type=str, default=None)
    p.add_argument("--print-badges", action="store_true")
    args = p.parse_args(argv)

    sha = args.git_sha or _get_env("GITHUB_SHA")
    ref = args.git_ref or _get_env("GITHUB_REF_NAME") or _get_env("GITHUB_REF")

    # Try to get a clean witnesses source string
    try:
        from pyhqiv.lean_witnesses import _default_witnesses_path  # type: ignore

        ws = str(_default_witnesses_path())
    except Exception:
        ws = "lean-witnesses"

    try:
        from pyhqiv._version import __version__ as pyv  # type: ignore
    except Exception:
        pyv = "dev"

    res = compute_score(
        metrics=build_default_metrics(),
        previous_results_path=args.baseline,
        git_sha=sha,
        git_ref=ref,
        witnesses_source=ws,
        pyhqiv_version=pyv,
    )

    data = serialize_score(res, args.out)

    # Preview
    print("=== HQIV Arena Score ===")
    print(f"overall_score: {res.overall_score}")
    print(f"sigma_weighted: {res.sigma_weighted}")
    print(f"protected_regressions: {res.num_regressed_protected}")
    print(f"metrics: {res.num_metrics} (protected: {res.num_protected})")
    print(f"wrote: {args.out}")

    if args.print_badges or _get_env("CI"):
        # Provisional badge computation (CI will re-compute on merge with full history)
        badges = award_badges(
            is_merge_to_main=(ref in ("main", "master")),
            sigma_improved=(res.sigma_weighted < 0.05),  # heuristic; real uses deltas
            num_regressed_protected=res.num_regressed_protected,
            new_tests_added=0,  # CI can pass via --new-tests or env
            rank=None,
            time_at_top=0,
        )
        print(f"provisional_badges: {badges}")

    # Also emit a tiny summary for PR comments
    summary = {
        "overall_score": res.overall_score,
        "sigma_weighted": res.sigma_weighted,
        "regressions": res.num_regressed_protected,
        "improvements": sum(1 for d in res.deltas.values() if d > 0),
    }
    print("SUMMARY_JSON:" + json.dumps(summary))

    return 0 if res.num_regressed_protected == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
