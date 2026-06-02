#!/usr/bin/env python3
"""
HQIV Arena leaderboard updater (called from CI on main, or manually).

Merges a new results.json into arena/leaderboard.json, awards badges using
history, updates baseline, and can commit.

This is the implementation behind Stage 5.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pyhqiv.arena.badges import award_badges  # type: ignore


def load_json(p: Path, default: dict):
    if p.exists():
        return json.load(open(p))
    return default


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="arena_results.json")
    ap.add_argument("--leaderboard", default="arena/leaderboard.json")
    ap.add_argument("--baseline", default="arena/baseline.json")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    res = json.load(open(args.results))
    lb = load_json(Path(args.leaderboard), {"entries": [], "current_best": None, "history": [], "badges": {}})
    base = load_json(Path(args.baseline), {})

    entry = {
        "branch": os.environ.get("GITHUB_REF_NAME", "local"),
        "sha": os.environ.get("GITHUB_SHA", "local"),
        "author": os.environ.get("GITHUB_ACTOR", os.environ.get("USER", "local")),
        "score": res.get("overall_score"),
        "sigma_weighted": res.get("sigma_weighted"),
        "timestamp": res.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "metrics": {m["name"]: m for m in res.get("metrics", [])},
        "regressions": res.get("num_regressed_protected", 0),
    }

    # History-aware badges
    prev_best = (lb.get("current_best") or {}).get("score") or 0
    is_best = (entry["score"] or 0) >= prev_best
    badges = award_badges(
        is_merge_to_main=True,
        sigma_improved=(entry.get("sigma_weighted", 99) <= ((lb.get("current_best") or {}).get("sigma_weighted") or 99)),
        num_regressed_protected=entry["regressions"],
        new_tests_added=int(os.environ.get("NEW_TESTS_ADDED", "0")),
        rank=1 if is_best else None,
        time_at_top=1 if is_best else 0,
        lean_cert_successes=1,
    )
    entry["badges"] = badges

    lb["entries"] = [e for e in lb.get("entries", []) if e.get("sha") != entry["sha"]]
    lb["entries"].append(entry)
    lb["entries"] = lb["entries"][-100:]

    if is_best or not lb.get("current_best"):
        lb["current_best"] = entry

    lb["history"].append({"ts": entry["timestamp"], "score": entry["score"], "sigma": entry.get("sigma_weighted")})
    lb["history"] = lb["history"][-500:]

    Path(args.leaderboard).parent.mkdir(parents=True, exist_ok=True)
    json.dump(lb, open(args.leaderboard, "w"), indent=2, sort_keys=True)

    # baseline for branch comparisons
    json.dump({"overall_score": entry["score"], "metrics": res.get("metrics", [])}, open(args.baseline, "w"), indent=2)

    print(f"Updated leaderboard with score {entry['score']} (best={is_best}) badges={badges}")

    if args.commit:
        subprocess.run(["git", "add", args.leaderboard, args.baseline], check=True)
        subprocess.run(["git", "commit", "-m", f"arena: leaderboard update {entry['sha'][:8]}"], check=True)
        subprocess.run(["git", "push"], check=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
