"""
HQIV Arena automatic badge system.

Badges are awarded based on:
- Merged impact (features + tests merged to main)
- Measurable sigma improvement on protected + broad observables
- Sustained contribution (time at #1, cumulative score, number of qualifying PRs)
- New capability (new test suites that increase coverage of physical regimes)

Badges live in the leaderboard JSON and are displayed on the site.
They are intentionally conservative — only awarded on *merged* work for the
"persistent" ones; PR comments may show "provisional" badges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass(frozen=True)
class Badge:
    key: str
    label: str
    desc: str
    tier: str = "standard"  # standard / silver / gold / legendary


BADGE_DEFS: Dict[str, Badge] = {
    "merged-feature": Badge(
        "merged-feature",
        "Merged Features",
        "Landed impactful new capability (new Lean theorem pack, new physics module, or major benchmark) to main.",
        "standard",
    ),
    "sigma-improver": Badge(
        "sigma-improver",
        "Sigma Improver",
        "Consistently reduced aggregate error (σ) across multiple observables with no protected regressions.",
        "silver",
    ),
    "highest-position": Badge(
        "highest-position",
        "Highest Position Held",
        "Held #1 on the overall leaderboard for the longest cumulative period or achieved the single highest all-time score.",
        "gold",
    ),
    "new-test-suite": Badge(
        "new-test-suite",
        "Best New Test Suite",
        "Added high-quality, physically meaningful new tests that measurably increased coverage (phase diagrams, new regimes, fluid/molecular observables, etc.).",
        "standard",
    ),
    "efficiency-leader": Badge(
        "efficiency-leader",
        "Efficiency Leader",
        "Best score per unit wall / CPU time or smallest Lean proof-checking delta while improving physics.",
        "standard",
    ),
    "lean-cert-champion": Badge(
        "lean-cert-champion",
        "Lean Certificate Champion",
        "Multiple successful full lake builds (including SO(8) closure + major theorems) with zero sorrys on high-impact branches that later merged.",
        "gold",
    ),
    "alignment-guardian": Badge(
        "alignment-guardian",
        "Alignment Guardian",
        "Strengthened or extended the Lean ↔ Python bidirectional validation (new keys in export, tighter tolerances, new mirror functions) that became permanent.",
        "silver",
    ),
}


def award_badges(
    *,
    is_merge_to_main: bool,
    sigma_improved: bool,
    num_regressed_protected: int,
    new_tests_added: int,
    rank: Optional[int],
    time_at_top: int = 0,  # in "merge epochs" or days; maintained by leaderboard updater
    efficiency_rank: Optional[int] = None,
    lean_cert_successes: int = 0,
    alignment_extensions: int = 0,
    cumulative_score: float = 0.0,
    previous_badges: Optional[List[str]] = None,
) -> List[str]:
    """
    Return list of badge keys earned by this contribution.

    This is called by the leaderboard updater (on main merges) and can be
    called in PR CI for "provisional" display.
    """
    earned: List[str] = []
    prev = set(previous_badges or [])

    if is_merge_to_main:
        # Merged feature / capability
        if new_tests_added >= 1 or lean_cert_successes >= 1:
            earned.append("merged-feature")

        # Lean cert champion (sustained)
        if lean_cert_successes >= 2:
            earned.append("lean-cert-champion")

        # Alignment guardian
        if alignment_extensions >= 1:
            earned.append("alignment-guardian")

        # New test suite quality
        if new_tests_added >= 3:
            earned.append("new-test-suite")

    # Sigma improver — only if no protected regressions and actual improvement
    if sigma_improved and num_regressed_protected == 0:
        earned.append("sigma-improver")

    # Highest position (either current rank 1 with history, or record holder)
    if rank == 1 or time_at_top >= 3:
        earned.append("highest-position")

    if efficiency_rank is not None and efficiency_rank <= 3:
        earned.append("efficiency-leader")

    # Keep previously earned persistent badges (they are not lost)
    for b in prev:
        if b not in earned:
            earned.append(b)

    # Dedup while preserving some order
    seen = set()
    out = []
    for b in earned:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def badge_label(key: str) -> str:
    b = BADGE_DEFS.get(key)
    return b.label if b else key


def describe_badges(keys: List[str]) -> List[Dict[str, str]]:
    out = []
    for k in keys:
        b = BADGE_DEFS.get(k)
        if b:
            out.append({"key": k, "label": b.label, "desc": b.desc, "tier": b.tier})
        else:
            out.append({"key": k, "label": k, "desc": "", "tier": "standard"})
    return out
