"""
HQIV Arena — branch/CI scoring and leaderboard primitives for pyhqiv + hqiv-lean.

This package provides:
- Modular metric definitions ("sigma everywhere")
- Scoring engine with protected core regression guards
- Badge awarding
- Result + leaderboard JSON schemas (pure stdlib + numpy optional)

All numerical references come from functional helpers or Lean witnesses (no
hard-coded magic numbers in scoring rules).

See scripts/arena/ for CLI entrypoints and the HQIV Arena CI workflow.
"""

from __future__ import annotations

__version__ = "0.1.0-arena"

# Re-exports for convenience in CI / notebooks
from .metrics import (  # noqa: F401
    METRIC_REGISTRY,
    Metric,
    build_default_metrics,
    register_metric,
)
from .scoring import (  # noqa: F401
    ScoreResult,
    compute_score,
    delta_vs_baseline,
    serialize_score,
)
from .badges import (  # noqa: F401
    Badge,
    award_badges,
    BADGE_DEFS,
)

__all__ = [
    "METRIC_REGISTRY",
    "Metric",
    "build_default_metrics",
    "register_metric",
    "ScoreResult",
    "compute_score",
    "delta_vs_baseline",
    "Badge",
    "award_badges",
    "BADGE_DEFS",
]
