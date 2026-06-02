"""
HQIV Arena scoring engine — "sigma everywhere" philosophy.

Given a set of Metrics (from registry or explicit), compute:
- per-metric absolute/relative errors vs reference
- vector of "sigmas" (here: relative error used as proxy; can be replaced by
  external variance estimates in future)
- overall score that REWARDS reduction of error across many observables
- HARD PENALTY (or failure) for regressions on protected core metrics

The engine is pure and deterministic (fixed order, no seeds unless the
underlying compute() uses them — contributors must document + pin seeds).

Results are serializable to results.json for CI artifacts + leaderboards.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping, Optional, Any
import json
import math
from datetime import datetime, timezone

from .metrics import Metric, build_default_metrics, METRIC_REGISTRY


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    reference: float
    abs_err: float
    rel_err: float
    protected: bool
    weight: float
    unit: str
    desc: str


@dataclass
class ScoreResult:
    # Summary
    overall_score: float
    sigma_avg: float  # mean relative error proxy ("sigma")
    sigma_weighted: float
    num_metrics: int
    num_protected: int
    num_regressed_protected: int

    # Details
    metrics: List[MetricResult]
    deltas: Dict[str, float]  # name -> (prev_rel_err - curr_rel_err); positive = improvement

    # Provenance (for reproducibility)
    git_sha: Optional[str]
    git_ref: Optional[str]
    timestamp: str
    pyhqiv_version: str
    witnesses_source: str

    # Raw for downstream (leaderboard, badges)
    raw: Dict[str, Any]


def _rel_err(v: float, r: float) -> float:
    if abs(r) < 1e-300:
        return abs(v - r)
    return abs(v - r) / abs(r)


def _load_previous_baseline(path: Optional[str]) -> Dict[str, MetricResult]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prev: Dict[str, MetricResult] = {}
        for m in data.get("metrics", []):
            prev[m["name"]] = MetricResult(**{k: m[k] for k in MetricResult.__dataclass_fields__ if k in m})
        return prev
    except Exception:
        return {}


def compute_score(
    metrics: Optional[List[Metric]] = None,
    previous_results_path: Optional[str] = None,
    git_sha: Optional[str] = None,
    git_ref: Optional[str] = None,
    witnesses_source: str = "lean-witnesses",
    pyhqiv_version: str = "unknown",
) -> ScoreResult:
    """
    Compute a full ScoreResult.

    previous_results_path: path to a prior results.json (from main baseline).
                           Used only for delta computation; not for hard refs.
    """
    if metrics is None:
        metrics = build_default_metrics()

    results: List[MetricResult] = []
    for m in metrics:
        try:
            val = float(m.compute())
            ref = float(m.reference())
        except Exception as e:
            # On compute failure, record NaN but keep going (CI will see it)
            val = math.nan
            ref = float(m.reference()) if callable(m.reference) else 0.0
        ae = abs(val - ref) if math.isfinite(val) else float("inf")
        re = _rel_err(val, ref) if math.isfinite(val) else float("inf")
        results.append(
            MetricResult(
                name=m.name,
                value=val,
                reference=ref,
                abs_err=ae,
                rel_err=re,
                protected=m.protected,
                weight=m.weight,
                unit=m.unit,
                desc=m.desc,
            )
        )

    # sigma proxies
    rels = [r.rel_err for r in results if math.isfinite(r.rel_err)]
    sigma_avg = sum(rels) / len(rels) if rels else 0.0
    sigma_weighted = (
        sum(r.rel_err * r.weight for r in results if math.isfinite(r.rel_err))
        / sum(r.weight for r in results if math.isfinite(r.rel_err))
        if rels
        else 0.0
    )

    # Deltas vs previous (positive = this run is better / lower error)
    prev_map = _load_previous_baseline(previous_results_path)
    deltas: Dict[str, float] = {}
    for r in results:
        if r.name in prev_map:
            deltas[r.name] = prev_map[r.name].rel_err - r.rel_err
        else:
            deltas[r.name] = 0.0  # no baseline yet

    # Core regression guard
    num_regressed_protected = 0
    for r in results:
        if r.protected and r.name in prev_map:
            prev_re = prev_map[r.name].rel_err
            # Allow tiny numerical jitter; > 5% relative worsening on protected is regression
            if r.rel_err > prev_re * 1.05 + 1e-12:
                num_regressed_protected += 1

    # Overall score:
    #   Base = 1000 / (1 + sigma_weighted)   — lower aggregate sigma → higher score
    #   + 200 * sum( max(0, delta) for deltas )  — reward actual improvements
    #   - 5000 * num_regressed_protected     — hard penalty (can make score negative)
    improvement_bonus = 200.0 * sum(max(0.0, d) for d in deltas.values())
    regression_penalty = 5000.0 * num_regressed_protected
    base = 1000.0 / (1.0 + max(sigma_weighted, 0.0))
    overall = base + improvement_bonus - regression_penalty

    ts = datetime.now(timezone.utc).isoformat()

    raw: Dict[str, Any] = {
        "metrics": [asdict(r) for r in results],
        "deltas": deltas,
        "sigma_avg": sigma_avg,
        "sigma_weighted": sigma_weighted,
    }

    return ScoreResult(
        overall_score=round(overall, 4),
        sigma_avg=round(sigma_avg, 8),
        sigma_weighted=round(sigma_weighted, 8),
        num_metrics=len(results),
        num_protected=sum(1 for r in results if r.protected),
        num_regressed_protected=num_regressed_protected,
        metrics=results,
        deltas=deltas,
        git_sha=git_sha,
        git_ref=git_ref,
        timestamp=ts,
        pyhqiv_version=pyhqiv_version,
        witnesses_source=witnesses_source,
        raw=raw,
    )


def delta_vs_baseline(current: ScoreResult, baseline: Optional[ScoreResult]) -> Dict[str, float]:
    """Return per-metric signed improvement (positive = better)."""
    if not baseline:
        return {m.name: 0.0 for m in current.metrics}
    prev = {m.name: m.rel_err for m in baseline.metrics}
    return {m.name: prev.get(m.name, m.rel_err) - m.rel_err for m in current.metrics}


def serialize_score(result: ScoreResult, path: Optional[str] = None) -> Dict[str, Any]:
    """Return JSON-serializable dict; optionally write to path."""
    data = {
        "overall_score": result.overall_score,
        "sigma_avg": result.sigma_avg,
        "sigma_weighted": result.sigma_weighted,
        "num_metrics": result.num_metrics,
        "num_protected": result.num_protected,
        "num_regressed_protected": result.num_regressed_protected,
        "timestamp": result.timestamp,
        "git_sha": result.git_sha,
        "git_ref": result.git_ref,
        "pyhqiv_version": result.pyhqiv_version,
        "witnesses_source": result.witnesses_source,
        "metrics": [asdict(m) for m in result.metrics],
        "deltas": result.deltas,
        "raw": result.raw,
    }
    if path:
        p = __import__("pathlib").Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    return data
