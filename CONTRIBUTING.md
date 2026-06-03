# Contributing to pyhqiv + HQIV Arena

Thank you for helping make HQIV a living, crowdsourced physics improvement engine.

## Core Principles

- **main is sacred and always certified.** Every commit on `main` must pass the full Lean lake build (including SO(8) closure, Spin(8) triality, GR+SM unification, horizon curvature, lapse relations, and all other proved theorems). These are certificates. No exceptions.
- **Lean ↔ Python bidirectional alignment is a hard gate.** Numerical results in pyhqiv must be consistent with formally derived values from hqiv-lean (within the tolerances defined in the alignment script). Functional helpers only — no hard-coded constants in scoring/alignment logic.
- **New feature → new test.** Adding capability (phase diagram generator, new fluid observable, improved lattice combinatorics, new resonance channel, …) requires corresponding new tests. Those tests become part of the permanent suite upon merge.
- **"Sigma everywhere".** We reward measurable, broad reductions in error/variance (σ) across multiple physical observables. Large regressions on any protected core metric (Ω_k horizons, lapse, mode counts, proton anchor, so(8) dimension, derived masses, …) are heavily penalized or cause the submission to be ineligible.

## Branch & PR Workflow (HQIV Arena)

All development happens on branches:

- `feat/`, `arena/`, `benchmark/`, or your-name/...
- CI (`hqiv-arena.yml`) runs on every push and every PR to these branches (and main).
- Leaderboard + scoring work on branches (you see your provisional rank and score in the PR).
- Only high-quality, aligned, improving changes that keep main green are merged.

### The Five-Stage Pipeline (all must pass for a score)

1. **Lean Certificate Gate** (hard)
   - Full `lake build HQIVSO8Closure` (or equivalent heavy targets covering all major theorems).
   - Zero `sorry`s. Proof-checking time is recorded.
   - See the workflow in hqiv-lean and the lean job in pyhqiv's arena workflow.

2. **Lean ↔ Python Alignment Gate** (hard)
   - `python scripts/check_arena_source_integrity.py` — AST gate on `lightcone.py` / `metric.py` (no import creep, no literal-return cheats in Ω_k mirrors).
   - `python scripts/validate_hqiv_alignment.py` — must pass 100%. Uses Lean-exported `witnesses.json` + functional mirror checks in pyhqiv (lightcone, metric, so8 generators, etc.).
   - If you changed a Lean definition that affects a numerical value, update the export and the py mirror (or the alignment will fail).

3. **Python Test Gate** (hard)
   - Full `pytest` (including paper validation, reproducibility checks, and your new tests).

4. **Scoring & Evaluation**
   - Only reached if 1-3 are green.
   - Computes a vector of metrics + deltas vs the current baseline on main.
   - "Sigma everywhere" logic: broad improvement rewarded; protected regressions penalized.
   - Produces `arena_results.json` (artifact + comment on PR).

5. **Leaderboard & Badges**
   - On merge to main the results are committed to `arena/leaderboard.json` (and `arena/baseline.json` for future deltas).
   - Badges are awarded automatically (see below).

## Adding a New Benchmark / Observable (the standard workflow)

1. Implement the physics (new module or extension).
2. Add tests that validate physical consistency (e.g. for a phase diagram: phase boundaries, thermodynamic stability, thermodynamic limit checks, consistency with Lean-derived shell quantities).
3. (Recommended) Register a new Arena metric so the improvement is scored:

   ```python
   # in your new test file or a small arena registration module
   from pyhqiv.arena.metrics import register_metric, Metric

   def my_phase_quality() -> float:
       # deterministic compute of a quality / error metric
       ...

   register_metric(Metric(
       name="phase_diagram_consistency_v1",
       compute=my_phase_quality,
       reference=lambda: 0.0,   # or load from witness / golden
       protected=False,
       weight=1.0,
       unit="",
       desc="Phase boundary + thermodynamic stability quality (new in this PR)",
   ))
   ```

4. The new metric will automatically participate in scoring on the next run.
5. Open PR. CI will show the impact on overall score and whether it improves σ.

Templates live in `arena/templates/`.

## Badges (awarded automatically on merge)

- **Merged Features** — impactful capability landed on main.
- **Sigma Improver** — consistent broad error reduction with no protected regressions.
- **Highest Position Held** — record holder or longest cumulative time at #1.
- **Best New Test Suite** — high-quality new tests that increase physical coverage.
- **Efficiency Leader** — excellent score per compute / proof time.
- **Lean Certificate Champion** — multiple full certified builds on work that merged.
- **Alignment Guardian** — extended or hardened the Lean ↔ py validation.

## Local Development & Testing the Arena

```bash
# alignment gate (uses committed + overlay witnesses)
python3 scripts/validate_hqiv_alignment.py --verbose

# scoring (compares to arena/baseline.json if present)
python3 scripts/arena/compute_score.py --out /tmp/arena.json --print-badges

# update leaderboard locally (dry)
python3 scripts/arena/update_leaderboard.py --results /tmp/arena.json --dry-run
```

To simulate a fresh Lean export:

```bash
cd /path/to/hqiv-lean
lake env lean -q --run scripts/export_witnesses.lean
# then run alignment from pyhqiv checkout (the overlay loader will pick it up)
```

## Determinism & Reproducibility

- All Arena benchmarks must be deterministic or use fixed seeds + report statistics.
- Golden / reference outputs for new benchmarks should be committed (under `tests/data/` or `arena/goldens/`).
- Never rely on wall time or non-reproducible network/FS state in scoring.

## Documentation & Alectryon

- The Lean side maintains Alectryon / docgen in its own CI (see hqiv-lean workflows).
- When you add new Lean modules that are cited by papers or the story, make sure the appropriate `paper_*` lake targets and doc facets still build.

## Questions / Coordination

For large cross-repo changes (Lean + py), open matching branches (`feat/xyz` in both) and reference each other. The Arena CI tries to detect same-named branches for alignment.

Welcome to the Arena. Let's reduce σ together.

## Using the hqiv-arena CLI (recommended for agents and rapid iteration)

```bash
pip install -e ".[dev]"
hqiv-arena install-skill   # for your coding agent
hqiv-arena clone my-hqiv-work
cd my-hqiv-work/pyhqiv
hqiv-arena setup
hqiv-arena run
# ... edit code + add tests ...
hqiv-arena submit --note-file progress.md --model "Claude 4 Opus"
```

Full reference lives in `arena/SKILL.md` (and gets installed to `~/.agents/skills/hqiv-arena/SKILL.md` etc.).
