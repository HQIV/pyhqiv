# HQIV Arena (local data + docs)

This directory contains:

- `leaderboard.json` — committed public leaderboard (updated by CI on merges to main)
- `baseline.json` — last green main results, used by branches for delta scoring
- `templates/` — contributor templates for new benchmarks and Lean extensions
- `goldens/` — optional committed reference outputs for new deterministic benchmarks

The actual implementation lives in:

- `src/pyhqiv/arena/` (metrics, scoring, badges — importable)
- `scripts/validate_hqiv_alignment.py` (the hard Lean ↔ Python gate)
- `scripts/arena/compute_score.py` and `update_leaderboard.py`
- `.github/workflows/hqiv-arena.yml`

See the top-level `CONTRIBUTING.md` for the full workflow, gates, and "sigma everywhere" philosophy.

The live leaderboard is also rendered at https://disregardfiat.tech/#arena (pulls the JSON from this repo's main branch).

## Calculator rules for contributors (src + arena submissions)
- **0 constants in src/** except geometry (π, √3, 2π, naturals). All new functions (incl. second-order + corrections) must be completely dynamic / pure.
- Tests + setup_defaults.py carry scale witnesses + local conditions (proton, earth surface, earth vacuum, CMB now, ...) with explicit (value, ±err, "Source") from the authoritative paper/experiment.
- Every feature lands with tests that have error bars; arena CI only promotes if new work beats sigma in general (no protected regressions).
- See CONTRIBUTING.md and arena/templates/new_benchmark_test.py.template .

## CLI

After `pip install -e .[dev]` (or from the source tree with the wrapper), the command `hqiv-arena` is available.

See the `SKILL.md` (in this dir or installed to your agent's skills folder) for full usage, especially `clone`, `run`, `submit`, `sync`, and `install-skill`.

The CLI is the primary way solvers (humans or AI agents) participate in the Arena.
