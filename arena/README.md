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

## Programme σ map (open problems)

`arena/programme_sigma.json` maps a curated subset of the
[Wikipedia list of unsolved problems in physics](https://en.wikipedia.org/wiki/List_of_unsolved_problems_in_physics)
to HQIV programme status and the current Arena metric snapshot. Regenerate after metric changes:

```bash
PYTHONPATH=src python scripts/export_programme_sigma.py
```

The disregardfiat.tech site loads this file at `#mysteries` (bundled under `public/arena/`).

The live leaderboard is also rendered at https://disregardfiat.tech/#arena (pulls the JSON from this repo's main branch).

## CLI

After `pip install -e .[dev]` (or from the source tree with the wrapper), the command `hqiv-arena` is available.

See the `SKILL.md` (in this dir or installed to your agent's skills folder) for full usage, especially `clone`, `run`, `submit`, `sync`, and `install-skill`.

The CLI is the primary way solvers (humans or AI agents) participate in the Arena.
