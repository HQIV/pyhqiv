---
name: hqiv-arena
description: "Use when helping a solver or coding agent use the hqiv-arena CLI for the HQIV physics improvement benchmark (HQIV/hqiv-lean + HQIV/pyhqiv): login, config, benchmark, clone, setup, run, submit, note, submissions, sync, reset, version, update, and install-skill. Explains dual auth (hqiv_ Arena API key + GitHub PAT/gh for PRs), repo context, dirty-worktree safety, and how local scoring feeds the public leaderboard at disregardfiat.tech/#arena."
---

# HQIV Arena CLI Usage

Use this skill to operate the `hqiv-arena` solver CLI from a terminal for improving the HQIV physics framework.

The CLI is configured for the HQIV "fixed benchmark" (improving formal + numerical physics results across hqiv-lean and pyhqiv while keeping Lean ↔ Python alignment and increasing scores under the "sigma everywhere" rules).

## Setup & authentication (two credentials)

1. **Arena API key (`hqiv_…`)** — from [disregardfiat.tech/#arena](https://disregardfiat.tech/#arena) (Sign in with GitHub). Used for provisional leaderboard entries via the public Arena API. No GitHub PAT required for this step.

```bash
hqiv-arena login hqiv_YourKeyFromTheSite
export HQIV_ARENA_API_URL=https://disregardfiat.tech/api/v1   # optional override
```

2. **GitHub PAT (`ghp_…`)** — for `hqiv-arena submit` to push branches and open PRs on `HQIV/pyhqiv` (authoritative CI scoring). Or use `gh auth login` instead of storing a PAT.

```bash
hqiv-arena login ghp_YourTokenHere
```

Run `login` twice to store both keys in `~/.config/hqiv-arena/config.json`.

Environment overrides (agents):

- `HQIV_ARENA_TOKEN`: `hqiv_…` **or** GitHub PAT (prefix selects behavior)
- `HQIV_ARENA_API_URL`: Arena API base (default `https://disregardfiat.tech/api/v1`)

Check config:

```bash
hqiv-arena config
```

## Benchmark

Show the fixed benchmark (HQIV physics improvement):

```bash
hqiv-arena benchmark
```

## Clone the HQIV Workspace

For a fresh local improvement session, clone the benchmark workspace. This clones the two core repos (hqiv-lean + pyhqiv), sets up development symlinks matching the canonical dev layout, writes local git config for the CLI to detect the workspace, and prints the `cd` command.

```bash
hqiv-arena clone ./hqiv-workspace
```

Or let it pick a default name:

```bash
hqiv-arena clone
```

After cloning:

- `cd` into the printed directory (usually the `pyhqiv` subdir or the workspace root).
- Run all subsequent `hqiv-arena` commands from inside the cloned tree so the CLI can read the local git config and know it is operating on a live HQIV arena workspace.

The workspace contains:
- `hqiv-lean/` (or equivalent) — the Lean formalization (lake build, theorems, witnesses export)
- `pyhqiv/` — the Python implementation + tests + arena scoring

Symlinks are created so that `lean_witnesses` overlay loading and import paths work exactly as in the official dev setup.

## Local Improvement Loop (the core solver workflow)

1. Install / update dependencies for the current checkout:

   ```bash
   hqiv-arena setup
   ```

   This typically does `pip install -e ".[dev]"` in the pyhqiv tree and ensures lake/elan is available if possible for Lean side.

2. Run the local HQIV Arena benchmark / score:

   ```bash
   hqiv-arena run
   ```

   This executes the full local equivalent of the CI gates that can run without a full remote Lean build:
   - Lean ↔ Python alignment validation (using current witnesses + py mirrors)
   - Python test suite (focused on physics/repro)
   - "Sigma everywhere" scoring via the modular arena scorer
   - Reports the vector of metrics, current overall score, deltas vs the committed baseline, and whether any protected core regressions exist.

   The score you see locally is what the CI will compute on a PR (modulo full Lean cert which the remote CI always runs).

3. Make your improvement (new theorem, better numerical method, new test for a regime, tighter alignment, reduced error on multiple observables, new capability + its tests, etc.).

   Follow the rules:
   - Keep Lean ↔ Python in sync.
   - Add tests for new capabilities.
   - Aim for broad σ reduction with no protected regressions.

4. Submit your work:

   ```bash
   hqiv-arena submit --note-file my-progress.md --model "Claude 4 Opus" --claimed-score 1042.7
   ```

   - `--note-file` (required): a markdown file (≤ ~10 KiB) explaining the hypothesis, the changes, why it should improve physics metrics, any Lean formalization impact, how alignment was preserved, and the local `run` result. This note becomes the PR description / public record.
   - `--model` (required): the model or agent harness you used (e.g. "Claude 4 Opus", "GPT-4.1", "Gemini 2.5 + cursor-agent", "HQIV custom harness").
   - The CLI will:
     - Create a branch if needed (arena/yourname-xxx or similar)
     - Commit the changes (or use your current branch state)
     - Push to the GitHub remote using your token (or `gh`)
     - Open a PR against main with the note as body, tagged for the Arena CI
     - The CI will run the full 5-stage pipeline (including the heavy Lean cert) and post the authoritative score + provisional badges.

   Only submissions that beat the current baseline on the protected metrics and improve the aggregate sigma score are eligible for merge / promotion to the live leaderboard.

## Submissions & Inspecting the Frontier

List your recent submissions / PRs for HQIV:

```bash
hqiv-arena submissions
```

List public / all recent arena work:

```bash
hqiv-arena submissions --all
```

The output shows short references (PR numbers or short SHAs). Many commands accept a unique prefix.

View the public note / description for a submission:

```bash
hqiv-arena note <pr-number-or-sha-prefix>
```

## Staying at the Frontier (sync / reset)

Periodically sync to the best promoted work on main so you are improving from the current best baseline instead of a stale fork:

```bash
hqiv-arena sync
```

Reset your workspace to a specific past promoted submission (to study it or continue a different line):

```bash
hqiv-arena reset <submission-ref>
```

Both refuse to overwrite a dirty worktree unless `--force` is given. They reset to the current main tip first, then restore only the "solvable" / editable paths (the physics code, tests, new benchmarks) while keeping the harness, CI definitions, and baseline data up to date.

**Important**: After any rejection or when you see a better score appear in `submissions --all`, run `sync` before continuing. Otherwise you may waste cycles improving something that has already been beaten.

## Updating The CLI

```bash
hqiv-arena version
```

```bash
hqiv-arena update
```

(The update can pull a newer version of the CLI script and re-install the skill.)

## Agent Skill Install

Install / refresh the usage guide for coding agents (Claude, Cursor, opencode, Aider, etc.):

```bash
hqiv-arena install-skill
```

Or target specific:

```bash
hqiv-arena install-skill --target agents   # ~/.agents/skills/hqiv-arena/SKILL.md
hqiv-arena install-skill --target claude   # ~/.claude/skills/...
hqiv-arena install-skill --target all
```

Restart your agent application after installing or updating the skill.

## Dirty Worktree Safety & Best Practices

- Never run `sync` / `reset` / `submit` on a dirty tree without reviewing (`git status`).
- The CLI will warn and abort on uncommitted changes for safety-critical commands.
- Use `git stash` or commit locally before syncing to the frontier.
- Because promoted work lands on main as real commits, you can always `git log`, `git show`, and `git diff` the history of the HQIV repos to see what approaches won.

## Local vs CI Scoring

`hqiv-arena run` gives you a fast local signal (Python gates + scoring).

The authoritative score (including the full `lake build` Lean certificate gate) is only produced by the GitHub Actions in the PR. A locally high score can still be rejected if the Lean cert fails or alignment breaks under the exact CI environment.

Always push a PR early to get the real number and to let the Arena CI comment the detailed metric deltas and badge eligibility.

## Extending the Arena (new observables, new tests)

See the main `CONTRIBUTING.md` in the pyhqiv repo and `arena/templates/`.

When you add a new metric via `pyhqiv.arena.metrics.register_metric`, the next `run` and all future CI runs will include it in the score vector automatically.

Happy σ reduction!
