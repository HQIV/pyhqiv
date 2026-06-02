#!/usr/bin/env bash
# HQIV Arena benchmark entrypoint (local solver run).
#
# This is the "benchmark.sh" equivalent for the HQIV Arena, modeled after ecdsa.fail challenges.
# It wipes any stale local results, runs the trusted scoring harness (alignment + sigma scoring),
# and writes canonical outputs (arena_results.json + summary).
#
# For full untrusted isolation you would add bubblewrap / sandbox-exec around the python run
# (the contestant only edits files under src/pyhqiv/ , Hqiv/ (via the lean tree), and tests/).
# The scoring itself (validate + compute) is the trusted evaluator.

set -euo pipefail

echo "=== HQIV Arena Local Benchmark ==="

# Ensure we are inside a pyhqiv tree
if [[ ! -f pyproject.toml ]] && [[ ! -f ../pyproject.toml ]]; then
  echo "Run this from the pyhqiv directory of an hqiv-arena clone (or the repo root)." >&2
  exit 1
fi

ROOT="$(pwd)"
if [[ -f pyproject.toml ]]; then
  PYROOT="$ROOT"
else
  PYROOT="$(cd .. && pwd)"
fi

export PYTHONPATH="$PYROOT/src:${PYTHONPATH:-}"

# Clean previous local results (so a "submission" can't pre-seed its score)
rm -f arena_results.json alignment_results.json 2>/dev/null || true

# Run the full local gates that don't require the heavy remote Lean cert
python3 -m pyhqiv.arena_cli run

echo
echo "Benchmark complete. See arena_results.json for the detailed score vector."
echo "Open a PR for the authoritative CI run (full lake build + alignment gate)."
