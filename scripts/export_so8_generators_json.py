#!/usr/bin/env python3
"""
Export the 28×8×8 so(8) Lean generator tables to ``src/pyhqiv/so8_generators.json``.

Run from the repository root::

    PYTHONPATH=src python scripts/export_so8_generators_json.py

Requires ``HQIV_LEAN/Hqiv/Generators.lean`` at the repo root (same layout as
:const:`pyhqiv.so8_generators.LEAN_GENERATORS_PATH`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SO(8) generators from Lean to JSON.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_REPO_ROOT / "src" / "pyhqiv" / "so8_generators.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    from pyhqiv.so8_generators import dump_so8_generators_json, load_so8_generators

    dump_so8_generators_json(args.output, load_so8_generators())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
