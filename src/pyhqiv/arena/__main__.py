"""
python -m pyhqiv.arena  — runs the default arena score computation.
"""
from __future__ import annotations
import sys
from . import compute_score  # re-export the cli entry for -m

if __name__ == "__main__":
    # delegate to the script impl
    from pathlib import Path
    import runpy, sys
    sys.exit(runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts" / "arena" / "compute_score.py"), run_name="__main__") or 0)
