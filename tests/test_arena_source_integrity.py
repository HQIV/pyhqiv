"""Tests for the static Arena source-integrity gate."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


def test_integrity_script_passes_on_main_tree():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_arena_source_integrity.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--verbose"],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_catches_literal_return_cheat():
    from scripts.check_arena_source_integrity import _bad_literal_returns

    tree = ast.parse("def f():\n    return 0.42\n")
    func = tree.body[0]
    assert _bad_literal_returns(func) == [0.42]
