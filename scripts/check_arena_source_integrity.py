#!/usr/bin/env python3
"""
Static integrity gate for HQIV Arena Lean-mirror Python modules.

Complements scripts/validate_hqiv_alignment.py by rejecting import creep and
literal-return shortcuts in lightcone.py / metric.py before numeric alignment runs.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_RETURN_FLOATS = frozenset({0.0, 1.0})
ALLOWED_MODULE_BINDINGS = frozenset({"ALPHA_EXACT", "__all__"})

IMPORT_ALLOWLIST: dict[str, frozenset[str | None]] = {
    "src/pyhqiv/lightcone.py": frozenset({None, "fractions", "math", "__future__"}),
    "src/pyhqiv/metric.py": frozenset({None, "dataclasses", "math", "__future__", "pyhqiv"}),
}

REQUIRED_CALLEES: dict[str, dict[str, frozenset[str]]] = {
    "src/pyhqiv/lightcone.py": {
        "omega_k_at_horizon": frozenset({"curvature_integral", "x_over_theta_from_horizons"}),
        "omega_k_partial": frozenset({"omega_k_at_horizon", "reference_m"}),
        "reference_m": frozenset({"qcd_shell", "lattice_step_count"}),
        "curvature_norm_combinatorial": frozenset(
            {"cube_directions", "octonion_imaginary_dim", "unit_cube_half_diagonal"}
        ),
    },
    "src/pyhqiv/metric.py": {
        "gamma_hqiv": frozenset({"alpha"}),
    },
}

PROTECTED_PATHS = tuple(IMPORT_ALLOWLIST.keys())


@dataclass
class Violation:
    file: str
    rule: str
    detail: str


def _import_roots(tree: ast.Module) -> set[str | None]:
    roots: set[str | None] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                roots.add(None)
            else:
                roots.add(node.module.split(".")[0])
    return roots


def _calls_in_function(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _bad_literal_returns(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[float]:
    bad: list[float] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        val = node.value
        if isinstance(val, ast.Constant) and isinstance(val.value, (int, float)):
            f = float(val.value)
            if f not in ALLOWED_RETURN_FLOATS:
                bad.append(f)
    return bad


def _module_float_assignments(tree: ast.Module) -> list[str]:
    bad: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if name in ALLOWED_MODULE_BINDINGS:
                        continue
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, float
                    ):
                        bad.append(name)
    return bad


def _function_defs(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def check_file(rel_path: str) -> list[Violation]:
    path = REPO_ROOT / rel_path
    if not path.exists():
        return [Violation(rel_path, "missing_file", str(path))]

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[Violation] = []

    allowed_imports = IMPORT_ALLOWLIST[rel_path]
    extra = _import_roots(tree) - allowed_imports
    if extra:
        violations.append(
            Violation(
                rel_path,
                "imports",
                f"disallowed import roots: {sorted(extra)}",
            )
        )

    for name in _module_float_assignments(tree):
        violations.append(Violation(rel_path, "module_float", f"module-level float: {name}"))

    funcs = _function_defs(tree)
    for func_name, must_call in REQUIRED_CALLEES.get(rel_path, {}).items():
        func = funcs.get(func_name)
        if func is None:
            violations.append(
                Violation(rel_path, "missing_function", f"expected {func_name}")
            )
            continue
        missing = must_call - _calls_in_function(func)
        if missing:
            violations.append(
                Violation(
                    rel_path,
                    "structural_calls",
                    f"{func_name} must call {sorted(missing)}",
                )
            )
        bad_returns = _bad_literal_returns(func)
        if bad_returns:
            violations.append(
                Violation(
                    rel_path,
                    "literal_return",
                    f"{func_name} forbidden literal return(s): {bad_returns}",
                )
            )
        if func_name == "hqvm_lapse":
            for node in ast.walk(func):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, (int, float)):
                        violations.append(
                            Violation(
                                rel_path,
                                "literal_return",
                                f"{func_name} must be 1 + Phi + phi*t, not a constant",
                            )
                        )

    return violations


def run_checks(paths: Iterable[str] | None = None) -> list[Violation]:
    all_v: list[Violation] = []
    for rel in list(paths) if paths else list(PROTECTED_PATHS):
        all_v.extend(check_file(rel))
    return all_v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HQIV Arena protected-source integrity gate")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("paths", nargs="*", help="Optional module paths under repo root")
    args = p.parse_args(argv)

    violations = run_checks(args.paths or None)
    if args.verbose or violations:
        print("=== HQIV Arena source integrity ===")
        for v in violations:
            print(f"[FAIL] {v.file} ({v.rule}): {v.detail}")
        if not violations:
            print("All protected modules passed.")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
