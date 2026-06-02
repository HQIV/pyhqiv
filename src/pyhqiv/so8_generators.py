"""
so(8) generator matrices (8×8) from Lean.

This module loads the 28 generator matrices from:
  `HQIV_LEAN/Hqiv/Generators.lean`

Lean defines `generator_0` … `generator_27` as explicit `Matrix.of` match tables.
We parse those tables to recover the numeric matrices in Python.

Motivation:
- This avoids maintaining a duplicated ~1800-float literal in Python.
- It keeps the Python energy-structure work aligned with the Lean single source.

Packaged wheels can load the same matrices from ``so8_generators.json`` (generated
via ``scripts/export_so8_generators_json.py``) using :func:`load_so8_generators_auto`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.resources as resources
import json
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Final, List, Mapping

import numpy as np

SO8_JSON_KEY_TENSOR: Final[str] = "so8_generators"
SO8_JSON_KEY_CHECKSUM: Final[str] = "sha256_hex"


@dataclass(frozen=True)
class So8Generators:
    """Container for the 28 basis matrices as a single (28,8,8) tensor."""

    tensor: np.ndarray  # shape (28,8,8), dtype float64

    def matrix(self, k: int) -> np.ndarray:
        return self.tensor[k]


LEAN_GENERATORS_PATH: Final[Path] = Path(__file__).resolve().parents[2] / "HQIV_LEAN" / "Hqiv" / "Generators.lean"


_CLAUSE_RE: Final[re.Pattern[str]] = re.compile(
    r"\|\s*(\d)\s*,\s*(\d)\s*=>\s*\(([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*:\s*ℝ\)"
)


def _extract_generator_block(text: str, k: int) -> str:
    m = re.search(
        rf"def generator_{k} : Matrix \(Fin 8\) \(Fin 8\) ℝ := Matrix\.of \(fun i j =>\s*\n\s*match i, j with\n(?P<body>.*?\n)\s*\)\n",
        text,
        flags=re.S,
    )
    if not m:
        raise ValueError(f"failed to locate generator_{k} definition in Lean file")
    return m.group("body")


def _parse_generator_matrix(body: str) -> np.ndarray:
    entries = np.full((8, 8), np.nan, dtype=np.float64)
    for i_s, j_s, v_s in _CLAUSE_RE.findall(body):
        i = int(i_s)
        j = int(j_s)
        entries[i, j] = float(v_s)
    if np.isnan(entries).any():
        missing = np.argwhere(np.isnan(entries))
        raise ValueError(f"generator missing {missing.shape[0]} entries, e.g. {missing[:5].tolist()}")
    return entries


@lru_cache(maxsize=1)
def load_so8_generators(*, lean_path: Path | None = None) -> So8Generators:
    """
    Load the 28 generator matrices from the Lean source.
    """
    path = LEAN_GENERATORS_PATH if lean_path is None else lean_path
    text = path.read_text(encoding="utf-8")
    mats: List[np.ndarray] = []
    for k in range(28):
        body = _extract_generator_block(text, k)
        mats.append(_parse_generator_matrix(body))
    tensor = np.stack(mats, axis=0)
    return So8Generators(tensor=tensor)


def so8_tensor_sha256_hex(tensor: np.ndarray) -> str:
    """Deterministic checksum for the (28, 8, 8) float64 payload."""
    arr = np.ascontiguousarray(tensor, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _parse_so8_generators_payload(
    data: Mapping[str, Any],
    *,
    verify_checksum: bool = True,
) -> So8Generators:
    if SO8_JSON_KEY_TENSOR not in data:
        raise ValueError(f"JSON object missing {SO8_JSON_KEY_TENSOR!r}")
    arr = np.asarray(data[SO8_JSON_KEY_TENSOR], dtype=np.float64)
    if arr.shape != (28, 8, 8):
        raise ValueError(f"expected {SO8_JSON_KEY_TENSOR} shape (28, 8, 8), got {arr.shape}")
    if verify_checksum:
        chk = data.get(SO8_JSON_KEY_CHECKSUM)
        if chk:
            want = str(chk)
            got = so8_tensor_sha256_hex(arr)
            if want != got:
                raise ValueError(
                    "SO(8) generator checksum mismatch: "
                    f"json has {want[:16]}…, computed {got[:16]}…"
                )
    return So8Generators(tensor=arr)


def load_so8_generators_from_json(
    path: str | Path,
    *,
    verify_checksum: bool = True,
) -> So8Generators:
    """
    Load the 28 matrices from a JSON export (see ``scripts/export_so8_generators_json.py``).
    """
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, Mapping):
        raise ValueError("SO(8) JSON root must be an object")
    return _parse_so8_generators_payload(data, verify_checksum=verify_checksum)


def load_so8_generators_from_packaged_json(*, verify_checksum: bool = True) -> So8Generators:
    """Load from ``pyhqiv/so8_generators.json`` if the package includes it."""
    traversable = resources.files("pyhqiv").joinpath("so8_generators.json")
    if not traversable.is_file():
        raise FileNotFoundError("packaged so8_generators.json not found")
    with traversable.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError("SO(8) JSON root must be an object")
    return _parse_so8_generators_payload(data, verify_checksum=verify_checksum)


def load_so8_generators_auto(
    *,
    json_path: str | Path | None = None,
    lean_path: Path | None = None,
    verify_checksum: bool = True,
) -> So8Generators:
    """
    Prefer ``json_path`` if given; else packaged ``so8_generators.json``; else parse Lean.
    """
    if json_path is not None:
        return load_so8_generators_from_json(json_path, verify_checksum=verify_checksum)
    try:
        return load_so8_generators_from_packaged_json(verify_checksum=verify_checksum)
    except FileNotFoundError:
        return load_so8_generators(lean_path=lean_path)


def dump_so8_generators_json(
    path: str | Path,
    generators: So8Generators | None = None,
    *,
    lean_path: Path | None = None,
) -> None:
    """
    Write a JSON certificate for the (28, 8, 8) tensor plus SHA-256 of float64 bytes.
    """
    if generators is None:
        generators = load_so8_generators(lean_path=lean_path)
    tensor = generators.tensor
    payload = {
        SO8_JSON_KEY_TENSOR: tensor.tolist(),
        SO8_JSON_KEY_CHECKSUM: so8_tensor_sha256_hex(tensor),
        "source": "HQIV_LEAN/Hqiv/Generators.lean",
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def lie_bracket(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix commutator [A,B] = AB - BA."""
    return A @ B - B @ A


def upper_triangle_coord(A: np.ndarray) -> np.ndarray:
    """
    Coordinate vector of an antisymmetric 8×8 matrix in the 28-dim upper triangle basis.
    Ordering: row-major over pairs (i<j).
    """
    coords = []
    for i in range(8):
        for j in range(i + 1, 8):
            coords.append(A[i, j])
    return np.array(coords, dtype=np.float64)


__all__ = [
    "SO8_JSON_KEY_CHECKSUM",
    "SO8_JSON_KEY_TENSOR",
    "So8Generators",
    "LEAN_GENERATORS_PATH",
    "dump_so8_generators_json",
    "lie_bracket",
    "load_so8_generators",
    "load_so8_generators_auto",
    "load_so8_generators_from_json",
    "load_so8_generators_from_packaged_json",
    "so8_tensor_sha256_hex",
    "upper_triangle_coord",
]

