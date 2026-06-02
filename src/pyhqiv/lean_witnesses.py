"""
Lean witness export loader (single source of truth for numerical anchors).

All numerical "witness" values (reference shell, proton mass anchor, SM coupling
outputs, projection factors, unit conversions, etc.) must come from a Lean
export artifact (JSON).

Python code must not hardcode physics literals; it should load them here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from functools import lru_cache
import importlib.resources as resources
from pathlib import Path
from typing import Any, Mapping, Sequence


class LeanWitnessError(RuntimeError):
    pass


@dataclass(frozen=True)
class LeanWitnesses:
    data: Mapping[str, Any]

    def require(self, key: str) -> Any:
        if key not in self.data:
            raise LeanWitnessError(f"Missing required witness key: {key!r}")
        return self.data[key]

    def get_int(self, key: str) -> int:
        value = self.require(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise LeanWitnessError(f"Witness {key!r} must be a number-like int, got {type(value).__name__}")
        try:
            iv = int(value)
        except Exception as e:  # noqa: BLE001
            raise LeanWitnessError(f"Witness {key!r} cannot be converted to int") from e
        return iv

    def get_float(self, key: str) -> float:
        value = self.require(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise LeanWitnessError(f"Witness {key!r} must be a number-like float, got {type(value).__name__}")
        try:
            fv = float(value)
        except Exception as e:  # noqa: BLE001
            raise LeanWitnessError(f"Witness {key!r} cannot be converted to float") from e
        return fv

    def get_float_list(self, key: str) -> list[float]:
        value = self.require(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise LeanWitnessError(f"Witness {key!r} must be a sequence of numbers")
        out: list[float] = []
        for i, v in enumerate(value):
            try:
                out.append(float(v))
            except Exception as e:  # noqa: BLE001
                raise LeanWitnessError(f"Witness {key!r}[{i}] cannot be converted to float") from e
        return out


def _default_witnesses_path() -> str:
    return str(resources.files("pyhqiv").joinpath("witnesses.json"))


def _optional_resonance_overlay_path() -> Path:
    """
    Optional repo-root overlay for the reverse-ladder resonance witnesses.

    The Lean exporter writes `data/hqiv_witnesses.json`. We merge it on top of the
    packaged `pyhqiv/witnesses.json` so existing keys remain available.
    """
    # lean_witnesses.py can be reached via symlinks; using `__file__` ancestry
    # may escape the main checkout directory.
    #
    # Instead, we find the repo root by walking upward from `cwd` looking for
    # the `Hqiv/` directory (which exists at the main HQIV-lean checkout root).
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if (parent / "Hqiv").exists():
            return parent / "data" / "hqiv_witnesses.json"

    # Fallback: try a best-effort guess relative to this file.
    resolved = Path(__file__).resolve()
    for parent in [resolved, *resolved.parents]:
        if (parent / "Hqiv").exists():
            return parent / "data" / "hqiv_witnesses.json"
    return resolved.parents[3] / "data" / "hqiv_witnesses.json"


@lru_cache(maxsize=1)
def load_lean_witnesses(path: str | None = None) -> LeanWitnesses:
    """
    Load Lean witnesses JSON. Defaults to the packaged `pyhqiv/witnesses.json`.
    """
    witness_path = _default_witnesses_path() if path is None else path
    try:
        with open(witness_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise LeanWitnessError(
            "Lean witnesses JSON not found. Expected a Lean-exported artifact at "
            f"{witness_path!r}. Generate it via the Lean export script."
        ) from e
    if not isinstance(data, Mapping):
        raise LeanWitnessError("Witnesses JSON must be a JSON object/dict")

    # Merge in the reverse-ladder resonance overlay by default.
    if path is None:
        overlay_path = _optional_resonance_overlay_path()
        if overlay_path.exists():
            with open(overlay_path, "r", encoding="utf-8") as f:
                overlay = json.load(f)
            if not isinstance(overlay, Mapping):
                raise LeanWitnessError(
                    f"Overlay witnesses JSON must be an object/dict: {overlay_path!s}"
                )
            merged = dict(data)
            merged.update(overlay)  # overlay wins on key conflicts
            data = merged

    return LeanWitnesses(data=data)


__all__ = ["LeanWitnessError", "LeanWitnesses", "load_lean_witnesses"]

