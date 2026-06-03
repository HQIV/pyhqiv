"""
Lean-backing checks for the HQIV quantum-computing formal layer.

This keeps Python simulation code aligned with formally named anchors in
`Hqiv/QuantumComputing/*.lean`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class LeanSymbolSpec:
    file: str
    required_symbols: List[str]


DEFAULT_QUANTUM_SPECS: List[LeanSymbolSpec] = [
    LeanSymbolSpec(
        file="Hqiv/QuantumComputing/DiscreteQuantumState.lean",
        required_symbols=[
            "def discreteIp",
            "def discreteNormSq",
            "theorem discreteNormSq_nonneg",
        ],
    ),
    LeanSymbolSpec(
        file="Hqiv/QuantumComputing/DigitalGates.lean",
        required_symbols=[
            "structure HQIVGate",
            "def phaseGate",
            "theorem cnot_preserves_unweighted_four",
        ],
    ),
    LeanSymbolSpec(
        file="Hqiv/QuantumComputing/DiscreteSchrodinger.lean",
        required_symbols=[
            "def digitalEvolution",
            "theorem digitalEvolution_preserves_normSq",
        ],
    ),
    LeanSymbolSpec(
        file="Hqiv/QuantumComputing/OctonionicFT.lean",
        required_symbols=[
            "def qftPhase",
            "def period4InterferenceProb",
            "def period4Support16",
            "def oft",
            "theorem oft_preserves_normSq",
        ],
    ),
    LeanSymbolSpec(
        file="Hqiv/QuantumComputing/ShoreOracle.lean",
        required_symbols=[
            "structure ShoreOracleCircuit",
            "def run",
            "def bornControlProb15",
            "def factors15Outcome",
            "def shorCircuit",
            "theorem shoreOracle_factors_15",
        ],
    ),
]


def validate_quantum_lean_backing(
    lean_repo_root: str | Path,
    specs: List[LeanSymbolSpec] | None = None,
) -> Dict[str, List[str]]:
    """
    Return missing symbols per Lean file.

    The empty dict means all required symbols were found.
    """
    root = Path(lean_repo_root).expanduser().resolve()
    active_specs = DEFAULT_QUANTUM_SPECS if specs is None else specs
    missing: Dict[str, List[str]] = {}

    for spec in active_specs:
        path = root / spec.file
        if not path.exists():
            missing[spec.file] = [f"<missing file: {spec.file}>"]
            continue

        content = path.read_text(encoding="utf-8")
        missing_symbols = [s for s in spec.required_symbols if s not in content]
        if missing_symbols:
            missing[spec.file] = missing_symbols

    return missing


def quantum_lean_backing_ok(lean_repo_root: str | Path) -> bool:
    return not validate_quantum_lean_backing(lean_repo_root)


__all__ = [
    "DEFAULT_QUANTUM_SPECS",
    "LeanSymbolSpec",
    "quantum_lean_backing_ok",
    "validate_quantum_lean_backing",
]

