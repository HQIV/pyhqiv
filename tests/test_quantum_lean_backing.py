from pyhqiv.quantum_lean_backing import (
    LeanSymbolSpec,
    quantum_lean_backing_ok,
    validate_quantum_lean_backing,
)


def test_validate_quantum_lean_backing_ok(tmp_path) -> None:
    root = tmp_path
    lean_file = root / "Hqiv/QuantumComputing/DiscreteQuantumState.lean"
    lean_file.parent.mkdir(parents=True, exist_ok=True)
    lean_file.write_text(
        "\n".join(
            [
                "def discreteIp := 0",
                "def discreteNormSq := 0",
                "theorem discreteNormSq_nonneg : True := by trivial",
            ]
        ),
        encoding="utf-8",
    )
    specs = [
        LeanSymbolSpec(
            file="Hqiv/QuantumComputing/DiscreteQuantumState.lean",
            required_symbols=["def discreteIp", "def discreteNormSq", "theorem discreteNormSq_nonneg"],
        )
    ]
    missing = validate_quantum_lean_backing(root, specs=specs)
    assert missing == {}
    assert quantum_lean_backing_ok(root) is False  # default spec set expects more files


def test_validate_quantum_lean_backing_reports_missing() -> None:
    missing = validate_quantum_lean_backing("/definitely/not/a/repo")
    assert missing

