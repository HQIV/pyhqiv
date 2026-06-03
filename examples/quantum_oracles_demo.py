"""
OSHoracle demo: period finding + Grover search.

Run:
    ./.venv/bin/python examples/quantum_oracles_demo.py
"""

from __future__ import annotations

from pyhqiv import (
    oshoracle_grover_search,
    oshoracle_period_find_quantum,
    optimal_grover_iterations,
)


def run_period_demo() -> None:
    print("=== Period Finding Demo (OSHoracle-style) ===")
    result = oshoracle_period_find_quantum(control_qubits=4, period=4)
    print(f"Estimated period: {result.estimated_period}")
    print(f"Support peaks: {result.support}")
    print("Probabilities:")
    for y in result.support:
        print(f"  y={y:2d} -> p={result.probabilities[y]:.6f}")
    print()


def run_grover_demo() -> None:
    print("=== Grover Demo (OSHoracle-style) ===")
    n_qubits = 3
    target_state = 5

    def oracle(idx: int) -> bool:
        return idx == target_state

    n_states = 1 << n_qubits
    k_opt = optimal_grover_iterations(n_states, marked_count=1)
    result = oshoracle_grover_search(n_qubits=n_qubits, oracle=oracle, iterations=k_opt)

    print(f"Qubits: {n_qubits} (states={n_states})")
    print(f"Target index: {target_state}")
    print(f"Iterations used: {result.iterations}")
    print(f"Top measured index: {result.top_index}")
    print(f"Top probability: {result.top_probability:.6f}")
    print()
    print("Top-4 probabilities:")
    ranked = sorted(enumerate(result.probabilities.tolist()), key=lambda x: x[1], reverse=True)[:4]
    for idx, p in ranked:
        print(f"  state={idx:2d} -> p={p:.6f}")
    print()


def main() -> None:
    run_period_demo()
    run_grover_demo()


if __name__ == "__main__":
    main()

