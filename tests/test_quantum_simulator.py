import math

from pyhqiv.quantum_simulator import Circuit, NoiseModel, QuantumState, run_circuit


def test_hadamard_on_zero_gives_half_probabilities() -> None:
    circuit = Circuit(1).h(0)
    result = run_circuit(circuit)
    probs = result.state.probabilities()
    assert math.isclose(float(probs[0]), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(probs[1]), 0.5, rel_tol=0.0, abs_tol=1e-12)


def test_bell_state_has_expected_support() -> None:
    circuit = Circuit(2).h(0).cnot(0, 1)
    result = run_circuit(circuit)
    probs = result.state.probabilities()
    assert math.isclose(float(probs[0]), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(probs[3]), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(probs[1] + probs[2]), 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_measurement_counts_sum_to_shots() -> None:
    circuit = Circuit(1).h(0)
    result = run_circuit(circuit, shots=200, seed=7)
    assert result.counts is not None
    assert sum(result.counts.values()) == 200
    est = result.probability_with_error(0)
    assert est.value >= 0.0
    assert est.sigma >= 0.0


def test_qft_inverse_round_trip() -> None:
    init = QuantumState(3)
    circuit = Circuit(3).x(0).x(2).qft().qft(inverse=True)
    result = run_circuit(circuit, initial_state=init)
    probs = result.state.probabilities()
    # initial basis state is |101> in little-endian indexing -> index 5
    assert int(probs.argmax()) == 5
    assert math.isclose(float(probs[5]), 1.0, rel_tol=0.0, abs_tol=1e-9)


def test_noise_model_validation() -> None:
    model = NoiseModel(bit_flip_p=0.1, phase_flip_p=0.2, depolarizing_p=0.3)
    model.validate()

