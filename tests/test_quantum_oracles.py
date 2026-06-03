import math

from pyhqiv.quantum_oracles import (
    estimate_period_from_peaks,
    hidden_tag_of,
    optimal_grover_iterations,
    oshoracle_grover_search,
    oshoracle_period_find_quantum,
    period_probability_distribution,
    period_support,
)
from pyhqiv.now_setters import nowSetter


def test_hidden_tag_of_period4() -> None:
    assert hidden_tag_of(0, 4) == 0
    assert hidden_tag_of(5, 4) == 1
    assert hidden_tag_of(12, 4) == 0


def test_period_support_for_q16_r4() -> None:
    support = period_support(control_qubits=4, period=4)
    assert support == [0, 4, 8, 12]


def test_period_distribution_uniform() -> None:
    probs = period_probability_distribution(control_qubits=4, period=4)
    assert set(probs.keys()) == {0, 4, 8, 12}
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    for v in probs.values():
        assert math.isclose(v, 0.25, rel_tol=0.0, abs_tol=1e-12)


def test_period_estimation_from_peaks() -> None:
    est = estimate_period_from_peaks([0, 4, 8, 12], q=16)
    assert est == 4


def test_oshoracle_period_find_period4() -> None:
    nowSetter(4.0, [{"id": 4, "lower": 4.0, "upper": 5.0, "reference_m": 0}])
    result = oshoracle_period_find_quantum(control_qubits=4, period=4)
    assert result.estimated_period == 4
    assert result.support == [0, 4, 8, 12]
    assert math.isclose(sum(result.probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert set(result.probability_errors.keys()) == set(result.probabilities.keys())


def test_grover_oracle_single_marked_state() -> None:
    target = 5

    def oracle(idx: int) -> bool:
        return idx == target

    result = oshoracle_grover_search(n_qubits=3, oracle=oracle)
    assert result.iterations == optimal_grover_iterations(8, 1)
    assert result.top_index == target
    assert result.top_probability > 0.90
    assert result.top_probability_error >= 0.0

