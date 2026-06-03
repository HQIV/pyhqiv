"""Classical integer next_prime helper."""

from pyhqiv.next_integer_prime import next_prime


def test_next_prime_examples() -> None:
    assert next_prime(5) == 7
    assert next_prime(500) == 503
    assert next_prime(4) == 5
    assert next_prime(1) == 2
    assert next_prime(0) == 2
    assert next_prime(-1) == 2
