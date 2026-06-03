"""Dense vs OSH-sparse protein-style benchmark (octonion paper technical section)."""

from pyhqiv.benchmark_osh_vs_dense import BenchCase, run_case


def test_sparse_faster_on_sparse_support_tiny_case() -> None:
    case = BenchCase(n_sites=64, steps=10, active_fraction=0.05, update_every=10, runs=2)
    out = run_case(case, seed=123)
    assert out["dense_median_ms"] > 0
    assert out["sparse_median_ms"] > 0
    assert out["dense_vs_sparse_speedup"] > 1.0
