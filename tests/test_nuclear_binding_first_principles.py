"""
Tests for first-principles nuclear binding (nuclear_binding_first_principles).

Passing = within 10% of experiment (AME/NNDC): ratio = B_pred/B_exp in [0.90, 1.10].
Run with: pytest tests/test_nuclear_binding_first_principles.py -s -v
to see each result and gap to experiment (not just pass/fail).
"""

from pyhqiv.nuclear_binding_first_principles import (
    binding_energy_mev,
    run_first_principles_scan,
    solve_equilibrium_x,
)

# Experiment (AME/NNDC) — reference values
B_2H_MEV_EXP = 2.224
B_HE4_MEV_EXP = 28.30
B_C12_MEV_EXP = 92.16
B_O16_MEV_EXP = 127.62

# Passing = within 10% of reality
RATIO_LO, RATIO_HI = 0.90, 1.10


def _report(label, B_pred, B_exp):
    """Print one line: PASS or FAIL with B_pred, B_exp, ratio, gap to experiment."""
    ratio = B_pred / B_exp if B_exp else 0.0
    gap = B_pred - B_exp
    ok = RATIO_LO <= ratio <= RATIO_HI
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {label}: B_pred={B_pred:.4f} MeV, B_exp={B_exp:.4f} MeV, ratio={ratio:.4f}, gap={gap:+.4f} MeV")
    return ok


def test_deuteron_binding_vs_experiment():
    """2H (deuteron): first-principles B within 10% of AME (2.224 MeV)."""
    B_mev, _, _ = binding_energy_mev(2, 1)
    ok = _report("2H (D)", B_mev, B_2H_MEV_EXP)
    assert ok, f"2H: ratio not in [{RATIO_LO}, {RATIO_HI}]; B_pred={B_mev:.4f}, B_exp={B_2H_MEV_EXP}"


def test_he4_binding_vs_experiment():
    """4He: first-principles B within 10% of AME (28.30 MeV)."""
    B_mev, _, _ = binding_energy_mev(4, 2)
    ok = _report("4He", B_mev, B_HE4_MEV_EXP)
    assert ok, f"4He: ratio not in [{RATIO_LO}, {RATIO_HI}]; B_pred={B_mev:.4f}, B_exp={B_HE4_MEV_EXP}"


def test_c12_binding_vs_experiment():
    """C-12: first-principles B within 10% of AME (92.16 MeV)."""
    B_mev, _, _ = binding_energy_mev(12, 6)
    ok = _report("C-12", B_mev, B_C12_MEV_EXP)
    assert ok, f"C-12: ratio not in [{RATIO_LO}, {RATIO_HI}]; B_pred={B_mev:.4f}, B_exp={B_C12_MEV_EXP}"


def test_o16_binding_vs_experiment():
    """O-16: first-principles B within 10% of AME (127.62 MeV)."""
    B_mev, _, _ = binding_energy_mev(16, 8)
    ok = _report("O-16", B_mev, B_O16_MEV_EXP)
    assert ok, f"O-16: ratio not in [{RATIO_LO}, {RATIO_HI}]; B_pred={B_mev:.4f}, B_exp={B_O16_MEV_EXP}"


def test_first_principles_scan_runs():
    """run_first_principles_scan returns list of dicts with expected keys."""
    results = run_first_principles_scan(A_min=2, A_max=8, gamma=0.4)
    assert len(results) >= 7
    for r in results:
        assert "A" in r and "Z" in r and "B_total_MeV" in r and "B_per_nucleon_MeV" in r
        assert "x_eq_fm" in r and "stable" in r


def test_solve_equilibrium_x_deuteron():
    """Deuteron: equilibrium scale x_eq in reasonable fm range."""
    from pyhqiv.subatomic import nucleon_effective_theta_m

    theta_p_m, theta_n_m = nucleon_effective_theta_m()
    x_eq = solve_equilibrium_x(2, 1, theta_p_m, theta_n_m)
    assert 0.1 < x_eq < 50.0, f"x_eq={x_eq:.4f} fm out of range"
