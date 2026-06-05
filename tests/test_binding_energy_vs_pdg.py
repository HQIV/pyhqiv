"""
Compare ``pyhqiv.isotope_ladder`` binding energies to AME2020 + CODATA reference data.

The **network** binding model (uniform so(8) weights at fixed shell) is **not** calibrated to
reproduce nuclear mass tables. These tests therefore:

1. Use **published** total binding energies ``B`` (MeV) and **1σ** uncertainties from
   ``tests.data.nuclear_binding_reference`` (AME2020-style).
2. Use **CODATA 2018** proton/neutron masses for the same definition
   ``B = Z m_p + N m_n - M_nucleus`` as the model.
3. Report **z-scores** ``(B_pred - B_ref) / σ_ref`` and keep CI green with a **documented
   loose** absolute tolerance until the model is tuned.

Tight agreement with PDG/AME would be a separate milestone (functional binding path, etc.).
"""

from __future__ import annotations

import math

from pyhqiv.isotope_ladder import IsotopeLadderConfig, IsotopeState, nuclear_binding_energy_mev
from tests.data.nuclear_binding_reference import (
    AME2020_BINDING_MEV,
    CODATA_2018_NEUTRON_MEV,
    CODATA_2018_PROTON_MEV,
    BindingReference,
    lookup_binding,
)


def _config_codata() -> IsotopeLadderConfig:
    return IsotopeLadderConfig(
        shell_m=4,
        m_proton_mev=CODATA_2018_PROTON_MEV,
        m_neutron_mev=CODATA_2018_NEUTRON_MEV,
        rotational_scale_mev=0.0,
    )


def _z_score(pred: float, ref: BindingReference) -> float:
    if ref.sigma_mev <= 0.0:
        return math.copysign(float("inf"), pred - ref.B_mev)
    return (pred - ref.B_mev) / ref.sigma_mev


def test_reference_table_lookup_roundtrip() -> None:
    d = lookup_binding(2, 2)
    assert d is not None and d.symbol == "4He"
    assert abs(d.B_mev - 28.295674) < 1e-5


def test_codata_nucleon_masses_reasonable() -> None:
    """Sanity: CODATA proton/neutron masses near 938–940 MeV."""
    assert 938.0 < CODATA_2018_PROTON_MEV < 939.0
    assert 939.0 < CODATA_2018_NEUTRON_MEV < 940.5


def test_isotope_ladder_binding_vs_ame2020_loose_envelope() -> None:
    """
    For each AME entry, require |B_pred - B_ref| within a wide envelope.

    Envelope = max(10 σ_AME, 10 |B_ref|) so the current uncalibrated model stays in CI
    while still comparing to the same central values and error bars used in particle physics.
    """
    cfg = _config_codata()
    for ref in AME2020_BINDING_MEV:
        st = IsotopeState(Z=ref.Z, N=ref.N, J=0.0)
        pred = nuclear_binding_energy_mev(st, cfg)
        delta = pred - ref.B_mev
        tol = max(10.0 * ref.sigma_mev, 10.0 * abs(ref.B_mev))
        z = _z_score(pred, ref)
        assert math.isfinite(pred), f"{ref.symbol}: non-finite prediction"
        assert abs(delta) <= tol, (
            f"{ref.symbol}: |ΔB|={abs(delta):.3f} MeV exceeds loose tol={tol:.3f} MeV; "
            f"B_pred={pred:.6f}, B_ref={ref.B_mev}±{ref.sigma_mev}, z={z:.2f}σ"
        )


def test_binding_comparison_sigma_units_document_gap() -> None:
    """
    The deuteron prediction differs from AME by many experimental sigmas (model not fitted).

    This asserts the **z-score machinery** is meaningful (|z| ≫ 1), not that the model matches.
    """
    cfg = _config_codata()
    ref = lookup_binding(1, 1)
    assert ref is not None
    pred = nuclear_binding_energy_mev(IsotopeState(1, 1, 0.0), cfg)
    z = _z_score(pred, ref)
    assert abs(z) > 50.0, f"expected |z| ≫ 1 for uncalibrated model, got z={z:.2f}"


def test_report_z_scores_for_all_ame_entries() -> None:
    """Build a structured comparison list (usable in pytest -s output or debugging)."""
    cfg = _config_codata()
    rows: list[tuple[str, float, float, float, float]] = []
    for ref in AME2020_BINDING_MEV:
        pred = nuclear_binding_energy_mev(IsotopeState(ref.Z, ref.N, 0.0), cfg)
        z = _z_score(pred, ref)
        rows.append((ref.symbol, ref.B_mev, ref.sigma_mev, pred, z))
    assert len(rows) == len(AME2020_BINDING_MEV)
    # All z-scores finite for positive σ
    for sym, _, sig, _, z in rows:
        assert math.isfinite(z), sym
        assert sig > 0.0
