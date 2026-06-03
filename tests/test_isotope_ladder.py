"""Isotope ladder: network masses, binding, decay Q-values, barns (explicit nucleon masses)."""

from __future__ import annotations

import math

from pyhqiv.isotope_ladder import (
    IsotopeLadderConfig,
    IsotopeState,
    cross_section_geometric_barns,
    cross_section_geometric_isotope_barns,
    enumerate_decay_steps,
    laplace_angular_factor,
    nuclear_binding_energy_mev,
    nuclear_radius_fm,
    nucleus_mass_mev,
    rotational_excitation_mev,
)


def _cfg() -> IsotopeLadderConfig:
    return IsotopeLadderConfig(
        shell_m=4,
        m_proton_mev=938.0,
        m_neutron_mev=939.0,
        rotational_scale_mev=0.5,
    )


def test_nucleus_mass_and_binding_finite() -> None:
    cfg = _cfg()
    st = IsotopeState(Z=6, N=6, J=0.0)
    m = nucleus_mass_mev(st, cfg)
    b = nuclear_binding_energy_mev(st, cfg)
    assert math.isfinite(m) and math.isfinite(b)


def test_rotational_excitation_increases_with_J() -> None:
    cfg = _cfg()
    g = IsotopeState(Z=6, N=6, J=0.0)
    e = IsotopeState(Z=6, N=6, J=2.0)
    assert rotational_excitation_mev(e, cfg) > rotational_excitation_mev(g, cfg)


def test_laplace_angular_factor() -> None:
    assert laplace_angular_factor(2) == 6.0


def test_barns_and_radius() -> None:
    r = nuclear_radius_fm(64, r0_fm=1.2)
    sigma = cross_section_geometric_barns(r)
    s2 = cross_section_geometric_isotope_barns(IsotopeState(Z=29, N=35, J=0.0), r0_fm=1.2)
    assert sigma > 0.0 and s2 > 0.0


def test_decay_steps_non_empty_heavy() -> None:
    cfg = _cfg()
    parent = IsotopeState(Z=92, N=146, J=0.0)
    steps = enumerate_decay_steps(parent, cfg)
    modes_found = {s.mode for s in steps}
    assert len(steps) >= 1
    assert len(modes_found) >= 1


def test_isobar_iter() -> None:
    from pyhqiv.isotope_ladder import iter_isobar_chain_same_A

    xs = list(iter_isobar_chain_same_A(4))
    assert len(xs) == 3
    assert all(s.A == 4 for s in xs)
