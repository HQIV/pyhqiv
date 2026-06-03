#!/usr/bin/env python3
"""Galactic façade: ``G_eff(φ)`` vs direct :func:`pyhqiv.metric.g_eff`."""

from __future__ import annotations

from pyhqiv import auxiliary_field as af
from pyhqiv import metric
from pyhqiv.state import HQIVState
from pyhqiv.regimes.galactic import galactic_g_eff, galactic_metric_summary


def main() -> None:
    m = 4
    phi = af.phi_of_shell(m)
    g1 = galactic_g_eff(phi)
    g2 = metric.g_eff(phi)
    print(f"m={m} phi={phi:.6g} galactic_g_eff={g1:.6g} metric.g_eff={g2:.6g} match={g1 == g2}")
    st = HQIVState.from_snapshot(m=m)
    print("metric_summary", galactic_metric_summary(st))


if __name__ == "__main__":
    main()
