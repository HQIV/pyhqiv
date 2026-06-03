#!/usr/bin/env python3
"""Print curvature + metric summaries from :class:`pyhqiv.state.HQIVState` (cross-check helper)."""

from __future__ import annotations

from pyhqiv.state import HQIVState


def main() -> None:
    st = HQIVState.from_snapshot(m=4, horizon_n=12, phi_newtonian=0.0, t=0.0)
    print("curvature:", st.curvature_summary())
    print("metric:", st.metric_summary())
    print("bundle:", st.as_dict())


if __name__ == "__main__":
    main()
