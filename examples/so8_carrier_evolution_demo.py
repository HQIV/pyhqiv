#!/usr/bin/env python3
"""Dense ``expm`` step on the Lean so(8) carrier (compare norms before/after)."""

from __future__ import annotations

import numpy as np

from pyhqiv.carrier import So8Carrier
from pyhqiv.regimes.quantum import evolve_so8_carrier_expm


def main() -> None:
    c = So8Carrier.from_unit_axis(0)
    coeffs = np.zeros(28, dtype=np.float64)
    coeffs[0] = 0.25
    coeffs[1] = -0.1
    out = evolve_so8_carrier_expm(c, coeffs, dt=0.1)
    print("norm before", float(np.linalg.norm(c.psi)))
    print("norm after ", float(np.linalg.norm(out.psi)))
    print("psi_after", out.psi)


if __name__ == "__main__":
    main()
