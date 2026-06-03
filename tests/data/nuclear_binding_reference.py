"""
Nuclear **total binding energy** reference values for interaction-layer tests.

Sources (central values and uncertainties are from standard tables used alongside
PDG/CODATA nucleon masses):

- **AME2020** (Atomic Mass Evaluation) / **NUBASE2020** style total binding energies
  for selected nuclides. Uncertainties are the published experimental 1σ values
  (keV converted to MeV) where available; otherwise a conservative MeV-scale placeholder.

These are **not** engine inputs for pyhqiv; they exist only so tests can report
``(B_pred - B_ref) / σ_ref`` against real error bars.

CODATA 2018 nucleon masses (MeV) — use the **same** values when computing
model binding ``B = Z m_p + N m_n - M`` for apples-to-apples comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Tuple

# CODATA 2018 (exactly) — masses in MeV, 1σ uncertainties
CODATA_2018_PROTON_MEV: float = 938.2720813
CODATA_2018_PROTON_UNC_MEV: float = 0.0000058
CODATA_2018_NEUTRON_MEV: float = 939.5654133
CODATA_2018_NEUTRON_UNC_MEV: float = 0.0000058


@dataclass(frozen=True)
class BindingReference:
    """One nuclide: total nuclear binding energy and 1σ uncertainty (MeV)."""

    symbol: str
    Z: int
    N: int
    B_mev: float
    sigma_mev: float
    note: str = ""

    @property
    def A(self) -> int:
        return self.Z + self.N


# AME2020 / NUBASE2020 — total binding energy B (MeV), experimental 1σ (MeV).
# σ is order 10^{-3} MeV for light nuclei; rounded conservative for tests.
AME2020_BINDING_MEV: Tuple[BindingReference, ...] = (
    BindingReference("2H", 1, 1, 2.224566, 0.000012, "deuteron"),
    BindingReference("4He", 2, 2, 28.295674, 0.000012, "alpha"),
    BindingReference("12C", 6, 6, 92.161753, 0.000025, "carbon-12"),
    BindingReference("16O", 8, 8, 127.619343, 0.000030, "oxygen-16"),
    BindingReference("56Fe", 26, 30, 492.259, 0.003, "iron-56"),
)


def reference_keys() -> FrozenSet[Tuple[int, int]]:
    return frozenset((r.Z, r.N) for r in AME2020_BINDING_MEV)


def lookup_binding(Z: int, N: int) -> BindingReference | None:
    for r in AME2020_BINDING_MEV:
        if r.Z == Z and r.N == N:
            return r
    return None


__all__ = [
    "AME2020_BINDING_MEV",
    "BindingReference",
    "CODATA_2018_NEUTRON_MEV",
    "CODATA_2018_NEUTRON_UNC_MEV",
    "CODATA_2018_PROTON_MEV",
    "CODATA_2018_PROTON_UNC_MEV",
    "lookup_binding",
    "reference_keys",
]
