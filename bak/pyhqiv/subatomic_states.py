"""
Epoch-dependent subatomic states from HQIV machinery.

This module provides a thin, LEAN-aligned wrapper around the existing
subatomic layer so that researchers can query **epoch-dependent** hadron
properties (energy and effective horizon) without touching PDG tables in
`src/`. PDG values live only in the test suite and are used there as
comparison targets.

Design
------

- Inputs:
  - `epoch`: same API as `t_qcd_gev_at_epoch` in the legacy layer:
    - `"now"`, `"lock"` / `"baryogenesis"`, or a float age in Gyr.
  - `flavor_content`: strings like `"uud"`, `"udd"`, `"udc"`, `"uudcc"`, etc.

- Outputs (dimensionful, first-principles at that epoch):
  - `energy_mev`: rest energy from the confined-state machinery
    (`confined_energy_mev(flavor, epoch)`).
  - `theta_m`: effective horizon Θ such that E = ħc/Θ.
  - `N_eff`: algebraic effective mode count N_eff from the LEAN `subatomic`
    module (`confined_effective_modes_for_flavor`).

Internally, the scale is set by the QCD temperature ladder
`t_qcd_gev_at_epoch` (see `subatomic_legacy`) and the same Fano-plane /
8×8 composite used for nucleon masses; PDG / CODATA do not enter this
module. When the flavor happens to be in the PDG-backed registry, tests may
compare the `"now"` slice of this generator against experimental values.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict

from pyhqiv.subatomic import (
    confined_effective_modes_for_flavor,
    e_bind_qcd_from_network,
    NetworkWeight,
)
from pyhqiv.subatomic_masses import QUARK_MASS_LADDER_NOW
from pyhqiv.constants import M_PROTON_MEV
from pyhqiv import coupling, subatomic_legacy as _legacy


@dataclass(frozen=True)
class SubatomicState:
    """Epoch-dependent subatomic state for a single confined flavor."""

    flavor: str
    epoch: str | float
    energy_mev: float
    theta_m: float
    N_eff: float


def _normalized_flavor(flavor_content: str) -> str:
    return flavor_content.strip().lower()


@lru_cache(maxsize=None)
def subatomic_state_at_epoch(
    flavor_content: str,
    epoch: str | float = "now",
    t_cmb: float = 2.725,
) -> SubatomicState:
    """
    First-principles subatomic state for a given flavor at a given epoch.

    Parameters
    ----------
    flavor_content:
        String of u,d,s,c,b,t (e.g. "uud", "udd", "udc", "uudcc").
    epoch:
        - "now": present-day QCD scale (T_LOCK_NOW_GEV).
        - "lock" / "baryogenesis": QCD lock-in epoch (T_LOCK_GEV).
        - float: age in Gyr (see `t_qcd_gev_at_epoch` in the legacy layer).
    t_cmb:
        CMB temperature in K; passed through for completeness to the legacy
        machinery, which uses it when mapping T_CMB → T_QCD. Defaults to
        the paper "now" value 2.725 K.
    """
    fc = _normalized_flavor(flavor_content)

    # Algebraic effective mode count (LEAN subatomic).
    N_eff = confined_effective_modes_for_flavor(fc)

    # Epoch-dependent coupling distance / horizon scale from the existing
    # first-principles machinery (QCD ladder + 8×8 composite).
    energy_mev = _legacy.confined_energy_mev(
        fc,
        t_cmb=t_cmb,
        epoch=epoch,
    )
    theta_m = _legacy.confined_effective_theta_m(
        fc,
        t_cmb=t_cmb,
        epoch=epoch,
    )

    return SubatomicState(
        flavor=fc,
        epoch=epoch,
        energy_mev=float(energy_mev),
        theta_m=float(theta_m),
        N_eff=float(N_eff),
    )


def subatomic_states_at_epoch(
    flavors: list[str],
    epoch: str | float = "now",
    t_cmb: float = 2.725,
) -> Dict[str, SubatomicState]:
    """
    Compute a dictionary of subatomic states for many flavors at one epoch.

    Example
    -------
    >>> states = subatomic_states_at_epoch(["uud", "udd", "udc"], epoch="now")
    >>> states["uud"].energy_mev
    >>> states["udc"].theta_m
    """
    return {
        _normalized_flavor(fc): subatomic_state_at_epoch(fc, epoch=epoch, t_cmb=t_cmb)
        for fc in flavors
    }


__all__ = [
    "SubatomicState",
    "subatomic_state_at_epoch",
    "subatomic_states_at_epoch",
]


# ---------------------------------------------------------------------------
# HQIV-only hadron masses from quark ladder + BoundStates network form
# ---------------------------------------------------------------------------


def _constituent_mass_sum_mev(flavor_content: str) -> float:
    """
    Sum of constituent quark masses for a given flavor string using the
    first-principles quark mass ladder (no PDG).
    """
    fc = flavor_content.strip().lower()
    masses = QUARK_MASS_LADDER_NOW.masses_mev
    total = 0.0
    for ch in fc:
        if ch not in masses:
            raise ValueError(f"Unknown quark flavor {ch!r} in {flavor_content!r}")
        total += masses[ch]
    return float(total)


def hadron_mass_mev_from_ladder_and_network(
    flavor_content: str,
    shell_qcd: int = 4,
    c: float = 1.0,
) -> float:
    """
    HQIV-only hadron mass from:

      - constituent masses from the quark ladder (proton-anchored, PDG-free),
      - QCD binding from the BoundStates network form with α_eff(shell_qcd).

    This is a minimal, structural implementation:

      M_hadron = Σ_q m_q  −  E_bind_QCD,
      E_bind_QCD = Σ_k w_k · α_eff(shell_qcd),

    To avoid double-counting binding on the proton witness, we *calibrate*
    the overall QCD binding strength so that the same formula reproduces the
    proton mass for flavor 'uud' at the chosen shell:

      Σ_q m_q('uud') − E_bind_QCD('uud') = M_PROTON_MEV.

    With the structural choice

      E_bind_QCD(flavor) = K · N_eff(flavor) · alpha_eff_shell(shell_qcd),

    the calibration fixes a single scalar K, which is then applied unchanged
    to all flavors (udd, ccd, ...). This is the simplest way to "unbind" the
    proton witness and reuse the same HQIV binding prescription elsewhere.
    """
    fc = flavor_content.strip().lower()
    # Constituent masses from ladder for target flavor and for proton witness
    m_const_target = _constituent_mass_sum_mev(fc)
    m_const_proton = _constituent_mass_sum_mev("uud")

    # Effective modes from LEAN subatomic (dimensionless)
    N_eff_target = confined_effective_modes_for_flavor(fc)
    N_eff_proton = confined_effective_modes_for_flavor("uud")

    # Shell-dependent coupling (same for all flavors at this QCD shell)
    alpha_qcd = coupling.alpha_eff_shell(shell_qcd, c=c)

    # Calibrate overall binding strength K from the proton witness:
    #   m_const_proton - K * N_eff_proton * alpha_qcd = M_PROTON_MEV
    denom = N_eff_proton * alpha_qcd
    if abs(denom) < 1e-30:
        K = 0.0
    else:
        K = (m_const_proton - M_PROTON_MEV) / denom

    # Define network weight function w_k = N_eff_target / 28 so that
    # e_bind_qcd_from_network reproduces K * N_eff_target * alpha_qcd.
    def w(k: int) -> float:  # NetworkWeight
        del k
        return (K * N_eff_target) / 28.0

    E_bind = e_bind_qcd_from_network(shell_qcd, w, c=c)

    return m_const_target - E_bind


__all__.extend(
    [
        "hadron_mass_mev_from_ladder_and_network",
    ]
)

