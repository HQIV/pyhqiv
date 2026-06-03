"""
Orbital / galactic dynamics calculator layer for HQIV.

Pure functions built on the foundation:
- lightcone (alpha, phi_of_shell via auxiliary or scale)
- metric (gamma_hqiv, G_eff, lapse)
- fluid (f_inertia)
- scale_witness / witnesses for anchors (reference shell, phi scales)

No constants in this module except geometry (imported). All scale, H0, c, G, solar masses etc.
are either passed as parameters or come from tests/setup_defaults + local_conditions (allowed).

Covers:
- Modified inertia f(a, φ) applications (already in fluid, re-exported for convenience)
- Angular Rindler denominator (shared with flyby + SPARC/galaxy rotation)
- Mass-horizon Doppler lapse / co-rotating terms
- Galaxy rotation curve correction (exponential disk + HQIV inertia + Rindler)
- Flyby anomaly correction terms (direction-dependent inertia, vacuum momentum, G_eff, lapse drag)

These are the "calculator" entry points for the orbital_flyby and octonionic_action papers.
Higher-level full integrators (ODE for hyperbolic flyby, full SPARC catalog fitting) can live in
tests or be added later as thin wrappers; the model physics lives here.

See papers/orbital_flyby/ and papers/octonionic_action/ scripts (and HQIV_LEAN/scripts/) for the
reference implementations this mirrors (cleaned of literals).
"""

from __future__ import annotations

from typing import Union

import numpy as np

from pyhqiv.fluid import f_inertia as _f_inertia
from pyhqiv.lightcone import alpha as get_alpha
from pyhqiv.metric import gamma_hqiv
from pyhqiv.auxiliary_field import phi_of_shell
from pyhqiv.scale_witness import load_local_conditions


def hqiv_inertia_factor(
    a_loc: Union[float, np.ndarray], phi: Union[float, np.ndarray], f_min: float = 0.01
) -> Union[float, np.ndarray]:
    """Re-export of the core f(a, φ) = a / (a + φ/6) with floor."""
    return _f_inertia(a_loc, phi, f_min=f_min)


def rindler_denominator(v: Union[float, np.ndarray], c: float = 1.0) -> Union[float, np.ndarray]:
    """
    Shared angular Rindler denominator D_R = 1 + (γ/2) (c/v)^2 .

    In natural units (c=1) pass c=1.0 (default). For SI, pass real c.
    Value comes from gamma_hqiv() (foundation).
    """
    gamma = gamma_hqiv()
    v = np.asarray(v, dtype=float)
    v = np.maximum(np.abs(v), 1e-30)
    return 1.0 + (gamma / 2.0) * (c / v) ** 2


def mass_horizon_doppler_lapse(
    v: Union[float, np.ndarray],
    *,
    projection: Union[float, np.ndarray] = 1.0,
    support_fraction: Union[float, np.ndarray] = 1.0,
    use_rindler_denominator: bool = True,
    c: float = 1.0,
) -> Union[float, np.ndarray]:
    """
    Co-rotating mass-horizon Doppler lapse term (galaxy / flyby analogue).

    eps = 2 (v/c) |proj| * support  / (rindler if enabled)
    Mirrors the term used for direction-dependent inertia screen in flybys and
    the disk term in SPARC/galaxy rotation.
    """
    v = np.asarray(v, dtype=float)
    proj = np.asarray(projection, dtype=float)
    supp = np.asarray(support_fraction, dtype=float)
    eps = 2.0 * np.abs(v) / c
    eps *= np.maximum(0.0, np.minimum(1.0, np.abs(proj)))
    eps *= np.maximum(0.0, np.minimum(1.0, supp))
    if use_rindler_denominator:
        eps = eps / rindler_denominator(v, c=c)
    return eps


def phi_acceleration_homogeneous(
    h0: float, c: float = 1.0
) -> float:
    """
    Homogeneous φ ≈ 2 c H0 (acceleration units).

    h0 and c must be supplied by the caller (from scale_witness local_conditions or
    tests/setup_defaults). No literals here.
    """
    return 2.0 * c * h0


def exponential_disk_enclosed_mass(
    radius: Union[float, np.ndarray], total_mass: float, scale_length: float
) -> Union[float, np.ndarray]:
    """Razor-thin exponential disk enclosed mass (standard, no HQIV)."""
    x = np.asarray(radius, dtype=float) / max(scale_length, 1e-30)
    return total_mass * (1.0 - np.exp(-x) * (1.0 + x))


def hqiv_galaxy_rotation_point(
    radius: float,
    disk_total_mass: float,
    disk_scale_length: float,
    observed_v: float | None = None,
    *,
    phi_shell: int = 0,
    use_rindler: bool = True,
    phi_hom: float | None = None,
) -> dict:
    """
    HQIV-corrected rotation at one radius for an exponential disk (SPARC-style first pass).

    Returns classical a_bary, phi, f, rindler, and the HQIV-adjusted centripetal acceleration
    or implied v_circ (depending on how the caller combines).

    This is the core "calculator" primitive used by the octonionic_action / SPARC paper
    first-pass tables. Higher-level catalog processing stays in tests or callers.
    """
    a_b = exponential_disk_enclosed_mass(radius, disk_total_mass, disk_scale_length) / max(radius, 1e-30) ** 2   # rough a = M(<r)/r^2  (G=1 units or scaled)

    phi = phi_of_shell(phi_shell)
    if phi_hom is not None:
        # allow caller to scale phi by homogeneous background
        phi = phi + phi_hom   # or more sophisticated blend per paper

    f = hqiv_inertia_factor(a_b, phi)

    v_tan = observed_v if observed_v is not None else (a_b * radius) ** 0.5   # placeholder
    rind = rindler_denominator(v_tan) if use_rindler else 1.0

    # Effective support from inertia screen + rindler (paper uses this for the correction to flat curve)
    a_hqiv = a_b * f / rind   # simplistic combination; real paper combines with direction etc.

    return {
        "radius": radius,
        "a_bary": a_b,
        "phi": phi,
        "f_inertia": f,
        "rindler": rind,
        "a_hqiv": a_hqiv,
    }


def hqiv_flyby_inertia_screen(
    a_loc: float,
    phi: float,
    h_z: float,
    h: float,
    h_ref: float,
    rho_pol: float,
    m_shell: int,
    *,
    direction_dependent: bool = True,
) -> float:
    """
    Direction-dependent inertia screen for flybys (polar fiber boost + release).

    Mirrors the logic in the orbital flyby paper / hqiv_orbital_flyby_omaxwell.py .
    Pure function; all scales (phi, m_shell) from foundation or caller.
    """
    f = hqiv_inertia_factor(a_loc, phi)

    if not direction_dependent:
        return f

    # Simplified polar fiber / release (full version has more eps_spin, gal, yr terms)
    floor = 1.0 / float(m_shell + 1) ** 2 if m_shell >= 0 else 0.0
    hz_eff = (h_z ** 2 + (h_ref * floor) ** 2) ** 0.5
    release = max(0.0, 1.0 - (h / max(hz_eff, 1e-30)) ** 2 )   # proxy
    boost = 1.0 + rho_pol * release
    return f * boost   # or more precise combination per equations


# Convenience re-exports for paper scripts / arena contributors
__all__ = [
    "hqiv_inertia_factor",
    "rindler_denominator",
    "mass_horizon_doppler_lapse",
    "phi_acceleration_homogeneous",
    "exponential_disk_enclosed_mass",
    "hqiv_galaxy_rotation_point",
    "hqiv_flyby_inertia_screen",
]
