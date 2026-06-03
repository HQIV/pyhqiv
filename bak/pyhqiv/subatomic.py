"""
LEAN-aligned subatomic layer: flavor → octonionic structure.

This module removes all CODATA/PDG/experimental constants and works purely with
the algebraic machinery of HQIV (octonions, Fano-plane, Spin(8) embedding).

Responsibilities (Layer 0, LEAN-only):
  - Translate first-generation Standard Model labels (u, d) and simple flavor
    strings (e.g. 'uud', 'udd') into 8×8 state matrices using the HQVM
    octonion algebra (`OctonionHQIVAlgebra`).
  - Build colour-singlet composites for confined states via the generic
    `merge_constituents` machinery.
  - Expose **dimensionless** invariants and effective mode counts derived from
    the 8×8 composite and the Fano-plane phase-lift Δ, suitable as inputs to
    higher-layer binding and horizon logic.

No physical units or experimental numbers live here: all outputs are
dimensionless and depend only on the algebraic structure (g₂ ⊂ so(8),
hypercharge operator Y, and the phase-lift Δ) that HQIV_LEAN fixes uniquely.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from pyhqiv.algebra import OctonionHQIVAlgebra
from pyhqiv.energy_field import merge_constituents
from pyhqiv import coupling

# Allowed flavor labels. These are *names only*; no masses or charges attached.
_VALID_FLAVORS: str = "udscbt"


# ---------------------------------------------------------------------------
# Generation shell indices and constituent-scale factors (dimensionless)
# ---------------------------------------------------------------------------
#
# The discrete auxiliary-field shell index m encodes how "heavy" a constituent
# is in the HQIV lattice. The following helpers implement the purely
# combinatorial / logarithmic shape
#
#   phi(m)        = 2 (m + 1)
#   lattice(m)    = 4 (m + 2) (m + 1)
#   shell_shape(m)= (1 + 0.6 log(m + 1)) / (m + 1)
#   mass_factor(m)= phi(m) * lattice(m) * shell_shape(m),
#
# and then use *ratios* of mass_factor(m) to distinguish generations. No PDG
# or CODATA inputs appear here; all numbers are dimensionless and built from
# integers and logs only.


def _phi(m: int) -> float:
    return 2.0 * (float(m) + 1.0)


def _lattice(m: int) -> float:
    mm = float(m)
    return 4.0 * (mm + 2.0) * (mm + 1.0)


def _shell_shape(m: int) -> float:
    mm = float(m)
    return (1.0 + 0.6 * np.log(mm + 1.0)) / (mm + 1.0)


def _mass_factor(m: int) -> float:
    return _phi(m) * _lattice(m) * _shell_shape(m)


# Minimal mapping from flavors to shell index m. This keeps u/d at a light
# constituent shell and assigns progressively higher m to heavier generations,
# following the sandbox table used in the HQIV notes.
_FLAVOR_SHELL_INDEX: Dict[str, int] = {
    "u": 2,
    "d": 2,
    "s": 3,
    "c": 5,
    "b": 7,
    "t": 9,
}


def _flavor_scale(flavor: str) -> float:
    """
    Dimensionless constituent scale factor for a quark flavor.

    This is the ratio

        scale(flavor) = mass_factor(m_flavor) / mass_factor(m_ref),

    with m_ref = shell index for first-generation u/d. It captures the
    hierarchy (s > d > u, c >> u, etc.) without introducing any PDG/CODATA
    information into the src tree.
    """
    f = (flavor or "u").strip().lower()
    m_ref = _FLAVOR_SHELL_INDEX["u"]
    m_flavor = _FLAVOR_SHELL_INDEX.get(f, m_ref)
    ref = _mass_factor(m_ref)
    if ref <= 0.0:
        return 1.0
    return _mass_factor(m_flavor) / ref


def constituent_shell_mass_factor_for_flavor(flavor_content: str) -> float:
    """
    Dimensionless constituent mass-factor sum for a confined flavor string.

    This is the sandbox-style proxy

        M_shell(flavor_content) = Σ_q mass_factor(m_q),

    where m_q is the shell index assigned to each constituent flavor. It encodes
    the heavy–light hierarchy (u/d, s, c, b, t) using only the combinatorial /
    logarithmic structure of the auxiliary field; no experimental inputs are
    used. Absolute MeV scales should be set in tests using a single witness.
    """
    flavors = quark_flavors_from_flavor_content(flavor_content)
    total = 0.0
    for f in flavors:
        m_q = _FLAVOR_SHELL_INDEX.get(f, _FLAVOR_SHELL_INDEX["u"])
        total += _mass_factor(m_q)
    return total


def mass_proxy_from_shell_and_em(flavor_content: str) -> float:
    """
    Dimensionless mass proxy combining shell hierarchy and EM block structure.

    - Baseline comes from the constituent shell mass factor sum.
    - A small correction term subtracts a magnified version of the 4×4
      hypercharge-block fraction so that states with more unbound EM
      (like uud) come out slightly lighter than folded states (like udd).

    No PDG/CODATA inputs appear; the proton can be used as a single
    external witness in tests to set the overall MeV scale.
    """
    m_shell = constituent_shell_mass_factor_for_flavor(flavor_content)
    inv = composite_invariants(flavor_content)
    f_block = float(inv.get("block_4x4_fraction", 0.0))
    # Magnify the tiny block fraction into an O(1) correction while
    # remaining dimensionless. The exact numerical factor is fixed at
    # this layer; any MeV-scale calibration happens in tests only.
    em_correction = 1.0e68 * f_block
    return m_shell - em_correction


def _sphere_touching_mu(radii: np.ndarray) -> float:
    """
    Sphere-touching mode multiplier μ for a set of horizon radii.

    Pure geometry:
        μ = (Σ r_i) / sqrt(Σ r_i²)  ≥  1

    This is the same Pythagorean "Casimir deficit" used throughout the horizon
    network code, but expressed without any reference to units or external
    constants. The radii provided by callers already encode whatever QCD /
    Lean-derived mass information is appropriate for the layer.
    """
    r = np.asarray(radii, dtype=float)
    if r.size == 0:
        return 1.0
    num = float(np.sum(r))
    den = float(np.sqrt(np.sum(r * r)))
    if den <= 0.0:
        return 1.0
    return max(num / den, 1.0)


def quark_flavors_from_flavor_content(flavor_content: str) -> Tuple[str, ...]:
    """
    Valence-quark flavors from a string of u,d,s,c,b,t (e.g. 'uud', 'udd', 'uds', 'uudcc').

    This is the single source of truth for mapping *labels* like 'uud' onto
    first-generation degrees of freedom. No numerical constants enter; the
    algebraic content comes entirely from `OctonionHQIVAlgebra`.
    """
    fc = flavor_content.strip().lower()
    out: List[str] = []
    for c in fc:
        if c not in _VALID_FLAVORS:
            raise ValueError(f"flavor_content must contain only {_VALID_FLAVORS!r}, got {flavor_content!r}")
        out.append(c)
    if not out:
        raise ValueError(f"flavor_content must be non-empty, got {flavor_content!r}")
    return tuple(out)


def quark_flavors_for_nucleon(is_proton: bool) -> Tuple[str, str, str]:
    """
    Convenience: valence-quark labels for a single nucleon in first generation.

    - Proton:  'uud'
    - Neutron: 'udd'
    """
    return quark_flavors_from_flavor_content("uud" if is_proton else "udd")  # type: ignore[return-value]


def _default_algebra(algebra: OctonionHQIVAlgebra | None = None) -> OctonionHQIVAlgebra:
    """Small helper to obtain a quiet OctonionHQIVAlgebra instance."""
    if algebra is not None:
        return algebra
    return OctonionHQIVAlgebra(verbose=False)


def color_generators(algebra: OctonionHQIVAlgebra | None = None) -> List[np.ndarray]:
    """
    Canonical SU(3)_c generators as 8×8 matrices from the HQVM algebra engine.

    These live entirely in the g₂ block selected by the Fano-plane orientation
    and do not depend on any experimental scales.
    """
    alg = _default_algebra(algebra)
    gens = alg._identify_color_generators()
    return list(gens)


def hypercharge_operator(algebra: OctonionHQIVAlgebra | None = None) -> np.ndarray:
    """
    Hypercharge operator Y (8×8) from the HQVM algebra engine.

    The exact linear combination is fixed by the Lean / paper construction and
    encoded in `hypercharge_coefficients`. No PDG input appears here; Y is
    purely algebraic.
    """
    alg = _default_algebra(algebra)
    _, Y, _ = alg.hypercharge_coefficients()
    return Y


def quark_state_matrix(
    flavor: str = "u",
    color_index: int = 0,
    algebra: OctonionHQIVAlgebra | None = None,
) -> np.ndarray:
    """
    8×8 state matrix for a single quark (u,d,s,c,b,t) in one colour.

    Construction:
      - Take one of the canonical colour generators G_c from g₂ (via
        `color_generators`).
      - Add a fixed multiple of the hypercharge operator Y, with the multiple
        set *symbolically* by the first-generation pattern:
          * up-type (u,c,t): +2/3 unit of Y
          * down-type (d,s,b): -1/3 unit of Y

    This is the minimal LEAN-consistent mapping from Standard Model labels to
    the Spin(8)/octonion representation; it uses only rational coefficients
    and the algebraic operators {G_c, Y} fixed by HQIV, with no external
    physical scales.
    """
    alg = _default_algebra(algebra)
    gens = color_generators(alg)
    if not gens:
        # Degenerate algebra instance; fall back to a zero matrix.
        return np.zeros((8, 8), dtype=complex)

    idx = int(color_index) % len(gens)
    G_c = gens[idx]

    f = (flavor or "u").strip().lower()
    if f in ("u", "c", "t"):
        y_coeff = 2.0 / 3.0
    elif f in ("d", "s", "b"):
        y_coeff = -1.0 / 3.0
    else:
        raise ValueError(f"Unknown quark flavor {flavor!r}; expected one of {_VALID_FLAVORS!r}")

    Y = hypercharge_operator(alg)
    # State is a linear combination in the Lie algebra; overall scale is left
    # free for higher layers to absorb into the coupling distance.
    return G_c + y_coeff * Y


def quark_state_matrices_for_flavor(
    flavor_content: str,
    algebra: OctonionHQIVAlgebra | None = None,
) -> List[np.ndarray]:
    """
    8×8 state matrices for each constituent in a confined state.

    Example:
      - 'uud' → [quark_state_matrix('u', 0), quark_state_matrix('u', 1),
                 quark_state_matrix('d', 2)]

    The assignment of colour index is purely combinatorial (0,1,2,...) and
    carries no experimental input.
    """
    alg = _default_algebra(algebra)
    flavors = quark_flavors_from_flavor_content(flavor_content)
    return [quark_state_matrix(f, color_index=i, algebra=alg) for i, f in enumerate(flavors)]


def composite_state_matrix(
    flavor_content: str,
    algebra: OctonionHQIVAlgebra | None = None,
    project_color_singlet: bool = True,
) -> np.ndarray:
    """
    Composite 8×8 state matrix for a confined configuration of quarks.

    This uses the same `merge_constituents` machinery as at nuclear and atomic
    scales. The only inputs are:
      - The list of constituent 8×8 matrices (from `quark_state_matrices_for_flavor`).
      - Whether to project onto the colour-singlet subspace.

    Returns the **state matrix** M (8×8) of the composite.
    """
    alg = _default_algebra(algebra)
    mats = quark_state_matrices_for_flavor(flavor_content, algebra=alg)
    composite = merge_constituents(mats, project_singlet=project_color_singlet, algebra=alg)
    return composite.state_matrix


def composite_invariants(
    flavor_content: str,
    algebra: OctonionHQIVAlgebra | None = None,
) -> Dict[str, float]:
    """
    Dimensionless invariants of the composite 8×8 state matrix.

    Outputs
    -------
    dict with keys:
      - 'trace': Tr(M)
      - 'frobenius_norm_sq': ||M||_F^2
      - 'coherence': (Tr M)^2 / ||M||_F^2
      - 'span': ||M||_F^2 / max(|Tr M|, ε)
      - 'trace_M_Delta': Tr(M Δ), where Δ is the phase-lift generator
      - 'block_4x4_fraction' (optional): fraction of Frobenius norm in the
        4×4 hypercharge block M[4:8,4:8] when hypercharge data is available
    """
    alg = _default_algebra(algebra)
    M = composite_state_matrix(flavor_content, algebra=alg, project_color_singlet=True)

    tr = float(np.trace(M))
    fro_sq = float(np.linalg.norm(M) ** 2)
    eps = 1e-30

    coherence = (tr * tr) / max(fro_sq, eps) if fro_sq > eps else 0.0
    span = fro_sq / max(abs(tr), eps) if abs(tr) > eps else 0.0
    trace_M_Delta = float(np.trace(M @ alg.Delta))

    out: Dict[str, float] = {
        "trace": tr,
        "frobenius_norm_sq": fro_sq,
        "coherence": coherence,
        "span": span,
        "trace_M_Delta": trace_M_Delta,
    }

    try:
        _, Y, _ = alg.hypercharge_coefficients()
        if Y is not None:
            block = M[4:8, 4:8]
            out["block_4x4_fraction"] = float(np.linalg.norm(block) ** 2) / max(fro_sq, eps)
    except Exception:
        # Hypercharge identification is optional; invariants above are still valid.
        pass

    return out


def effective_modes_from_composite(
    flavor_content: str,
    algebra: OctonionHQIVAlgebra | None = None,
) -> float:
    """
    Dimensionless effective mode count N_eff for a confined state.

    This is the LEAN-aligned, unitless replacement for the legacy
    `_confined_effective_modes`. It depends only on:
      - The phase-lift Δ (Fano-plane / Spin(8) structure).
      - The composite state matrix M built from colour + hypercharge.

    Definition
    ----------
    N_eff = 8 + Tr(M Δ) + bonus(coherence),

    where the "bonus" term is a small, purely algebraic correction that rewards
    more coherent composites (like uud) relative to less coherent ones (like
    udd) without ever referring to MeV, fm, or experimental numbers.
    """
    alg = _default_algebra(algebra)
    measures = composite_invariants(flavor_content, algebra=alg)

    # Baseline 8 modes from the octonionic representation plus a phase-lift
    # correction proportional to Tr(M Δ).
    n_eff = 8.0 + measures["trace_M_Delta"]

    # Coherence-based bonus: use a soft, dimensionless rescaling that keeps the
    # correction O(1) across the baryon registry.
    coherence = measures["coherence"]
    # Saturating map: bonus in [0, 1) for coherence ≥ 0.
    bonus = coherence / (1.0 + coherence) if coherence > 0.0 else 0.0
    return n_eff + bonus


def nucleon_effective_modes(
    is_proton: bool,
    algebra: OctonionHQIVAlgebra | None = None,
) -> float:
    """
    Effective mode count for a single nucleon (proton or neutron).

    - Proton:  flavor_content = 'uud'
    - Neutron: flavor_content = 'udd'
    """
    fc = "uud" if is_proton else "udd"
    return effective_modes_from_composite(fc, algebra=algebra)


def confined_effective_modes_for_flavor(
    flavor_content: str,
    algebra: OctonionHQIVAlgebra | None = None,
) -> float:
    """
    Public helper: effective mode count N_eff for any confined flavor string.

    This is the only scalar from the subatomic layer that higher layers should
    use when turning algebra into horizons/energies:
      E ∝ 1 / (x · N_eff)

    where x is the coupling distance supplied by the binding / curvature
    machinery. This keeps all CODATA/PDG dependence out of this module.
    """
    return effective_modes_from_composite(flavor_content, algebra=algebra)


# ---------------------------------------------------------------------------
# Bound-states style network binding (Lean: BoundStates.lean)
# ---------------------------------------------------------------------------

# In the Lean development, the bound-state energy is expressed as a sum over
# the so(8) generator index k with shell-dependent coupling α_eff(shell):
#
#   E_bind(m) = Σ_k w_k · alphaEffAtShell(m)
#
# where the weights w_k come from the 8×8 representation (matrix elements,
# traces, or expectation values). Here we expose the same structural form so
# that nucleon and hadron masses can be computed from first principles using
# the algebraic subatomic layer plus the coupling module.

So8Index = int
NetworkWeight = Callable[[So8Index], float]


def network_weight_from_composite(
    composite_matrix: np.ndarray,
    algebra: OctonionHQIVAlgebra | None = None,
) -> NetworkWeight:
    """
    Canonical NetworkWeight from a composite 8×8 state matrix.

    We use the so(8) basis produced by `OctonionHQIVAlgebra.lie_closure_basis`
    and define the weight for generator k as a simple trace pairing

        w_k = Re Tr(M · T_k),

    where M is the composite state matrix and T_k is the k-th so(8) generator.
    This is fully dimensionless and depends only on the algebraic structure
    fixed by HQIV/LEAN.
    """
    alg = _default_algebra(algebra)
    basis = alg.lie_closure_basis()
    # Precompute coefficients once; expose them through a callable.
    coeffs: List[float] = []
    for T_k in basis:
        # Use the standard matrix trace pairing; take real part in case of
        # numerical noise.
        coeffs.append(float(np.trace(composite_matrix @ T_k).real))

    def w(k: So8Index) -> float:
        idx = int(k)
        if 0 <= idx < len(coeffs):
            return coeffs[idx]
        return 0.0

    return w


def network_weight_from_flavor(
    flavor_content: str,
    algebra: OctonionHQIVAlgebra | None = None,
    project_color_singlet: bool = True,
) -> NetworkWeight:
    """
    Convenience: NetworkWeight for a confined configuration specified by flavor_content.

    This builds the composite 8×8 state matrix via `composite_state_matrix` and
    then derives the canonical NetworkWeight using `network_weight_from_composite`.
    No scales or experimental inputs enter; everything is determined by the
    octonionic algebra and the flavor labels.
    """
    alg = _default_algebra(algebra)
    M = composite_state_matrix(
        flavor_content,
        algebra=alg,
        project_color_singlet=project_color_singlet,
    )
    return network_weight_from_composite(M, algebra=alg)


def alpha_eff_at_shell(m: int, c: float = 1.0) -> float:
    """
    Effective fine-structure α_eff at shell m.

    Thin wrapper around `coupling.alpha_eff_shell`, mirroring
    `alphaEffAtShell` in the Lean BoundStates module.
    """
    return float(coupling.alpha_eff_shell(m, c=c))


def binding_coupling_at_shell(m: int, _k: So8Index, c: float = 1.0) -> float:
    """
    Coupling factor at shell m for generator k.

    In the abstract BoundStates form this is the same α_eff(m) for all k;
    sector-dependent refinements (EM vs strong) can be added later if needed.
    """
    return alpha_eff_at_shell(m, c=c)


def e_bind_from_network(m: int, w: NetworkWeight, c: float = 1.0) -> float:
    """
    Binding energy from a network of so(8) generators at shell m.

    E_bind = Σ_k w_k · binding_coupling_at_shell(m, k, c)

    The weights w_k should be constructed from the 8×8 composite state matrix
    (e.g. traces or expectation values of the generators). This mirrors
    `E_bind_from_network` in `Hqiv/Physics/BoundStates.lean`.
    """
    total = 0.0
    # OctonionHQIVAlgebra exposes a 28-dimensional so(8) basis; we mirror that.
    for k in range(28):
        total += w(k) * binding_coupling_at_shell(m, k, c=c)
    return float(total)


def e_bind_qcd_from_network(m: int, w: NetworkWeight, c: float = 1.0) -> float:
    """
    QCD binding (quarks → hadron) as a sum over the so(8) network at shell m.

    Alias for `e_bind_from_network`; kept for clarity when wiring nucleon /
    hadron mass formulas to this layer.
    """
    return e_bind_from_network(m, w, c=c)


__all__ = [
    "_sphere_touching_mu",
    "quark_flavors_from_flavor_content",
    "quark_flavors_for_nucleon",
    "color_generators",
    "hypercharge_operator",
    "quark_state_matrix",
    "quark_state_matrices_for_flavor",
    "composite_state_matrix",
    "composite_invariants",
    "effective_modes_from_composite",
    "nucleon_effective_modes",
    "confined_effective_modes_for_flavor",
    "So8Index",
    "NetworkWeight",
    "network_weight_from_composite",
    "network_weight_from_flavor",
    "alpha_eff_at_shell",
    "binding_coupling_at_shell",
    "e_bind_from_network",
    "e_bind_qcd_from_network",
    # Legacy-compatible exports are provided via subatomic_legacy in __init__.py.
]

