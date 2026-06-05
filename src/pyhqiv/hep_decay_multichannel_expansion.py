"""
Full multi-channel HEP decay expansion (OZI, templates, combinatorics).

Port of Lean-side hqiv_hep_multichannel_expansion.py into the pyhqiv calculator.
Uses HepDecayReadout for OZI + production weights, and a mass lookup callable.

Enables large open channel counts for J/ψ, Υ etc in the discharged calculator.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import pyhqiv.hep_decay_readout as hdr

ChannelTag = Literal["strong", "weak", "electromagnetic", "weak_hadron", "stable"]
MassLookup = Callable[[str], float]
MassXi = float

# Pools (light hadrons for expansion)
LIGHT_HADRONS: tuple[str, ...] = (
    "pi_plus", "pi_minus", "pi_zero",
    "K_plus", "K_minus", "K0",
    "rho_zero", "rho_plus", "rho_minus",
    "omega_meson", "phi",
    "eta", "eta_prime",
)
VECTOR_POOL: frozenset[str] = frozenset({"rho_zero", "rho_plus", "rho_minus", "omega_meson", "phi", "Jpsi", "Upsilon"})

# Simple templates for open charm/bottom (extend as needed)
OPEN_CHARM_WEAK_TEMPLATES: tuple[tuple[tuple[str, ...], float], ...] = (
    (("K_minus", "pi_plus"), 0.55),
    (("K0", "pi_plus"), 0.45),
    (("K_plus", "pi_minus"), 0.20),
    (("pi_plus", "pi_minus", "pi_zero"), 0.10),
)
OPEN_BOTTOM_WEAK_TEMPLATES: tuple[tuple[tuple[str, ...], float], ...] = (
    (("D0", "pi_plus"), 0.50),
    (("D_plus", "pi_minus"), 0.30),
)

# Baryon weak
CHARMED_BARYON_WEAK: tuple[tuple[tuple[str, ...], float], ...] = (
    (("p", "K_minus", "pi_plus"), 0.70),
    (("p", "pi_zero"), 0.12),
    (("n", "K_plus"), 0.15),
)
BOTTOM_BARYON_WEAK: tuple[tuple[tuple[str, ...], float], ...] = (
    (("lambda_c", "pi_minus"), 0.85),
    (("lambda_c", "K_minus"), 0.45),
    (("p", "D0", "pi_minus"), 0.35),
)

OPEN_CHARM_MESONS = frozenset({"D_plus", "D0", "Ds_plus"})
OPEN_BOTTOM_MESONS = frozenset({"B_plus", "B0", "Bs"})
CHARMED_BARYONS = frozenset({"lambda_c", "sigma_c", "xi_c", "omega_c"})
BOTTOM_BARYONS = frozenset({"lambda_b", "xi_b", "omega_b"})
HIDDEN_QUARKONIA = frozenset({"Jpsi", "Upsilon"})

@dataclass(frozen=True)
class GeneratedMode:
    parent_id: str
    channel: ChannelTag
    daughter_ids: tuple[str, ...]
    relative_branch: float
    source: str = "multichannel"

    @property
    def key(self) -> str:
        d = "+".join(self.daughter_ids) if self.daughter_ids else "stable"
        return f"{self.parent_id}->{self.channel}:{d}"


def ozi_suppression_factor(parent_id: str, daughter_ids: Sequence[str]) -> float:
    """
    OZI / Zweig suppression for hidden quarkonia → light hadrons.

    Lean slot: ``γ/4`` leading; extra ``γ/8`` per light vector in final state.
    """
    if parent_id not in ("Jpsi", "Upsilon"):
        return 1.0
    base = hdr.GAMMA / 4.0
    n_vector = sum(1 for d in daughter_ids if d in VECTOR_POOL)
    return base * (1.0 + hdr.GAMMA * n_vector / 8.0)


def _strange_count(daughter_ids: Sequence[str]) -> int:
    strange_ids = {"K_plus", "K_minus", "K0", "K0_bar", "phi", "Ds_plus"}
    return sum(1 for d in daughter_ids if d in strange_ids or d.startswith("K"))


def _daughter_mass_sum(daughters: Sequence[str], mass_of: MassLookup) -> float:
    total = 0.0
    for did in daughters:
        try:
            total += mass_of(did)
        except (KeyError, TypeError):
            total += 0.0
    return total


def _channel_open(parent_mass_mev: float, daughters: Sequence[str], mass_of: MassLookup) -> bool:
    return _daughter_mass_sum(daughters, mass_of) < parent_mass_mev - 1.0


def _topology_prior(parent_id: str, channel: ChannelTag, daughter_ids: Sequence[str]) -> float:
    """Base prior before OZI / phase / CKM."""
    if channel == "electromagnetic":
        if any(d in ("e_plus", "e_minus", "mu_plus", "mu_minus") for d in daughter_ids):
            return 0.06 if parent_id in ("Jpsi",) else 0.03
        return 0.01
    if parent_id in ("Jpsi", "Upsilon") and channel == "strong":
        # OZI will suppress further
        return 0.3 if any("rho" in d or "phi" in d for d in daughter_ids) else 0.1
    if channel == "strong":
        return 1.0
    if channel in ("weak", "weak_hadron"):
        # charm/bottom weak priors (will be reweighted by CKM in production layer)
        if parent_id.startswith("D") or parent_id in ("lambda_c",):
            return 0.5
        if parent_id.startswith("B") or "lambda_b" in parent_id:
            return 0.2
        return 0.1
    return 0.01


def _add_mode(
    out: list[GeneratedMode],
    parent: str,
    channel: ChannelTag,
    daughters: tuple[str, ...],
    prior: float,
    mass_of: MassLookup,
    parent_mass: float,
) -> None:
    if not _channel_open(parent_mass, daughters, mass_of):
        return
    ozi = ozi_suppression_factor(parent, daughters)
    br = max(1e-9, prior * ozi)
    out.append(GeneratedMode(parent, channel, daughters, br))


def _two_body_combos(pool: Sequence[str]) -> list[tuple[str, ...]]:
    return [(a, b) for a in pool for b in pool if a <= b]  # avoid pure dupes for count


def _three_body_combos(pool: Sequence[str], *, max_count: int = 120) -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = []
    for a in pool:
        for b in pool:
            for c in pool:
                if len(combos) >= max_count:
                    return combos
                if a <= b <= c:  # canonical order
                    combos.append((a, b, c))
    return combos


def _generate_quarkonium_modes(
    parent_id: str,
    parent_mass_mev: float,
    mass_of: MassLookup,
) -> list[GeneratedMode]:
    out: list[GeneratedMode] = []
    # EM leptonic
    for lep in [("e_plus", "e_minus"), ("mu_plus", "mu_minus")]:
        _add_mode(out, parent_id, "electromagnetic", lep, 0.06 if "e" in lep[0] else 0.03, mass_of, parent_mass_mev)
    # Radiative + hadronic via pools (light vectors + pions + K)
    light_pool = [h for h in LIGHT_HADRONS if h not in ("Jpsi", "Upsilon")]
    for d in light_pool:
        _add_mode(out, parent_id, "strong", (d,), 0.02, mass_of, parent_mass_mev)
    for combo in _two_body_combos(light_pool)[:80]:
        _add_mode(out, parent_id, "strong", combo, 0.01, mass_of, parent_mass_mev)
    for combo in _three_body_combos(light_pool, max_count=60):
        _add_mode(out, parent_id, "strong", combo, 0.005, mass_of, parent_mass_mev)
    # Hidden to hidden (small)
    if parent_id == "Jpsi":
        _add_mode(out, parent_id, "electromagnetic", ("Upsilon", "gamma"), 0.001, mass_of, parent_mass_mev)
    return out


def _generate_from_templates(
    parent_id: str,
    templates: Sequence[tuple[tuple[str, ...], float]],
    channel: ChannelTag,
    parent_mass: float,
    mass_of: MassLookup,
) -> list[GeneratedMode]:
    out: list[GeneratedMode] = []
    for ds, prior in templates:
        _add_mode(out, parent_id, channel, ds, prior, mass_of, parent_mass)
    # Add some extra combos from light for more channels
    pool = [h for h in LIGHT_HADRONS if "K" in h or "pi" in h][:6]
    for combo in _two_body_combos(pool)[:10]:
        _add_mode(out, parent_id, channel, combo, 0.02, mass_of, parent_mass)
    return out


def generate_multichannel_modes(
    parent_id: str,
    *,
    parent_mass_mev: float,
    mass_of: MassLookup,
    xi: MassXi = 5.0,
) -> list[GeneratedMode]:
    """Generate rich set of decay modes for the parent (multichannel + OZI)."""
    pid = parent_id
    out: list[GeneratedMode] = []
    if pid in ("Jpsi", "Upsilon"):
        out.extend(_generate_quarkonium_modes(pid, parent_mass_mev, mass_of))
        # Add more light combos for volume
        pool = [h for h in LIGHT_HADRONS if h not in (pid,)]
        for combo in list(itertools.combinations(pool, 2))[:40]:
            _add_mode(out, pid, "strong", combo, 0.008, mass_of, parent_mass_mev)
        for combo in list(itertools.combinations(pool, 3))[:30]:
            _add_mode(out, pid, "strong", combo, 0.003, mass_of, parent_mass_mev)
    elif pid in OPEN_CHARM_MESONS:
        out.extend(_generate_from_templates(pid, OPEN_CHARM_WEAK_TEMPLATES, "weak", parent_mass_mev, mass_of))
    elif pid in OPEN_BOTTOM_MESONS:
        out.extend(_generate_from_templates(pid, OPEN_BOTTOM_WEAK_TEMPLATES, "weak", parent_mass_mev, mass_of))
    elif pid in CHARMED_BARYONS:
        out.extend(_generate_from_templates(pid, CHARMED_BARYON_WEAK, "weak", parent_mass_mev, mass_of))
    elif pid in BOTTOM_BARYONS:
        out.extend(_generate_from_templates(pid, BOTTOM_BARYON_WEAK, "weak", parent_mass_mev, mass_of))
    else:
        # light fallback: few strong + weak
        for d in ("pi_plus", "pi_minus", "pi_zero", "K_plus"):
            _add_mode(out, pid, "strong", (d,), 0.2, mass_of, parent_mass_mev)
    # Dedup by key, keep highest prior
    by_key: dict[str, GeneratedMode] = {}
    for m in out:
        k = m.key
        if k not in by_key or m.relative_branch > by_key[k].relative_branch:
            by_key[k] = m
    return sorted(by_key.values(), key=lambda m: -m.relative_branch)


def open_channel_count(parent_id: str, parent_mass_mev: float, xi: MassXi) -> int:
    def mass_of(sid: str) -> float:
        # lightweight; real callers pass hep.particle_mass_mev
        try:
            # try import to avoid cycle at module load
            import pyhqiv.hep_decay_chain as ch

            return ch.particle_mass_mev(sid, xi=xi)
        except Exception:
            return 140.0 if "pi" in sid else 500.0 if "K" in sid else 1000.0
    modes = generate_multichannel_modes(parent_id, parent_mass_mev=parent_mass_mev, mass_of=mass_of, xi=xi)
    return len([m for m in modes if _channel_open(parent_mass_mev, m.daughter_ids, mass_of)])
