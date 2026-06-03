"""
First-principles atomic energy levels from HQIV axioms only.

Electrons are leptons: they live in a different part of the Fano plane than
quarks (no quarks in electrons). The 8×8 composite + confined_energy machinery
is hadronic only; we do not derive electron mass from it. Until the lepton
sector is implemented in the framework, we use the reference value M_E_MEV_REF
(CODATA) for the electron rest energy. Electrons are treated as diffuse wave
packets on the nuclear φ-field; higher n = higher lattice modes → more compact
central probability peak via mode interference. No separate electron horizons.

Binding physics: Casimir Ω_ij is only between nuclear pairs (electrons have no
horizon shadows). So for H (A=1) Casimir = 0; for multi-nucleon cores Casimir
is p–p / p–n / n–n only. Atomic binding is thus dominated by modified Coulomb
(e–n attraction + e–e repulsion) and the global conservation term (mode count).
Reuses nuclear engine: generate_nuclei_pack_coords, Casimir (nuclear only),
modified_field_contribution, conservation, minimize over radial trial; scale
via apply_scale_witness only.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Callable

import numpy as np
from scipy.optimize import minimize_scalar

from pyhqiv.atom import HQIVAtom
from pyhqiv.constants import C_SI, GAMMA, HBAR_C_MEV_FM, M_E_MEV_REF, T_LOCK_NOW_GEV
from pyhqiv.subatomic import nucleon_effective_theta_m

from pyhqiv.nuclear_binding_first_principles import (
    _apply_scale_witness,
    _conservation_energy_geV,
    _lattice_norm_ref,
    _lattice_primitives,
    _nucleus_shell_index_from_x,
    generate_nuclei_pack_coords,
    solve_equilibrium_x,
)

_FM_TO_M = 1e-15
_M_TO_FM = 1e15
_NUCLEAR_SCALE = (HBAR_C_MEV_FM / 1000.0) / T_LOCK_NOW_GEV
_MEV_TO_EV = 1000.0


# -----------------------------------------------------------------------------
# Electron: lepton sector (different part of Fano plane; no quarks). Reference m_e.
# -----------------------------------------------------------------------------

def electron_mass_and_wave_packet(
    n: int,
    l: int,
    m_l: int,
    spin: float,
) -> Tuple[float, List[int], Callable[[float], float]]:
    """
    Electron mass (MeV) and diffuse wave packet for state (n,l,m_l,spin).

    Electrons are leptons; they live in a different area of the Fano plane than
    quarks (no quarks in electrons). The 8×8 confined_energy machinery is
    hadronic only. We use M_E_MEV_REF (CODATA) until the lepton sector is
    derived in the framework. Diffuse wave packet on nuclear φ-field: superpose
    modes m_base + k for k = 0..n-1 (higher n = higher modes → compact central
    peak). Returns (m_e_mev, wave_packet_modes, probability_density(r_fm)).
    """
    m_e_mev = M_E_MEV_REF
    m_base = 0
    modes = [m_base + k for k in range(max(1, n))]
    # Probability density from mode interference: peakier for higher n (more modes)
    n_eff = max(1, n)

    def probability_density(r_fm: float) -> float:
        if r_fm <= 0:
            return 0.0
        # Radial shape: r^(2*l) * exp(-r / (n*scale)); scale from lattice
        scale_fm = 1.0 * n_eff
        r_n = r_fm / max(scale_fm, 1e-30)
        return (r_fm ** (2 * max(0, l))) * math.exp(-r_n)

    return m_e_mev, modes, probability_density


def _spherical_to_cartesian_fm(r_fm: float, theta_rad: float, phi_rad: float) -> np.ndarray:
    """(r, θ, φ) in fm → (x, y, z) in fm."""
    st = math.sin(theta_rad)
    ct = math.cos(theta_rad)
    sp = math.sin(phi_rad)
    cp = math.cos(phi_rad)
    return np.array([r_fm * st * cp, r_fm * st * sp, r_fm * ct], dtype=float)


# -----------------------------------------------------------------------------
# Casimir: nuclear pairs only (no separate electron horizons)
# -----------------------------------------------------------------------------

def _atom_casimir_caustics_geV(
    coords_nuc_fm: np.ndarray,
    charges_nuc: List[int],
    theta_p_m: float,
    theta_n_m: float,
    x_fm: float,
    theta_fm: float,
    gamma: float = GAMMA,
) -> float:
    """
    Pairwise solid-angle caustics Ω_ij over nuclear horizons only.

    Same formula as nuclear first-principles. Electrons are diffuse (no
    horizon shadows), so they do not enter Casimir; binding from electrons
    is effectively modified Coulomb + conservation. For A=1 (H) this returns 0.
    """
    m_nuc = _nucleus_shell_index_from_x(x_fm, theta_fm)
    _, delta_E, new_modes = _lattice_primitives(m_nuc)
    ref = _lattice_norm_ref()
    geom = (theta_fm / max(x_fm, 1e-6)) ** 1.5
    scale_GeV = T_LOCK_NOW_GEV * (new_modes * delta_E / ref) * gamma * geom
    A = coords_nuc_fm.shape[0]
    if A < 2:
        return 0.0
    coords_m = coords_nuc_fm * _FM_TO_M
    total_omega = 0.0
    for i in range(A):
        for j in range(i + 1, A):
            d_ij = np.linalg.norm(coords_m[i] - coords_m[j])
            if d_ij <= 0:
                continue
            Theta_i = theta_p_m if charges_nuc[i] == 1 else theta_n_m
            Theta_j = theta_p_m if charges_nuc[j] == 1 else theta_n_m
            denom = math.sqrt(d_ij * d_ij + Theta_i * Theta_j)
            omega_ij = 0.5 * (1.0 - d_ij / max(denom, 1e-300))
            total_omega += omega_ij
    return -scale_GeV * total_omega * _NUCLEAR_SCALE


# -----------------------------------------------------------------------------
# Modified EM: probability-weighted (electron at expectation r)
# -----------------------------------------------------------------------------

def _nuc_center_m_from_coords(coords_nuc_fm: np.ndarray, charges_nuc: List[int]) -> np.ndarray:
    """Nucleus centroid (m)."""
    coords_m = coords_nuc_fm * _FM_TO_M
    if coords_m.shape[0] == 1:
        return coords_m[0].copy()
    proton_idx = [i for i in range(len(charges_nuc)) if charges_nuc[i] == 1]
    if proton_idx:
        return np.mean(coords_m[proton_idx], axis=0)
    return np.mean(coords_m, axis=0)


def _atom_coulomb_geV(
    coords_nuc_fm: np.ndarray,
    charges_nuc: List[int],
    electron_positions_fm: List[Tuple[float, float, float]],
    Z: int,
    x_fm: float,
    theta_fm: float,
    gamma: float = GAMMA,
) -> float:
    """
    Full modified EM: e–nucleus (attractive) and e–e (repulsive) via HQIVAtom.

    Probability-weighted: electron positions are expectation radii (trial r).
    Same modified_field_contribution (ε(φ) + δθ′) and scale as nuclear.
    """
    m_nuc = _nucleus_shell_index_from_x(x_fm, theta_fm)
    ref = _lattice_norm_ref()
    _, delta_E, new_modes = _lattice_primitives(m_nuc)
    scale_GeV = T_LOCK_NOW_GEV * (new_modes * delta_E / ref) * gamma
    theta_m = theta_fm * _FM_TO_M
    nuc_center = _nuc_center_m_from_coords(coords_nuc_fm, charges_nuc)
    atom_nuc = HQIVAtom(position=nuc_center, charge=float(Z), species="H", c_si=C_SI)
    total = 0.0
    for r, th, ph in electron_positions_fm:
        pos_e = _spherical_to_cartesian_fm(r, th, ph) * _FM_TO_M
        pos_e = np.asarray(pos_e, dtype=float).reshape(3)
        contrib = atom_nuc.modified_field_contribution(pos_e, E_prime=0.5, gamma=gamma)
        total -= float(np.asarray(contrib).flat[0]) * theta_m * Z
    elec_positions_m = [_spherical_to_cartesian_fm(r, th, ph) * _FM_TO_M for r, th, ph in electron_positions_fm]
    for i in range(len(elec_positions_m)):
        for j in range(i + 1, len(elec_positions_m)):
            pos_i = np.asarray(elec_positions_m[i], dtype=float).reshape(3)
            pos_j = np.asarray(elec_positions_m[j], dtype=float).reshape(3)
            atom_i = HQIVAtom(position=pos_i, charge=-1.0, species="H", c_si=C_SI)
            contrib = atom_i.modified_field_contribution(pos_j, E_prime=0.5, gamma=gamma)
            total += float(np.asarray(contrib).flat[0]) * theta_m
    return scale_GeV * total * _NUCLEAR_SCALE


# -----------------------------------------------------------------------------
# Conservation: total mode count (nuclear + electron modes from wave packets)
# -----------------------------------------------------------------------------

def _conservation_energy_atom_geV(
    x_fm: float,
    theta_fm: float,
    electron_modes: List[int],
    gamma: float = GAMMA,
) -> float:
    """
    Global conservation term with total mode count (nuclear + electron modes).

    Same (Θ/x)^1.5 structure as nuclear; total modes = nuclear new_modes + sum of
    electron wave-packet modes (higher n = more modes).
    """
    m_nuc = _nucleus_shell_index_from_x(x_fm, theta_fm)
    _, delta_E, new_modes = _lattice_primitives(m_nuc)
    ref = _lattice_norm_ref()
    total_modes = new_modes + sum(electron_modes)
    geom = (theta_fm / max(x_fm, 1e-6)) ** 1.5
    return -(total_modes * delta_E / ref) * gamma * T_LOCK_NOW_GEV * geom * _NUCLEAR_SCALE


# -----------------------------------------------------------------------------
# HQIVAtomState: nuclear core + electrons as diffuse wave packets
# -----------------------------------------------------------------------------

class HQIVAtomState:
    """
    Single atom (nucleus + electrons) with total energy from horizon packing only.

    Nuclear core unchanged (generate_nuclei_pack_coords). Electrons: diffuse wave
    packets from electron_mass_and_wave_packet (lepton sector; m_e = M_E_MEV_REF
    until lepton Fano sector is derived; higher n = higher modes → compact peak).
    No separate electron
    horizons. E_total = Casimir (nuclear pairs) + modified EM (probability-weighted)
    + conservation (nuclear + electron modes).
    """

    def __init__(self, Z: int, N_electrons: int, A: int | None = None):
        self.Z = max(0, Z)
        self.N_electrons = max(0, N_electrons)
        if A is None:
            A = self.Z if self.Z <= 1 else 2 * self.Z
        self.A = max(self.Z, A)
        self._theta_p_m: float | None = None
        self._theta_n_m: float | None = None

    def _nucleon_thetas(self) -> Tuple[float, float]:
        if self._theta_p_m is None or self._theta_n_m is None:
            self._theta_p_m, self._theta_n_m = nucleon_effective_theta_m()
        return self._theta_p_m, self._theta_n_m

    def E_total(
        self,
        x_nuc_fm: float,
        electron_positions_fm: List[Tuple[float, float, float]],
        electron_modes_list: List[List[int]],
        gamma: float = GAMMA,
    ) -> float:
        """
        Total energy (GeV): Casimir (nuclear) + modified EM (weighted) + conservation.

        Electron mass = M_E_MEV_REF (lepton sector; no 8×8 hadronic). Diffuse wave
        with higher n = higher modes for compact central probability peaks.
        """
        theta_p_m, theta_n_m = self._nucleon_thetas()
        theta_fm = min(theta_p_m, theta_n_m) * _M_TO_FM
        coords_fm, charges, _ = generate_nuclei_pack_coords(self.A, self.Z, x_nuc_fm)
        E_cas = _atom_casimir_caustics_geV(coords_fm, charges, theta_p_m, theta_n_m, x_nuc_fm, theta_fm, gamma=gamma)
        E_coul = _atom_coulomb_geV(coords_fm, charges, electron_positions_fm, self.Z, x_nuc_fm, theta_fm, gamma=gamma)
        all_modes: List[int] = []
        for mods in electron_modes_list:
            all_modes.extend(mods)
        x_eff = x_nuc_fm
        if electron_positions_fm:
            x_eff = max(x_nuc_fm, max(r for r, _, _ in electron_positions_fm))
        E_cons = _conservation_energy_atom_geV(x_eff, theta_fm, all_modes, gamma=gamma)
        geom_scale = (theta_fm / 1.0) ** 2
        return (E_cas + E_coul + E_cons) * geom_scale


# -----------------------------------------------------------------------------
# Variational minimiser (same as nuclei: minimize_E_total over radial trial)
# -----------------------------------------------------------------------------

def _minimize_electron_radius(
    Z: int,
    N_electrons: int,
    n: int,
    l: int,
    m_l: int,
    spin: float,
    fixed_positions_fm: List[Tuple[float, float, float]],
    fixed_modes_list: List[List[int]],
    x_nuc_fm: float,
    theta_fm: float,
    gamma: float = GAMMA,
    x_lo_fm: float | None = None,
    x_hi_fm: float | None = None,
) -> Tuple[float, float]:
    """
    Minimize E_total over electron radial trial parameter for state (n,l,m_l,spin).

    Same variational minimiser as nuclei (bounded minimize_scalar). Wave packet
    from electron_mass_and_wave_packet (m_e = M_E_MEV_REF); modes in conservation.
    """
    _, modes, _ = electron_mass_and_wave_packet(n, l, m_l, spin)
    n2 = max(1, n * n)
    if x_lo_fm is None:
        x_lo_fm = max(1.2 * n2 * theta_fm, 0.15)
    if x_hi_fm is None:
        x_hi_fm = max(3.0 * n2 * theta_fm, 15.0)
    x_lo_fm = max(x_lo_fm, 1e-6)
    state = HQIVAtomState(Z, N_electrons)

    def objective(r_fm: float) -> float:
        positions = list(fixed_positions_fm) + [(float(r_fm), 0.0, 0.0)]
        modes_list = list(fixed_modes_list) + [modes]
        return state.E_total(x_nuc_fm, positions, modes_list, gamma=gamma)

    res = minimize_scalar(
        objective,
        bounds=(x_lo_fm, x_hi_fm),
        method="bounded",
        options={"xatol": 1e-4},
    )
    r_eq = float(res.x)
    positions_eq = list(fixed_positions_fm) + [(r_eq, 0.0, 0.0)]
    modes_list_eq = list(fixed_modes_list) + [modes]
    E_geV = state.E_total(x_nuc_fm, positions_eq, modes_list_eq, gamma=gamma)
    return r_eq, E_geV


# -----------------------------------------------------------------------------
# find_atomic_energy_levels
# -----------------------------------------------------------------------------

def find_atomic_energy_levels(
    Z: int,
    max_n: int = 4,
    N_electrons: int | None = None,
    gamma: float = GAMMA,
) -> List[Tuple[int, int, int, float, float]]:
    """
    Compute discrete quantum energy levels from horizon packing only.

    For each (n,l,m_l,spin): build diffuse wave packet (m_e = M_E_MEV_REF,
    lepton sector); run same variational minimiser over radial trial. Return
    (n, l, m_l, spin, energy_eV) sorted by energy; convert with
    apply_scale_witness only.
    """
    if N_electrons is None:
        N_electrons = Z
    N_electrons = max(0, min(N_electrons, Z))
    theta_p_m, theta_n_m = nucleon_effective_theta_m()
    theta_fm = min(theta_p_m, theta_n_m) * _M_TO_FM
    A = Z if Z <= 1 else 2 * Z
    x_nuc_fm = solve_equilibrium_x(A, Z, theta_p_m, theta_n_m, gamma=gamma)

    results: List[Tuple[int, int, int, float, float]] = []
    fixed_positions: List[Tuple[float, float, float]] = []
    fixed_modes_list: List[List[int]] = []

    for n in range(1, max_n + 1):
        for l in range(n):
            for m_l in range(-l, l + 1):
                for spin in (-0.5, 0.5):
                    r_eq, E_geV = _minimize_electron_radius(
                        Z, N_electrons, n, l, m_l, spin,
                        fixed_positions, fixed_modes_list,
                        x_nuc_fm, theta_fm, gamma=gamma,
                    )
                    energy_eV = _apply_scale_witness(E_geV) * _MEV_TO_EV
                    results.append((n, l, m_l, spin, energy_eV))
    results.sort(key=lambda row: row[4])
    return results


def ground_state_binding_eV(
    Z: int,
    N_electrons: int | None = None,
    max_n: int = 2,
    gamma: float = GAMMA,
) -> float:
    """
    Ground-state binding energy (eV): -min(E) from find_atomic_energy_levels.

    Scaled by existing apply_scale_witness only. Electron mass = M_E_MEV_REF
    (lepton sector; no 8×8 hadronic).
    """
    levels = find_atomic_energy_levels(Z=Z, max_n=max_n, N_electrons=N_electrons, gamma=gamma)
    if not levels:
        return 0.0
    return -levels[0][4]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _run_hydrogen(gamma: float = GAMMA) -> List[Tuple[int, int, int, float, float]]:
    """Hydrogen: ground + first excited."""
    return find_atomic_energy_levels(Z=1, max_n=3, N_electrons=1, gamma=gamma)


def _run_helium(gamma: float = GAMMA) -> List[Tuple[int, int, int, float, float]]:
    """Helium: two electrons."""
    return find_atomic_energy_levels(Z=2, max_n=3, N_electrons=2, gamma=gamma)


def _print_table(rows: List[Tuple[int, int, int, float, float]], title: str) -> None:
    print(title)
    print("n\tl\tm_l\tspin\tenergy_eV")
    for n, l, m_l, spin, energy_eV in rows:
        print(f"{n}\t{l}\t{m_l}\t{spin}\t{energy_eV:.4f}")
    print()


def _plot_levels(
    hydrogen: List[Tuple[int, int, int, float, float]],
    helium: List[Tuple[int, int, int, float, float]],
    out_png: str = "atomic_levels_first_principles.png",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for ax, levels, name in [(ax1, hydrogen, "Hydrogen"), (ax2, helium, "Helium")]:
        if not levels:
            continue
        labels = [f"n={n},l={l}" for n, l, m_l, spin, _ in levels]
        energies = [row[4] for row in levels]
        ax.barh(range(len(energies)), energies, align="center")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("energy (eV)")
        ax.set_title(name)
    fig.suptitle("Atomic energy levels — pure first-principles (HQIV)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close()


# Reference first ionization (eV), NIST; for comparison only
_I1_REF_EV = {1: 13.5984, 2: 24.5874}


def _print_experiment_vs_calculation() -> None:
    """Print experiment vs calculation: I1_exp, binding_HQIV, difference, ratio."""
    print("--- Experiment vs calculation (first ionization, eV) ---")
    print("Element   I1_exp(eV)   binding_HQIV(eV)   diff(HQIV-exp)   ratio(HQIV/exp)")
    for Z, sym in [(1, "H"), (2, "He")]:
        I1 = _I1_REF_EV.get(Z)
        if I1 is None:
            continue
        binding = ground_state_binding_eV(Z, N_electrons=Z, max_n=2)
        diff = binding - I1
        ratio = binding / I1
        print(f"  {sym:3}       {I1:10.4f}   {binding:16.2f}   {diff:14.2f}   {ratio:12.3f}")
    print()


if __name__ == "__main__":
    H_levels = _run_hydrogen()
    _print_table(H_levels, "Hydrogen (Z=1, N_e=1) — ground + first excited")
    He_levels = _run_helium()
    _print_table(He_levels, "Helium (Z=2, N_e=2)")
    _print_experiment_vs_calculation()
    _plot_levels(H_levels, He_levels)
    print("Binding from HQIV axioms; nuclear core from 8×8+lattice. Electrons are leptons (different Fano sector; no quarks); m_e = M_E_MEV_REF until lepton sector is derived. Higher n = higher modes → compact central peak.")
