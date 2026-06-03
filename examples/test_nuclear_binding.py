"""
HQIV Nuclear Binding Test — relaxed sphere packing (0 < x < Θ) + conservation + Casimir + EM.
Solve for separation x in (0, Θ) so the Casimir overlap term is dampened while the conservation
term (lattice dN×δE×γ, independent of overlap) stays intact. Single proton-witness sets scale.
Casimir at nuclear scale uses two nucleon horizons (Θ_p ≠ Θ_n): lens overlap is strongly
affected by the two geometries (e.g. d < |Θ_p−Θ_n| → one sphere inside the other → overlap 1).
See docs/coupled_system_design.md §2 (x couples to Casimir; ∂E/∂x = 0 → x_eq).
Run: python examples/test_nuclear_binding.py
"""

import numpy as np
from scipy.optimize import minimize_scalar
from pyhqiv.lattice import discrete_mode_count, curvature_imprint_delta_E
from pyhqiv.constants import GAMMA, T_PL_GEV, T_LOCK_NOW_GEV, M_TRANS, L_PLANCK_M
from pyhqiv.atom import HQIVAtom
from pyhqiv.constants import HBAR_C_MEV_FM

# Bohr radius (m); 4πε₀ℏ²/(m_e e²) ≈ 5.29e-11
A0_M: float = 5.29177210903e-11
# H ground-state binding (eV); single atomic witness
H_BINDING_EV: float = 13.60569301
# Universal constant for atomic–nuclear scale: exponent = 1/π (inverse π), not fitted
INV_PI: float = 1.0 / np.pi

# ============== GEOMETRY: scale x (0 < x < Θ) sets separation ==============
def coords_from_scale_x(
    x_fm: float, A: int, Z: int, Theta_fm: float
) -> tuple[np.ndarray, list[int]]:
    """Positions (fm) with characteristic separation scale x_fm; 0 < x_fm < Theta_fm.
    A=2: one pair at separation x_fm. A=4: tetrahedron with edge = x_fm. A>4: radial shell scaled by x_fm."""
    if A <= 0:
        return np.zeros((0, 3)), []
    if A == 1:
        return np.zeros((1, 3)), [0] if Z >= 1 else []
    if A == 2:
        coords = np.array([[0.0, 0.0, -0.5 * x_fm], [0.0, 0.0, 0.5 * x_fm]])
        proton_idx = [0] if Z >= 1 else []
        return coords, proton_idx
    if A == 4:
        # Tetrahedron edge = x_fm: vertices at (±r,±r,±r) with r = x_fm / (2*sqrt(2))
        r = x_fm / (2.0 * np.sqrt(2.0))
        coords = np.array(
            [
                [r, r, r],
                [r, -r, -r],
                [-r, r, -r],
                [-r, -r, r],
            ]
        )
        coords -= np.mean(coords, axis=0)
        dists = np.linalg.norm(coords, axis=1)
        order = np.argsort(dists)
        proton_idx = list(order[:Z])
        return coords, proton_idx
    # A > 4: radial shell scaled so mean pair distance ~ x_fm
    n = int(np.ceil(np.sqrt(A)))
    theta = np.linspace(0, np.pi, max(n, 1))
    phi = np.linspace(0, 2 * np.pi, max(n, 1))
    R_pack = x_fm * (0.5 + 0.5 * A ** (1 / 3))
    coords = []
    for i in range(A):
        r = R_pack * (i / max(A, 1)) ** 0.33
        th, ph = theta[i % n], phi[i % n]
        coords.append(
            [
                r * np.sin(th) * np.cos(ph),
                r * np.sin(th) * np.sin(ph),
                r * np.cos(th),
            ]
        )
    coords = np.array(coords)
    coords -= np.mean(coords, axis=0)
    dists = np.linalg.norm(coords, axis=1)
    order = np.argsort(dists)
    proton_idx = list(order[:Z])
    return coords, proton_idx


def generate_nuclei_pack_coords(
    A: int, Z: int, R_pack_fm: float | None = None, Theta_fm: float | None = None
) -> tuple[np.ndarray, list[int]]:
    """Legacy: PESS-style packing with R_pack or Theta-based radius. Prefer coords_from_scale_x(x, A, Z, Theta)."""
    if R_pack_fm is None:
        if Theta_fm is not None and Theta_fm > 0:
            R_pack_fm = (0.5 + 0.5 * A ** (1 / 3)) * Theta_fm
        else:
            R_pack_fm = 1.0 * A ** (1 / 3)
    coords, proton_idx = coords_from_scale_x(R_pack_fm, A, Z, Theta_fm or 1.0)
    return coords, proton_idx


# ============== HORIZON: OVERLAP vs WRAP-AROUND CONCENTRATION ==============
#
# Two geometric pictures:
#
# (1) OVERLAP (current): Volume of the 3D region inside BOTH spheres (intersection of two
#     balls). Solid geometry only — no rays, no shadows. Proxy for "shared space" / mode
#     suppression in the overlap region.
#
# (2) WRAP-AROUND CONCENTRATION (intended model): Allowed modes wrap around the horizon
#     of each nucleon (proton and neutron), like flow around a sphere in an airstream.
#     As they wrap around, their effects concentrate — e.g. in the gap between the two
#     spheres (narrow channel → higher mode density) or in the "shadow" side. That
#     concentration enhances the Casimir-like coupling between the two. Geometric
#     proxy: solid angle one sphere subtends at the other (how much of the "wrap-around"
#     from one horizon is directed at the other) and/or inverse gap (concentration when
#     the gap is small).
#
# Nucleon masses (MeV) for horizon radii: Θ = ℏc / m
M_PROTON_MEV: float = 938.272
M_NEUTRON_MEV: float = 939.565


def effective_nucleon_horizon_m(mass_MeV: float = 938.272) -> float:
    """Θ_local in metres (informational-energy axiom)."""
    hbar_c_mev_m = HBAR_C_MEV_FM * 1e-15
    return hbar_c_mev_m / max(mass_MeV, 1e-30)


def _lens_volume_two_spheres(R1_fm: float, R2_fm: float, d_fm: float) -> float:
    """Volume of the 3D region inside BOTH spheres (intersection of two balls). Solid geometry only.
    R1, R2 = radii (fm), d = center separation (fm). Uses spherical-cap formula. Returns 0 if no overlap."""
    if d_fm >= R1_fm + R2_fm or d_fm < 1e-30:
        return 0.0
    if d_fm <= abs(R1_fm - R2_fm):
        r = min(R1_fm, R2_fm)
        return (4.0 / 3.0) * np.pi * r**3
    R1, R2, d = float(R1_fm), float(R2_fm), float(d_fm)
    x = (d**2 - R2**2 + R1**2) / (2.0 * d)
    h1 = R1 - x
    h2 = R2 - (d - x)
    if h1 <= 0 or h2 <= 0:
        return 0.0
    vol1 = np.pi * h1**2 * (R1 - h1 / 3.0)
    vol2 = np.pi * h2**2 * (R2 - h2 / 3.0)
    v = vol1 + vol2
    return v if np.isfinite(v) and v > 0 else 0.0


def horizon_overlap_fraction(coords_fm: np.ndarray, Theta_fm: float) -> float:
    """Exact geometric lens overlap for every pair; single horizon (equal spheres)."""
    if Theta_fm <= 0 or not np.isfinite(Theta_fm):
        return 0.0
    overlaps = 0.0
    vol_norm = 4.0 / 3.0 * np.pi * Theta_fm**3
    for i in range(len(coords_fm)):
        for j in range(i + 1, len(coords_fm)):
            d = np.linalg.norm(coords_fm[i] - coords_fm[j])
            if d >= 2 * Theta_fm or d < 1e-30:
                continue
            vol = (
                np.pi
                * (2 * Theta_fm - d) ** 2
                * (d**2 + 2 * d * Theta_fm - 3 * Theta_fm**2)
                / (12 * d)
            )
            if np.isfinite(vol) and vol > 0:
                overlaps += vol / vol_norm
    return overlaps


def horizon_overlap_fraction_two_geometries(
    coords_fm: np.ndarray,
    proton_idx: list[int],
    Theta_p_fm: float,
    Theta_n_fm: float,
) -> float:
    """Lens overlap for every pair using two nucleon horizons: Θ_p (proton), Θ_n (neutron).
    Each pair (i,j) uses Theta_i, Theta_j from nucleon type; normalizes by min of the two sphere volumes."""
    if Theta_p_fm <= 0 and Theta_n_fm <= 0:
        return 0.0
    overlaps = 0.0
    n = len(coords_fm)
    for i in range(n):
        for j in range(i + 1, n):
            R1 = Theta_p_fm if i in proton_idx else Theta_n_fm
            R2 = Theta_p_fm if j in proton_idx else Theta_n_fm
            d = np.linalg.norm(coords_fm[i] - coords_fm[j])
            vol = _lens_volume_two_spheres(R1, R2, d)
            if vol <= 0:
                continue
            vol_norm = (4.0 / 3.0) * np.pi * (min(R1, R2) ** 3)
            if vol_norm > 0:
                overlaps += vol / vol_norm
    return overlaps


def _solid_angle_subtended(R_fm: float, d_fm: float) -> float:
    """Solid angle (in units of 4π) that a sphere of radius R subtends at a point at distance d.
    Modes wrapping around the horizon 'see' the partner over this fraction of the sky."""
    if d_fm < 1e-30:
        return 1.0
    if R_fm >= d_fm:
        return 1.0
    # Ω/(4π) = (1 - cos θ)/2, cos θ = sqrt(1 - (R/d)²)
    r_over_d = R_fm / d_fm
    if r_over_d >= 1.0:
        return 1.0
    cos_theta = np.sqrt(1.0 - r_over_d**2)
    return (1.0 - cos_theta) / 2.0


def wrap_around_concentration_two_spheres(
    R1_fm: float, R2_fm: float, d_fm: float,
) -> float:
    """Wrap-around concentration: as modes wrap around each horizon (like flow around a sphere
    in an airstream), how much they concentrate toward the partner. Proxy: mean solid angle
    each sphere subtends at the other's center (fraction of 'sky' the partner occupies)."""
    if d_fm < 1e-30:
        return 0.0
    omega_12 = _solid_angle_subtended(R2_fm, d_fm)
    omega_21 = _solid_angle_subtended(R1_fm, d_fm)
    return (omega_12 + omega_21) / 2.0


def wrap_around_concentration_fraction(
    coords_fm: np.ndarray,
    proton_idx: list[int],
    Theta_p_fm: float,
    Theta_n_fm: float,
) -> float:
    """Total wrap-around concentration over all nucleon pairs (Θ_p, Θ_n per nucleon)."""
    if Theta_p_fm <= 0 and Theta_n_fm <= 0:
        return 0.0
    total = 0.0
    n = len(coords_fm)
    for i in range(n):
        for j in range(i + 1, n):
            R1 = Theta_p_fm if i in proton_idx else Theta_n_fm
            R2 = Theta_p_fm if j in proton_idx else Theta_n_fm
            d = np.linalg.norm(coords_fm[i] - coords_fm[j])
            total += wrap_around_concentration_two_spheres(R1, R2, d)
    return total


# ============== BINDING: conservation term (intact) + Casimir overlap (dampened by x) + EM ==============
def _energy_at_x(
    x_fm: float,
    A: int,
    Z: int,
    Theta_fm: float,
    m_nuc: int,
    Theta_p_fm: float | None = None,
    Theta_n_fm: float | None = None,
    use_wrap_around: bool = True,
) -> tuple[float, float, float, float, float, float, float]:
    """E_total(x), E_cons, E_cas_overlap(x), E_em(x), overlap_single, overlap_two, wrap_conc.
    If use_wrap_around: Casimir term uses wrap-around concentration (modes wrapping around
    horizons, concentrating toward partner). Else: uses overlap volume."""
    coords_fm, proton_idx = coords_from_scale_x(x_fm, A, Z, Theta_fm)
    overlap_single = horizon_overlap_fraction(coords_fm, Theta_fm)
    overlap_single = np.clip(float(overlap_single), 1e-30, 1.0)
    if not np.isfinite(overlap_single):
        overlap_single = 1e-30

    if Theta_p_fm is not None and Theta_n_fm is not None:
        overlap_two = horizon_overlap_fraction_two_geometries(
            coords_fm, proton_idx, Theta_p_fm, Theta_n_fm
        )
        overlap_two = np.clip(float(overlap_two), 1e-30, 1.0)
        if not np.isfinite(overlap_two):
            overlap_two = 1e-30
        wrap_conc = wrap_around_concentration_fraction(
            coords_fm, proton_idx, Theta_p_fm, Theta_n_fm
        )
        wrap_conc = np.clip(float(wrap_conc), 1e-30, 10.0)
        if not np.isfinite(wrap_conc):
            wrap_conc = 1e-30
        overlap_frac = wrap_conc if use_wrap_around else overlap_two
    else:
        overlap_two = overlap_single
        wrap_conc = overlap_single
        overlap_frac = overlap_single

    # Shell from A only (conservation); same m_shell for overlap
    m_shell = max(0, min(m_nuc, M_TRANS))
    T_shell = np.array([T_PL_GEV / max(m_shell + 1, 1)])
    delta_E = curvature_imprint_delta_E(
        np.array([float(m_shell)]), T_shell
    )[0]
    dN = discrete_mode_count(m_shell)

    # Conservation term: intact (no overlap factor)
    E_cons = -dN * delta_E * GAMMA
    # Casimir overlap term: dampened as x increases; uses two geometries (Θ_p ≠ Θ_n) when provided
    E_cas_overlap = -dN * overlap_frac * delta_E * GAMMA
    E_cas = E_cons + E_cas_overlap

    coords_m = coords_fm * 1e-15
    atoms = [
        HQIVAtom(position=coords_m[i], charge=1.0 if i in proton_idx else 0.0)
        for i in range(A)
    ]
    E_em_Pl = 0.0
    E_prime = 0.5
    for i in range(A):
        for j in range(i + 1, A):
            d_m = np.linalg.norm(coords_m[i] - coords_m[j])
            if d_m < 1e-30:
                continue
            x_j = np.array([atoms[j].position])
            contrib = atoms[i].modified_field_contribution(
                x_j, E_prime=E_prime, gamma=GAMMA
            )[0]
            if i in proton_idx and j in proton_idx:
                E_em_Pl += contrib * d_m

    E_tot = E_cas + E_em_Pl
    return E_tot, E_cons, E_cas_overlap, E_em_Pl, overlap_single, overlap_two, wrap_conc


def nuclear_binding_potential_nuclei_pack(
    A: int, Z: int, m_nuc: int | None = None, solve_x: bool = True
) -> tuple[float, float, dict]:
    """Binding potential with optional relaxation: solve for x in (0, Θ) to dampen Casimir overlap.
    Conservation term (lattice dN×δE×γ) stays intact; overlap term shrinks at equilibrium x_eq.
    Returns (E_pot_Pl, R_nuclear_m, info) with info = {x_eq_fm, E_cons, E_cas_overlap, E_em}."""
    if m_nuc is None:
        m_nuc = A
    Theta_m = effective_nucleon_horizon_m()
    Theta_fm = Theta_m * 1e15
    Theta_p_fm = effective_nucleon_horizon_m(M_PROTON_MEV) * 1e15
    Theta_n_fm = effective_nucleon_horizon_m(M_NEUTRON_MEV) * 1e15
    # Keep d > |Θ_p − Θ_n| so we stay in lens regime (no "one sphere inside the other")
    d_lens_min = abs(Theta_p_fm - Theta_n_fm) + 1e-9
    x_min_fm = max(1e-6 * Theta_fm, 1e-6, d_lens_min)
    x_max_fm = Theta_fm * (1.0 - 1e-6)

    if not solve_x or A <= 1:
        x_fm = 0.5 * Theta_fm
        E_tot, E_cons, E_cas_ov, E_em, ov_single, ov_two, wrap_conc = _energy_at_x(
            x_fm, A, Z, Theta_fm, m_nuc, Theta_p_fm, Theta_n_fm, use_wrap_around=True
        )
        pairs = [
            np.linalg.norm(
                coords_from_scale_x(x_fm, A, Z, Theta_fm)[0][i]
                - coords_from_scale_x(x_fm, A, Z, Theta_fm)[0][j]
            )
            for i in range(A)
            for j in range(i + 1, A)
        ]
        R_nuclear_m = (np.mean(pairs) * 1e-15) if pairs else 1e-15
        info = {
            "x_eq_fm": x_fm, "E_cons": E_cons, "E_cas_overlap": E_cas_ov, "E_em": E_em,
            "overlap_single": ov_single, "overlap_two": ov_two, "wrap_conc": wrap_conc,
        }
        return E_tot, R_nuclear_m, info

    def objective(x_fm: float) -> float:
        E_tot, _, _, _, _, _, _ = _energy_at_x(
            float(x_fm), A, Z, Theta_fm, m_nuc, Theta_p_fm, Theta_n_fm, use_wrap_around=True
        )
        return float(E_tot)

    res = minimize_scalar(
        objective,
        bounds=(x_min_fm, x_max_fm),
        method="bounded",
        options={"xatol": 1e-6 * Theta_fm},
    )
    x_eq_fm = float(res.x)
    E_tot, E_cons, E_cas_ov, E_em, ov_single, ov_two, wrap_conc = _energy_at_x(
        x_eq_fm, A, Z, Theta_fm, m_nuc, Theta_p_fm, Theta_n_fm, use_wrap_around=True
    )
    pairs = [
        np.linalg.norm(
            coords_from_scale_x(x_eq_fm, A, Z, Theta_fm)[0][i]
            - coords_from_scale_x(x_eq_fm, A, Z, Theta_fm)[0][j]
        )
        for i in range(A)
        for j in range(i + 1, A)
    ]
    R_nuclear_m = (np.mean(pairs) * 1e-15) if pairs else (x_eq_fm * 1e-15)
    info = {
        "x_eq_fm": x_eq_fm, "E_cons": E_cons, "E_cas_overlap": E_cas_ov, "E_em": E_em,
        "overlap_single": ov_single, "overlap_two": ov_two, "wrap_conc": wrap_conc,
    }
    return E_tot, R_nuclear_m, info


# ============== ATOM WITH ELECTRON: H as witness for scale ==============
def atomic_binding_raw_hydrogen(
    ratio_exponent: float | None = None,
) -> float:
    """H atom: proton at origin, electron at a0. Same φ/Θ machinery → raw binding (dimensionless).
    Uses 2*GAMMA*(a0/Θ_proton)^ratio_exponent. Universal Θ from proton scale; local Θ's from masses.
    ratio_exponent=1 (linear) or 1/π (inverse π): universal constant, not fitted."""
    if ratio_exponent is None:
        ratio_exponent = INV_PI
    Theta_m = effective_nucleon_horizon_m()
    ratio = A0_M / max(Theta_m, 1e-30)
    ratio = ratio ** ratio_exponent
    return 2.0 * GAMMA * ratio


def apply_scale_witness_atomic(
    E_pot_Pl: float,
    atomic_raw: float,
    H_binding_eV: float = H_BINDING_EV,
) -> float:
    """Use H-atom binding (eV) as single witness: scale = H_binding_eV / atomic_raw; return binding in MeV."""
    if atomic_raw <= 0 or not np.isfinite(atomic_raw):
        return 0.0
    scale_ev = H_binding_eV / atomic_raw
    binding_eV = (-E_pot_Pl) * scale_ev
    return binding_eV / 1e6


# ============== COMPOSITE-HORIZON BINDING (coord_factor + QCD witness) ==============
def coord_factor_simple(A: int, use_pairs: bool = True) -> float:
    """Coordination factor for E_bound_Pl = |E_cons| × coord_factor.
    use_pairs=False: A=2 → 1; else (A-1)/2 (per-nucleon average).
    use_pairs=True: A=2 → 1; else number of pairs A*(A-1)/2 (tetrahedral ~6 for ⁴He).
    Full fidelity: use exact 8×8 μ_comp from atom/system (same as proton mass)."""
    if A <= 1:
        return 0.0
    if A == 2:
        return 1.0
    if use_pairs:
        return float(A * (A - 1) // 2)  # tetrahedral coordination ~6 for ⁴He
    return (A * (A - 1) / 2) / A  # (A-1)/2


def binding_pl_from_cons(E_cons_Pl: float, A: int) -> float:
    """Raw binding in Planck-like units from conservation term and coordination.
    E_bound_Pl = |E_cons_Pl| * coord_factor."""
    cf = coord_factor_simple(A)
    return abs(float(E_cons_Pl)) * cf


def apply_scale_witness_qcd(
    E_bound_Pl: float,
    T_lock_GeV: float = T_LOCK_NOW_GEV,
    proton_witness_Pl: float | None = None,
) -> float:
    """Single witness (QCD-scale): binding_MeV = E_bound_Pl * (T_LOCK_NOW_GEV*1e3 / proton_witness_Pl).
    If proton_witness_Pl is None, returns NaN (set from deuteron run)."""
    if proton_witness_Pl is None or proton_witness_Pl <= 0:
        return np.nan
    return E_bound_Pl * (T_lock_GeV * 1e3 / proton_witness_Pl)


# ============== SINGLE WITNESS (proton rest energy) ==============
def apply_scale_witness(
    E_pot_Pl: float,
    proton_MeV: float = 938.272,
    R_nuclear_m: float | None = None,
    L_PLANCK_M: float | None = None,
) -> float:
    """Only place physical scale enters — exactly as in cosmology / Higgs derivations.
    E_pot_Pl: potential energy in combinatorial/Planck-related units (negative = attraction).
    Returns binding B = -E_pot in MeV.
    Optional R_nuclear_m, L_PLANCK_M: paper-derived geometric factor (R/L_P) to convert
    combinatorial output to energy at nuclear scale (no fitted constants). Matching
    experiment (D ~ 2.22 MeV, ⁴He ~ 28.3 MeV) may require a paper-derived exponent
    or normalization in the Casimir/EM combination."""
    E_Planck_MeV = T_PL_GEV * 1e3
    scale = proton_MeV / E_Planck_MeV
    if R_nuclear_m is not None and L_PLANCK_M is not None and R_nuclear_m > 0:
        scale = scale * (R_nuclear_m / L_PLANCK_M)
    return (-E_pot_Pl) * scale


# ============== RUN TESTS ==============
if __name__ == "__main__":
    print(
        "HQIV Nuclear Binding Test — relaxed x (0 < x < Θ), composite-horizon + QCD witness\n"
    )

    # Single witness from deuteron: scale set so D → 2.224 MeV (T_LOCK_NOW_GEV from constants)
    E_pot_D, _, info_D = nuclear_binding_potential_nuclei_pack(2, 1, solve_x=True)
    E_bound_Pl_D = binding_pl_from_cons(info_D["E_cons"], 2)
    proton_witness_Pl = E_bound_Pl_D * (T_LOCK_NOW_GEV * 1e3) / 2.224
    print(f"   T_LOCK_NOW_GEV   = {T_LOCK_NOW_GEV}  (QCD scale, constants.py)")
    print(f"   Deuteron E_cons  → E_bound_Pl = {E_bound_Pl_D:.4g}  → witness scale set for 2.224 MeV")
    print()

    atomic_raw_1 = atomic_binding_raw_hydrogen(ratio_exponent=1.0)
    atomic_raw_inv_pi = atomic_binding_raw_hydrogen(ratio_exponent=INV_PI)
    print(f"   H atom witness = {H_BINDING_EV:.4f} eV  (optional: (a0/Θ)^(1/π) comparison)")
    print(f"   atomic_raw (a0/Θ)^1     = {atomic_raw_1:.4g}  (linear)")
    print(f"   atomic_raw (a0/Θ)^(1/π) = {atomic_raw_inv_pi:.4g}  (inverse π)\n")

    for A, Z, name, target_MeV in [
        (2, 1, "Deuteron", 2.224),
        (4, 2, "⁴He (alpha)", 28.3),
    ]:
        E_pot_Pl, R_nuclear_m, info = nuclear_binding_potential_nuclei_pack(
            A, Z, solve_x=True
        )
        # Composite-horizon: E_bound_Pl = |E_cons| * coord_factor; single QCD witness
        E_bound_Pl = binding_pl_from_cons(info["E_cons"], A)
        binding_MeV_qcd = apply_scale_witness_qcd(
            E_bound_Pl, proton_witness_Pl=proton_witness_Pl
        )
        cf = coord_factor_simple(A)
        binding_MeV = apply_scale_witness(E_pot_Pl)
        binding_MeV_atom_1 = apply_scale_witness_atomic(E_pot_Pl, atomic_raw_1)
        binding_MeV_atom_inv_pi = apply_scale_witness_atomic(E_pot_Pl, atomic_raw_inv_pi)
        binding_MeV_geom = apply_scale_witness(
            E_pot_Pl, R_nuclear_m=R_nuclear_m, L_PLANCK_M=L_PLANCK_M
        )
        x_eq = info["x_eq_fm"]
        Theta_fm = effective_nucleon_horizon_m() * 1e15

        ov_s = info.get("overlap_single", 0.0)
        ov_t = info.get("overlap_two", 0.0)
        wrap_c = info.get("wrap_conc", 0.0)
        ratio_ov = (ov_t / ov_s) if ov_s > 0 else 1.0
        Theta_p_fm = effective_nucleon_horizon_m(M_PROTON_MEV) * 1e15
        Theta_n_fm = effective_nucleon_horizon_m(M_NEUTRON_MEV) * 1e15

        print(f"{name:12} (A={A}, Z={Z}):")
        print(f"   coord_factor     = {cf:.4f}  (1 if A=2 else A×(A-1)/2 pairs)")
        print(f"   E_cons           = {info['E_cons']:.4g}  → E_bound_Pl = {E_bound_Pl:.4g}")
        print(f"   Binding (QCD)    = {binding_MeV_qcd:.4g} MeV  ← E_bound_Pl × (T_LOCK_NOW×1e3 / proton_witness_Pl)")
        print(f"   x_eq / Θ         = {x_eq:.4f} / {Theta_fm:.4f} fm  (0 < x < Θ)")
        print(f"   Θ_p, Θ_n         = {Theta_p_fm:.4f}, {Theta_n_fm:.4f} fm  (two nucleon horizons)")
        print(f"   overlap (vol)    = {ov_t:.4g}  (volume inside both spheres)")
        print(f"   wrap_conc        = {wrap_c:.4g}  (modes wrap around horizons → concentrate toward partner)")
        if ratio_ov > 1e6:
            print(f"   Casimir uses     = wrap-around concentration  (not overlap volume)")
        else:
            print(f"   overlap ratio    = {ratio_ov:.4f}×  (two vs single Θ)")
        print(f"   E_cas_overlap    = {info['E_cas_overlap']:.4g}  (from wrap-around)")
        print(f"   E_em             = {info['E_em']:.4g}")
        print(f"   E_pot (raw)      = {E_pot_Pl:.4g}  (R_nuclear = {R_nuclear_m:.2e} m)")
        print(f"   Binding (proton) = {binding_MeV:.4g} MeV")
        print(f"   Binding (H, a0/Θ^1)   = {binding_MeV_atom_1:.4g} MeV")
        print(f"   Binding (H, (a0/Θ)^(1/π)) = {binding_MeV_atom_inv_pi:.4g} MeV")
        print(f"   Binding (R/L_P)  = {binding_MeV_geom:.4g} MeV")
        print(f"   Target           = {target_MeV:.3f} MeV")
        if target_MeV > 0 and np.isfinite(binding_MeV_qcd):
            print(f"   Ratio (QCD)      = {binding_MeV_qcd / target_MeV:.4g}")
        print()

    print("✓ Composite-horizon: E_bound_Pl = |E_cons| × coord_factor; coord_factor = 1 (A=2) or A×(A-1)/2 pairs (A>2).")
    print("✓ Single witness: binding_MeV = E_bound_Pl × (T_LOCK_NOW_GEV×1e3 / proton_witness_Pl); witness set from D → 2.224 MeV.")
    # Summary: D and ⁴He with composite-horizon + QCD witness
    results = []
    for A, Z, name, target_MeV in [(2, 1, "D", 2.224), (4, 2, "⁴He", 28.3)]:
        _, _, info = nuclear_binding_potential_nuclei_pack(A, Z, solve_x=True)
        E_bound_Pl = binding_pl_from_cons(info["E_cons"], A)
        B_qcd = apply_scale_witness_qcd(E_bound_Pl, proton_witness_Pl=proton_witness_Pl)
        results.append((name, B_qcd, target_MeV))
    print("\n--- Binding summary (D and ⁴He): composite-horizon + QCD witness (no (a0/Θ)^(1/π)) ---")
    for name, B_qcd, target in results:
        ratio = B_qcd / target if target > 0 and np.isfinite(B_qcd) else np.nan
        print(f"   {name}: QCD witness = {B_qcd:.3f} MeV, target = {target:.2f} MeV, ratio = {ratio:.3f}")
