# Constants audit: paper-derived only + unit conversions (2026 arena rules)

**Rule:** Absolutely zero constants in `src/pyhqiv/*.py` (other than geometry necessities: π, √3 for the unit-cube half-diagonal, 2π for phase, natural-unit 1.0). All scale witnesses and local conditions (proton mass anchor, CMB T0 + unc, earth surface g, vacuum ε₀/μ₀, ...) live in:
- `src/pyhqiv/witnesses.json` (Lean-derived, via overlay when available)
- `src/pyhqiv/local_conditions.json` (test/applied "local" values with sources)
- `tests/setup_defaults.py` + `tests/data/*_with_errors.py` (explicit (central, ±err, "Source from PDG/Planck/CODATA/...") for every experimental comparison)

Measurement tables and bare literals are forbidden in src (enforced by `tests/test_src_no_measurement_references.py` + CI). New arena contributions (functions or features) must obey this + ship tests with error bars.

## Single source: `src/pyhqiv/constants.py`

### Paper-derived
- `GAMMA`, `ALPHA` — entanglement / G_eff exponent
- `T_PL_GEV`, `T_LOCK_GEV` — temperature scales
- `T_CMB_K` — CMB today
- `M_TRANS` — discrete-to-continuous shell
- `COMBINATORIAL_INVARIANT` — 6^7 √3
- `OMEGA_TRUE_K_PAPER`, `LAPSE_COMPRESSION_PAPER`, `AGE_WALL_GYR_PAPER`, `AGE_APPARENT_GYR_PAPER`
- `H0_KM_S_MPC_PAPER` — H0 is the radial gradient (the same as time); from paper apparent age, 1/(age) → km/s/Mpc. Not an independent constant.

### Unit conversions only
- `C_SI`, `E_PL_SI`, `HBAR_SI`, `K_B_SI` — SI
- `SEC_PER_GYR`, `MPC_M` — time/distance
- `K_B_GEV_PER_K`, `GEV_TO_K`, `T_PL_K`, `T_CMB_MUK` — temperature
- `Z_RECOMB` — standard recombination redshift reference
- `HBAR_C_EV_ANG`, `A_LOC_ANG` — molecular/Å

### Fiducial (not from paper)
- `OMEGA_M0_FIDUCIAL`, `OMEGA_L0_FIDUCIAL` — used only for optional `comoving_distance(z)` geometry
- `K_PIVOT_FIDUCIAL` — for primordial P(k) pivot; phenomenological sigma8 may use `N_S`, `K_PIVOT` in cosmology_full (documented there)

## Files updated to use constants
- `perturbations.py` — T_CMB_K, T_PL_K, K_B_GEV_PER_K, Z_RECOMB, OMEGA_TRUE_K_PAPER, LAPSE_COMPRESSION_PAPER, H0_KM_S_MPC_PAPER, COMBINATORIAL_INVARIANT
- `cosmology/background.py` — H0_KM_S_MPC_PAPER, OMEGA_M0_FIDUCIAL, OMEGA_L0_FIDUCIAL, Z_RECOMB
- `cosmology/hqiv_cmb.py`, `cosmology/cmb_map.py` — T_CMB_K, T_CMB_MUK, Z_RECOMB, OMEGA_TRUE_K_PAPER
- `cosmology_full.py` — T_CMB_K, T_CMB_MUK, T_PL_K, H0_KM_S_MPC_PAPER, LAPSE_COMPRESSION_PAPER, OMEGA_M0_FIDUCIAL, OMEGA_L0_FIDUCIAL, COMBINATORIAL_INVARIANT
- `lattice.py` — K_PIVOT_FIDUCIAL
- `bulk_seed.py` — T_CMB_K, LAPSE_COMPRESSION_PAPER
- `protocols.py` — T_CMB_K, AGE_WALL_GYR_PAPER, AGE_APPARENT_GYR_PAPER, LAPSE_COMPRESSION_PAPER
- `solar_core.py` — LAPSE_COMPRESSION_PAPER
- `redshift.py` — AGE_WALL_GYR_PAPER, AGE_APPARENT_GYR_PAPER, LAPSE_COMPRESSION_PAPER, OMEGA_TRUE_K_PAPER, GAMMA

## Remaining literals (allowed or documented)
- **Numerical scaling in transfer/growth:** e.g. `COMBINATORIAL_INVARIANT/4.85`, `COMBINATORIAL_INVARIANT/1.52`, clip bounds `0.05`, `0.4` — dimensionless ratios from lattice output; can be refined to full lattice-derived expressions in a future pass.
- **cosmology_full:** `N_S`, `K_PIVOT`, `sigma8_z0_ref` — fiducial for phenomenological sigma8; not from paper, documented in code.
- **Docstrings** that mention “0.0098” or “3.96” — documentation only; values come from constants at runtime.
