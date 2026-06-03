# SM Mass Ladder Issues Summary

## 1) Generation degeneracy (muon/tau/electron collapse)
The current Python implementation in `src/pyhqiv/sm_mass_ladder.py` computes the generation shell as
`sm_mass_shell_real(label) = electron_shell_real() + {0,1,2}`.

Problem: the electron “anchor” shell index is enormous:
- `electron_shell_real()` comes from `shell_index_for_temperature(T_CMB_natural)`
- with `T_CMB_natural = 1.9e-32`, the shell index is ~`5.263157894736842e31`

In double-precision floats, adding `+1` or `+2` to a number of that magnitude does not change the value (float resolution), so the geometric factor evaluates at essentially the same `s` for:
- `electron`, `muon`, `tau`
- `up`, `charm`, `top`
- `down`, `strange`, `bottom`

Impact: the ladder currently produces no visible generation splitting in `smMassFromGeometry` outputs.

Fix directions:
- evaluate the shell index and/or geometry factor using higher precision (e.g. `decimal`/`mpmath`) or exact arithmetic where possible
- include generation-dependent corrections beyond the coarse `+0/+1/+2` shell shift (the next required Lean-port likely supplies nontrivial projection factors)

## 2) `T_lockin` exists in Lean, but Python SM ladder isn’t using it
In Lean, `T_lockin` is defined in:
- `HQIV_LEAN/Hqiv/Physics/Baryogenesis.lean`

Specifically:
- `m_lockin := referenceM` (lock-in/reference horizon shell index)
- `T_lockin := T m_lockin`

Python currently defines the ladder differently:
- `src/pyhqiv/sm_mass_ladder.py` anchors the “electron shell” to CMB temperature:
  - `electronTemperatureAnchor := T_CMB_natural`
  - implemented as `T_CMB_NATURAL = 1.9e-32` in Python

Therefore:
- the current SM mass ladder is “set at now” via the electron/temperature anchor (`T_CMB_natural`)
- it does not yet implement a dedicated lock-in/η calibration path like Lean’s baryogenesis `T_lockin` horizon

Impact: if the intended workflow is “lock-in sets the geometric normalization at the observer’s horizon cutoff θ”, the current Python SM mass ladder is not wired that way.

Fix directions:
- add a Python `T_lockin_natural()` mirroring `T m_lockin = 1/(m_lockin+1)` using `reference_m()` from `src/pyhqiv/lightcone.py`
- implement a `now_set_from_lockin_witness()` or re-parameterize existing `now_set_from_temperature_witness()` to allow anchoring by lock-in horizon rather than directly by `T_CMB_natural`
- add tests that confirm `T_lockin` mapping and verify whether replacing the electron anchor with a lock-in anchor changes the SM ladder

## 3) Two different “now” characterizations can be conflated
Lean documents a conceptual distinction in `HQIV_LEAN/Hqiv/Geometry/Now.lean` between:
- the framework-natural “now” condition (`phi = H0`)
- the “paper now” obtained by the CMB temperature ladder (temperature-mapped shell index)

If Python callers assume a single “now slice” without encoding which characterization is being used, confusion can arise about whether a witness should map to:
- a time slice (via dynamics), or
- a lattice/radial shell (via the temperature ladder)

Fix direction:
- explicitly name which “now anchoring method” is being used by each setter/witness:
  - temperature-ladder anchored now
  - lock-in/reference horizon anchored now
  - dynamics/`phi = H0` now

