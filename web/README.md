# HQIV Web Calculator (Pyodide / WASM)

This directory contains the static site that runs the **exact same** `pyhqiv` package inside the browser using Pyodide.

## How it gets built

- A GitHub Action (`.github/workflows/web.yml`) runs **only on pushes to `main`** (never on PRs or other branches).
- It builds the Python wheel (`python -m build`), stages the `web/` assets, injects the freshly-built wheel into `wheels/`, and deploys everything to GitHub Pages via `actions/deploy-pages`.
- Result: https://disregardfiat.github.io/pyhqiv/ (or your Pages URL) always reflects the code on main.

## Local development / testing the site

1. Build the wheel locally:
   ```bash
   python -m build
   ```

2. Copy the wheel where the site expects it (and write a manifest so the page finds the right one):
   ```bash
   mkdir -p web/wheels
   cp dist/pyhqiv-*-py3-none-any.whl web/wheels/
   cd web/wheels && ls pyhqiv-*-py3-none-any.whl | head -n -1 | xargs -r rm -f -- && \
     python3 -c '
   import glob, json, os
   os.chdir(".")
   w = sorted(glob.glob("pyhqiv-*-py3-none-any.whl"))[-1]
   open("manifest.json","w").write(json.dumps({"wheel": w}))
   print("local manifest ->", w)
   '
   ```

3. Serve the `web/` directory (any static server):
   ```bash
   # Python (recommended; its dir listing is auto-parsed by the JS as fallback)
   python3 -m http.server -d web 8080

   # or npx
   npx serve web
   ```

4. Open http://localhost:8080 . It will auto-detect the wheel (via manifest or dir listing).

The JS will also try `wheels/manifest.json` (written by the Action + above) or a `data-wheel` attribute for production deploys. See `main.js:resolveWheelUrl`.

## What the calculator demonstrates

- Pure geometry / Lean witnesses (identical numbers)
- Nucleus binding + mass via the isotope ladder network (same as `isotope_ladder`)
- z-score / σ comparison against the same AME2020 references used in `tests/test_binding_energy_vs_pdg.py`
- Thermo free energy etc. via `thermo.compute_free_energy`
- A full Python REPL so you can call *any* public pyhqiv API exactly as in a normal install

## Adding more calculator surfaces

Edit `web/main.js` (the `compute*` async functions + UI wiring in `initUI`). Keep the Python snippets minimal and close to real usage so that "same inputs, same outputs".

Charts currently use Chart.js (CDN). If you want matplotlib output you can load `matplotlib` via `pyodide.loadPackage("matplotlib")` and render to a canvas or base64 PNG.

## Notes / limitations

- Scipy is loaded because a few top-level imports (`spherical_harmonics`, some nuclear paths) pull it in. The core binding/thermo/geometry paths used here are light.
- Pint is pulled automatically as a declared dep of the wheel.
- No heavy optionals (jax, qutip, pyvista, healpy...) are loaded.

## Updating Pyodide version

Bump `PYODIDE_VERSION` in `web/main.js` and test. Newer versions have better scipy/numpy coverage.

## GitHub Pages setup (one time)

In the repo:
- Settings → Pages → Source: **GitHub Actions**
- The workflow already has the correct permissions and jobs.

After the first successful run on main you should see the site live.
