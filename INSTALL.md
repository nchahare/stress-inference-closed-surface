# Installation & getting started

Curvature, normals, local frames and **membrane stress** on closed surface meshes —
both the **inverse** solvers (GFDM and stress-based FEM) and the **forward** neo-Hookean
membrane inflation (method M3). Pure `numpy` / `scipy` / `vedo`; no FEM framework required.

## 1. Requirements

- **Python 3.11**
- **git** (needed to install `vedo` from source)
- An OpenGL-capable display **only** for the interactive `--show` windows. Everything runs
  headless otherwise, writing PNG / CSV / NPZ / VTP files into `out/`.

## 2. Install

### Option A — conda / miniforge (recommended)

```bash
conda env create -f environment.yml      # or: mamba env create -f environment.yml
conda activate fem_env
```

This creates an env named `fem_env` (Python 3.11) and pip-installs everything from
`requirements.txt`.

### Option B — pip + virtualenv

```bash
python3.11 -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

> **vedo from git** — the curvature pipeline uses `vedo.project_point_on_variety`, which is
> in the vedo *git* build (PyPI `vedo >= 2026.6.1` also exposes it). `pip` will build it from
> source, so `git` must be on your PATH. The forward/FEM solvers themselves use only stable
> vedo features.

## 3. Verify the install

These need **no external data** (they build their own analytic meshes):

```bash
python surface_fd.py                       # GFDM operator self-test (prints ~2nd-order convergence)
python forward_neohookean.py --shape sphere   # forward inflation; expect "stretch ... err 0.00%"
python membrane_stress_fem.py --no-gfdm       # stress-based FEM on sphere + spheroid
```

A successful `forward_neohookean.py --shape sphere` prints something like:

```
stretch lambda : solver 1.001677   analytic 1.001679   err 0.00%
isotropy |s1-s2|/s1 (mean) : 0.37%
tension err vs Laplace     : 0.10%
```

Add `--show` to any of these to open an interactive 3-D window (needs a display).

## 4. Data files (not in the repository)

These are **gitignored** — obtain them separately and place them in the repo root. The
analytic shapes (sphere / prolate / oblate / capsule) need none of them.

| file / folder | needed for | source |
|---|---|---|
| `2025-09-18-16-46-HH17.vtk`, `2025-10-23-13-06-HH20.vtk` | real-mesh runs (`forward_neohookean.py --mesh`, `real_mesh_stress.py`, `forward_compare_hh.py`) | chick neural-tube surfaces (not redistributed) |
| `cMSM_ref/` | reference only (not used at runtime) | Zenodo [10.5281/zenodo.7921052](https://doi.org/10.5281/zenodo.7921052) |
| `41467_2023_38879_MOESM1_ESM.pdf` | `cmsm_sphere_compare.py` (crops cMSM Fig. 15) | Marín-Llauradó et al., Nat. Commun. (2023) supplement |

## 5. What to run

Everything is headless by default and writes to `out/`; add `--show` for an interactive
window. A few entry points (see **README.md** for the full list):

```bash
# forward neo-Hookean inflation (M3): analytic shapes with built-in validation
python forward_neohookean.py --shape prolate --show --field sigma_max
python forward_neohookean.py --shape capsule --show

# forward inflation on an arbitrary closed mesh (needs the .vtk data above)
python forward_neohookean.py --mesh 2025-10-23-13-06-HH20.vtk --field trace --show

# inverse stress-based FEM (the primary inverse engine)
python membrane_stress_fem.py --show                # colour by trace + principal-stress crosses

# HH17 vs HH20 forward comparison; saves .vtp so you can re-view without re-solving
python forward_compare_hh.py            # solve + save out/forward_{HH17,HH20}.vtp
python forward_compare_hh.py --load --show --field sigma_min   # instant re-view
```

## 6. Notes / troubleshooting

- **Headless servers:** omit `--show`; the file outputs (PNG/VTP) are produced offscreen.
  `.vtp` files open in [ParaView](https://www.paraview.org/).
- **Buffered output:** Python buffers stdout when not attached to a TTY, so prints from a
  backgrounded run appear only at exit — run in the foreground (or `python -u`) to watch progress.
- **Windows / PowerShell:** after `conda activate fem_env`, just use `python`. (The project was
  developed under miniforge; some README snippets show the full interpreter path for one machine —
  `python` works once the env is active.)
- **`pip` can't build vedo:** ensure `git` is installed, or instead `pip install "vedo>=2026.6.1"`
  from PyPI (also exposes `project_point_on_variety`).
