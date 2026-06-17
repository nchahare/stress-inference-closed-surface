# stress-inference-closed-surface

**Curvature, normals, local axes & membrane stress on closed surfaces (vedo + GFDM).**

Estimate, per vertex of a surface mesh: **principal curvatures / radii**, the **vertex
normal**, a **local frame (local axes)**, and the **membrane stress** $(\sigma_1,\sigma_2)$
that balances an internal pressure. Curvature is adapted from the
[spatchcocking](https://github.com/nchahare/spatchcocking) routine via
`vedo.project_point_on_variety`. Membrane stress is solved on the surface with a
**generalized finite difference method (GFDM)** — no FEM framework, no axisymmetry
assumption, `scipy.sparse` only — which extends curved Monolayer Stress Microscopy (cMSM) to
**closed / arbitrary-topology** surfaces.

The **sphere** is the analytic check (`K=1/R²`, `H=1/R`, `σ₁=σ₂=ΔpR/2t`); a **prolate
spheroid** (stretched sphere) breaks the symmetry (`σ_hoop/σ_merid → 1.75` at the equator).

## Data & external references (not committed)

To keep the repo source-only, the following are **gitignored** — obtain them separately:
- **Input meshes** `2025-09-18-16-46-HH17.vtk`, `2025-10-23-13-06-HH20.vtk` — chick neural-tube
  surfaces (place in the project root to run the real-mesh scripts).
- **cMSM reference code** `cMSM_ref/` — Marín-Llauradó et al., Zenodo
  [10.5281/zenodo.7921052](https://doi.org/10.5281/zenodo.7921052) (implementation reference only).
- **Reference paper** — *Mapping mechanical stress in curved epithelia of designed size and
  shape*, Marín-Llauradó et al., Nat. Commun. (2023), `41467_2023_38879`; Supplementary Note 2
  describes cMSM. The supplement PDF is not redistributed here (copyright).
- **Generated outputs** (`out/`) and **compiled PDFs** rebuild from the scripts and `.tex`.

## Running the code

All scripts use the **`fem_env`** conda environment. Set the interpreter once (PowerShell);
everything is headless by default and writes PNG/CSV/NPZ/VTP into `out/`. Add `--show` to open
an interactive vedo window. Run from the project root.

```powershell
$py = "C:\Users\nimes\miniforge3\envs\fem_env\python.exe"
```

### 1. Curvature, normals, local axes

```powershell
# per-vertex curvature, normals, local frame -> out/sphere_curvature_*.csv
& $py sphere_curvature.py                       # flags: --radius --res --depth --degree --show

# mean-curvature colouring + normal arrows; sphere vs stretched -> out/curvature_compare.png
& $py curvature_compare.py                      # flags: --radius --res --depth --stretch --every --show
```

Curvature CSV columns: `X,Y,Z, nx,ny,nz, v1x..v2z, K,H, k1,k2, r1,r2`
(`n`=normal, `v1`/`v2`=tangent local axes).

### 2. GFDM operators (self-test)

```powershell
# validates surface grad / Laplace-Beltrami vs analytic sphere fields (~2nd-order convergence)
& $py surface_fd.py
```

### 3. Membrane stress on the analytic cases (sphere + spheroid)

```powershell
# direct GFDM solve; default IcoSphere subdiv4, depth2 -> out/membrane_stress.png
& $py membrane_stress_fd.py                     # flags: --radius --subdiv --depth --dp --t --lam --stretch --show

# best-accuracy config used in the notes (finer mesh, larger stencil, light smoothing weight)
& $py membrane_stress_fd.py --subdiv 5 --depth 3 --lam 0.02
```

### 4. Removing the structured-mesh "lines"

```powershell
# (a) Laplacian (umbrella) post-smoothing: raw vs smoothed vs mean, 2x3 -> out/stress_smoothing_compare.png
& $py stress_smoothing_compare.py               # flags: --subdiv --depth --dp --t --lam --stretch --iters --alpha --vmin --vmax --show

# (b) Beltrami / Airy stress-function solve (single scalar, no smoothing) -> out/membrane_stress_beltrami.png
& $py membrane_stress_beltrami.py --wn 1.0      # flags: --subdiv --depth --dp --t --wn --stretch --vmin --vmax --show

# (c) cMSM-style regularization vs our Laplacian smoothing, sphere+spheroid -> out/reg_compare.png
& $py reg_compare.py --subdiv 4 --depth 3 --lam 0.02 --lam_t 0.05 --lam_c 0.05
# L-curve-style sweep of the cMSM grad-trace weight on the sphere (prints residual/scatter table):
& $py reg_compare.py --sweep --subdiv 4 --depth 3 --lam_c 0.05
```

### 5. Real meshes (chick neural tube HH17 / HH20)

```powershell
# auto-detects the two .vtk files, decimates HH17 to ~4000 pts, solves + Laplacian-smooths;
# writes per-vertex CSV + NPZ + VTP and out/real_mesh_stress_compare.png
& $py real_mesh_stress.py --dp 20 --t 0.05 --decimate 4000
#   flags: --dp --t --depth --lam --iters --alpha --decimate --outdir --out --show

# inspect the saved smoothed fields interactively (independent cameras per panel)
& $py view_smoothed.py                          # flags: --field --vtps --save
& $py view_smoothed.py --field sigma_max_smooth --save out/HH_smoothed_view.png
```

> Note: Python buffers stdout when not attached to a TTY, so background-run prints appear only
> at exit; run in the foreground (or with `-u`) to see progress live.

### 6. Final-results simulations (method comparison: Local vs cMSM vs FEM)

The "final results" follow a 12-sim matrix: **2 geometries** (sphere, prolate ellipsoid 2:1)
× **2 thickness** (uniform, dorsoventral-varying — *pending*) × **3 methods** — **M1 Local**
(curvature-only), **M2 cMSM** (our GFDM inference + Laplacian smoothing), **M3 FEM**
(**our own** neo-Hookean inflation — *pending*; the cMSM `.mat` data is not reused, see To-do).
All runs use **IcoSphere subdiv-5, depth-3, dp=20 Pa, t=0.05**. M1 = axisymmetric two-curvature membrane theory on revolutions
(`σ_merid=Δp·r_hoop/2t`, `σ_hoop=Δp·r_hoop/t·(1−r_hoop/2r_merid)`), or isotropic
mean-curvature `σ=Δp/2tH` on arbitrary meshes (the only valid local estimate there).

```powershell
# Sphere + ellipsoid, M1 + M2 -> per-vertex CSV/NPZ/VTP in out/final/, metrics vs analytic
& $py final_sims.py                             # flags: --subdiv --depth --dp --t --lam --stretch

# HH17 (decimated to HH20 size) + HH20, M1 (isotropic) + M2 -> out/final/hh*_{local,cmsm}.*
& $py final_real.py                             # flags: --dp --t --depth --lam --iters --alpha

# interactive viewer (you rotate); SHARED colour limits within each group
& $py view_final.py --group analytic --method m2   # sphere+ellipsoid share limits
& $py view_final.py --group real     --method m2   # HH17+HH20 share limits
#   flags: --group {analytic,real} --method {m1,m2} --field sigma_max|sigma_min --save

# box-plot comparisons
& $py box_compare.py                            # sphere+ellipsoid: Local vs cMSM vs analytical
& $py real_box_compare.py                       # HH17 vs HH20 side-by-side (Local & cMSM)
```

## Method summary (membrane stress)

Solve the membrane equilibrium PDE on the surface:
`div_s(N) + Δp·n = 0`, where `N` is the tangential stress-resultant tensor. This single
ambient (Cartesian) vector equation encodes **both** tangential equilibrium and the normal
Laplace law (`σ₁κ₁+σ₂κ₂=Δp/t`), which is what makes it general for arbitrary surfaces.
Discretization:
- **GFDM operators** `G_ξ, G_η`: per-vertex weighted-least-squares quadratic Taylor fit in
  the tangent plane → sparse surface-derivative operators (`surface_fd.py`).
- **Stress DOFs**: 3 per vertex `(p,q,r)` in the local tangent basis (builds in `N·n=0`).
- **Sparse solve** with Tikhonov smoothing: `min ‖L S − b‖² + λ²‖R S‖²` (suppresses the
  closed-surface spurious null modes). Then `σ₁,σ₂ = eig([[p,r],[r,q]]) / t`.

## Status

Last updated: **2026-06-16 (UTC-04:00)**

### Done
- [x] Selected `fem_env` (scikit-fem 12.0.1, Python 3.11); installed `vedo` (git, 2026.6.2.dev7) + `vtk 9.6.2`; `scipy 1.14.1` present — _2026-06-15_
- [x] `sphere_curvature.py` — per-vertex curvature, normals, local axes; validated vs analytic sphere (radii ~1–2% @res40, ~0.5% @res80; normals <1.6° of radial; correct R-scaling & convergence) — _2026-06-15_
- [x] `curvature_compare.py` — mean-curvature colouring + normal arrows; sphere (H≈const) vs stretched ellipsoid (H tip≈2.0 matches analytic) — _2026-06-15_
- [x] `stress_estimation.tex/.pdf` — equations for membrane stress; updated to the actual GFDM method + results — _2026-06-15_
- [x] `surface_fd.py` — GFDM surface-derivative operators (no FEM framework); validated vs analytic grad/Laplace–Beltrami, **~2nd-order convergence** — _2026-06-15_
- [x] `membrane_stress_fd.py` — general GFDM membrane-equilibrium solve for σ₁,σ₂; validated vs analytic sphere & spheroid; **iterative `lsqr` fallback for large meshes** — _2026-06-15_
- [x] Diagnosed/handled artefacts: switched to uniform **IcoSphere** (removes UV poles); **Tikhonov smoothing** kills oscillatory null modes — _2026-06-15_
- [x] `stress_smoothing_compare.py` — Laplacian (umbrella) smoothing of σ fields; 2×3 comparison raw vs smoothed vs mean, shared scale — _2026-06-15_
- [x] `membrane_stress_beltrami.py` — **Beltrami/Airy stress-function solve** (single scalar Φ, no tensor null space); 2nd-derivative GFDM operators added to `surface_fd.py`; **removes the lines structurally (no smoothing)** — _2026-06-15_
- [x] **Real mesh pipeline** — `real_mesh_stress.py` runs GFDM solve + Laplacian smoothing on **HH17** (64001 pts, auto-decimated to ~4000) and **HH20** (3766 pts); saves per-vertex CSV + NPZ + VTP; renders comparison PNG — _2026-06-16_
- [x] `view_smoothed.py` — interactive vedo viewer of smoothed σ fields from saved `.vtp` (independent cameras per panel, `sharecam=False`) — _2026-06-16_
- [x] `manuscript_outline.tex/.pdf` — new manuscript outline (GFDM method, uniform vs non-uniform thickness, hyperelastic FEM comparison, HH17/HH20 results) — _2026-06-16_
- [x] **Studied the reference cMSM code** (Zenodo `7921052`, downloaded to `cMSM_ref/`); documented the method↔ours comparison + a full GFDM derivation in the working notes — _2026-06-16_
- [x] `reg_compare.py` — ported cMSM's first-order (grad-trace + curl) regularization into our GFDM solve and compared it to Laplacian smoothing on sphere + spheroid — _2026-06-16_
- [x] **Final-results matrix started** — `local_stress.py` (M1), `final_sims.py` (sphere+ellipsoid M1+M2), `final_real.py` (HH17/HH20 M1+M2), `view_final.py` (interactive, grouped colour limits), `box_compare.py` + `real_box_compare.py` (box plots); ran Sims 1,2,7,8 + the two real meshes at **dp=20** — _2026-06-16_

### Final-results matrix (2 geom × 2 thickness × 3 methods = 12)

| # | geometry | thickness | method | status |
|---|---|---|---|---|
| 1,2 | sphere | uniform | M1 Local, M2 cMSM | ✅ done |
| 3 | sphere | uniform | M3 FEM | ⏸ pending (FEM) |
| 4,5,6 | sphere | DV-varying | M1, M2, M3 | ⏸ pending (t field / FEM) |
| 7,8 | ellipsoid | uniform | M1 Local, M2 cMSM | ✅ done |
| 9 | ellipsoid | uniform | M3 FEM | ⏸ pending (FEM) |
| 10,11,12 | ellipsoid | DV-varying | M1, M2, M3 | ⏸ pending (t field / FEM) |

Note: for M1/M2 thickness is only a `1/t` divisor (statically-determinate resultant), so the
uniform-vs-DV science lives in the **FEM** (M3), where `t` is constitutive. Also ran the same
M1+M2 on **HH17 (decimated to HH20's 3766 pts) + HH20** for the real-mesh comparison.

### Results (what works)
- Mean stress `(σ₁+σ₂)/2` recovered to **a few %** (median); sphere → 10.0 (target ΔpR/2t=10).
- Spheroid **anisotropy reproduced**: σ_max≈15.7 (≈analytic), σ_min≈9.2; equator ratio → **1.75**.
- Equilibrium residual `‖LS−b‖/‖b‖ ~ 10⁻³` on analytic meshes; **~0.20 on real irregular meshes** (expected).
- **Two fixes for the lines, with a tradeoff:**
  - **Laplacian smoothing** (`stress_smoothing_compare.py`) — cuts sphere σ_max std 0.69→0.15 (~4.5×); on real meshes smoothing cuts std ~2.5×. **Currently the best practical method.**
  - **Beltrami stress function** (`membrane_stress_beltrami.py`) — removes lines structurally; under-predicts spheroid anisotropy ~40% with current operators.
- **cMSM regularization tested and ruled out for closed surfaces** (`reg_compare.py`): porting their first-order grad-trace + curl penalty into our GFDM solve does **not** remove the lines on the closed sphere/spheroid (sphere σ_max std stays 0.69 = our *raw* Tikhonov; spheroid σ_min error 38% vs **8% for Laplacian**). A λ_t/λ_c sweep plateaus the spurious anisotropy at ~10% — the penalty can't crush the icosahedral null mode without flattening the real signal. The Laplacian post-smoothing dominates regardless of which regularizer precedes it (cMSM+Laplacian ≈ Tikhonov+Laplacian). cMSM's penalty needs the **open-dome boundary** (which pins the null modes) that closed surfaces lack. Comparison figure: `out/reg_compare.png`; detail in working notes §6.

  | case / metric | Tikhonov raw | **Tik+Laplacian (ours)** | cMSM reg (no smooth) |
  |---|---|---|---|
  | sphere σ_max std | 0.69 | **0.15** | 0.69 |
  | spheroid σ_min error | 30% | **8%** | 38% |
- **Real meshes (Δp=20, t=0.05 placeholder):**

| mesh | pts | resid | σ_max std: raw → smoothed |
|---|---|---|---|
| HH17 (decimated) | 4001 | 0.20 | 3.2e5 → 1.3e5 |
| HH20 | 3766 | 0.18 | 6.1e5 → 2.1e5 |

- **Final-results runs (dp=20 Pa, t=0.05, IcoSphere subdiv-5):** on the analytic cases M1 (local
  axisym) is essentially exact and M2 (cMSM) is within a few % (`out/final/box_compare.png`):

  | case | M1 σ_max/σ_min (err) | M2 σ_max/σ_min (err) | analytic |
  |---|---|---|---|
  | sphere | 200.1/198.6 (0.1/0.7%) | 208.9/191.9 (4.2/4.0%) | 200/200 |
  | ellipsoid 2:1 | 313.4/184.4 (0.2/0.0%) | 316.7/181.1 (0.6/4.3%) | 314.2/184.4 |

  On the sphere M2 shows the ~4% spurious deviatoric (σ_max biased high, σ_min low); on the
  ellipsoid both recover the 1.75 anisotropy. **HH17→HH20** (cMSM, `out/final/real_box_compare.png`;
  stress is in **Pa** but the magnitude is **uncalibrated** while `t` is a placeholder — read
  patterns, not absolute values): σ_max median 2.6e5→4.8e5 (~1.9×), σ_min −1.4e5→−3.9e5 (~2.8×
  more compressive) — HH20 carries a stronger, more anisotropic field. **M1 (isotropic mean-curvature) is unusable on the real
  meshes**: `1/H` blows up near flat/saddle patches (std ~1e7), so it can't resolve the
  tension/compression split — exactly why the inference solve is needed there.

### To do
- [ ] **Thickness field** — run with measured non-uniform `t(x)` vs uniform placeholder; this is the headline R2 result
- [ ] **Hyperelastic FEM cross-check** (R3 / method M3) — **run our own neo-Hookean FEM** inflations of our geometries (sphere, ellipsoid, neural tube) at **Δp=20 Pa** and compare σ₁,σ₂ to the GFDM inference. The archived cMSM `.mat` fields are **not** reusable as ground truth — their geometries, material parameters, and 400 Pa loading differ from ours; the comparison must use a forward model consistent with our inference inputs.
- [ ] Stage comparison HH17→HH20 DV stress profiles (R4)
- [ ] Sensitivity sweeps: Δp range, decimation level, λ (R5)
- [ ] **Improve Beltrami accuracy** — proper K·∇Φ handling or two-potential (Gauss–Codazzi) representation
- [ ] Cross-check curvature against vedo's built-in `compute_curvature`

## Files
- `sphere_curvature.py` — per-vertex curvature, normals, local axes
- `curvature_compare.py` — mean-curvature + normals; sphere vs stretched
- `surface_fd.py` — GFDM surface-derivative operators (+ self-test)
- `membrane_stress_fd.py` — direct GFDM membrane-stress solve (σ₁, σ₂); auto lsqr for large meshes
- `stress_smoothing_compare.py` — Laplacian smoothing of σ; raw vs smoothed vs mean
- `membrane_stress_beltrami.py` — Beltrami/Airy stress-function solve (single scalar Φ)
- `reg_compare.py` — cMSM-style (grad-trace + curl) regularization vs our Laplacian smoothing
- `real_mesh_stress.py` — pipeline for real VTK meshes; auto-decimates; saves CSV/NPZ/VTP
- `view_smoothed.py` — interactive vedo viewer of smoothed σ from saved .vtp files
- `local_stress.py` — **M1 local stress**: `local_stress_axisym` (two-curvature, revolutions) + `local_stress_isotropic` (mean-curvature, any surface)
- `final_sims.py` — final-results runner, sphere+ellipsoid, M1+M2 (dp=20) → `out/final/sim0*.*`
- `final_real.py` — final-results runner, HH17 (decimated to HH20 size)+HH20, M1+M2 → `out/final/hh*_*.*`
- `view_final.py` — interactive viewer with **grouped shared colour limits** (sphere↔ellipsoid, HH17↔HH20)
- `box_compare.py` — box plot Local vs cMSM vs analytical (sphere+ellipsoid) → `out/final/box_compare.png`
- `real_box_compare.py` — box plot HH17 vs HH20 (Local & cMSM) → `out/final/real_box_compare.png`
- `stress_estimation.tex` / `.pdf` — working-notes equations, method, results (full GFDM derivation, cMSM comparison, real meshes)
- `manuscript_outline.tex` / `.pdf` — manuscript outline, **mechanics-first reframing**: thickness-driven stress *dissipation* + *bending* (transmural gradient); Models A (Local) / B (CMSM) / C (3D neo-Hookean FEM), pHH3 mitotic correlation
- `requirements.txt` — pinned `fem_env` dependencies (Python 3.11)
- `CLAUDE.md` — project context for future sessions
- _(not committed: `out/` generated data, `*.vtk` input meshes, `cMSM_ref/`, the supplement PDF, compiled `*.pdf` — see "Data & external references" above)_
