# CLAUDE.md

Project context for Claude Code. See `README.md` for status/task tracking.

## Goal
Compute per-vertex **radius of curvature**, **vertex normals**, a **local frame
(local axes)**, and the **membrane stress** `(σ₁, σ₂)` on surface meshes. Curvature adapts
the [spatchcocking](https://github.com/nchahare/spatchcocking) method (Nature Comms
`41467_2023_38879`, supplementary PDF in repo root). Stress solves the membrane
equilibrium PDE on the surface by **generalized finite differences (GFDM)** — must stay
**general for arbitrary surfaces** (user requirement), no FEM framework, `scipy.sparse`
only. Sphere + prolate spheroid are the analytic validation cases before real meshes.

## User decisions / preferences (carry forward)
- Stress output: **only `σ₁` and `σ₂`** — do NOT compute world-frame stress or von Mises.
- Solve approach: **finite differences on the surface (GFDM)**, NOT scikit-fem, NOT
  FEniCS/Abaqus. (We explored scikit-fem 1D LSFEM and scipy.solve_bvp but the user wants a
  general surface FDM.) scikit-fem is still installed/used elsewhere if needed.
- Must generalize to **arbitrary (non-axisymmetric) surfaces**, not just bodies of revolution.
- **Always use `dp = 20` Pa** for stress runs (standing default; stress scales linearly as dp/t).
- **Final-results matrix:** 2 geometries (sphere, prolate ellipsoid 2:1) × 2 thickness
  (uniform, dorsoventral-varying) × 3 methods — **M1 Local** (curvature-only), **M2 cMSM**
  (our GFDM + Laplacian), **M3 FEM** (our OWN neo-Hookean forward FE — NOT the cMSM archive)
  = 12 sims. Config: IcoSphere subdiv-5, depth-3, dp=20 Pa, t=0.05. **M1 defined (user choice) as axisymmetric two-curvature**
  on revolutions (isotropic mean-curvature σ=Δp/2tH on arbitrary meshes, since the axisym
  split is undefined there). **Done:** Sims 1,2,7,8 + HH17/HH20 (M1+M2). **Pending:** M3 FEM
  (Sims 3,6,9,12) and the DV-thickness field (Sims 4–6,10–12). KEY: for M1/M2 thickness is
  just a 1/t divisor (statically determinate), so uniform-vs-DV science lives in the FEM.
  Results: M1≈exact on revolutions; M2 within a few % (sphere ~4% spurious deviatoric);
  HH17→HH20 cMSM σ_max 2.6e5→4.8e5, σ_min −1.4e5→−3.9e5 (HH20 stronger/more anisotropic);
  M1 isotropic unusable on real meshes (1/H blow-up). Plots: `out/final/box_compare.png`,
  `out/final/real_box_compare.png`.

## Environment — IMPORTANT
- **Always use the `fem_env` conda environment for all Python.** It has scikit-fem 12.0.1
  (Python 3.11, numpy 2.2.6) plus the vedo stack we installed.
- Interpreter: `C:\Users\nimes\miniforge3\envs\fem_env\python.exe`
- `conda` is **not on PATH**; invoke the interpreter by full path. miniforge lives at
  `C:\Users\nimes\miniforge3`.
- Installed for this project: `vedo` 2026.6.2.dev7 (git build from
  `git+https://github.com/marcomusy/vedo.git`), `vtk` 9.6.2; `scipy` already present.
- Shell is **PowerShell** (Windows 11). Run scripts as:
  `& "C:\Users\nimes\miniforge3\envs\fem_env\python.exe" sphere_curvature.py`
- Other envs exist (`ni-vedo-env` has vedo, `solidspy_env`, etc.) but this project
  standardises on `fem_env`.

## Key technical fact
The whole curvature calc rests on one vedo call:
```python
proj, poly, grid = vedo.project_point_on_variety(pt, neighbour_pts, degree=2, normal=vertex_normal)
# poly = (coeffs, R, centroid, gauss_curv K, mean_curv H)
```
- It **always returns a 3-tuple** `(proj, poly, grid)` (grid is None unless `return_grid=True`).
- `R` is a 3x3 matrix = the **local frame**: row0 = tangent `v1`, row1 = tangent `v2`,
  row2 = `normal`. This is the requested "local axes of the calculation".
- Pass the vertex normal so the mean-curvature sign is consistent.
- Principal curvatures: `disc = max(H**2 - K, 0); k1 = H + sqrt(disc); k2 = H - sqrt(disc)`;
  principal radii `r1 = 1/k1`, `r2 = 1/k2`.

## Mesh / neighbourhood pattern (vedo)
```python
mesh = vedo.Sphere(r=R, res=40)
mesh.compute_normals()
adlist = mesh.compute_adjacency()
neigh = mesh.find_adjacent_vertices(i, depth=depth, adjacency_list=adlist)  # k-ring
pts = mesh.coordinates ; normals = mesh.vertex_normals
```
`depth` controls neighbourhood size for the fit (default 3). Need >= 6 points for degree-2.

## Membrane stress — method & key facts
- PDE solved: `div_s(N) + Δp·n = 0` for the tangential stress tensor `N`. Worked in
  **ambient (global Cartesian) components** so frame-rotation/connection terms are
  implicit → works on arbitrary surfaces. The normal component of this equation IS the
  Laplace law `σ₁κ₁+σ₂κ₂=Δp/t`; tangential components are in-plane equilibrium.
- **GFDM operators** (`surface_fd.py`): per-vertex WLS quadratic Taylor fit in the tangent
  plane gives sparse `G_ξ, G_η` with `(G_ξ f)_i ≈ ∂f/∂ξ`. Validated → ~2nd-order convergence.
- **DOFs**: 3 per vertex `(p,q,r)` for `N = p t1⊗t1 + q t2⊗t2 + r(t1⊗t2+t2⊗t1)` (builds in `N·n=0`).
- **Solve** (`membrane_stress_fd.py`): assemble sparse `L` (3n×3n), Tikhonov-smoothed
  least squares `(LᵀL + λ²RᵀR) S = Lᵀb`, `b = -Δp·n`. `σ₁,σ₂ = eig([[p,r],[r,q]])/t`.
- **Principal curvature frame variant** (`membrane_stress_fd_v2.py`): same solve, but uses
  `e1, e2` (principal curvature directions from `compute_curvature_frame`) as the tangent
  frame instead of the arbitrary fit frame `v1, v2`. DOFs then represent
  `N = p e1⊗e1 + q e2⊗e2 + r(e1⊗e2+e2⊗e1)`. Result: `σ₁,σ₂` are **identical** to v1
  (eigenvalues are frame-independent). New outputs: **`d1`, `d2`** (principal stress directions
  in world R³) via `θ_s = ½ arctan2(2r, p−q)` → `d1 = cos θ_s · e1 + sin θ_s · e2`, and
  the **`r` shear diagnostic** (|r|/(|p|+|q|) ≈ 1–2% on axisymmetric surfaces → ~0 when stress
  and curvature axes coincide; non-trivial on general meshes).
- **Principal curvature dirs ≠ principal stress dirs in general.** They coincide only when
  geometry and loading share the same symmetry (axisymmetric surface + uniform pressure). On
  general meshes `r≠0` in the curvature frame — the stress principal axes tilt away from the
  curvature axes to satisfy global in-plane equilibrium. The shear `r` quantifies this tilt.
- **Also returns `d1`, `d2`**: principal stress direction unit vectors in world R³; arrows from
  these on the 3-panel vedo plot (sphere + spheroid + capsule) make the stress field visually
  interpretable.
- **Closed-surface null modes**: `L` has spurious near-null ("hourglass") modes that show up
  as streaky "lines" along the icosphere triangulation. Tikhonov smoothing (`λ≈0.01–0.05`)
  suppresses oscillatory ones; use **IcoSphere** (not UV `Sphere`) to avoid pole artifacts.
  Spurious deviatoric ~5–7%, converges slowly to ~4% with subdiv5+depth3.
- **Best fix for the lines = Laplacian (umbrella) post-smoothing** (`stress_smoothing_compare.py`):
  `σ ← (1-α)σ + α·mean_1ring(σ)`, ~12 iters, α=0.5. Sphere σ_max std 0.69→0.15 (~4.5×);
  spheroid streaks gone, real hoop band preserved. Display/report the smoothed field; keep
  raw σ in data.
- **IMPORTANT correction:** the artifact is *whole-field*, NOT just deviatoric. The MEAN
  stress (σ₁+σ₂)/2 also carries the lines (sphere mean-stress std ≈6.8%). Do NOT claim "mean
  stress is clean" — it isn't. Mean is accurate only in the *median*; smoothing is the cure.
- A genus-0 surface has NO smooth divergence-free traceless tensor field, so on the sphere
  σ₁=σ₂ exactly — any anisotropy there is pure discretization error (useful sanity check).
- **Beltrami stress function (IMPLEMENTED, `membrane_stress_beltrami.py`)**: unknown is a
  single scalar Φ → no tensor null space → removes the lines STRUCTURALLY (no smoothing;
  sphere σ_max std 0.69→0.16). Key curved-surface facts learned:
    - Airy is NOT div-free on curved surfaces (`div_s N(Φ) ∝ K·∇Φ`), and it can only carry
      self-equilibrated stress — NOT the pressure on a closed surface. So split
      `N = N_p + N_Airy(Φ)`: particular `N_p = (Δp/tr B)·g` (isotropic, exact on sphere) +
      Airy `N_Airy = cof(Hess_s Φ)` (local: `N11=Φ_ηη, N22=Φ_ξξ, N12=-Φ_ξη`, from the WLS
      Hessian / `build_derivative_operators` in `surface_fd.py`).
    - `B` (2nd fundamental form, local frame) from Weingarten `+grad_s n` (SIGN: use `+`, so
      tr B = 2H > 0 and σ is tensile under Δp>0; the `-` gives flipped/negative σ — a bug we hit).
    - Solve LS for Φ: tangential-equilibrium residual + `wn·(N_Airy:B)` (normal-preservation)
      + tiny `eps·Φ` (Φ has a constant null space). `wn≈1` is the sweet spot.
  - **Tradeoff (honest):** removes lines but the 1-scalar ansatz + 3rd-order GFDM doesn't
    satisfy tangential equilibrium well on the spheroid (resid ~0.9) → UNDER-predicts anisotropy
    (σ_min ~40% off; σ_max ~9%). So **Laplacian-smoothed direct solve is still the best
    *accurate* method**; Beltrami is the principled line-removal that needs accuracy refinement
    (better high-order operators / handle K∇Φ / two-potential Gauss–Codazzi).
- Analytic validation refs (axisymmetric, in `analytic_axisym`): sphere σ=ΔpR/2t; spheroid
  `N_φ=Δp·r2/2`, `N_θ=Δp·r2(1−r2/2r1)`, with `r1=J³/(ab)`, `r2=bJ/a`, `J=√(a²sin²β+b²cos²β)`.
- **cMSM = the reference method we adapt** (Marín-Llauradó et al., Nat Commun 2023, Supp Note 2;
  MATLAB code downloaded to `cMSM_ref/`, Zenodo 7921052). Same physics (`div_s σ=0`, `σ:κ=ΔP`,
  statically determinate, constitutive-free), but they use globally-parametrized linear-triangle
  FEM → **open domes only**; we use local-frame GFDM → **closed/arbitrary surfaces** (our niche).
  Both avoid Christoffel symbols in the balance operator by the same trick (they interpolate
  `s^b=√g σ^ab e_a`; we use ambient components). For the R3 cross-check we **run our own
  neo-Hookean FEM** — do NOT reuse the archived cMSM `.mat` fields (`cMSM_ref/MeshData/NeoHookean/`):
  their geometries, material parameters, and 400 Pa loading differ from ours, so they are not a
  valid ground truth. `cMSM_ref/` is an implementation reference only.
- **cMSM regularization TESTED and RULED OUT for closed surfaces** (`reg_compare.py`): ported their
  1st-order grad-trace (`λ_t`) + curl (`λ_c`) penalty into our GFDM solve. Covariant derivative
  needs no Christoffel: `∇_α N_βγ = (v_β)_a(v_γ)_b (G_α N_ab)`; in 2D curl-penalty == full
  covariant-gradient norm. **Finding:** on closed sphere/spheroid it performs like our *raw*
  Tikhonov (sphere σ_max std stays 0.69; spheroid σ_min err ~38%), and a λ_t/λ_c sweep plateaus
  spurious anisotropy at ~10%. The **Laplacian post-smoothing dominates** regardless of regularizer
  (cMSM+Laplacian ≈ Tikhonov+Laplacian). Reason: closed-surface null mode tracks the icosahedral
  pattern (low-k), which a gradient penalty can't kill without flattening the real signal; their
  penalty relies on the open-dome pinned boundary we don't have. So **do NOT adopt cMSM reg** —
  Laplacian smoothing is necessary and sufficient. Details in `stress_estimation.tex` §6.
- **Mesh resolution requirement (embryo) — `mesh_resolution_study.py`, tension_inference §10.7.**
  The controlling variable is the dimensionless **`h·κ`** (mesh spacing × curvature = 1/elements-
  per-curvature-radius). Error vs `h·κ`: sphere needs `h·κ≲0.05` (~20 elem/radius) for ~3%;
  spheroid more forgiving (<4% at `h·κ≈0.18`). **HH20 embryo (n=3766) sits at h·κ≈0.17 median,
  0.44 p90 — COARSER than our coarsest test (sphere subdiv-3 h·κ=0.14).** ⇒ worst-case (high-
  curvature folds) error ≳10%; smooth tube-body ~few %. Uniform refinement to h·κ=0.05 needs
  ~4e4 verts (~3e5 to fix the p90 tail) → **use curvature-ADAPTIVE refinement, not uniform**;
  ship a per-vertex `h·κ` quality map, flag `h·κ>0.1` as low-confidence.
- **λ=0.05 confirmed near-optimal** (same study): spheroid λ-sweep is U-shaped — σ_max err min
  0.9% at λ=0.05, rises to 5.8% at λ=0.2 (over-smooths anisotropy); λ=0.005 leaves 6.6% (null
  modes). At λ=0 on the sphere ε≈1e-7 but error ~1900% → **residual is a diagnostic, NOT the
  objective; the correct ε at the optimum is ~2e-2, not the smallest achievable.**
- **Solve timing & scaling** (sphere, depth-3, this machine): sd3 (n=642) 0.7s · sd4 (2562) 7.6s ·
  sd5 (10242) **267s** · sd6 (40962, lsqr) 532s. Direct sparse solve on the normal equations
  `(LᵀL+λ²RᵀR)` scales ~×35 per ×4 dof (depth-3 stencil ~30 nnz/row → LᵀL ~900 nnz/row, heavy
  fill-in). `lsqr_thresh=60000` dof (=20k verts) is **too high**: sd5 (30726 dof) stays on the
  slow direct path (267s) when lsqr would be faster — lower the threshold (~20–30k dof) so sd5+
  use iterative. Embryo (n=3766, ~11k dof) ≈ 15–30s direct. A 4e4-vertex adaptive target ≈ sd6
  territory (~9 min, lsqr).
- **Two added reference papers (root PDFs), bracket our method (tension_inference §11):**
  **Romo et al. 2014** (*J. Biomech.* 47(3):607–616, ATAA bulge-inflation) = the SAME inverse
  method on real tissue — their Eq.(6) IS our `div_s N+Δp·n=0`, solved nodally, explicitly
  "without a constitutive model"; open dome w/ boundary tractions (like cMSM); **passive**;
  headline = **rupture ≠ max-stress location** (it's where stress meets local thinning/weakness).
  **Bal et al. 2026** (*JMPS* 208:106477, active-gel→continuum shell) = forward active theory;
  stress `σ=σ_elastic+ξρg` (myosin active term ξ); needs Cosserat (through-thickness) for
  apicobasal-asymmetry **bending**. Lineage we complete: Romo/cMSM (open) → us (closed/arbitrary
  = the niche).
- **Fluid-vs-elastic / heterogeneous tension (tension_inference §3.4 — conceptual key fact):**
  a CLEAN FLUID interface has isotropic `N=γP`; tangential equilibrium then demands `∇_s γ=0`,
  so a static bubble CANNOT hold heterogeneous tension (gradient = unbalanced Marangoni force →
  flow homogenises it). On a SOLID-LIKE membrane `N` is a full tensor, and a varying scalar
  tension `N=γ(x)P+N_dev` is only in tangential equilibrium if `∇_s γ` is balanced by
  `div_s N_dev` ⇒ **heterogeneous tension FORCES anisotropy/shear** (this is why `r≠0` and d₁
  tilts off the curvature axes on general meshes). Determinacy by dof-count: fluid = 1 unknown
  (over-constrained to uniform), elastic = 3 unknowns vs 3 eqs (the real inverse problem). The
  inference is meaningful ONLY because the tissue is solid-like.
- **Neural-tube novel angle (tension_inference §11, NOT YET data-verified):** inference gives the
  TOTAL membrane tension; static determinacy fixes the sum but CANNOT split `N=N_elastic+N_active`
  (Bal) — need ablation/pMLC or a forward model for that. Two equilibrium-derived **active
  signatures** we ALREADY compute (forbidden for a passive pressurised convex membrane):
  (i) **principal-axis tilt** (shear `δ=|r|/(|p|+|q|)`) and (ii) **compressive σ_min<0**. Testable
  thesis: HH17→HH20 tension rise/anisotropy is accompanied by growing δ and compressive zones
  localising at high-`h·κ` folds; those active-signature regions (not peak-tension) mark active
  shape change. **TODO when ready: verify δ + σ_min<0 actually localise coherently on saved HH20.**

## Files
- `sphere_curvature.py` — per-vertex curvature, normals, local axes (`compute_vertex_frames`).
- `curvature_compare.py` — mean curvature + normal arrows, sphere vs stretched.
- `surface_fd.py` — GFDM surface-derivative operators + self-test.
- `membrane_stress_fd.py` — direct GFDM membrane-stress solve (σ₁, σ₂); arbitrary fit frame.
- `membrane_stress_fd_v2.py` — same solve in the **principal curvature frame** (e1, e2 from
  `compute_curvature_frame`); adds `d1`, `d2` (principal stress directions, world R³), `r`
  shear diagnostic, and per-vertex `resid_pv`; includes `make_capsule` and `plot_stress_frame`
  (3-panel vedo plot: sphere + spheroid + capsule, mesh coloured by σ₁, white d1 segments).
  **`save_results`/`smooth_results`** write per-vertex NPZ + VTP (resultants `N1`/`N2` stored
  separately from σ so varying-`t` needs no re-solve; `t_field` always per-vertex).
- `stress_smoothing_compare.py` — Laplacian smoothing of σ; raw vs smoothed vs mean (2×3 grid).
- `membrane_stress_beltrami.py` — Beltrami/Airy stress-function solve (single scalar Φ).
- `reg_compare.py` — cMSM-style (grad-trace + curl) regularization vs our Laplacian smoothing,
  sphere + spheroid (imports operators from `membrane_stress_fd`/`surface_fd`). → `out/reg_compare.png`.
- `real_mesh_stress.py` — full pipeline for HH17/HH20 VTK meshes; auto-decimates; CSV/NPZ/VTP.
- `view_smoothed.py` — interactive viewer of smoothed σ from saved `.vtp` (sharecam=False).
- `local_stress.py` — **M1 local stress**: `local_stress_axisym` (axisym two-curvature for
  revolutions: σ_merid=Δp·r_hoop/2t, σ_hoop=Δp·r_hoop/t·(1−r_hoop/2r_merid), with κ_hoop=(n·ρ̂)/ρ
  geometric + κ_merid=2H−κ_hoop) and `local_stress_isotropic` (σ=Δp/2tH, any surface).
- `final_sims.py` — final-results runner: sphere+ellipsoid, M1+M2, IcoSphere subdiv5 dp=20 →
  `out/final/sim0*.{csv,npz,vtp}` + metrics vs analytic.
- `final_real.py` — final-results runner: HH17 (decimated to HH20's ~3766 pts) + HH20, M1
  (isotropic) + M2 → `out/final/hh*_{local,cmsm}.*`.
- `view_final.py` — interactive viewer, SHARED colour limits per group (sphere↔ellipsoid;
  HH17↔HH20). CLI: `--group {analytic,real} --method {m1,m2} --field`.
- `box_compare.py` — box plot Local vs cMSM vs analytical (sphere+ellipsoid) → `out/final/box_compare.png`.
- `real_box_compare.py` — box plot HH17 vs HH20 side-by-side (Local & cMSM) → `out/final/real_box_compare.png`.
- `cMSM_ref/` — downloaded reference cMSM MATLAB code (Zenodo 7921052), implementation
  reference only — its FEM data is NOT our ground truth (we run our own FEM for M3).
- `surface_fd.py` also has `build_derivative_operators` → 1st+2nd derivative ops (g_xi,g_eta,h_xixi,h_xieta,h_etaeta).
- `stress_estimation.tex` / `.pdf` — equations, method, results (compile with `pdflatex` TWICE
  for refs; MiKTeX present). Embeds figures from `out/`.
- `tension_inference.tex` / `.pdf` — standalone mathematical derivation (continuously updated;
  has a **bibliography** now: Bal 2026, Calladine, Flügge, Marín-Llauradó 2023, Romo 2014,
  Timoshenko). **NOTE: the new front section shifted ALL section numbers +1** (sectioning was
  re-derived 2026-06-18). Current layout: **§1 elastostatics framing** (problem class, governing
  eq + 2 projections, static determinacy, 4 method families, constitutive ladder M1/M2/M3 — NEW),
  §2 surface geometry, §3 membrane stress model (thickness role **§3.3**; **§3.4 fluid-vs-elastic
  limit** — NEW: `N=γP` Young-Laplace + Marangoni, heterogeneous tension FORCES anisotropy/shear,
  1-dof fluid vs 3-dof elastic determinacy), §4 balance of momentum (GFDM trick), §5 GFDM
  discretisation (principal curvature frame solve **§5.3**, null modes + mesh pattern + Rician
  bias **§5.5**, Tikhonov + direct-vs-lsqr **§5.6**), §6 curvature frame, §7 principal stress
  directions d₁/d₂, §8 summary, §9 validation cases, **§10 validation suite (mostly run)**:
  §10.1 benchmarks, §10.2 convergence, §10.3 linearity, §10.4 residual maps, §10.7 mesh-resolution
  (`h·κ`) + timing; §10.5/10.6/10.8 (shear, smoothing, FEM) still proposed. **§11 biological
  interpretation: neural-tube tension landscape** — NEW, synthesises Romo 2014 + Bal 2026 (see
  key-facts bullet below).
- `benchmark_analytic.py` — §10.1 figure: sphere/spheroid/capsule vs exact (latitude scatter).
- `convergence_study.py` — §10.2: error & spurious-deviatoric vs h (subdiv 3-6).
- `linearity_test.py` — §10.3: σ ∝ Δp/t over 6 (Δp,t) combos.
- `residual_test.py` — §10.4: per-vertex equilibrium-residual surface maps (vedo 3-panel).
- `mesh_resolution_study.py` — §10.7: error vs `h·κ` (embryo band) + λ tradeoff U-curve.
- `out/` — generated `.npy` / `.csv` / `.png` / `.npz` / `.vtp` results.

## Conventions
- Keep code self-contained; do **not** install the full spatchcocking package (drags in
  opencv/tifffile). Adapt the needed function only.
- Headless by default; only open vedo windows behind `--show`. Long solves: run in background.
- Note: Python buffers stdout when not a TTY → background-run prints appear only at exit.
- Validate every new path against the analytical sphere (and spheroid) before trusting it.
