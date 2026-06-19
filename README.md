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
- **Additional reference papers** (cited in `tension_inference` §11; PDFs not redistributed) —
  Romo et al., *In vitro analysis of localized aneurysm rupture*, J. Biomech. 47(3):607–616
  (2014), [doi:10.1016/j.jbiomech.2013.12.012](https://doi.org/10.1016/j.jbiomech.2013.12.012);
  Bal et al., *Continuum theory for the mechanics of curved epithelial shells…*, JMPS 208:106477
  (2026), [doi:10.1016/j.jmps.2025.106477](https://doi.org/10.1016/j.jmps.2025.106477).
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

### 7. Principal curvature frame and sign consistency (new — 2026-06-17)

Extends the curvature pipeline to produce, per vertex, the full **curvature-aligned
orthonormal frame** {**e₁**, **e₂**, **n**} — i.e. the principal curvature *directions* as
unit vectors in world R³ — in addition to the scalar curvatures κ₁ ≥ κ₂.

Strategy: build GFDM first-order operators on the normal field, evaluate the Weingarten map
**B**ᵢⱼ = (∂ᵢ**n**) · **e**ⱼ (shape operator in the local fit frame), then diagonalise the
2×2 symmetric **B** analytically via `θ = ½ arctan2(2B₁₂, B₁₁−B₂₂)`. The eigenvectors are
rotated back to world R³, giving **e₁** (along κ₁) and **e₂** = **n** × **e₁**.

**Sign consistency:** principal directions are only defined up to sign (±**e₁** equally valid).
The `arctan2` formula picks the sign independently per vertex; at near-umbilic points
(κ₁≈κ₂, discriminant → 0) it can flip sign relative to neighbours. Larger patch `depth`
does **not** help (tested depth 2–5: 34–40 flipped vertices, no trend). The fix is a single
BFS walk from vertex 0 that flips **e₁**[j], **e₂**[j] whenever they disagree with the
already-visited parent — reducing flips from 34 to 2. The remaining 2 are the umbilic tips
of the spheroid (genuine topological singularities; Poincaré-Hopf requires ≥ 2 on a closed
genus-0 surface; unfixable by any algorithm).

**LaTeX document:** `tension_inference.tex` / `.pdf` — standalone derivation (continuously
updated; now with a **bibliography**). Current sections: a **§1 elastostatics framing** (problem
class membrane-vs-shell, the governing equation and its normal/tangential projections, static
determinacy = why tension is geometric not material, the four method families, and the
constitutive ladder placing M1/M2/M3), surface parametrisation and fundamental forms, membrane
stress resultant, the **role of thickness** (resultant vs stress, static determinacy, varying-`t`
regimes), a **§3.4 fluid-vs-elastic limit** (isotropic `N=γP` Young-Laplace + the Marangoni
argument that a fluid bubble cannot hold heterogeneous tension statically; on a solid-like
membrane a varying scalar tension *forces* deviatoric/shear stress; 1-dof-fluid vs 3-dof-elastic
determinacy), balance of linear momentum (normal → Young-Laplace; tangential → in-plane
equilibrium), ambient-component GFDM trick, 2×2 analytic diagonalisation, static
indeterminacy and null modes (mesh-dependent pattern + Rician eigenvalue bias), Tikhonov
regularisation (incl. **direct-vs-lsqr solver choice**), principal curvature frame solve,
principal stress directions (d₁, d₂, θ_s), a **validation suite (§10, mostly run)**:
analytic benchmarks, mesh convergence, pressure/thickness linearity, equilibrium-residual
maps, **mesh-resolution requirement (`h·κ`) + solve timing**, shear diagnostic, smoothing
sensitivity, FEM cross-check, direction-field biology — and a **§11 biological interpretation:
the neural-tube tension landscape** synthesising the two added reference papers
(Romo et al. 2014 passive bulge-inflation inference; Bal et al. 2026 active-gel shell theory):
the shared equilibrium core, the total-tension vs elastic/active split equilibrium cannot
resolve, two equilibrium-derived active signatures (principal-axis tilt and compressive
`σ_min`), and a falsifiable HH17→HH20 thesis. Finally a **§12 stress-based FEM chapter**
(**implemented**, `membrane_stress_fem.py`): the primal virtual-work weak form
`∫ N:ε_s(w) = ∫ Δp·n·w`, P1 nodal stress + P1 vector test → square 3n system (faithfully
cMSM on a *closed* surface), element assembly + consistent load, the singular-system solve
(min-norm `lsqr` then Tikhonov with a FEM-native 1-ring roughness), and the results: the
**raw min-norm FEM shows the same "lines" as GFDM** → the closed-surface artefact is intrinsic
to the indeterminacy, not a GFDM-stencil effect (and cMSM is singular on a closed surface);
regularised FEM is **as/more accurate** than GFDM (sphere dev-std 4.4 vs 6.5, spheroid σ_max
1.7% vs 3.6%) and **~5–12× faster** (subdiv-4 2.0 s vs 10.3 s; subdiv-5 9.2 s vs 110 s), since
the P1 `K` is 6.6× sparser than the depth-3 GFDM `LᵀL`.

```powershell
# per-vertex principal curvature frame: kappa1, kappa2, e1, e2 (world R3), n
# validates on sphere (kappa1≈kappa2≈1) and spheroid equator (kappa1≈1, kappa2≈0.25)
& $py surface_curvature_frame.py                  # headless validation
& $py surface_curvature_frame.py --show           # interactive view: kappa1 colour + e1/e2/n arrows
& $py surface_curvature_frame.py --csv            # save to out/curvature_frame_*.csv
#   flags: --file <mesh> --subdiv --depth --show --csv --out-dir

# sign-consistency on the prolate spheroid (the typical case)
# cyan = consistent after BFS, red = sign-flipped vs majority of 1-ring neighbours
& $py show_e2_spheroid.py   # result: 34 → 2 flipped (2 umbilic poles)

# sign-consistency on the sphere (worst case: totally umbilic everywhere)
# hairy-ball theorem: BFS field must have singularities; 314/2562 vertices (12%) affected
& $py show_e2_sphere.py

# capsule: cylinder (R=1, H=2) + hemispherical caps — three-region validation case
# cylinder: kappa1=1/R (hoop), kappa2=0 (axial)  |  caps: kappa1=kappa2=1/R (umbilic)
# mesh coloured by discriminant d=|k1-k2|/2 (0=umbilic blue, 1/2R red); e1 arrows cyan/red
& $py show_capsule.py
```

| case | post-BFS inconsistent | why |
|---|---|---|
| sphere | 314 (12%) | totally umbilic — all directions undefined; hairy-ball theorem |
| spheroid | 2 (<0.1%) | only 2 umbilic poles; rest well-defined |
| capsule | 3 (0.1%) | 2 umbilic cap poles + 1 junction vertex; cylinder body fully consistent |

Capsule curvature validation (computed vs analytic, `ntheta=40 nphi=14`, `n=2522`):

| region | κ₁ mean±std | κ₂ mean±std | analytic (κ₁, κ₂) |
|---|---|---|---|
| cylinder body (n=1400) | 1.000 ± 0.000 | 0.034 ± 0.083 | (1.000, 0.000) |
| spherical caps (n=1042) | 1.037 ± 0.047 | 0.895 ± 0.076 | (1.000, 1.000) |

For real biological meshes (non-umbilic almost everywhere) the spheroid/capsule result
is the relevant model: BFS leaves at most a handful of isolated singular vertices.

CSV columns: `X,Y,Z, kappa1,kappa2, r1,r2, H,K, e1x,e1y,e1z, e2x,e2y,e2z, nx,ny,nz`

### 8. Principal-curvature-frame stress solve and principal stress directions (new — 2026-06-17)

`membrane_stress_fd_v2.py` is a revised version of the GFDM solve that uses the
**principal curvature frame** (e₁, e₂, n from `compute_curvature_frame`) as the local tangent
frame, instead of the arbitrary polynomial-fit frame (v₁, v₂).

**What changes vs v1:**
| | `membrane_stress_fd.py` (v1) | `membrane_stress_fd_v2.py` (v2) |
|---|---|---|
| Tangent frame | arbitrary fit (v₁, v₂) | principal curvature (e₁, e₂) |
| σ₁, σ₂ result | same | same (eigenvalues are frame-independent) |
| Extra outputs | — | d₁, d₂ (principal stress directions); r (shear diagnostic) |
| Validation cases | sphere, spheroid | sphere, spheroid, capsule |
| Visualization | 2-panel PNG | 3-panel PNG: σ₁ colour + white line segments for d₁ |

**Principal stress directions** d₁, d₂ are extracted from the shear DOF r:
```
theta_s = 0.5 * arctan2(2*r, p - q)
d1 = cos(theta_s) * e1 + sin(theta_s) * e2   # world-frame principal stress dir 1
d2 = n x d1
```

**Why line segments, not arrows.**
A principal stress direction is a **line field** — tension along +d₁ is physically identical
to tension along −d₁. A directed arrow would incorrectly imply a signed (vector) quantity.
The visualization uses symmetric line segments centered at each vertex, extending in ±d₁,
which is the standard glyph for any line-field quantity (stress, curvature, fabric tensor).

**Glyph sparseness logic** (`plot_stress_frame`): two independent filters control which
vertices get a segment:
1. **Umbilic suppression** — drops vertices where `|κ₁−κ₂| < 0.25 × max(|κ₁−κ₂|)`.
   At those points d₁ is ill-defined (BFS noise dominates); showing a segment there is
   misleading. On the capsule this cleanly removes the hemispherical caps; on the spheroid
   it removes the two poles. On the sphere (all umbilic, `max ≈ 0`) this filter is
   disabled and all vertices are shown.
2. **Density cap** — from the surviving vertices, takes every Nth in index order so that
   ~150 segments appear total. Sampling is by vertex index (approximately spatially uniform
   on the regular IcoSphere/capsule meshes we use).

**Principal curvature dirs ≠ principal stress dirs in general.**
They coincide only when geometry and loading share the same symmetry — e.g., axisymmetric
surface + uniform pressure. On such surfaces r ≈ 0 (confirmed: sphere 1%, spheroid 2.2%).
On a general closed mesh (HH17/HH20) r will be non-trivially non-zero, quantifying how far
the stress principal axes tilt away from the curvature axes to satisfy global equilibrium.

**Capsule stress validation** (R=1, H=2, Δp=20, t=0.05):

| region | σ_max mean | analytic | σ_min mean | analytic |
|---|---|---|---|---|
| cylinder body | 422.8 Pa | 400 Pa (Δp·R/t) | 204 Pa | 200 Pa (Δp·R/2t) |
| hemispherical caps | ~200 Pa | 200 Pa (sphere: Δp·R/2t) | ~200 Pa | 200 Pa |

```powershell
# solve in principal curvature frame; sphere + spheroid + capsule -> out/membrane_stress_v2.png
# ALSO saves per-vertex results to out/{sphere_s4,spheroid_s4,capsule}.{npz,vtp}
& $py membrane_stress_fd_v2.py                   # flags: --subdiv --depth --dp --t --lam --save-dir --no-save --show
& $py membrane_stress_fd_v2.py --show            # opens interactive 3-panel vedo window
```

**Saved results.** `solve_membrane` returns (and `save_results` writes to NPZ + VTP) every
per-vertex field for later analysis: geometry (`pts`, `faces`, `normals`), curvature
(`kappa1`, `kappa2`, `H`, `K`, `e1`, `e2`), stress DOFs (`p`, `q`, `r`), **resultants**
(`N1`, `N2`) and **stresses** (`sigma1`, `sigma2`) plus their Laplacian-smoothed versions,
principal stress **directions** (`d1`, `d2`), diagnostics (`delta` shear, `resid_pv`
per-vertex residual), and the loading/solve metadata (`dp`, `t_field`, `lam`, `depth`).
Because the membrane is statically determinate, `N1`/`N2` are independent of `t` — a new
thickness map is applied by re-dividing `σ = N/t_field` with no re-solve. `t_field` is always
stored per-vertex so varying-thickness cases need no special handling.

### 9. Validation suite (§10 of `tension_inference`)

Each script loads/produces analytic-case results and writes a figure into `out/`.

```powershell
# 10.1 analytic benchmarks: sphere/spheroid/capsule vs exact, latitude scatter -> out/benchmark_analytic.png
& $py benchmark_analytic.py        # reads saved out/{sphere_s4,spheroid_s4,capsule}.npz

# 10.2 convergence: error & spurious-deviatoric vs mesh spacing h (subdiv 3-6) -> out/convergence.png
& $py convergence_study.py

# 10.3 linearity: sigma ∝ Δp/t across 6 (Δp,t) combos, two routes overlap -> out/linearity_test.png
& $py linearity_test.py

# 10.4 equilibrium residual: per-vertex residual map on each surface -> out/residual_map.png
& $py residual_test.py             # flags: --subdiv --show

# 10.7 mesh-resolution & λ tradeoff: error vs h·κ with embryo band + λ U-curve -> out/mesh_resolution_study.png
& $py mesh_resolution_study.py
```

Key findings (smoothed, subdiv-4, Δp=20, t=0.05): sphere mean-stress **1.0%**, spheroid
equatorial σ_max/σ_min **1.5% / 1.9%**, capsule cylinder hoop/axial **5.0% / 2.0%**.
Linearity: `σ/(Δp/t)` constant to 4 decimals. **λ=0.05 confirmed near-optimal** (U-shaped
error). **Resolution**: error is governed by the dimensionless `h·κ`; the HH20 embryo
(`n=3766`) sits at `h·κ≈0.17` median / `0.44` p90 — coarser than our coarsest analytic
test, so high-curvature folds need curvature-**adaptive** refinement (a uniform `h·κ=0.05`
target is ~4e4 verts). **Timing** (sphere, depth-3): sd3 0.7s · sd4 7.6s · sd5 **267s**
(direct) · sd6 532s (lsqr) — the direct normal-equation solve scales ~×35 per ×4 DOFs.

### 10. Stress-based FEM — an alternative discretisation (§12 of `tension_inference`)

`membrane_stress_fem.py` solves the **same** balance law `div_s N + Δp·n = 0` by a
finite-element method instead of GFDM, as a cross-check (and to put cMSM's formulation on a
*closed* surface).

**Methodology.** Primal **virtual-work** weak form (dot the balance with a vector test field
`w`, integrate by parts on the closed surface): `∫_Γ N : ε_s(w) dS = ∫_Γ Δp (n·w) dS` — internal
virtual work = external virtual work of the pressure. Both fields are **continuous P1** in
ambient Cartesian components: the stress trial carries 3 DOF/vertex `(p,q,r)` in the local frame
(`N = p v1⊗v1 + q v2⊗v2 + r(v1⊗v2+v2⊗v1)`, tangential by build); the test is a P1 vector field.
This gives a **square 3n×3n** system (the count of cMSM). Element block on a triangle `T`
(area `A_T`, constant P1 hat gradient `g_j = n_T×opp_j/(2A_T)`):
`a|_T(N, e_d φ_j) = (A_T/3) Σ_{m∈T} (N_m g_j)_d`; consistent load via the P1 mass matrix
`M^T_jm = (A_T/12)(1+δ_jm)`. `K` is **singular** on a closed surface (self-equilibrating null
modes), but the pressure load is consistent (net force/torque = 0). Two-stage solve:
(i) **raw min-norm** `lsqr` (lines diagnostic); (ii) **Tikhonov** `‖Ks−b‖²+w²‖Rs‖²` with a
**FEM-native 1-ring roughness** `R` (`∫|∇_s N|²` from the P1 element gradients — no GFDM
operators), where `w = λ·‖K‖_F/‖R‖_F` rescales the area-weighted `K` (~h) against `R` (~O(1)) so
`λ=0.05` means the same as in GFDM (without this rescale a bare λ mis-scales by ~h⁻⁴ and collapses
the solve). Direct solve below 20k DOFs, iterative `lsqr` on `[K; wR]` above. Principal stresses
`σ₁,σ₂ = eig([[p,r],[r,q]])/t`, same post-processing as GFDM.

**Visualization.** The viewer colours each mesh by a stress scalar (default **trace**
`σ₁+σ₂` — the hydrostatic / mean surface tension that cMSM uses, and its most robustly-recovered
quantity; also `--field vonmises|mean|shear|sigma_max|sigma_min`, where von Mises
`√(σ₁²−σ₁σ₂+σ₂²)` is the plane-stress equivalent, mean `(σ₁+σ₂)/2` the isotropic tension, and
shear `(σ₁−σ₂)/2` the anisotropy) and overlays a **principal-stress cross** at each sampled vertex:
two symmetric segments along ±d₁ and ±d₂ (stress is a line field), each arm length ∝ `|σᵢ|` (so
isotropic regions look like a `+`, anisotropic ones elongate along the larger stress), coloured red
for tension and blue for compression. This mirrors cMSM's own stress glyph (orthogonal arrow pairs,
length ∝ principal tension, divergence/convergence = sign).

```powershell
# solve sphere + spheroid, FEM vs GFDM head-to-head + analytic -> out/membrane_stress_fem.png
& $py membrane_stress_fem.py                     # flags: --subdiv --depth --dp --t --lam --stretch --no-gfdm --raw --field --show
& $py membrane_stress_fem.py --show              # interactive 2-panel viewer: von Mises + principal-stress crosses
& $py membrane_stress_fem.py --show --field shear   # colour by max-shear (anisotropy) instead
& $py membrane_stress_fem.py --raw --show        # colour by the RAW min-norm field (the null-mode "lines")

# reproduce the GFDM §10 validation battery for the FEM (convergence, linearity,
# analytic benchmark sphere/spheroid/capsule, λ tradeoff) -> out/fem_validation.png
& $py fem_validation.py
```

**Key findings** (subdiv-4, Δp=20, t=0.05, vs GFDM):
- **The "lines" are intrinsic, not a GFDM artefact.** The raw min-norm FEM — a completely
  different discretisation — produces the same spurious deviatoric (dev-std 73 at subdiv-3, 216 at
  subdiv-4 on a sphere field that must be 0 — far above the regularised ~3–5). So the closed-surface streaks come from the
  static indeterminacy itself; equivalently cMSM is singular on a closed surface and relies on its
  open-dome boundary.
- **Regularised FEM ≥ GFDM accuracy:** sphere mean **0.3%** (GFDM 1.1%), dev-std **4.4** (GFDM 6.5);
  spheroid σ_max **1.7%** (GFDM 3.6%); capsule cylinder hoop **0.15%** / axial 3.8%. Linearity:
  `σ/(Δp/t)` constant to 4 decimals (sphere ratio 0.498 vs analytic 0.5). λ-tradeoff: FEM's σ_min
  keeps improving with larger λ, so its optimum sits **higher** than GFDM's 0.05.
- **~5–12× faster:** subdiv-4 **2.0 s** vs 10.3 s, subdiv-5 **9.2 s** vs 110 s. `KᵀK` is 6.6× sparser
  than the depth-3 GFDM `LᵀL` (57 vs 376 nnz/row), with no per-vertex WLS operator to build →
  the better engine for embryo-scale, curvature-adaptive meshes.

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

Last updated: **2026-06-18 (UTC-04:00)**

### Done
- [x] **Stress-based FEM IMPLEMENTED + validated** (`membrane_stress_fem.py`, `tension_inference` **§12**) — primal virtual-work (cMSM-style) weak form `∫ N:ε_s(w)=∫ Δp·n·w`, hand-rolled `scipy.sparse`, P1 nodal local-frame DOFs (square 3n system); element assembly + consistent load; singular-system solve = raw min-norm `lsqr` then Tikhonov with a **FEM-native 1-ring roughness** (Frobenius-matched so λ=0.05 matches GFDM); auto-iterative `lsqr` above 20k DOFs. **Findings:** raw FEM shows the same "lines" as GFDM ⇒ artefact is intrinsic to the indeterminacy, not a GFDM stencil effect (cMSM singular on closed surfaces); regularised FEM ≥ GFDM accuracy (sphere dev-std 4.4 vs 6.5, spheroid σ_max 1.7% vs 3.6%); **~5–12× faster** (subdiv-4 2.0 vs 10.3 s, subdiv-5 9.2 vs 110 s) — KᵀK is 6.6× sparser (57 vs 376 nnz/row) — _2026-06-18_
- [x] **Stress-based FEM methodology decided + documented** (`tension_inference` **§12**) — chose **primal virtual-work (cMSM-style)**, **hand-rolled `scipy.sparse`**, P1 nodal local-frame DOFs; runners-up (LSFEM, mixed Hellinger–Reissner) weighed and recorded — _2026-06-18_
- [x] **`tension_inference` conceptual expansion** — added **§1 elastostatics framing** (problem class, static determinacy, four method families, M1/M2/M3 constitutive ladder), **§3.4 fluid-vs-elastic limit** (`N=γP`/Young–Laplace + Marangoni; heterogeneous tension forces deviatoric shear; 1-dof-fluid vs 3-dof-elastic determinacy), **§11 neural-tube interpretation** synthesising the two added reference papers (Romo et al. 2014 passive bulge-inflation; Bal et al. 2026 active-gel shell) — total-vs-active split, tilt + compressive-`σ_min` active signatures, falsifiable HH17→HH20 thesis — and a **bibliography**. All section numbers shifted +1; PDF recompiled — _2026-06-18_
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
- [x] `surface_curvature_frame.py` — principal curvature **directions** (**e₁**, **e₂**) as world-frame unit vectors via shape-operator (Weingarten map) diagonalisation + BFS sign propagation; validated on sphere (κ₁≈κ₂≈1) and spheroid equator (κ₁≈1, κ₂≈0.25); sign flips reduced 34 → 2 (unavoidable umbilic singularities) — _2026-06-17_
- [x] `tension_inference.tex/.pdf` — new standalone 6-page derivation: surface geometry, membrane equilibrium (normal → Young-Laplace; tangential → in-plane balance), ambient GFDM discretisation, curvature-frame extraction — _2026-06-17_
- [x] `show_e2_spheroid.py` — sign-consistency visualiser on spheroid: 1-ring dot-product check, cyan/red colouring; confirmed 2 topological singularities at spheroid poles after BFS (34→2) — _2026-06-17_
- [x] `show_e2_sphere.py` — sign-consistency visualiser on sphere (totally umbilic worst case): 314/2562 vertices (12%) inconsistent after BFS — hairy ball theorem in action; confirms spheroid is the relevant model for real meshes — _2026-06-17_
- [x] `show_capsule.py` — capsule mesh builder (`make_capsule`) + curvature-frame validation + sign-consistency viewer; 3 region validation (cylinder κ₁=1/R κ₂=0 exact, caps κ₁=κ₂=1/R within 4–10%); 3/2522 (0.1%) inconsistent after BFS — _2026-06-17_
- [x] `membrane_stress_fd_v2.py` — GFDM stress solve in the **principal curvature frame** (e₁,e₂); same σ₁,σ₂ as v1 (frame-independent eigenvalues); adds **d₁,d₂** (principal stress directions, world R³) via `θ_s = ½arctan2(2r, p−q)` and `r` shear diagnostic; `plot_stress_frame` renders 3-panel vedo plot (sphere + spheroid + capsule, σ₁ colour + **symmetric line segments** for d₁; stress is a line field so segments not arrows; umbilic regions suppressed by discriminant filter; ~150 segments per panel via stride subsampling) → `out/membrane_stress_v2.png` — _2026-06-17_
- [x] **Final-results matrix started** — `local_stress.py` (M1), `final_sims.py` (sphere+ellipsoid M1+M2), `final_real.py` (HH17/HH20 M1+M2), `view_final.py` (interactive, grouped colour limits), `box_compare.py` + `real_box_compare.py` (box plots); ran Sims 1,2,7,8 + the two real meshes at **dp=20** — _2026-06-16_
- [x] **Result saving in `membrane_stress_fd_v2.py`** — `save_results` writes per-vertex NPZ + VTP (curvature, stress DOFs, resultants `N1`/`N2`, σ₁/σ₂ raw + smoothed, directions d₁/d₂, `delta`, `resid_pv`, `t_field`); `smooth_results` adds Laplacian-smoothed fields; resultants stored separately from σ so varying-thickness needs no re-solve — _2026-06-17_
- [x] **Validation suite (§10 of `tension_inference`)** — `benchmark_analytic.py` (10.1: sphere/spheroid/capsule vs exact, ≤5% smoothed), `convergence_study.py` (10.2: ~h¹·¹⁻¹·³ to error floor), `linearity_test.py` (10.3: σ∝Δp/t to 4 decimals), `residual_test.py` (10.4: per-vertex residual maps — null-mode fingerprint on sphere, junction spikes on capsule), `mesh_resolution_study.py` (10.7: error vs `h·κ` + embryo band, λ U-curve, solve timing) — _2026-06-17_

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
- **Validation suite (§9, smoothed, subdiv-4, Δp=20, t=0.05):** benchmark errors — sphere mean
  **1.0%**, spheroid σ_max/σ_min **1.5%/1.9%**, capsule cylinder hoop/axial **5.0%/2.0%**;
  linearity `σ/(Δp/t)` constant to 4 decimals; **λ=0.05 near-optimal** (U-shaped error);
  residual maps show the imbalance is structural (icosahedral fingerprint, capsule-junction spikes),
  not physical. **Mesh resolution** governed by `h·κ`; **HH20 embryo coarser than subdiv-3**
  (`h·κ≈0.17` median, `0.44` p90) → needs adaptive refinement at high-curvature folds.
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
- [ ] **Verify the §11 neural-tube thesis on saved HH20 fields** — check whether the two active signatures (principal-axis tilt `δ=|r|/(|p|+|q|)` and compressive `σ_min<0`) actually localise *coherently* (and to the high-`h·κ` folds) rather than as noise, before the interpretation goes in a manuscript. This is the data-check deferred from the conceptual write-up.
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
- `membrane_stress_fem.py` — **stress-based FEM** (tension_inference §12): primal virtual-work (cMSM-style), P1 nodal local-frame DOFs, square 3n system, FEM-native 1-ring roughness, auto-iterative solve; returns σ₁/σ₂ + principal directions d₁/d₂; viewer colours by **trace σ₁+σ₂** (cMSM's metric; `--field` for vonmises/mean/shear/sigma_max/sigma_min) with **principal-stress crosses** (±d₁/±d₂, arms ∝|σᵢ|, red=tension/blue=compression); `--show` interactive, `--raw` to see the lines. ~5–12× faster than GFDM and ≥ its accuracy
- `fem_validation.py` — **FEM validation suite** reproducing the GFDM §10 battery for `membrane_stress_fem`: convergence (subdiv 3–5, FEM vs GFDM), linearity (σ ∝ Δp/t), analytic benchmark (sphere/spheroid/capsule), λ tradeoff → `out/fem_validation.png` + printed tables
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
- `surface_curvature_frame.py` — principal curvature frame (κ₁,κ₂, **e₁**,**e₂**,**n**): shape-operator diagonalisation + BFS sign propagation for global consistency
- `membrane_stress_fd_v2.py` — GFDM solve in the principal curvature frame (e₁,e₂); adds **d₁**,**d₂** (principal stress directions, world R³) and `r` shear diagnostic; `plot_stress_frame` renders 3-panel plot with σ₁ colour map and **symmetric line segments** for d₁ (line field, not arrows; umbilic-suppressed + stride-subsampled to ~150 glyphs); **`save_results`/`smooth_results`** write per-vertex NPZ + VTP (resultants `N1`/`N2` + smoothed σ + directions + `resid_pv` + `t_field`)
- `benchmark_analytic.py` — §10.1 analytic benchmark figure (sphere/spheroid/capsule vs exact, latitude scatter) → `out/benchmark_analytic.png`
- `convergence_study.py` — §10.2 error & spurious-deviatoric vs mesh spacing h (IcoSphere subdiv 3-6) → `out/convergence.png`
- `linearity_test.py` — §10.3 σ ∝ Δp/t check over 6 (Δp,t) combos → `out/linearity_test.png`
- `residual_test.py` — §10.4 per-vertex equilibrium-residual surface maps (sphere/spheroid/capsule) → `out/residual_map.png`
- `mesh_resolution_study.py` — §10.7 error vs dimensionless `h·κ` with embryo band + λ tradeoff U-curve → `out/mesh_resolution_study.png`
- `tension_inference.tex` / `.pdf` — standalone derivation: **§1 elastostatics framing** (problem class, static determinacy, method families, constitutive ladder), surface geometry, membrane balance (normal + tangential), thickness role, **§3.4 fluid-vs-elastic limit** (`N=γP`/Marangoni; heterogeneous tension forces shear), GFDM (+ solver choice), curvature-frame extraction, principal stress directions, the §10 validation suite (benchmarks, convergence, linearity, residual maps, resolution/timing), **§11 neural-tube interpretation** (Romo 2014 + Bal 2026: total-vs-active split, tilt/compression active signatures, HH17→HH20 thesis), **§12 stress-based FEM** (implemented: primal virtual-work / cMSM-style, P1 square system; results — lines intrinsic, accuracy ≥ GFDM, ~5–12× faster), and a **bibliography**
- `show_e2_spheroid.py` — sign-consistency visualiser for **e₂** on spheroid (cyan=consistent, red=flipped; 2 residual singularities at umbilic poles)
- `show_e2_sphere.py` — sign-consistency visualiser on sphere (totally umbilic worst case; 314 inconsistent after BFS — hairy ball theorem)
- `show_capsule.py` — capsule mesh builder (`make_capsule`) + three-region curvature validation + sign-consistency viewer (mesh coloured by discriminant d=|κ₁−κ₂|/2)
- `stress_estimation.tex` / `.pdf` — working-notes equations, method, results (full GFDM derivation, cMSM comparison, real meshes)
- `manuscript_outline.tex` / `.pdf` — manuscript outline, **mechanics-first reframing**: thickness-driven stress *dissipation* + *bending* (transmural gradient); Models A (Local) / B (CMSM) / C (3D neo-Hookean FEM), pHH3 mitotic correlation
- `requirements.txt` — pinned `fem_env` dependencies (Python 3.11)
- `CLAUDE.md` — project context for future sessions
- _(not committed: `out/` generated data, `*.vtk` input meshes, `cMSM_ref/`, the supplement PDF, compiled `*.pdf` — see "Data & external references" above)_
