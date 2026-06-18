"""Mesh-resolution & regularisation study — how fine must an embryo mesh be?

Two experiments, both on analytic geometries where the exact stress is known:

  1. lambda sweep (fixed sphere subdiv-4): equilibrium residual eps and the
     null-mode indicator (spurious deviatoric std) vs the Tikhonov weight lambda.
     Shows the tradeoff: small lambda -> small eps but large null-mode streaks.

  2. resolution sweep (sphere + spheroid, subdiv 3-5): stress error and eps vs the
     DIMENSIONLESS resolution  h*kappa  (mesh spacing relative to curvature radius).
     The HH20 embryo mesh occupies h*kappa ~ 0.17 (median) to 0.44 (p90), shown as
     a shaded band, so the plot reads directly as "what resolution does the embryo
     need for a target error".

Saves: out/mesh_resolution_study.png

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe mesh_resolution_study.py
"""

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vedo

from membrane_stress_fd import solve_membrane

R   = 1.0
DP  = 20.0
T   = 0.05
SIG_REF = DP * R / (2 * T)        # 200 Pa  (sphere analytic sigma1 = sigma2)

# HH20 embryo mesh resolution (computed from the .vtk, decimated to ~3766 pts):
EMBRYO_HK_MEDIAN = 0.173
EMBRYO_HK_P90    = 0.440
EMBRYO_N         = 3766


def one_ring_avg(mesh: vedo.Mesh) -> sp.csr_matrix:
    adlist = mesh.compute_adjacency()
    n = mesh.npoints
    rows, cols, vals = [], [], []
    for i in range(n):
        nb = np.asarray(mesh.find_adjacent_vertices(i, depth=1, adjacency_list=adlist), dtype=int)
        nb = nb[nb != i]
        if len(nb) == 0:
            rows.append(i); cols.append(i); vals.append(1.0); continue
        w = 1.0 / len(nb)
        rows += [i] * len(nb); cols += list(nb); vals += [w] * len(nb)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def smooth(f, A, iters=12, alpha=0.5):
    out = f.copy()
    for _ in range(iters):
        out = (1 - alpha) * out + alpha * (A @ out)
    return out


# ========================================================================== #
# Experiment 1 — lambda sweep
#   sphere    : demonstrates the lambda=0 null-mode catastrophe
#   spheroid  : shows the genuine tradeoff (too high lambda smears anisotropy)
# ========================================================================== #
from membrane_stress_fd import analytic_axisym

print("=" * 70)
print("Experiment 1: lambda sweep")

# --- sphere: the lambda=0 catastrophe (single illustrative point) ---------- #
sphere4 = vedo.IcoSphere(r=R, subdivisions=4)
A4s     = one_ring_avg(sphere4)
res0    = solve_membrane(sphere4, DP, T, depth=3, lam=0.0)
s1sm0   = smooth(res0["sigma1"], A4s)
cata_eps = res0["resid"]
cata_err = abs(s1sm0.mean() - SIG_REF) / SIG_REF
print(f"  [sphere] lambda=0  eps={cata_eps:.2e}  err_sm={cata_err:.0%}  "
      f"(null-mode collapse: tiny residual, useless stress)")

# --- spheroid: real anisotropy, sweep lambda ------------------------------- #
A_AX, B_AX = 2.0, 1.0
ell4 = vedo.IcoSphere(r=R, subdivisions=4).scale([A_AX, 1.0, 1.0])
A4e  = one_ring_avg(ell4)
x_e  = ell4.coordinates[:, 0]
belt_e = np.abs(x_e) < 0.5 * A_AX
sm_an, sh_an = analytic_axisym(ell4.coordinates, DP, T, a=A_AX, b=B_AX)
an_max_e = np.maximum(sm_an, sh_an)
an_min_e = np.minimum(sm_an, sh_an)

lams = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
lam_rows = []
for lam in lams:
    res  = solve_membrane(ell4, DP, T, depth=3, lam=lam)
    s1sm = smooth(res["sigma1"], A4e)
    s2sm = smooth(res["sigma2"], A4e)
    e_max = np.abs(s1sm[belt_e] - an_max_e[belt_e]).mean() / an_max_e[belt_e].mean()
    e_min = np.abs(s2sm[belt_e] - an_min_e[belt_e]).mean() / an_min_e[belt_e].mean()
    lam_rows.append((lam, res["resid"], e_max, e_min))
    print(f"  [spheroid] lambda={lam:6.3f}  eps={res['resid']:.3e}  "
          f"err_max={e_max:.2%}  err_min={e_min:.2%}")

lam_arr = np.array(lam_rows)

# ========================================================================== #
# Experiment 2 — resolution sweep, expressed in h*kappa
# ========================================================================== #
print("=" * 70)
print("Experiment 2: resolution sweep (sphere + spheroid)")

def sphere_run(subdiv):
    mesh = vedo.IcoSphere(r=R, subdivisions=subdiv)
    n = mesh.npoints
    h = np.sqrt(4 * np.pi * R**2 / n)         # mesh spacing
    hk = h / R                                # kappa = 1/R = 1, so h*kappa = h/R
    res = solve_membrane(mesh, DP, T, depth=3, lam=0.05)
    A = one_ring_avg(mesh)
    s1sm = smooth(res["sigma1"], A)
    s2sm = smooth(res["sigma2"], A)
    err_raw = abs(res["sigma1"].mean() - SIG_REF) / SIG_REF
    err_sm  = abs(s1sm.mean()       - SIG_REF) / SIG_REF
    dev_raw = (res["sigma1"] - res["sigma2"]).std() / 2
    return dict(n=n, hk=hk, eps=res["resid"],
                err_raw=err_raw, err_sm=err_sm, dev_raw=dev_raw)

def spheroid_run(subdiv, a=2.0, b=1.0):
    mesh = vedo.IcoSphere(r=R, subdivisions=subdiv).scale([a, 1.0, 1.0])
    n = mesh.npoints
    h = np.sqrt(mesh.area() / n)
    kchar = 1.0 / b                            # max (hoop) curvature at equator
    hk = h * kchar
    res = solve_membrane(mesh, DP, T, depth=3, lam=0.05)
    A = one_ring_avg(mesh)
    s1sm = smooth(res["sigma1"], A)
    # analytic sigma_max field for the spheroid
    from membrane_stress_fd import analytic_axisym
    sm, sh = analytic_axisym(mesh.coordinates, DP, T, a=a, b=b)
    an_max = np.maximum(sm, sh)
    # equatorial belt error
    x = mesh.coordinates[:, 0]
    belt = np.abs(x) < 0.5 * a
    err_sm = np.abs(s1sm[belt] - an_max[belt]).mean() / an_max[belt].mean()
    return dict(n=n, hk=hk, eps=res["resid"], err_sm=err_sm)

sph_rows = [sphere_run(sd)   for sd in (3, 4, 5)]
ell_rows = [spheroid_run(sd) for sd in (3, 4, 5)]
for r in sph_rows:
    print(f"  Sphere   sd n={r['n']:6d}  h*k={r['hk']:.4f}  eps={r['eps']:.2e}  "
          f"err_raw={r['err_raw']:.2%}  err_sm={r['err_sm']:.2%}  dev_raw={r['dev_raw']:.2f}")
for r in ell_rows:
    print(f"  Spheroid    n={r['n']:6d}  h*k={r['hk']:.4f}  eps={r['eps']:.2e}  "
          f"err_sm={r['err_sm']:.2%}")

# ========================================================================== #
# Figure
# ========================================================================== #
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle("Regularisation tradeoff and mesh-resolution requirement "
             r"($\Delta p=20$ Pa, $t=0.05$, depth-3)", fontsize=11)

# ---- Panel A: lambda tradeoff (spheroid) ----------------------------- #
ax = axes[0]
lam_x   = lam_arr[:, 0]
eps_y   = lam_arr[:, 1]
emax_y  = lam_arr[:, 2] * 100
emin_y  = lam_arr[:, 3] * 100

ax.semilogx(lam_x, emin_y, "o-", color="tab:red",
            label=r"$\sigma_{\min}$ error (anisotropy)")
ax.semilogx(lam_x, emax_y, "s-", color="tab:blue",
            label=r"$\sigma_{\max}$ error")
ax.set_xlabel(r"Tikhonov weight $\lambda$")
ax.set_ylabel(r"equatorial stress error (%)")
ax.axvline(0.05, color="gray", ls=":", lw=1)
ax.text(0.053, ax.get_ylim()[1] * 0.9, "default\n$\\lambda=0.05$",
        fontsize=7, color="gray", va="top")
ax.annotate(r"$\lambda\!=\!0$: $\varepsilon\!\approx\!0$ but error $\sim$1900%"
            "\n(null-mode collapse)",
            xy=(0.005, emin_y[0]), xytext=(0.006, max(emax_y) * 0.55),
            fontsize=7, color="black",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

ax2 = ax.twinx()
ax2.semilogx(lam_x, eps_y, "^--", color="tab:green", alpha=0.7,
             label=r"residual $\varepsilon$")
ax2.set_ylabel(r"equilibrium residual $\varepsilon$", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")

ax.set_title(r"(A) $\lambda$ tradeoff (spheroid subdiv-4)")
l1, lab1 = ax.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper center")
ax.grid(alpha=0.3)

# ---- Panel B: error vs h*kappa --------------------------------------- #
ax = axes[1]
sph_hk  = np.array([r["hk"]      for r in sph_rows])
sph_es  = np.array([r["err_sm"]  for r in sph_rows]) * 100
sph_er  = np.array([r["err_raw"] for r in sph_rows]) * 100
sph_n   = [r["n"] for r in sph_rows]
ell_hk  = np.array([r["hk"]      for r in ell_rows])
ell_es  = np.array([r["err_sm"]  for r in ell_rows]) * 100

ax.loglog(sph_hk, sph_es, "o-", color="tab:blue",   label=r"sphere $\sigma_1$ (smoothed)")
ax.loglog(sph_hk, sph_er, "o--", color="tab:blue", alpha=0.4, label=r"sphere $\sigma_1$ (raw)")
ax.loglog(ell_hk, ell_es, "s-", color="tab:green",  label=r"spheroid $\sigma_{\max}$ (smoothed)")

# embryo resolution band
ax.axvspan(EMBRYO_HK_MEDIAN, EMBRYO_HK_P90, color="tab:orange", alpha=0.18, zorder=0)
ax.axvline(EMBRYO_HK_MEDIAN, color="tab:orange", lw=1.2)
ax.axvline(EMBRYO_HK_P90,    color="tab:orange", lw=1.2, ls=":")
ax.text(EMBRYO_HK_MEDIAN, 0.55, f"HH20 embryo\nmedian {EMBRYO_HK_MEDIAN:.2f} - p90 {EMBRYO_HK_P90:.2f}\n($n={EMBRYO_N}$)",
        rotation=0, fontsize=7, color="tab:orange", ha="left", va="bottom")

# target error guides
for tgt in (5, 10):
    ax.axhline(tgt, color="gray", ls=":", lw=0.8)
    ax.text(ax.get_xlim()[0], tgt, f" {tgt}%", fontsize=7, color="gray", va="bottom")

# annotate subdiv labels
for hk, e, n in zip(sph_hk, sph_es, sph_n):
    ax.annotate(f"n={n}", (hk, e), textcoords="offset points", xytext=(4, 5),
                fontsize=7, color="tab:blue")

ax.set_xlabel(r"resolution  $h\,\kappa$  (mesh spacing $\times$ curvature)  — coarser $\rightarrow$")
ax.set_ylabel(r"relative stress error (%)")
ax.set_title("(B) error vs resolution (embryo band shown)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, which="both", alpha=0.3)
ax.invert_xaxis()   # finer mesh (small h*k) on the right

plt.tight_layout()
out_path = "out/mesh_resolution_study.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}")
