"""Analytic benchmark figure for §9.1 of tension_inference.tex.

Loads saved NPZ results (from membrane_stress_fd_v2.py, subdiv=4, dp=20, t=0.05)
and produces a 3-panel comparison of GFDM vs analytic membrane stresses for:
  Panel 1 — Sphere:     histogram of σ₁ raw + smoothed vs 200 Pa reference
  Panel 2 — Spheroid:   scatter σ₁,σ₂ vs latitude β, with analytic curves
  Panel 3 — Capsule:    scatter σ₁,σ₂ vs long-axis x, with analytic reference lines

Saves: out/benchmark_analytic.png

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe benchmark_analytic.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ---- configuration ------------------------------------------------------- #
NPZ_SPHERE   = "out/sphere_s4.npz"
NPZ_SPHEROID = "out/spheroid_s4.npz"
NPZ_CAPSULE  = "out/capsule.npz"
OUT_PNG      = "out/benchmark_analytic.png"

DP = 20.0
T  = 0.05
R  = 1.0
A  = 2.0   # spheroid long semi-axis (x)
B  = 1.0   # spheroid equatorial radius
CAP_R, CAP_H = 1.0, 2.0


def analytic_spheroid_at_pts(pts):
    """Analytic membrane stresses for spheroid (long axis x, semi-axes A,B)."""
    x    = pts[:, 0]
    rho  = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2)
    beta = np.arctan2(rho / B, x / A)
    sb, cb = np.sin(beta), np.cos(beta)
    J    = np.sqrt((A * sb)**2 + (B * cb)**2)
    r1   = J**3 / (A * B)
    r2   = B * J / A
    sm   = DP * r2 / (2 * T)                         # meridional
    sh   = DP * r2 * (1 - r2 / (2 * r1)) / T        # hoop
    return beta, sm, sh


def analytic_spheroid_curve(n=500):
    """Fine analytic curves over β ∈ [0, π]."""
    beta = np.linspace(0, np.pi, n)
    sb, cb = np.sin(beta), np.cos(beta)
    J    = np.sqrt((A * sb)**2 + (B * cb)**2)
    r1   = J**3 / (A * B)
    r2   = B * J / A
    sm   = DP * r2 / (2 * T)
    sh   = DP * r2 * (1 - r2 / (2 * r1)) / T
    return beta, sm, sh


def analytic_capsule(z):
    """Analytic hoop and axial stresses for a capsule.

    Cylinder body (|z| < CAP_H): σ_hoop = ΔpR/t, σ_axial = ΔpR/2t.
    Hemispherical caps (|z| > CAP_H): both curvatures = 1/R → σ = ΔpR/2t.
    """
    sig_hoop  = np.where(np.abs(z) < CAP_H, DP * CAP_R / T,
                         DP * CAP_R / (2 * T))
    sig_axial = np.full_like(z, DP * CAP_R / (2 * T))
    return np.maximum(sig_hoop, sig_axial), np.minimum(sig_hoop, sig_axial)


# ---- load results -------------------------------------------------------- #
d_sph = np.load(NPZ_SPHERE)
d_ell = np.load(NPZ_SPHEROID)
d_cap = np.load(NPZ_CAPSULE)

# ---- figure --------------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    r"Analytic benchmarks: GFDM vs exact membrane stresses"
    f"\n(IcoSphere subdiv-4, $\\Delta p={DP}$ Pa, $t={T}$, Tikhonov $\\lambda=0.05$, "
    r"Laplacian-smoothed $\alpha=0.5$, 12 iters)",
    fontsize=10,
)

# =====================================================================
# Panel 1 — Sphere (latitude scatter, same style as spheroid)
# =====================================================================
ax = axes[0]

pts_s  = d_sph["pts"]
s1r    = d_sph["sigma1"]
s1sm   = d_sph["sigma1_smooth"]
s2sm   = d_sph["sigma2_smooth"]
sig_ref = DP * R / (2 * T)   # 200 Pa

# latitude β = arccos(z / R), pole at z-axis (arbitrary for isotropic sphere)
beta_s = np.arccos(np.clip(pts_s[:, 2] / R, -1, 1))

stride_s = max(1, len(beta_s) // 500)
ax.scatter(beta_s[::stride_s], s1sm[::stride_s], s=6, alpha=0.5,
           color="tab:blue", zorder=2, label=r"$\sigma_1$ (smoothed)")
ax.scatter(beta_s[::stride_s], s2sm[::stride_s], s=6, alpha=0.5,
           color="tab:red",  zorder=2, label=r"$\sigma_2$ (smoothed)")
ax.axhline(sig_ref, color="k", lw=2, ls="--", zorder=3,
           label=f"analytic {sig_ref:.0f} Pa")

err_sm = np.abs(s1sm - sig_ref).mean() / sig_ref * 100
ax.text(0.97, 0.97,
        f"mean err (smoothed): {err_sm:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax.set_xlabel(r"Latitude $\beta$ (0 = north pole, $\pi/2$ = equator, $\pi$ = south pole)")
ax.set_ylabel("Stress (Pa)")
ax.set_title(r"Sphere ($R=1$,  $\sigma_1=\sigma_2=200$ Pa)")
ax.legend(fontsize=8)
ax.set_xlim(0, np.pi)
ax.set_ylim(0)
ax.grid(alpha=0.3)

# =====================================================================
# Panel 2 — Prolate spheroid
# =====================================================================
ax = axes[1]

pts_e  = d_ell["pts"]
s1_e   = d_ell["sigma1"]
s2_e   = d_ell["sigma2"]
s1s_e  = d_ell["sigma1_smooth"]
s2s_e  = d_ell["sigma2_smooth"]

beta_v, sm_v, sh_v = analytic_spheroid_at_pts(pts_e)
an_max_v = np.maximum(sm_v, sh_v)
an_min_v = np.minimum(sm_v, sh_v)

# Fine analytic curves
beta_c, sm_c, sh_c = analytic_spheroid_curve()
an_max_c = np.maximum(sm_c, sh_c)
an_min_c = np.minimum(sm_c, sh_c)

# Scatter (subsampled)
stride = max(1, len(beta_v) // 500)
ax.scatter(beta_v[::stride], s1s_e[::stride], s=6, alpha=0.5,
           color="tab:blue", zorder=2, label=r"$\sigma_1$ (smoothed)")
ax.scatter(beta_v[::stride], s2s_e[::stride], s=6, alpha=0.5,
           color="tab:red",  zorder=2, label=r"$\sigma_2$ (smoothed)")
ax.plot(beta_c, an_max_c, "b-",  lw=2, zorder=3, label=r"analytic $\sigma_{\max}$")
ax.plot(beta_c, an_min_c, "r-",  lw=2, zorder=3, label=r"analytic $\sigma_{\min}$")

# Error stats (exclude poles: |beta - π/2| < π/3 ≈ equatorial belt)
belt = np.abs(beta_v - np.pi / 2) < np.pi / 3
err_max = np.abs(s1s_e[belt] - an_max_v[belt]).mean() / an_max_v[belt].mean() * 100
err_min = np.abs(s2s_e[belt] - an_min_v[belt]).mean() / an_min_v[belt].mean() * 100
ax.text(0.97, 0.97,
        f"equatorial belt (smoothed):\n"
        f"$\\sigma_{{\\max}}$ err: {err_max:.1f}%\n"
        f"$\\sigma_{{\\min}}$ err: {err_min:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax.set_xlabel(r"Latitude $\beta$ (0 = pole, $\pi/2$ = equator, $\pi$ = pole)")
ax.set_ylabel("Stress (Pa)")
ax.set_title(r"Spheroid ($a=2,\ b=1$)")
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(0, np.pi)
ax.set_ylim(0)
ax.grid(alpha=0.3)

# =====================================================================
# Panel 3 — Capsule
# =====================================================================
ax = axes[2]

pts_c  = d_cap["pts"]
s1_c   = d_cap["sigma1"]
s2_c   = d_cap["sigma2"]
s1s_c  = d_cap["sigma1_smooth"]
s2s_c  = d_cap["sigma2_smooth"]

# Long axis is x (capsule was rotate_y(90) before solve)
long_ax = pts_c[:, 0]

# Fine analytic reference
z_fine = np.linspace(-(CAP_H + CAP_R), CAP_H + CAP_R, 500)
an_hi_fine, an_lo_fine = analytic_capsule(z_fine)

stride_c = max(1, len(long_ax) // 500)
ax.scatter(long_ax[::stride_c], s1s_c[::stride_c], s=6, alpha=0.5,
           color="tab:blue", zorder=2, label=r"$\sigma_1$ (smoothed)")
ax.scatter(long_ax[::stride_c], s2s_c[::stride_c], s=6, alpha=0.5,
           color="tab:red",  zorder=2, label=r"$\sigma_2$ (smoothed)")
ax.plot(z_fine, an_hi_fine, "b-", lw=2, zorder=3,
        label=r"analytic $\sigma_{\max}$")
ax.plot(z_fine, an_lo_fine, "r-", lw=2, zorder=3,
        label=r"analytic $\sigma_{\min}$")

# Cylinder/cap boundary markers
for xv in [-CAP_H, CAP_H]:
    ax.axvline(xv, color="gray", lw=1, ls=":", zorder=1)
ax.text( CAP_H * 0.4,  ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 450,
        "cylinder", ha="center", va="top", fontsize=7, color="gray")

# Error stats — cylinder body only
cyl = np.abs(long_ax) < CAP_H * 0.85
err_hoop  = np.abs(s1s_c[cyl] - DP * CAP_R / T ).mean() / (DP * CAP_R / T)  * 100
err_axial = np.abs(s2s_c[cyl] - DP * CAP_R / (2*T)).mean() / (DP * CAP_R / (2*T)) * 100
ax.text(0.97, 0.97,
        f"cylinder body (smoothed):\n"
        f"$\\sigma_{{\\rm hoop}}$ err: {err_hoop:.1f}%\n"
        f"$\\sigma_{{\\rm axial}}$ err: {err_axial:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax.set_xlabel("Long-axis coordinate $x$")
ax.set_ylabel("Stress (Pa)")
ax.set_title(r"Capsule ($R=1$, $H=2$)")
ax.legend(fontsize=7)
ax.set_ylim(0)
ax.grid(alpha=0.3)

# ---- save ----------------------------------------------------------------- #
plt.tight_layout()
os.makedirs("out", exist_ok=True)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_PNG}")

# ---- print summary table -------------------------------------------------- #
print("\nBenchmark summary (Laplacian-smoothed, subdiv-4)")
print(f"{'Case':<22} {'Field':<16} {'Computed':>10} {'Analytic':>10} {'Err%':>8}")
print("-" * 70)
# sphere
sm_sph = (d_sph["sigma1_smooth"] + d_sph["sigma2_smooth"]) / 2
print(f"{'Sphere':<22} {'mean stress':16} {sm_sph.mean():10.1f} {sig_ref:10.1f} "
      f"{abs(sm_sph.mean()-sig_ref)/sig_ref*100:8.1f}%")
print(f"{'':22} {'dev std (Pa)':16} {(d_sph['sigma1_smooth']-d_sph['sigma2_smooth']).std()/2:10.2f} "
      f"{'0':>10} {'—':>8}")
# spheroid equatorial belt
s1b, s2b = d_ell["sigma1_smooth"][belt], d_ell["sigma2_smooth"][belt]
print(f"{'Spheroid (equat.)':<22} {'sigma_max':16} {s1b.mean():10.1f} {an_max_v[belt].mean():10.1f} "
      f"{abs(s1b.mean()-an_max_v[belt].mean())/an_max_v[belt].mean()*100:8.1f}%")
print(f"{'':22} {'sigma_min':16} {s2b.mean():10.1f} {an_min_v[belt].mean():10.1f} "
      f"{abs(s2b.mean()-an_min_v[belt].mean())/an_min_v[belt].mean()*100:8.1f}%")
# capsule cylinder
print(f"{'Capsule (cylinder)':<22} {'sigma_hoop':16} {s1s_c[cyl].mean():10.1f} {DP*CAP_R/T:10.1f} "
      f"{err_hoop:8.1f}%")
print(f"{'':22} {'sigma_axial':16} {s2s_c[cyl].mean():10.1f} {DP*CAP_R/(2*T):10.1f} "
      f"{err_axial:8.1f}%")
