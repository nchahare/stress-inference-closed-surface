"""Linearity test (§9.3): σ ∝ dp/t for the GFDM membrane solve.

Runs the solver at multiple (Δp, t) combinations on the sphere and prolate
spheroid.  Two routes reach the same dp/t value: varying Δp with fixed t,
and varying t with fixed Δp.  If the resulting mean stresses coincide on a
σ vs dp/t scatter, linearity is confirmed.

Saves: out/linearity_test.png

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe linearity_test.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vedo

from membrane_stress_fd import solve_membrane, analytic_axisym

# ---- geometry ------------------------------------------------------------ #
R  = 1.0
A  = 2.0   # spheroid long semi-axis (x)
B  = 1.0
SUBDIV = 4
DEPTH  = 3
LAM    = 0.05

# ---- parameter combinations ---------------------------------------------- #
# Each entry: (dp, t).  Two groups share a dp/t value achieved different ways:
#   dp/t = 200  →  (dp=10, t=0.05)  and  (dp=20, t=0.10)
#   dp/t = 800  →  (dp=40, t=0.05)  and  (dp=20, t=0.025)
vary_dp = [(5, 0.05), (10, 0.05), (20, 0.05), (40, 0.05)]   # fix t, vary dp
vary_t  = [(20, 0.10), (20, 0.05), (20, 0.025)]              # fix dp, vary t
# Note: (20, 0.05) appears in both groups; run once, assign to both.
all_combos = list(dict.fromkeys(vary_dp + vary_t))           # deduplicate, keep order

print("Building meshes …")
sphere   = vedo.IcoSphere(r=R, subdivisions=SUBDIV)
spheroid = vedo.IcoSphere(r=R, subdivisions=SUBDIV).scale([A, 1.0, 1.0])

# ---- run solves ---------------------------------------------------------- #
results_sph = {}   # (dp, t) → mean σ₁
results_ell = {}   # (dp, t) → (mean σ₁, mean σ₂, analytic σ_max mean, analytic σ_min mean)

for dp, t in all_combos:
    print(f"  Sphere   dp={dp:5.2f}  t={t:.3f}  (dp/t={dp/t:.0f})")
    rs = solve_membrane(sphere, dp, t, depth=DEPTH, lam=LAM)
    results_sph[(dp, t)] = rs["sigma1"].mean()

    print(f"  Spheroid dp={dp:5.2f}  t={t:.3f}  (dp/t={dp/t:.0f})")
    re = solve_membrane(spheroid, dp, t, depth=DEPTH, lam=LAM)
    pts = re["pts"]
    sm_an, sh_an = analytic_axisym(pts, dp, t, a=A, b=B)
    an_max = np.maximum(sm_an, sh_an)
    an_min = np.minimum(sm_an, sh_an)
    results_ell[(dp, t)] = (
        re["sigma1"].mean(), re["sigma2"].mean(),
        an_max.mean(),       an_min.mean(),
    )

# ---- collect into arrays ------------------------------------------------- #
def collect(combos, results_dict, col):
    return np.array([(dp / t, results_dict[(dp, t)][col] if isinstance(results_dict[(dp, t)], tuple)
                      else results_dict[(dp, t)])
                     for dp, t in combos if (dp, t) in results_dict])

def grp(combos, res, col=None):
    out = []
    for dp, t in combos:
        if (dp, t) not in res:
            continue
        val = res[(dp, t)]
        if isinstance(val, tuple):
            val = val[col]
        out.append((dp / t, val))
    return np.array(out)

# ---- analytic reference lines -------------------------------------------- #
dpt_fine   = np.linspace(0, 900, 200)
sph_line   = 0.5 * R * dpt_fine            # σ = ΔpR/(2t)

# Spheroid global analytic mean: compute once at (dp=1,t=1) and scale
pts_ell = spheroid.coordinates
sm_norm, sh_norm = analytic_axisym(pts_ell, dp=1.0, t=1.0, a=A, b=B)
an_max_slope = np.maximum(sm_norm, sh_norm).mean()   # σ_max/unit(dp/t)
an_min_slope = np.minimum(sm_norm, sh_norm).mean()

# ---- plot ---------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    r"Linearity test: $\sigma \propto \Delta p/t$ (GFDM, IcoSphere subdiv-4, "
    r"$\lambda=0.05$, no smoothing)",
    fontsize=11,
)

marker_kw = dict(vary_dp=dict(marker="o", s=80, zorder=4),
                 vary_t =dict(marker="s", s=80, zorder=4))
colors    = dict(sigma1="tab:blue", sigma2="tab:red")

# ---- Panel 1: Sphere ---------------------------------------------------- #
ax = axes[0]
ax.plot(dpt_fine, sph_line, "k-", lw=2, label=r"analytic $\sigma = R\,\Delta p / 2t$", zorder=3)

g = grp(vary_dp, results_sph)
ax.scatter(g[:, 0], g[:, 1], color="tab:blue", label=r"vary $\Delta p$ ($t=0.05$)", **marker_kw["vary_dp"])
g2 = grp(vary_t, results_sph)
ax.scatter(g2[:, 0], g2[:, 1], color="tab:orange", label=r"vary $t$ ($\Delta p=20$)", **marker_kw["vary_t"])

# annotate (dp, t) next to each point
for dp, t in vary_dp:
    if (dp, t) in results_sph:
        ax.annotate(f"({dp},{t})", (dp/t, results_sph[(dp,t)]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7, color="tab:blue")
for dp, t in vary_t:
    if (dp, t) in results_sph and (dp, t) not in vary_dp:
        ax.annotate(f"({dp},{t})", (dp/t, results_sph[(dp,t)]),
                    textcoords="offset points", xytext=(5, -12), fontsize=7, color="tab:orange")

ax.set_xlabel(r"$\Delta p\,/\,t$  (Pa/m)")
ax.set_ylabel(r"Mean $\sigma_1$  (Pa)")
ax.set_title(r"Sphere ($R=1$)")
ax.legend(fontsize=8)
ax.set_xlim(0); ax.set_ylim(0)
ax.grid(alpha=0.3)

# ---- Panel 2: Spheroid -------------------------------------------------- #
ax = axes[1]
ax.plot(dpt_fine, an_max_slope * dpt_fine, "b-",  lw=2,
        label=r"analytic $\sigma_{\max}$ global mean", zorder=3)
ax.plot(dpt_fine, an_min_slope * dpt_fine, "r-",  lw=2,
        label=r"analytic $\sigma_{\min}$ global mean", zorder=3)

g1_max = grp(vary_dp, results_ell, col=0)
g1_min = grp(vary_dp, results_ell, col=1)
ax.scatter(g1_max[:, 0], g1_max[:, 1], color="tab:blue",
           label=r"$\sigma_1$ vary $\Delta p$", **marker_kw["vary_dp"])
ax.scatter(g1_min[:, 0], g1_min[:, 1], color="tab:red",
           label=r"$\sigma_2$ vary $\Delta p$", **marker_kw["vary_dp"])

g2_max = grp(vary_t, results_ell, col=0)
g2_min = grp(vary_t, results_ell, col=1)
ax.scatter(g2_max[:, 0], g2_max[:, 1], color="tab:blue", alpha=0.5,
           label=r"$\sigma_1$ vary $t$", **marker_kw["vary_t"])
ax.scatter(g2_min[:, 0], g2_min[:, 1], color="tab:red", alpha=0.5,
           label=r"$\sigma_2$ vary $t$", **marker_kw["vary_t"])

ax.set_xlabel(r"$\Delta p\,/\,t$  (Pa/m)")
ax.set_ylabel(r"Mean $\sigma$  (Pa)")
ax.set_title(r"Prolate spheroid ($a=2$, $b=1$)")
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(0); ax.set_ylim(0)
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = "out/linearity_test.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}")

# ---- print ratio table --------------------------------------------------- #
print("\nLinearity check: sigma(dp,t) / (dp/t)  [should be constant]")
print(f"{'dp':>6} {'t':>6} {'dp/t':>8} {'s_sph/(dp/t)':>15} {'s_ell s1/(dp/t)':>18}")
print("-" * 60)
for dp, t in all_combos:
    dpt = dp / t
    s_sph = results_sph[(dp, t)]
    s_ell = results_ell[(dp, t)][0]
    print(f"{dp:6.2f} {t:6.3f} {dpt:8.1f} {s_sph/dpt:15.4f} {s_ell/dpt:18.4f}")
