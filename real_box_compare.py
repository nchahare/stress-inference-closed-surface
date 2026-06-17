"""Box-plot comparison of HH17 vs HH20 per-vertex membrane stress.

The comparison axis is the MESH (HH17 vs HH20): each panel places the two meshes side by
side. Rows = method (Local M1 isotropic mean-curvature, cMSM M2 GFDM+Laplacian); columns =
sigma_max / sigma_min. Fields are signed (cMSM sigma_min is compressive), so the y-axis is
not floored at 0; a zero line marks the tension/compression boundary. Local has 1/H blow-up
tails on irregular meshes, suppressed by whiskers=5-95% and no fliers. Each row shares
y-limits across its two columns so HH17<->HH20 and sigma_max<->sigma_min are comparable
within a method.

-> out/final/real_box_compare.png

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe real_box_compare.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "out/final"
MESHES = ["HH17", "HH20"]
MESH_COLORS = {"HH17": "#3B7DD8", "HH20": "#E07B39"}
METHODS = [("Local (M1)", "local"), ("cMSM (M2)", "cmsm")]
FIELDS = ["sigma_max", "sigma_min"]


def load(mesh, method):
    d = np.load(os.path.join(OUTDIR, f"{mesh.lower()}_{method}.npz"))
    return {f: d[f] for f in FIELDS}


def main():
    # data[method][mesh] = {field: array}
    data = {m: {mesh: load(mesh, m) for mesh in MESHES} for _, m in METHODS}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    print(f"{'method':<12}{'mesh':<8}{'field':<10}{'median':>12}{'IQR':>12}")
    for row, (mlabel, m) in enumerate(METHODS):
        # per-row (method) shared y-limits across both fields & meshes, 5-95 pct
        vals = []
        for mesh in MESHES:
            for f in FIELDS:
                arr = data[m][mesh][f]
                vals += [np.percentile(arr, 5), np.percentile(arr, 95)]
        lo, hi = min(vals), max(vals)
        pad = 0.10 * (hi - lo)
        ylim = (lo - pad, hi + pad)

        for col, f in enumerate(FIELDS):
            ax = axes[row, col]
            trip = [data[m][mesh][f] for mesh in MESHES]
            bp = ax.boxplot(trip, tick_labels=MESHES, showfliers=False,
                            patch_artist=True, widths=0.5, whis=(5, 95),
                            medianprops=dict(color="black", lw=2.2),
                            whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3),
                            boxprops=dict(lw=1.3))
            for patch, mesh in zip(bp["boxes"], MESHES):
                patch.set_facecolor(MESH_COLORS[mesh]); patch.set_alpha(0.85)
            ax.axhline(0, ls="--", c="crimson", lw=1.4, zorder=0, label="zero stress")
            ax.set_ylim(*ylim)
            ax.set_title(f"{mlabel} — {f}", fontsize=13, pad=8)
            ax.set_ylabel("membrane stress  (Pa)", fontsize=11)
            ax.tick_params(labelsize=10)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.0)

            for mesh, arr in zip(MESHES, trip):
                q1, med, q3 = np.percentile(arr, [25, 50, 75])
                print(f"{mlabel:<12}{mesh:<8}{f:<10}{med:>12.3g}{q3 - q1:>12.3g}")

    fig.suptitle("HH17 vs HH20 membrane stress per vertex "
                 "(dp=20 Pa, t=0.05 placeholder → magnitude uncalibrated, whiskers 5–95%)",
                 y=1.0)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "real_box_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
