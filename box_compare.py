"""Box-plot comparison of per-vertex membrane stress on the sphere and prolate ellipsoid:
Local (M1, axisymmetric two-curvature) vs cMSM (M2, GFDM) vs Analytical.

Loads the saved final-sim fields (out/final/sim0*.npz), recomputes the analytic
axisymmetric reference at each vertex, excludes the long-axis poles, and draws box plots
of sigma_max and sigma_min for each method. -> out/final/box_compare.png

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe box_compare.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from membrane_stress_fd import analytic_axisym

DP, T, R = 20.0, 0.05, 1.0
OUTDIR = "out/final"
CASES = [
    ("Sphere", R, R,
     "sim01_sphere_uniform_local.npz", "sim02_sphere_uniform_cmsm.npz"),
    ("Prolate ellipsoid (2:1)", 2 * R, R,
     "sim07_ellipsoid_uniform_local.npz", "sim08_ellipsoid_uniform_cmsm.npz"),
]
COLORS = ["#4C9F70", "#E07B39", "#888888"]   # Local, cMSM, Analytical
LABELS = ["Local\n(M1)", "cMSM\n(M2)", "Analytical"]


def load(tag):
    d = np.load(os.path.join(OUTDIR, tag))
    return d["pts"], d["sigma_max"], d["sigma_min"]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    print(f"{'case':<26}{'field':<10}{'method':<12}"
          f"{'median':>10}{'IQR':>10}{'rel-err(med)':>14}")
    for row, (name, a, b, floc, fcmsm) in enumerate(CASES):
        pts, l_max, l_min = load(floc)
        _, c_max, c_min = load(fcmsm)

        cen = pts.mean(0)
        rad = pts - cen
        rad /= np.linalg.norm(rad, axis=1, keepdims=True)
        mask = np.abs(rad[:, 0]) < 0.92                 # exclude long-axis poles

        sm, sh = analytic_axisym(pts, DP, T, a, b)
        a_max = np.maximum(sm, sh); a_min = np.minimum(sm, sh)

        for col, (trip, an, field) in enumerate([
                ([l_max[mask], c_max[mask], a_max[mask]], a_max[mask], "sigma_max"),
                ([l_min[mask], c_min[mask], a_min[mask]], a_min[mask], "sigma_min")]):
            ax = axes[row, col]
            bp = ax.boxplot(trip, tick_labels=LABELS, showfliers=False,
                            patch_artist=True, widths=0.55, whis=(5, 95),
                            medianprops=dict(color="black", lw=2.2),
                            whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3),
                            boxprops=dict(lw=1.3))
            for patch, c in zip(bp["boxes"], COLORS):
                patch.set_facecolor(c); patch.set_alpha(0.85)
            anmed = np.median(an)
            ax.axhline(anmed, ls="--", c="crimson", lw=1.5, zorder=0,
                       label=f"analytic = {anmed:.0f}")
            top = 1.12 * max(np.percentile(arr, 95) for arr in trip)
            ax.set_ylim(0, top)
            ax.set_title(f"{name} — {field}", fontsize=13, pad=8)
            ax.set_ylabel("membrane stress  (Pa)", fontsize=11)
            ax.tick_params(labelsize=10)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.legend(loc="lower right", fontsize=9, framealpha=0.0)

            # console summary
            for lab, arr in zip(["Local", "cMSM", "Analytical"], trip):
                q1, med, q3 = np.percentile(arr, [25, 50, 75])
                rel = np.median(np.abs(arr - an) / np.maximum(np.abs(an), 1e-12))
                print(f"{name:<26}{field:<10}{lab:<12}"
                      f"{med:>10.2f}{q3 - q1:>10.2f}{rel:>13.1%}")

    fig.suptitle(f"Membrane stress per vertex: Local vs cMSM vs Analytical "
                 f"(dp={DP:.0f} Pa, t={T}, whiskers=5–95%, poles excluded)", y=1.0)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "box_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
