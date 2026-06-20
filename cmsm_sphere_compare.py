"""One-to-one reproduction of cMSM Supplementary Fig. 15 (spherical cap) for our
closed-surface stress FEM.

cMSM Fig. 15 panels (open spherical cap, pinned edge, global parametrization):
  (a) relative error [%] vs lambda_t  -> U-shaped, <1% in a mid band
  (c) force-balance residual vs lambda_t -> flat then sharp rise at the corner
  (d) L-curve: regularization norm vs residual -> corner ~ optimal lambda

We reproduce the SAME three diagnostics on the CLOSED unit sphere with our single
Tikhonov weight lambda (FEM-native roughness R). The closed sphere has no boundary,
so the true field is constant isotropic sigma = dp*R/(2t); any anisotropy is pure
discretization error. The point of this script is to check whether our (residual, roughness)
tradeoff still produces a cMSM-style L-curve corner, and how our error-vs-lambda compares.

Their rendered Fig. 15 page (out/cmsm_ref/fig15_page.png) is shown alongside for a
direct visual one-to-one.

-> out/cmsm_sphere_compare.png

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe cmsm_sphere_compare.py --subdiv 4
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vedo

from membrane_stress_fem import solve_membrane_fem

R, DP, T = 1.0, 20.0, 0.05
SIGMA_TRUE = DP * R / (2.0 * T)        # exact isotropic principal stress on the closed sphere
TRACE_TRUE = 2.0 * SIGMA_TRUE

PDF = "41467_2023_38879_MOESM1_ESM.pdf"
REF_PNG = "out/cmsm_ref/grid_abcd.png"


def ensure_ref():
    """Crop cMSM Supplementary Fig. 15's a-d plot grid out of the supplement PDF (page index
    29) so the comparison figure is self-contained. No-op if already present."""
    if os.path.exists(REF_PNG) or not os.path.exists(PDF):
        return
    try:
        import fitz                       # pymupdf
        from PIL import Image
    except ImportError:
        print("  (pymupdf/Pillow missing -> skipping cMSM reference panel)")
        return
    os.makedirs("out/cmsm_ref", exist_ok=True)
    pix = fitz.open(PDF)[29].get_pixmap(dpi=220)
    hi = "out/cmsm_ref/fig15_hi.png"
    pix.save(hi)
    Image.open(hi).crop((150, 185, 1785, 955)).save(REF_PNG)   # a,b,c,d grid (excludes glyph e)
    os.remove(hi)
    print(f"  wrote {REF_PNG}")


def sweep(subdiv: int, lams: np.ndarray):
    mesh = vedo.IcoSphere(r=R, subdivisions=subdiv)
    rows, saved = [], {}
    for lam in lams:
        r = solve_membrane_fem(mesh, DP, T, depth=3, lam=float(lam), raw=False, solver="direct")
        s1, s2 = r["sigma1"], r["sigma2"]
        smax, smin = np.maximum(s1, s2), np.minimum(s1, s2)
        tr = s1 + s2
        # full principal-stress error vs the exact isotropic solution
        full_err = (np.linalg.norm(np.r_[smax - SIGMA_TRUE, smin - SIGMA_TRUE])
                    / np.linalg.norm(np.r_[np.full_like(smax, SIGMA_TRUE),
                                           np.full_like(smin, SIGMA_TRUE)]))
        trace_err = np.linalg.norm(tr - TRACE_TRUE) / np.linalg.norm(np.full_like(tr, TRACE_TRUE))
        rows.append(dict(lam=float(lam), trace_err=trace_err, full_err=full_err,
                         resid=r["resid"], reg=r["reg_norm"]))
        saved[float(lam)] = r
        print(f"  lam={lam:9.4g}  trace_err={trace_err:7.3%}  full_err={full_err:7.3%}  "
              f"resid={r['resid']:.3e}  reg={r['reg_norm']:.4e}")
    return mesh, rows, saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--nlam", type=int, default=18)
    args = ap.parse_args()

    ensure_ref()
    lams = np.geomspace(1e-4, 50.0, args.nlam)
    print(f"Closed-sphere lambda sweep (subdiv-{args.subdiv}, dp={DP}, t={T}); "
          f"sigma_true={SIGMA_TRUE:.1f} Pa, trace_true={TRACE_TRUE:.1f} Pa")
    mesh, rows, saved = sweep(args.subdiv, lams)

    lam = np.array([r["lam"] for r in rows])
    trace_err = np.array([r["trace_err"] for r in rows])
    full_err = np.array([r["full_err"] for r in rows])
    resid = np.array([r["resid"] for r in rows])
    reg = np.array([r["reg"] for r in rows])
    lam_opt = lam[int(np.argmin(full_err))]
    print(f"min full-sigma error = {full_err.min():.3%} at lambda = {lam_opt:.4g}")

    os.makedirs("out", exist_ok=True)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    # ---- top row: cMSM Fig 15 reference image (their open cap) ----
    axref = fig.add_subplot(gs[0, :])
    if os.path.exists(REF_PNG):
        axref.imshow(plt.imread(REF_PNG))
    axref.axis("off")
    axref.set_title("cMSM Supplementary Fig. 15 — OPEN spherical cap, pinned edge, multi λc "
                    "(a: error vs λt   b: edge σ   c: residual vs λt   d: L-curve)", fontsize=10)

    # ---- bottom row: our closed-sphere reproduction ----
    axa = fig.add_subplot(gs[1, 0])
    axa.loglog(lam, 100 * full_err, "s-", label="full σ error", color="C0")
    axa.loglog(lam, 100 * trace_err, "o-", label="trace(σ) error", color="C1")
    axa.axvline(lam_opt, color="k", ls=":", lw=1)
    axa.set_xlabel("λ"); axa.set_ylabel("relative error [%]")
    axa.set_title("(a) OUR error vs λ (closed sphere)")
    axa.legend(fontsize=8); axa.grid(True, which="both", alpha=0.3)

    axc = fig.add_subplot(gs[1, 1])
    axc.loglog(lam, resid, "o-", color="C2")
    axc.axvline(lam_opt, color="k", ls=":", lw=1)
    axc.set_xlabel("λ"); axc.set_ylabel("‖Ks−b‖ / ‖b‖")
    axc.set_title("(c) OUR residual vs λ")
    axc.grid(True, which="both", alpha=0.3)

    axd = fig.add_subplot(gs[1, 2])
    axd.loglog(resid, reg, "o-", color="C3")
    iopt = int(np.argmin(full_err))
    axd.loglog(resid[iopt], reg[iopt], "*", ms=18, color="k",
               label=f"min-error λ={lam_opt:.3g}")
    for i in range(len(lam)):
        axd.annotate(f"{lam[i]:.2g}", (resid[i], reg[i]), fontsize=6,
                     textcoords="offset points", xytext=(3, 3))
    axd.set_xlabel("residual ‖Ks−b‖ / ‖b‖"); axd.set_ylabel("roughness ‖Rs‖")
    axd.set_title("(d) OUR L-curve (★ = min-error point ≈ corner)")
    axd.legend(fontsize=8); axd.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"cMSM Fig. 15 vs our closed-sphere FEM  "
                 f"(subdiv-{args.subdiv}, Δp={DP:.0f} Pa, t={T}); min err {full_err.min():.2%} "
                 f"@ λ={lam_opt:.3g}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("out/cmsm_sphere_compare.png", dpi=130)
    print("Saved out/cmsm_sphere_compare.png")


if __name__ == "__main__":
    main()
