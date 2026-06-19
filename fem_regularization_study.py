"""cMSM Supplementary-Fig-15-style regularization analysis for the stress-based FEM.

Performed on the prolate spheroid (2:1) -- the closed-surface analogue of cMSM's spherical
cap with genuine anisotropy, so that over-regularization actually penalizes the solution and
a tradeoff/optimum appears (on a closed sphere the true field is constant, so more smoothing
always helps and no optimum exists). Sweeps the Tikhonov weight lambda and produces:
  (a) relative error of trace(sigma)=sigma1+sigma2 AND of the full principal stress vs lambda
      (the trace error stays lower -- the cMSM observation);
  (b) sigma_max vs latitude for representative lambda values + the analytic curve;
  (c) force-balance residual ||K s - b|| / ||b|| vs lambda (a diagnostic, not the objective);
  (d) the L-curve: roughness ||R s|| vs residual, parametrized by lambda (corner ~ optimum);
  (e) 3D spheroid coloured by trace(sigma) with principal-stress crosses, inferred (red) vs the
      analytic axisymmetric solution (green) -- mirrors cMSM's arrow-pair stress glyph.

-> out/fem_regularization.png

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe fem_regularization_study.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vedo

from membrane_stress_fem import solve_membrane_fem
from membrane_stress_fd import analytic_axisym

R, DP, T = 1.0, 20.0, 0.05
A, B = 2.0, 1.0                 # prolate spheroid semi-axes (long axis = x)


def _analytic(pts):
    """Per-vertex analytic (sigma_max, sigma_min, trace) for the prolate spheroid."""
    sm, sh = analytic_axisym(pts, DP, T, a=A, b=B)       # meridional, hoop
    return np.maximum(sm, sh), np.minimum(sm, sh), sm + sh, sm, sh


def _belt(pts):
    """Equatorial belt mask (exclude the umbilic long-axis poles where the frame degenerates)."""
    rad = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return np.abs(rad[:, 0]) < 0.9


def sweep(subdiv=4, n_lam=10):
    mesh = vedo.IcoSphere(r=R, subdivisions=subdiv).scale([A, 1.0, 1.0])
    lams = np.geomspace(0.005, 0.5, n_lam)
    rows, saved = [], {}
    for lam in lams:
        r = solve_membrane_fem(mesh, DP, T, depth=3, lam=lam, raw=False, solver="direct")
        s1, s2 = r["sigma1"], r["sigma2"]
        smax, smin = np.maximum(s1, s2), np.minimum(s1, s2)
        an_max, an_min, an_tr, _, _ = _analytic(r["pts"])
        m = _belt(r["pts"])
        tr = (s1 + s2)[m]
        trace_err = np.linalg.norm(tr - an_tr[m]) / np.linalg.norm(an_tr[m])
        full_err = (np.linalg.norm(np.r_[smax[m] - an_max[m], smin[m] - an_min[m]])
                    / np.linalg.norm(np.r_[an_max[m], an_min[m]]))
        rows.append(dict(lam=lam, trace_err=trace_err, full_err=full_err,
                         resid=r["resid"], reg=r["reg_norm"]))
        saved[lam] = r
        print(f"  lam={lam:.4f}  trace_err={trace_err:.3%}  full_err={full_err:.3%}  "
              f"resid={r['resid']:.2e}  reg={r['reg_norm']:.3e}")
    return mesh, lams, rows, saved


def render_glyph(mesh, r, out_png, n_glyph=160):
    """3D spheroid coloured by trace(sigma) with inferred (red) and analytic (green) crosses."""
    pts, d1, d2, s1, s2 = r["pts"], r["d1"], r["d2"], r["sigma1"], r["sigma2"]
    nrm = r["normals"]
    tr = s1 + s2
    an_max, an_min, _, sm, sh = _analytic(pts)
    diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    base = diag * 0.05
    ref = np.percentile(an_max, 98)
    idx = np.arange(len(pts))[:: max(1, len(pts) // n_glyph)]

    m = mesh.clone()
    m.pointdata["trace"] = tr
    clim = (float(np.percentile(tr, 2)), float(np.percentile(tr, 98)))
    m.cmap("coolwarm", "trace", vmin=clim[0], vmax=clim[1]).add_scalarbar(title="trace(sigma) (Pa)")
    objs = [m]
    # inferred crosses (red): arms proportional to |sigma_i|
    for d, s in ((d1, s1), (d2, s2)):
        h = (base * np.clip(np.abs(s) / ref, 0.0, 1.3))[:, None]
        objs.append(vedo.Lines((pts - h * d)[idx], (pts + h * d)[idx], c="red", lw=3, alpha=0.9))
    # analytic crosses (green): meridional / hoop directions, arms proportional to analytic sigma
    xhat = np.array([1.0, 0.0, 0.0])
    hoop = np.cross(np.broadcast_to(xhat, pts.shape), pts)
    hn = np.linalg.norm(hoop, axis=1, keepdims=True)
    hoop = np.divide(hoop, hn, out=np.zeros_like(hoop), where=hn > 1e-9)
    merid = np.cross(nrm, hoop)
    for d, s in ((merid, sm), (hoop, sh)):
        h = (base * np.clip(s / ref, 0.0, 1.3))[:, None]
        objs.append(vedo.Lines((pts - h * d)[idx], (pts + h * d)[idx], c="green", lw=2, alpha=0.6))
    p = vedo.Plotter(offscreen=True, size=(1100, 850))
    p.show(*objs, axes=0, azimuth=20, elevation=-15)
    p.screenshot(out_png)
    p.close()


def main():
    print("Sweeping lambda on the prolate spheroid (subdiv-4)...")
    mesh, lams, rows, saved = sweep()
    trace_err = np.array([r["trace_err"] for r in rows])
    full_err = np.array([r["full_err"] for r in rows])
    resid = np.array([r["resid"] for r in rows])
    reg = np.array([r["reg"] for r in rows])
    lam_opt = float(lams[np.argmin(full_err)])
    print(f"Optimal lambda (min full-sigma error) = {lam_opt:.4f}")

    glyph_png = "out/_fem_glyph_tmp.png"
    os.makedirs("out", exist_ok=True)
    render_glyph(mesh, saved[lam_opt], glyph_png)

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])

    axa = fig.add_subplot(gs[0, 0])
    axa.loglog(lams, 100 * trace_err, "o-", label="trace(σ) error")
    axa.loglog(lams, 100 * full_err, "s-", label="full σ error")
    axa.axvline(lam_opt, color="k", ls=":", lw=1)
    axa.set_xlabel("λ"); axa.set_ylabel("relative error [%]")
    axa.set_title("(a) Relative error vs λ"); axa.legend(); axa.grid(True, which="both", alpha=0.3)

    axb = fig.add_subplot(gs[0, 1])
    pts0 = saved[lams[0]]["pts"]
    beta = np.degrees(np.arctan2(np.sqrt(pts0[:, 1] ** 2 + pts0[:, 2] ** 2) / B, pts0[:, 0] / A))
    order = np.argsort(beta)
    an_max0, _, _, _, _ = _analytic(pts0)
    for lam in (lams[0], lam_opt, lams[-1]):
        smax = np.maximum(saved[lam]["sigma1"], saved[lam]["sigma2"])
        axb.plot(beta, smax, ".", ms=2.5, alpha=0.45, label=f"λ={lam:.3f}")
    axb.plot(beta[order], an_max0[order], "k-", lw=2, label="analytic σ_max")
    axb.set_xlabel("latitude β (deg)"); axb.set_ylabel("σ_max (Pa)")
    axb.set_title("(b) σ_max vs latitude"); axb.legend(fontsize=7); axb.grid(True, alpha=0.3)

    axc = fig.add_subplot(gs[1, 0])
    axc.loglog(lams, resid, "o-", color="C2"); axc.axvline(lam_opt, color="k", ls=":", lw=1)
    axc.set_xlabel("λ"); axc.set_ylabel("‖Ks−b‖ / ‖b‖")
    axc.set_title("(c) Force-balance residual vs λ"); axc.grid(True, which="both", alpha=0.3)

    axd = fig.add_subplot(gs[1, 1])
    axd.loglog(resid, reg, "o-", color="C3")
    for i, lam in enumerate(lams):
        axd.annotate(f"{lam:.3f}", (resid[i], reg[i]), fontsize=6,
                     textcoords="offset points", xytext=(3, 3))
    axd.set_xlabel("residual ‖Ks−b‖ / ‖b‖"); axd.set_ylabel("roughness ‖Rs‖")
    axd.set_title("(d) L-curve (corner ≈ optimal λ)"); axd.grid(True, which="both", alpha=0.3)

    axe = fig.add_subplot(gs[2, :])
    axe.imshow(plt.imread(glyph_png)); axe.axis("off")
    axe.set_title("(e) trace(σ) colour + principal-stress crosses: red = inferred, "
                  "green = analytic (meridional/hoop)")

    fig.suptitle(f"Stress-FEM regularization analysis — prolate spheroid 2:1 "
                 f"(subdiv-4, Δp={DP:.0f} Pa, t={T}); optimal λ = {lam_opt:.3f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig("out/fem_regularization.png", dpi=130)
    os.remove(glyph_png)
    print("Saved out/fem_regularization.png")


if __name__ == "__main__":
    main()
