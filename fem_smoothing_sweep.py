"""cMSM-Fig-15-style sweep of the FEM Tikhonov weight lambda at several Laplacian
post-smoothing iteration counts, for several closed surfaces (sphere, prolate spheroid,
oblate spheroid, capsule).

Two regularization knobs act on the closed-surface null modes: the Tikhonov weight
lambda (inside the solve) and a Laplacian (umbrella) post-smoothing of the stress field
(`iters` passes, blend alpha). For each shape this sweeps lambda for a few `iters` values
and reports, in the style of Marin-Llaurado et al. (2023) Supplementary Fig. 15:

  (a) relative error of the principal stress vs lambda, one curve per iters
  (b) sigma1 (= sigma_max) vs latitude/axial coordinate at the best lambda, one series per
      iters, with the analytic solution overlaid
  (c) force-balance residual vs lambda, one curve per iters
  (d) L-curve: regularization functional ||R s|| vs relative residual, one curve per iters

IMPORTANT: the Laplacian smoothing is applied to the DOF fields (p,q,r) -- the local-frame
tensor components -- and the residual ||K s - b||/||b|| and roughness ||R s|| are then
RECOMPUTED on the smoothed DOFs, so both respond to `iters` exactly as lambda does.
sigma1,sigma2 for the error/latitude panels come from the same smoothed DOFs, so all panels
are mutually consistent.

Per-vertex fields for every (lambda, iters) are saved to
out/fem_smoothing_sweep_<shape>.npz for later 3D / glyph plotting.

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe fem_smoothing_sweep.py
    & .../python.exe fem_smoothing_sweep.py --shapes sphere prolate oblate capsule --subdiv 4
"""

from __future__ import annotations

import argparse
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vedo

from membrane_stress_fem import (assemble_fem, fem_roughness_operator,
                                  _frames_outward, _principal)
from membrane_stress_fd import analytic_axisym
from membrane_stress_fd_v2 import make_capsule

DP, T = 20.0, 0.05
ALPHA = 0.5                            # Laplacian blend per pass (iters is the swept knob)
CAP_R, CAP_H = 1.0, 2.0
CAP_NTHETA, CAP_NPHI = 40, 14

# shape -> geometry parameters
SHAPES = {
    "sphere":  dict(kind="spheroid", A=1.0, B=1.0),
    "prolate": dict(kind="spheroid", A=2.0, B=1.0),   # long axis = x
    "oblate":  dict(kind="spheroid", A=0.5, B=1.0),   # flattened along x
    "capsule": dict(kind="capsule"),
}


def build_mesh(shape: str, subdiv: int) -> vedo.Mesh:
    cfg = SHAPES[shape]
    if cfg["kind"] == "spheroid":
        return vedo.IcoSphere(r=1.0, subdivisions=subdiv).scale([cfg["A"], cfg["B"], cfg["B"]])
    return make_capsule(R=CAP_R, H=CAP_H, ntheta=CAP_NTHETA, nphi=CAP_NPHI)


def analytic_coord_mask(shape: str, pts: np.ndarray):
    """Per-vertex analytic (sigma_max, sigma_min, trace), an abscissa coordinate for panel
    (b), its label, and a mask excluding frame-degenerate / discontinuous vertices."""
    cfg = SHAPES[shape]
    if cfg["kind"] == "spheroid":
        A, B = cfg["A"], cfg["B"]
        sm, sh = analytic_axisym(pts, DP, T, a=A, b=B)             # meridional, hoop
        an_max, an_min = np.maximum(sm, sh), np.minimum(sm, sh)
        rho = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2)
        coord = np.degrees(np.arctan2(rho / B, pts[:, 0] / A))     # latitude beta
        label = "latitude β (deg)"
        if A == B:                                                 # sphere: no umbilic issue
            mask = np.ones(len(pts), bool)
        else:                                                      # exclude umbilic x-poles
            radial = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            mask = np.abs(radial[:, 0]) < 0.9
        return an_max, an_min, sm + sh, coord, label, mask

    # capsule: cylinder hoop = dp*R/t, axial = dp*R/2t; caps isotropic dp*R/2t
    z = pts[:, 2]
    s_hoop, s_axial = DP * CAP_R / T, DP * CAP_R / (2.0 * T)
    on_cyl = np.abs(z) <= CAP_H
    an_max = np.where(on_cyl, s_hoop, s_axial)
    an_min = np.full(len(pts), s_axial)
    band = 0.12
    mask = (np.abs(z) < CAP_H + CAP_R - band) & (np.abs(np.abs(z) - CAP_H) > band)
    return an_max, an_min, an_max + an_min, z, "axial z", mask


def one_ring_avg(mesh: vedo.Mesh) -> sp.csr_matrix:
    adlist = mesh.compute_adjacency()
    n = mesh.npoints
    rows, cols, vals = [], [], []
    for i in range(n):
        nb = np.asarray(mesh.find_adjacent_vertices(i, depth=1, adjacency_list=adlist), dtype=int)
        nb = nb[nb != i]
        if len(nb) == 0:
            rows.append(i); cols.append(i); vals.append(1.0); continue
        rows += [i] * len(nb); cols += list(nb); vals += [1.0 / len(nb)] * len(nb)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def smooth_dofs(s: np.ndarray, A: sp.csr_matrix, iters: int, alpha: float = ALPHA) -> np.ndarray:
    if iters <= 0:
        return s.copy()
    p, q, r = s[0::3].copy(), s[1::3].copy(), s[2::3].copy()
    for _ in range(iters):
        p = (1 - alpha) * p + alpha * (A @ p)
        q = (1 - alpha) * q + alpha * (A @ q)
        r = (1 - alpha) * r + alpha * (A @ r)
    out = np.empty_like(s)
    out[0::3], out[1::3], out[2::3] = p, q, r
    return out


def run(shape: str, subdiv: int, lams: np.ndarray, iters_list: list[int]):
    mesh = build_mesh(shape, subdiv)
    pts, t1, t2, n, radial = _frames_outward(mesh, depth=3)
    faces = np.asarray(mesh.cells, dtype=int)
    an_max, an_min, an_tr, coord, coord_label, mask = analytic_coord_mask(shape, pts)

    K, b = assemble_fem(pts, faces, t1, t2, n, DP)
    Rop = fem_roughness_operator(pts, faces, t1, t2)
    A = one_ring_avg(mesh)
    bnorm = np.linalg.norm(b)
    kfac = np.linalg.norm(K.data) / np.linalg.norm(Rop.data)
    nv = len(pts)

    am, ai = an_max[mask], an_min[mask]
    den_full = np.linalg.norm(np.r_[am, ai])
    den_trace = np.linalg.norm(an_tr[mask])

    nl, ni = len(lams), len(iters_list)
    err_full = np.zeros((nl, ni)); err_trace = np.zeros((nl, ni))
    resid = np.zeros((nl, ni)); reg = np.zeros((nl, ni))
    sig1 = np.zeros((nl, ni, nv)); sig2 = np.zeros((nl, ni, nv)); pqr = np.zeros((nl, ni, nv, 3))

    print(f"[{shape}] n={nv}, masked={mask.sum()}")
    for il, lam in enumerate(lams):
        w = lam * kfac
        Amat = (K.T @ K + (w ** 2) * (Rop.T @ Rop)).tocsc()
        s0 = spla.spsolve(Amat, K.T @ b)
        for ik, it in enumerate(iters_list):
            s = smooth_dofs(s0, A, it)
            resid[il, ik] = np.linalg.norm(K @ s - b) / bnorm
            reg[il, ik] = np.linalg.norm(Rop @ s)
            _, _, s1, s2 = _principal(s, T)
            smax, smin = np.maximum(s1, s2), np.minimum(s1, s2)
            err_full[il, ik] = np.linalg.norm(np.r_[smax[mask] - am, smin[mask] - ai]) / den_full
            err_trace[il, ik] = np.linalg.norm((s1 + s2)[mask] - an_tr[mask]) / den_trace
            sig1[il, ik] = s1; sig2[il, ik] = s2
            pqr[il, ik] = np.c_[s[0::3], s[1::3], s[2::3]]
        print(f"  lam={lam:9.4g} | " + "  ".join(
            f"it{iters_list[ik]}:e{err_full[il,ik]*100:4.1f}%/r{resid[il,ik]:.1e}"
            for ik in range(ni)))

    return dict(shape=shape, pts=pts, normals=n, t1=t1, t2=t2, faces=faces,
                coord=coord, coord_label=coord_label, mask=mask,
                an_max=an_max, an_min=an_min, an_trace=an_tr,
                lams=lams, iters=np.array(iters_list), alpha=ALPHA,
                err_full=err_full, err_trace=err_trace, resid=resid, reg=reg,
                sigma1=sig1, sigma2=sig2, pqr=pqr)


def make_figure(d: dict, subdiv: int):
    shape = d["shape"]
    iters_list = list(d["iters"]); lams = d["lams"]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(iters_list)))
    il_b, ik_b = np.unravel_index(np.argmin(d["err_full"]), d["err_full"].shape)
    lam_b = lams[il_b]
    print(f"  [{shape}] best: lam={lam_b:.4g}, iters={iters_list[ik_b]}, "
          f"err={d['err_full'][il_b, ik_b]:.3%}  (iters0 best "
          f"{d['err_full'][:,0].min():.3%})")

    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    (axa, axb), (axc, axd) = axs

    for ik, it in enumerate(iters_list):
        axa.loglog(lams, 100 * d["err_full"][:, ik], "o-", color=colors[ik], ms=4, label=f"iters={it}")
    axa.axvline(lam_b, color="k", ls=":", lw=1)
    axa.set_xlabel("λ"); axa.set_ylabel("relative error of σ [%]")
    axa.set_title("(a) Relative error vs λ"); axa.grid(True, which="both", alpha=0.3)
    axa.legend(fontsize=8, title="Laplacian")

    coord = d["coord"]; order = np.argsort(coord)
    for ik, it in enumerate(iters_list):
        axb.plot(coord, d["sigma1"][il_b, ik], ".", ms=3, alpha=0.5, color=colors[ik], label=f"iters={it}")
    axb.plot(coord[order], d["an_max"][order], "k-", lw=2, label="analytic σ_max")
    axb.set_xlabel(d["coord_label"]); axb.set_ylabel("σ₁ = σ_max (Pa)")
    axb.set_title(f"(b) σ₁ vs {d['coord_label'].split(' ')[0]} at λ={lam_b:.3g}")
    axb.grid(True, alpha=0.3); axb.legend(fontsize=8)

    for ik, it in enumerate(iters_list):
        axc.loglog(lams, d["resid"][:, ik], "o-", color=colors[ik], ms=4, label=f"iters={it}")
    axc.axvline(lam_b, color="k", ls=":", lw=1)
    axc.set_xlabel("λ"); axc.set_ylabel("‖Ks−b‖ / ‖b‖")
    axc.set_title("(c) Force-balance residual vs λ")
    axc.grid(True, which="both", alpha=0.3); axc.legend(fontsize=8, title="Laplacian")

    for ik, it in enumerate(iters_list):
        axd.loglog(d["resid"][:, ik], d["reg"][:, ik], "o-", color=colors[ik], ms=4, label=f"iters={it}")
    axd.set_xlabel("residual ‖Ks−b‖ / ‖b‖"); axd.set_ylabel("roughness ‖Rs‖")
    axd.set_title("(d) L-curve (corner ≈ optimal λ)")
    axd.grid(True, which="both", alpha=0.3); axd.legend(fontsize=8, title="Laplacian")

    fig.suptitle(f"FEM regularization — {shape}: λ sweep × Laplacian iters "
                 f"(subdiv-{subdiv}, Δp={DP:.0f} Pa, t={T}, α={ALPHA})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = f"out/fem_smoothing_sweep_{shape}.png"
    fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"  saved {out_png}")
    return lam_b, iters_list[ik_b], d["err_full"][il_b, ik_b], d["err_full"][:, 0].min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["sphere", "prolate", "oblate", "capsule"],
                    choices=list(SHAPES))
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--nlam", type=int, default=16)
    ap.add_argument("--iters", type=int, nargs="+", default=[0, 4, 8, 12, 16])
    args = ap.parse_args()

    lams = np.geomspace(1e-4, 50.0, args.nlam)
    os.makedirs("out", exist_ok=True)
    summary = []
    for shape in args.shapes:
        print(f"\n=== {shape} (subdiv-{args.subdiv}, dp={DP}, t={T}, α={ALPHA}, iters={args.iters}) ===")
        d = run(shape, args.subdiv, lams, args.iters)
        npz = f"out/fem_smoothing_sweep_{shape}.npz"
        np.savez_compressed(npz, shape=shape, pts=d["pts"], normals=d["normals"],
                            t1=d["t1"], t2=d["t2"], faces=d["faces"], coord=d["coord"],
                            coord_label=d["coord_label"], mask=d["mask"], an_max=d["an_max"],
                            an_min=d["an_min"], an_trace=d["an_trace"], lams=d["lams"],
                            iters=d["iters"], alpha=d["alpha"], dp=DP, t=T, subdiv=args.subdiv,
                            err_full=d["err_full"], err_trace=d["err_trace"], resid=d["resid"],
                            reg=d["reg"], sigma1=d["sigma1"], sigma2=d["sigma2"], pqr=d["pqr"])
        print(f"  saved {npz}")
        lam_b, it_b, e_b, e_i0 = make_figure(d, args.subdiv)
        summary.append((shape, lam_b, it_b, e_b, e_i0))

    print("\n=== SUMMARY (best over λ×iters vs best at iters=0) ===")
    print(f"{'shape':9s} {'λ*':>8s} {'iters*':>7s} {'err*':>8s} {'err(it0)':>9s} {'gain':>7s}")
    for shape, lam_b, it_b, e_b, e_i0 in summary:
        print(f"{shape:9s} {lam_b:8.3g} {it_b:7d} {e_b*100:7.2f}% {e_i0*100:8.2f}% "
              f"{(1 - e_b / e_i0) * 100:6.1f}%")


if __name__ == "__main__":
    main()
