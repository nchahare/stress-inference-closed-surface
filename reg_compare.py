"""Compare regularization strategies for the GFDM membrane-stress solve.

  (A) OUR method: Tikhonov roughness penalty (gradient of all 6 ambient stress
      components) + Laplacian (umbrella) post-smoothing of the recovered sigma field.
      This is the current "best practical" pipeline.

  (B) cMSM-style first-order regularization (Marin-Llaurado et al., Nat. Commun. 2023,
      Supp. Note 2): penalize the gradient of the TRACE (mean tension) with weight
      lambda_t, plus the surface-covariant gradient of the stress tensor ("curl" term,
      which in 2D equals the full covariant-gradient norm) with weight lambda_c.
      NO post-smoothing -- the regularization alone is meant to remove the lines.

Key identity used for (B): the surface covariant derivative of a tangential tensor equals
the local-frame sandwich of the ambient-component derivatives,
    nabla_alpha N_{beta,gamma} = (v_beta)_a (v_gamma)_b (G_alpha N_ab),
so no explicit Christoffel symbols are needed (same trick that builds L). Unlike our
ambient-component roughness operator, this penalizes only the tangential-tangential part,
not the frame-rotation leakage.

Validated on the analytic sphere (sigma=dp*R/2t, isotropic) and prolate spheroid
(anisotropy hoop/merid -> 1.75 at the equator). Produces out/reg_compare.png and a
metrics table.

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe reg_compare.py
    & ... reg_compare.py --sweep        # lambda_t sweep on the sphere (cMSM L-curve style)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import vedo

from sphere_curvature import compute_vertex_frames
from surface_fd import get_neighborhoods, build_grad_operators
from membrane_stress_fd import (_component_operator, assemble_equilibrium,
                                 _roughness_operator, analytic_axisym, solve_membrane)


# --------------------------------------------------------------------------- #
# Shared setup (frames, operators, equilibrium L) -- mirrors solve_membrane
# --------------------------------------------------------------------------- #
def setup(mesh, depth):
    f = compute_vertex_frames(mesh, depth=depth)
    pts, t1, t2, n = f["pts"], f["v1"], f["v2"], f["normals"]
    c = pts.mean(axis=0)
    radial = pts - c
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    flip = np.sum(n * radial, axis=1) < 0
    n = n.copy(); n[flip] *= -1
    t1 = t1.copy(); t2 = t2.copy(); t2[flip] *= -1   # keep (t1,t2,n) right-handed
    neigh = get_neighborhoods(mesh, depth=depth)
    G_xi, G_eta = build_grad_operators(pts, t1, t2, n, neigh)
    L = assemble_equilibrium(pts, t1, t2, n, G_xi, G_eta)
    return dict(pts=pts, t1=t1, t2=t2, n=n, radial=radial,
                G_xi=G_xi, G_eta=G_eta, L=L)


def vertex_areas(mesh, pts):
    faces = np.asarray(mesh.cells, dtype=int)
    tri = pts[faces]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    af = 0.5 * np.linalg.norm(cr, axis=1)
    A = np.zeros(len(pts))
    np.add.at(A, faces.ravel(), np.repeat(af / 3.0, 3))
    A[A <= 0] = A[A > 0].mean()
    return A


# --------------------------------------------------------------------------- #
# cMSM-style regularizers
# --------------------------------------------------------------------------- #
def cmsm_regularizers(pts, t1, t2, G_xi, G_eta, area):
    """Return (R_t, R_c), sqrt-area-weighted, with
       ||R_t S||^2 = int |grad_s tr(N)|^2 dS         (grad-of-trace, L_1t)
       ||R_c S||^2 = int |grad_s N|^2 dS  (cov.)     (curl == full cov. gradient, L_1c)
    """
    n = len(pts)
    sa = sp.diags(np.sqrt(area))

    # trace selector T: S -> p+q
    rows = np.concatenate([np.arange(n), np.arange(n)])
    cols = np.concatenate([3 * np.arange(n), 3 * np.arange(n) + 1])
    T = sp.csr_matrix((np.ones(2 * n), (rows, cols)), shape=(n, 3 * n))
    R_t = sp.vstack([sa @ (G_xi @ T), sa @ (G_eta @ T)]).tocsr()

    # ambient component maps S -> N_ab (all 9, symmetric)
    Cab = {}
    for a in range(3):
        for b in range(a, 3):
            Cab[(a, b)] = _component_operator(t1, t2, a, b)
            if a != b:
                Cab[(b, a)] = Cab[(a, b)]

    frames = {1: t1, 2: t2}
    mult = {(1, 1): 1.0, (2, 2): 1.0, (1, 2): 2.0}   # off-diagonal counted twice
    blocks = []
    for G in (G_xi, G_eta):
        GC = {key: (G @ C) for key, C in Cab.items()}     # G_alpha N_ab
        for (bb, gg), m in mult.items():
            vb, vg = frames[bb], frames[gg]
            D = sp.csr_matrix((n, 3 * n))
            for a in range(3):
                for b in range(3):
                    D = D + sp.diags(vb[:, a] * vg[:, b]) @ GC[(a, b)]
            blocks.append(np.sqrt(m) * (sa @ D))
    R_c = sp.vstack(blocks).tocsr()
    return R_t, R_c


def fields_from_S(S, t, resid):
    p, q, r = S[0::3], S[1::3], S[2::3]
    tr = p + q
    det = p * q - r * r
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0.0))
    N1, N2 = tr / 2 + disc, tr / 2 - disc
    return dict(sigma1=N1 / t, sigma2=N2 / t, resid=resid)


def solve_cmsm(s, dp, t, lam_t, lam_c, area):
    L, nrm = s["L"], s["n"]
    rhs = -dp * np.concatenate([nrm[:, 0], nrm[:, 1], nrm[:, 2]])
    R_t, R_c = cmsm_regularizers(s["pts"], s["t1"], s["t2"], s["G_xi"], s["G_eta"], area)
    A = (L.T @ L + lam_t ** 2 * (R_t.T @ R_t) + lam_c ** 2 * (R_c.T @ R_c)).tocsc()
    S = spla.spsolve(A, L.T @ rhs)
    resid = np.linalg.norm(L @ S - rhs) / np.linalg.norm(rhs)
    return fields_from_S(S, t, resid)


# --------------------------------------------------------------------------- #
# Laplacian (umbrella) post-smoothing
# --------------------------------------------------------------------------- #
def avg_operator(mesh):
    n = mesh.npoints
    neigh1 = get_neighborhoods(mesh, depth=1)
    rows, cols, vals = [], [], []
    for i in range(n):
        nb = neigh1[i]
        if len(nb) == 0:
            rows.append(i); cols.append(i); vals.append(1.0); continue
        w = 1.0 / len(nb)
        rows += [i] * len(nb); cols += list(nb); vals += [w] * len(nb)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def laplacian_smooth(avg, field, iters=12, alpha=0.5):
    f = field.copy()
    for _ in range(iters):
        f = (1 - alpha) * f + alpha * (avg @ f)
    return f


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(sig1, sig2, pts, dp, t, a, b, radial):
    smax = np.maximum(sig1, sig2)
    smin = np.minimum(sig1, sig2)
    mask = np.abs(radial[:, 0]) < 0.92          # exclude long-axis poles
    sm, sh = analytic_axisym(pts, dp, t, a, b)
    an_max = np.maximum(sm, sh); an_min = np.minimum(sm, sh)

    def relerr(num, an):
        e = np.abs(num[mask] - an[mask]) / np.maximum(np.abs(an[mask]), 1e-12)
        return np.median(e)
    return dict(
        mean_max=smax[mask].mean(), mean_min=smin[mask].mean(),
        std_max=smax[mask].std(),
        err_max=relerr(smax, an_max), err_min=relerr(smin, an_min),
        anis=(smax[mask] - smin[mask]).mean() / max(np.abs(smax[mask].mean()), 1e-12),
    )


def fmt(tag, m):
    return (f"  {tag:<26s} sig_max={m['mean_max']:6.2f} (err {m['err_max']:5.1%})  "
            f"sig_min={m['mean_min']:6.2f} (err {m['err_min']:5.1%})  "
            f"std_max={m['std_max']:5.3f}  anis={m['anis']:6.1%}")


# --------------------------------------------------------------------------- #
# Render 2x3 comparison
# --------------------------------------------------------------------------- #
def render(rows, out):
    """rows: list of (mesh, [(title, smax_field), x3])."""
    plt = vedo.Plotter(shape=(2, 3), size=(1650, 1050), sharecam=False, offscreen=True)
    for ri, (m, panels) in enumerate(rows):
        allv = np.concatenate([f for _, f in panels])
        clim = (float(np.percentile(allv, 2)), float(np.percentile(allv, 98)))
        for ci, (title, f) in enumerate(panels):
            mm = m.clone()
            mm.pointdata["s"] = f
            mm.cmap("plasma", "s", vmin=clim[0], vmax=clim[1])
            if ci == 2:
                mm.add_scalarbar(title="sigma_max")
            plt.at(ri * 3 + ci).show(mm, vedo.Text2D(title, pos="top-left"), axes=0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.screenshot(out); plt.close()
    print(f"\nSaved {out}")


# --------------------------------------------------------------------------- #
def run_case(mesh, tag, dp, t, depth, lam, lam_t, lam_c, a, b, iters, alpha):
    s = setup(mesh, depth)
    area = vertex_areas(mesh, s["pts"])
    avg = avg_operator(mesh)

    # (A) our Tikhonov solve (raw) + Laplacian post-smoothing
    rawf = solve_membrane(mesh, dp, t, depth=depth, lam=lam, solver="direct")
    s1_raw, s2_raw = rawf["sigma1"], rawf["sigma2"]
    smax_raw = np.maximum(s1_raw, s2_raw)
    s1_sm = laplacian_smooth(avg, s1_raw, iters, alpha)
    s2_sm = laplacian_smooth(avg, s2_raw, iters, alpha)
    smax_sm = np.maximum(s1_sm, s2_sm)

    # (B) cMSM-style regularization, no smoothing
    cm = solve_cmsm(s, dp, t, lam_t, lam_c, area)
    smax_cm = np.maximum(cm["sigma1"], cm["sigma2"])

    rad = rawf["radial"]; pts = rawf["pts"]
    m_raw = metrics(s1_raw, s2_raw, pts, dp, t, a, b, rad)
    m_sm = metrics(s1_sm, s2_sm, pts, dp, t, a, b, rad)
    m_cm = metrics(cm["sigma1"], cm["sigma2"], pts, dp, t, a, b, rad)

    print(f"\n[{tag}]  (residual: Tikhonov={rawf['resid']:.2e}, cMSM={cm['resid']:.2e})")
    print(fmt("(A) Tikhonov raw", m_raw))
    print(fmt("(A) Tikhonov + Laplacian", m_sm))
    print(fmt("(B) cMSM grad-trace+curl", m_cm))

    panels = [("Tikhonov raw", smax_raw),
              ("Tikhonov + Laplacian (ours)", smax_sm),
              ("cMSM grad-trace+curl", smax_cm)]
    return (mesh, panels)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dp", type=float, default=1.0)
    ap.add_argument("--t", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.02, help="our Tikhonov weight")
    ap.add_argument("--lam_t", type=float, default=0.05, help="cMSM grad-trace weight")
    ap.add_argument("--lam_c", type=float, default=0.01, help="cMSM curl weight")
    ap.add_argument("--stretch", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", default="out/reg_compare.png")
    ap.add_argument("--sweep", action="store_true", help="lambda_t sweep on the sphere")
    args = ap.parse_args()

    R = args.radius
    if args.sweep:
        mesh = vedo.IcoSphere(r=R, subdivisions=args.subdiv)
        s = setup(mesh, args.depth)
        area = vertex_areas(mesh, s["pts"])
        print(f"cMSM lambda_t sweep (lambda_c={args.lam_c}), sphere subdiv{args.subdiv} "
              f"depth{args.depth}; target sigma=10, true anisotropy=0")
        print(f"{'lambda_t':>10}  {'residual':>10}  {'sig_max':>8}  {'std_max':>8}  "
              f"{'spurious anis':>13}")
        for lt in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            cm = solve_cmsm(s, args.dp, args.t, lt, args.lam_c, area)
            m = metrics(cm["sigma1"], cm["sigma2"], s["pts"], args.dp, args.t, R, R, s["radial"])
            print(f"{lt:>10.3f}  {cm['resid']:>10.2e}  {m['mean_max']:>8.2f}  "
                  f"{m['std_max']:>8.3f}  {m['anis']:>12.1%}")
        return

    sphere = vedo.IcoSphere(r=R, subdivisions=args.subdiv)
    row_s = run_case(sphere, "sphere", args.dp, args.t, args.depth, args.lam,
                     args.lam_t, args.lam_c, R, R, args.iters, args.alpha)

    ell = vedo.IcoSphere(r=R, subdivisions=args.subdiv).scale([args.stretch, 1.0, 1.0])
    row_e = run_case(ell, f"spheroid x{args.stretch}", args.dp, args.t, args.depth,
                     args.lam, args.lam_t, args.lam_c, args.stretch * R, R,
                     args.iters, args.alpha)

    render([row_s, row_e], args.out)


if __name__ == "__main__":
    main()
