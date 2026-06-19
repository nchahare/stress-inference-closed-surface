"""Stress-based finite-element membrane solve on a closed surface mesh.

An independent finite-element discretisation of the SAME balance law solved by the
GFDM scheme in ``membrane_stress_fd`` (see tension_inference.tex Section 12):

        div_s(N) + dp * n = 0.

Formulation: primal virtual-work (the cMSM-style stress FEM). Dotting the balance with
a vector test field w and integrating by parts on the closed surface gives

        a(N, w) = INT_Gamma  N : eps_s(w)  dS  =  INT_Gamma  dp (n . w) dS = l(w),

i.e. internal virtual work (stress : virtual strain) = external virtual work of the
pressure. Both fields are continuous P1 on the triangulation, in ambient Cartesian
components:
  * trial (stress): 3 DOF per vertex (p, q, r) in the local frame (v1, v2, n),
        N_i = p (v1 x v1) + q (v2 x v2) + r (v1 x v2 + v2 x v1)  -> N_i n_i = 0 by build;
  * test  (virtual displacement): P1 vector field, ambient nodal basis {e_d phi_j}.
This is a SQUARE 3n x 3n system (the count of cMSM).

Element block on a triangle T (area A_T, P1 hat gradient g_j = n_T x opp_j / (2 A_T),
constant per triangle):
        a|_T(N, e_d phi_j) = (A_T / 3) sum_{m in T} (N_m g_j)_d,
consistent pressure load via the P1 mass matrix M^T_jm = (A_T/12)(1 + delta_jm).

K is square but SINGULAR on a closed surface (the self-equilibrating null modes); the
load is consistent (net pressure force/torque vanish). We solve in two stages:
  (i)  raw minimum-norm least squares (lsqr, no regularisation) -- look for the
       triangulation "lines";
  (ii) Tikhonov  min ||K s - b||^2 + w^2 ||R s||^2  with a FEM-native 1-ring roughness
       R (P1 surface gradient of the ambient stress components -- no GFDM operators),
       solved directly (small meshes) or by iterative lsqr on [K; w R] (large meshes).
Principal stresses sigma_1, sigma_2 = eig([[p, r], [r, q]]) / t, identical to GFDM.

Validation (project rule -- sphere first):
  * sphere   -> sigma_1 = sigma_2 = dp*R/(2t)  (isotropic); deviatoric std = lines indicator
  * prolate spheroid (2:1) -> equatorial hoop/merid anisotropy -> 1.75

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe membrane_stress_fem.py
    & ... membrane_stress_fem.py --subdiv 4 --show
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import vedo

from sphere_curvature import compute_vertex_frames
from membrane_stress_fd import (
    analytic_axisym, report, _component_operator, solve_membrane as solve_gfdm,
)


def _frames_outward(mesh: vedo.Mesh, depth: int):
    """Per-vertex orthonormal frame (v1, v2, n) with n flipped to point outward and the
    frame kept right-handed -- identical convention to membrane_stress_fd.solve_membrane."""
    f = compute_vertex_frames(mesh, depth=depth)
    pts, t1, t2, n = f["pts"], f["v1"].copy(), f["v2"].copy(), f["normals"].copy()
    c = pts.mean(axis=0)
    radial = pts - c
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    flip = np.sum(n * radial, axis=1) < 0
    n[flip] *= -1
    t2[flip] *= -1            # keep (t1, t2, n) right-handed when n flips
    return pts, t1, t2, n, radial


def assemble_fem(pts, faces, v1, v2, nrm, dp):
    """Assemble the square 3n x 3n virtual-work stiffness K and consistent load b.

    Rows are test DOFs (3*j + d): vertex j, ambient component d.
    Cols are stress DOFs (3*m + {0,1,2}): vertex m, local-frame (p, q, r).
    """
    n = len(pts)
    tri = np.asarray(faces, dtype=int)
    P = pts[tri]                                  # (ntri, 3, 3): [tri, localvert, xyz]
    P0, P1, P2 = P[:, 0], P[:, 1], P[:, 2]
    cross = np.cross(P1 - P0, P2 - P0)            # (ntri, 3)
    twoA = np.linalg.norm(cross, axis=1)          # = 2 * area
    area = 0.5 * twoA
    nhat = cross / twoA[:, None]                  # triangle normal (from vertex ordering)
    # opposite edges: opp[:, j] is the edge opposite local vertex j
    opp = np.stack([P2 - P1, P0 - P2, P1 - P0], axis=1)        # (ntri, 3, 3)
    g = np.cross(nhat[:, None, :], opp) / twoA[:, None, None]  # hat gradients (ntri, 3, 3)

    V1 = v1[tri]                                   # (ntri, 3, 3): [tri, localm, xyz]
    V2 = v2[tri]
    Nv = nrm[tri]                                  # outward normals at nodes

    rows, cols, vals = [], [], []
    b = np.zeros(3 * n)
    for j in range(3):                             # test node (local)
        gj = g[:, j, :]                            # (ntri, 3)
        rj = 3 * tri[:, j]                          # row base for test node j
        for m in range(3):                         # source node (local)
            v1m, v2m = V1[:, m, :], V2[:, m, :]
            s1 = np.einsum("ij,ij->i", v1m, gj)    # v1m . gj
            s2 = np.einsum("ij,ij->i", v2m, gj)    # v2m . gj
            coef = area / 3.0
            cm = 3 * tri[:, m]                       # col base for source node m
            Mjm = area / 12.0 * (1.0 + (1.0 if j == m else 0.0))   # P1 mass entry
            for d in range(3):                     # ambient component
                row = rj + d
                vp = coef * v1m[:, d] * s1
                vq = coef * v2m[:, d] * s2
                vr = coef * (v1m[:, d] * s2 + v2m[:, d] * s1)
                rows.append(np.concatenate([row, row, row]))
                cols.append(np.concatenate([cm + 0, cm + 1, cm + 2]))
                vals.append(np.concatenate([vp, vq, vr]))
                np.add.at(b, row, dp * Nv[:, m, d] * Mjm)
    K = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3 * n, 3 * n)).tocsr()
    return K, b


def fem_roughness_operator(pts, faces, v1, v2):
    """FEM-native roughness R: ||R s||^2 = INT_Gamma sum_ab |grad_s N_ab|^2 dS, built from
    the P1 element gradients (1-ring coupling) and the per-vertex frames -- no GFDM operators.

    R = stack_ab ( D @ C_ab ), where C_ab maps the (p,q,r) DOFs to the nodal ambient stress
    component N_ab (frame algebra, _component_operator), and D is the area-weighted P1
    surface-gradient operator mapping a nodal scalar to its per-triangle gradient."""
    tri = np.asarray(faces, dtype=int)
    n, ntri = len(pts), len(tri)
    P = pts[tri]
    P0, P1, P2 = P[:, 0], P[:, 1], P[:, 2]
    cross = np.cross(P1 - P0, P2 - P0)
    twoA = np.linalg.norm(cross, axis=1)
    nhat = cross / twoA[:, None]
    opp = np.stack([P2 - P1, P0 - P2, P1 - P0], axis=1)
    g = np.cross(nhat[:, None, :], opp) / twoA[:, None, None]    # hat gradients (ntri, 3, 3)
    w = np.sqrt(0.5 * twoA)                                       # sqrt(area), L2 weight

    rows, cols, vals = [], [], []
    for jloc in range(3):
        for d in range(3):
            rows.append(3 * np.arange(ntri) + d)
            cols.append(tri[:, jloc])
            vals.append(w * g[:, jloc, d])
    D = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3 * ntri, n)).tocsr()
    blocks = [D @ _component_operator(v1, v2, a, b) for a in range(3) for b in range(a, 3)]
    return sp.vstack(blocks).tocsr()


def _principal(s, t):
    """Principal resultants/stresses from the (p, q, r) DOF vector."""
    p, q, r = s[0::3], s[1::3], s[2::3]
    tr = p + q
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - (p * q - r * r), 0.0))
    N1, N2 = tr / 2.0 + disc, tr / 2.0 - disc
    return N1, N2, N1 / t, N2 / t


def solve_membrane_fem(mesh: vedo.Mesh, dp: float, t: float, depth: int = 3,
                       lam: float = 0.05, raw: bool = True, solver: str = "auto",
                       lsqr_thresh: int = 20000, lsqr_iters: int = 5000):
    """Primal virtual-work stress FEM. Returns the Tikhonov-regularised principal stresses
    (and, if raw=True, the raw minimum-norm field for the lines diagnostic).

    solver: "direct" forms the normal equations (K^T K + w^2 R^T R) and factors with sparse
    LU; "lsqr" runs iterative least squares on the augmented system [K; w R]; "auto" switches
    to lsqr once the DOF count (3*npoints) exceeds lsqr_thresh. The FEM K is sparse (P1 1-ring),
    so the iterative path scales gently to large meshes."""
    pts, t1, t2, n, radial = _frames_outward(mesh, depth)
    faces = np.asarray(mesh.cells, dtype=int)
    K, b = assemble_fem(pts, faces, t1, t2, n, dp)
    bnorm = np.linalg.norm(b)

    # (i) optional raw minimum-norm least squares on the singular square system (lines diagnostic)
    if raw:
        s_raw = spla.lsqr(K, b, atol=1e-10, btol=1e-10, iter_lim=lsqr_iters)[0]
        resid_raw = np.linalg.norm(K @ s_raw - b) / bnorm
        _, _, s1r, s2r = _principal(s_raw, t)
    else:
        s1r = s2r = None
        resid_raw = np.nan

    # (ii) Tikhonov with the FEM-native 1-ring roughness. lam is made a dimensionless relative
    # weight by matching R to K's Frobenius norm (K is area-weighted ~h, R ~O(1)); without this
    # a raw lam mis-scales by ~1/h^4 and collapses the solution.
    R = fem_roughness_operator(pts, faces, t1, t2)
    w = lam * np.linalg.norm(K.data) / np.linalg.norm(R.data)
    ndof = K.shape[1]
    use_lsqr = solver == "lsqr" or (solver == "auto" and ndof > lsqr_thresh)
    if use_lsqr:
        Aug = sp.vstack([K, w * R]).tocsr()
        baug = np.concatenate([b, np.zeros(R.shape[0])])
        s_tik = spla.lsqr(Aug, baug, atol=1e-8, btol=1e-8, iter_lim=lsqr_iters)[0]
    else:
        A = (K.T @ K + (w ** 2) * (R.T @ R)).tocsc()
        s_tik = spla.spsolve(A, K.T @ b)
    resid_tik = np.linalg.norm(K @ s_tik - b) / bnorm

    N1t, N2t, s1t, s2t = _principal(s_tik, t)
    # principal stress directions in world R^3: eigenvector of [[p,r],[r,q]] for the
    # larger eigenvalue makes angle theta_s with t1; d1 <-> sigma1, d2 <-> sigma2.
    p, q, r = s_tik[0::3], s_tik[1::3], s_tik[2::3]
    theta_s = 0.5 * np.arctan2(2.0 * r, p - q)
    d1 = np.cos(theta_s)[:, None] * t1 + np.sin(theta_s)[:, None] * t2
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True) + 1e-15
    d2 = np.cross(n, d1)
    return dict(pts=pts, normals=n, radial=radial, faces=faces,
                sigma1_raw=s1r, sigma2_raw=s2r, resid_raw=resid_raw,
                sigma1=s1t, sigma2=s2t, N1=N1t, N2=N2t, d1=d1, d2=d2, resid=resid_tik)


def stress_scalar(r, name):
    """Scalar stress metric per vertex. von Mises is the default equivalent stress for a
    membrane (plane-stress); mean = isotropic tension (sigma1+sigma2)/2; shear =
    in-plane max shear (sigma1-sigma2)/2 = anisotropy magnitude."""
    s1, s2 = r["sigma1"], r["sigma2"]
    smax, smin = np.maximum(s1, s2), np.minimum(s1, s2)
    return {
        "vonmises": np.sqrt(s1 ** 2 - s1 * s2 + s2 ** 2),
        "mean": (s1 + s2) / 2.0,
        "shear": (smax - smin) / 2.0,
        "sigma_max": smax,
        "sigma_min": smin,
    }[name]


def cross_glyphs(r, n_glyph=180):
    """Principal-stress crosses: at ~n_glyph vertices, two symmetric segments along +-d1
    and +-d2 (stress is a line field), each arm scaled by |sigma_i| (so isotropic regions
    look like a +, anisotropic ones elongate along the larger stress). Tensile arms red,
    compressive arms blue."""
    pts, d1, d2 = r["pts"], r["d1"], r["d2"]
    s1, s2 = r["sigma1"], r["sigma2"]
    diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    base = diag * 0.06
    sref = np.percentile(np.maximum(np.abs(s1), np.abs(s2)), 98) + 1e-12
    idx = np.arange(len(pts))[:: max(1, len(pts) // n_glyph)]
    objs = []
    for d, s in ((d1, s1), (d2, s2)):
        h = (base * np.clip(np.abs(s) / sref, 0.0, 1.0))[:, None]
        P0, P1 = (pts - h * d)[idx], (pts + h * d)[idx]
        si = s[idx]
        for mask, col in ((si >= 0, "red"), (si < 0, "blue")):
            if mask.any():
                objs.append(vedo.Lines(P0[mask], P1[mask], c=col, lw=3, alpha=0.9))
    return objs


def plot_fem(panels, field="vonmises", out=None, show=False):
    """Each mesh coloured by the chosen stress scalar (shared scale) with principal-stress
    crosses overlaid. Saves to `out` and/or opens an interactive window."""
    vals = [stress_scalar(r, field) for _, r in panels]
    alls = np.concatenate(vals)
    clim = (float(np.percentile(alls, 2)), float(np.percentile(alls, 98)))
    plt = vedo.Plotter(N=len(panels), size=(1700, 850), sharecam=False, offscreen=not show,
                       title="Membrane stress (stress-based FEM)")
    for k, ((m, r), v) in enumerate(zip(panels, vals)):
        m.pointdata[field] = v
        m.cmap("viridis", field, vmin=clim[0], vmax=clim[1]).add_scalarbar(title=f"{field} (Pa)")
        plt.at(k).show(m, *cross_glyphs(r), vedo.Text2D(r["title"], pos="top-left"), axes=1)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        plt.screenshot(out)
        print(f"Saved {out}")
    if show:
        plt.interactive()
    plt.close()


def _dev_std(s1, s2, mask=None):
    """Std of the deviatoric (s1 - s2)/2 -- the spurious-anisotropy / lines indicator."""
    d = (s1 - s2) / 2.0
    return float(np.std(d if mask is None else d[mask]))


def report_sphere(tag, res, dp, t, R):
    """Sphere-specific summary: mean stress vs analytic dp*R/2t, deviatoric std (lines),
    for both the raw and the Tikhonov fields, plus residuals."""
    target = dp * R / (2.0 * t)
    print(f"\n[{tag}]  analytic sigma = dp*R/2t = {target:.3f} Pa  (isotropic)")
    items = []
    if res["sigma1_raw"] is not None:
        items.append(("raw (min-norm)", res["sigma1_raw"], res["sigma2_raw"], res["resid_raw"]))
    items.append(("Tikhonov      ", res["sigma1"], res["sigma2"], res["resid"]))
    for label, s1, s2, rr in items:
        mean = (s1 + s2) / 2.0
        err = abs(mean.mean() - target) / target
        print(f"    {label}: mean={mean.mean():7.2f}  err={err:6.2%}  "
              f"dev-std={_dev_std(s1, s2):6.3f}  resid={rr:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--subdiv", type=int, default=4, help="icosphere subdivisions")
    ap.add_argument("--depth", type=int, default=3, help="frame/roughness neighbourhood")
    ap.add_argument("--dp", type=float, default=20.0, help="pressure jump (project default 20 Pa)")
    ap.add_argument("--t", type=float, default=0.05, help="wall thickness")
    ap.add_argument("--lam", type=float, default=0.05, help="Tikhonov weight")
    ap.add_argument("--stretch", type=float, default=2.0)
    ap.add_argument("--no-gfdm", action="store_true", help="skip the GFDM head-to-head")
    ap.add_argument("--raw", action="store_true", help="colour by the raw (unregularised) field")
    ap.add_argument("--field", default="vonmises",
                    choices=["vonmises", "mean", "shear", "sigma_max", "sigma_min"],
                    help="mesh colour metric (default von Mises)")
    ap.add_argument("--out", default="out/membrane_stress_fem.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    R = args.radius

    print("=" * 70)
    print("STRESS-BASED FEM (primal virtual-work) vs analytic" +
          ("" if args.no_gfdm else " + GFDM"))
    print("=" * 70)

    # ---- sphere ----
    sphere = vedo.IcoSphere(r=R, subdivisions=args.subdiv)
    fs = solve_membrane_fem(sphere, args.dp, args.t, depth=args.depth, lam=args.lam)
    fs["title"] = f"FEM sphere R={R}"
    report_sphere("FEM sphere", fs, args.dp, args.t, R)
    if not args.no_gfdm:
        gs = solve_gfdm(sphere, args.dp, args.t, depth=args.depth, lam=args.lam)
        target = args.dp * R / (2.0 * args.t)
        mean = (gs["sigma1"] + gs["sigma2"]) / 2.0
        print(f"    GFDM          : mean={mean.mean():7.2f}  "
              f"err={abs(mean.mean()-target)/target:6.2%}  "
              f"dev-std={_dev_std(gs['sigma1'], gs['sigma2']):6.3f}  resid={gs['resid']:.2e}")

    # ---- prolate spheroid ----
    ell = vedo.IcoSphere(r=R, subdivisions=args.subdiv).scale([args.stretch, 1.0, 1.0])
    fe = solve_membrane_fem(ell, args.dp, args.t, depth=args.depth, lam=args.lam)
    fe["title"] = f"FEM stretched x{args.stretch}"
    report(f"FEM spheroid x{args.stretch}", fe, args.dp, args.t, a=args.stretch * R, b=R)
    if not args.no_gfdm:
        ge = solve_gfdm(ell, args.dp, args.t, depth=args.depth, lam=args.lam)
        report(f"GFDM spheroid x{args.stretch}", ge, args.dp, args.t, a=args.stretch * R, b=R)

    if args.raw:                      # colour by the unregularised field to see the lines
        fs["sigma1"], fs["sigma2"] = fs["sigma1_raw"], fs["sigma2_raw"]
        fe["sigma1"], fe["sigma2"] = fe["sigma1_raw"], fe["sigma2_raw"]
    panels = [(sphere, fs), (ell, fe)]
    plot_fem(panels, field=args.field, out=args.out, show=args.show)


if __name__ == "__main__":
    main()
