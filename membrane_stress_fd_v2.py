"""Membrane stress solve in the principal-curvature frame (GFDM v2).

Identical physics and solver to membrane_stress_fd.py, but replaces the
arbitrary polynomial-fit frame (v1, v2) with the principal curvature frame
(e1, e2) from compute_curvature_frame(), and uses polynomial-fit normals
instead of mesh.compute_normals().

Key differences:
  1. Normals: polynomial patch fit (consistent with curvature calc), not
     area-weighted face-normal average from mesh.compute_normals().
  2. Tangent frame: e1 (direction of kappa1), e2 (direction of kappa2),
     after BFS sign propagation.
  3. DOF interpretation: the solved (p, q, r) represent
         N = p e1*e1 + q e2*e2 + r (e1*e2 + e2*e1)
     so on axisymmetric surfaces (where principal curvature dirs = principal
     stress dirs) r~0 and p~sigma_merid*t, q~sigma_hoop*t directly.
     On general surfaces r != 0; principal stresses are still found by
     diagonalising [[p,r],[r,q]] / t.

Note on umbilic regions:
  BFS sign propagation of e1/e2 has unavoidable singularities on genus-0
  surfaces (Poincare-Hopf: index sum = 2). Near those singularities, adjacent
  vertices have inconsistent e1 directions, which degrades the GFDM operators
  locally. On the sphere (totally umbilic) this is severe; on the spheroid it
  is limited to the two poles. The v1/v2 frame in membrane_stress_fd.py does
  not have this issue. Use Laplacian post-smoothing regardless of which frame.

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe membrane_stress_fd_v2.py
    & ... membrane_stress_fd_v2.py --show
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import vedo

from surface_curvature_frame import compute_curvature_frame
from surface_fd import get_neighborhoods, build_grad_operators


# --------------------------------------------------------------------------- #
# Sparse assembly helpers (identical to membrane_stress_fd.py)
# --------------------------------------------------------------------------- #

def _component_operator(t1, t2, a, b):
    """Sparse (n x 3n) map S -> nodal ambient component N_{ab} of the stress tensor."""
    n = len(t1)
    cp = t1[:, a] * t1[:, b]
    cq = t2[:, a] * t2[:, b]
    cr = t1[:, a] * t2[:, b] + t2[:, a] * t1[:, b]
    rows = np.repeat(np.arange(n), 3)
    cols = np.column_stack([3 * np.arange(n),
                            3 * np.arange(n) + 1,
                            3 * np.arange(n) + 2]).ravel()
    vals = np.column_stack([cp, cq, cr]).ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, 3 * n))


def assemble_equilibrium(pts, t1, t2, normals, G_xi, G_eta):
    """Assemble L (3n x 3n) for the ambient-component form of div_s(N)."""
    n = len(pts)
    blocks = []
    for a in range(3):
        La = sp.csr_matrix((n, 3 * n))
        for b in range(3):
            Dt1 = sp.diags(t1[:, b])
            Dt2 = sp.diags(t2[:, b])
            Pab = _component_operator(t1, t2, a, b)
            La = La + (Dt1 @ G_xi + Dt2 @ G_eta) @ Pab
        blocks.append(La)
    return sp.vstack(blocks).tocsr()


def _roughness_operator(t1, t2, G_xi, G_eta):
    """Tikhonov roughness: penalise spatial variation of each ambient stress component."""
    blocks = []
    for a in range(3):
        for b in range(a, 3):
            Cab = _component_operator(t1, t2, a, b)
            blocks.append(G_xi @ Cab)
            blocks.append(G_eta @ Cab)
    return sp.vstack(blocks).tocsr()


# --------------------------------------------------------------------------- #
# Main solver
# --------------------------------------------------------------------------- #

def solve_membrane(mesh: vedo.Mesh, dp: float, t: float,
                   depth: int = 3, lam: float = 0.05,
                   solver: str = "auto",
                   lsqr_thresh: int = 60_000, lsqr_iters: int = 4000):
    """Solve regularised membrane equilibrium in the principal curvature frame.

    Uses polynomial-fit normals and BFS-corrected principal curvature directions
    (e1, e2) as the per-vertex tangent frame.  Everything else follows the same
    Tikhonov least-squares approach as membrane_stress_fd.solve_membrane.

    Returns
    -------
    dict with per-vertex arrays:
      pts, normals, e1, e2  — geometry
      kappa1, kappa2        — principal curvatures (from the frame computation)
      p, q, r               — raw DOFs: N = p e1*e1 + q e2*e2 + r sym(e1*e2)
      sigma1, sigma2        — principal stresses (eigenvalues of [[p,r],[r,q]] / t)
      N1, N2                — principal resultants (sigma * t)
      resid                 — relative equilibrium residual ||LS - rhs|| / ||rhs||
    """
    # ---- 1. Principal curvature frame ---------------------------------------- #
    # compute_curvature_frame already: fits polynomial patches, orients n outward,
    # builds the Weingarten-map shape operator, and BFS-propagates e1/e2 signs.
    f   = compute_curvature_frame(mesh, depth=depth)
    pts = f["pts"]
    t1  = f["e1"]       # principal direction for kappa1
    t2  = f["e2"]       # principal direction for kappa2
    n   = f["normals"]  # outward unit normals from the polynomial fit

    # ---- 2. GFDM gradient operators in the principal curvature frame ---------- #
    neigh       = get_neighborhoods(mesh, depth=depth)
    G_xi, G_eta = build_grad_operators(pts, t1, t2, n, neigh)

    # ---- 3. Assemble L and rhs ----------------------------------------------- #
    L   = assemble_equilibrium(pts, t1, t2, n, G_xi, G_eta)
    rhs = -dp * np.concatenate([n[:, 0], n[:, 1], n[:, 2]])

    # ---- 4. Tikhonov-smoothed least squares ----------------------------------- #
    R    = _roughness_operator(t1, t2, G_xi, G_eta)
    ndof = L.shape[1]
    use_lsqr = solver == "lsqr" or (solver == "auto" and ndof > lsqr_thresh)
    if use_lsqr:
        Aug  = sp.vstack([L, lam * R]).tocsr()
        baug = np.concatenate([rhs, np.zeros(R.shape[0])])
        out  = spla.lsqr(Aug, baug, atol=1e-8, btol=1e-8, iter_lim=lsqr_iters)
        S    = out[0]
        print(f"    [lsqr] istop={out[1]} iters={out[2]} "
              f"normr={out[3]:.3e} (ndof={ndof})")
    else:
        A    = (L.T @ L + (lam ** 2) * (R.T @ R)).tocsc()
        bvec = L.T @ rhs
        S    = spla.spsolve(A, bvec)

    resid = np.linalg.norm(L @ S - rhs) / np.linalg.norm(rhs)

    # ---- 5. Extract principal stresses ---------------------------------------- #
    # DOFs in the principal curvature frame: N = p e1*e1 + q e2*e2 + r sym(e1*e2)
    # Diagonalise [[p, r], [r, q]] to get principal resultants N1 >= N2.
    p = S[0::3]; q = S[1::3]; r = S[2::3]
    tr   = p + q
    det  = p * q - r * r
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0.0))
    N1   = tr / 2 + disc
    N2   = tr / 2 - disc
    sigma1 = N1 / t
    sigma2 = N2 / t

    # ---- 6. Principal stress directions in world frame ----------------------- #
    # The eigenvector of [[p,r],[r,q]] for the larger eigenvalue N1 makes an
    # angle theta_s with t1 in the tangent plane.
    theta_s = 0.5 * np.arctan2(2.0 * r, p - q)
    d1 = np.cos(theta_s)[:, None] * t1 + np.sin(theta_s)[:, None] * t2
    d2 = np.cross(n, d1)
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True) + 1e-15
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True) + 1e-15

    return dict(
        pts=pts, normals=n, e1=t1, e2=t2,
        kappa1=f["kappa1"], kappa2=f["kappa2"],
        p=p, q=q, r=r,
        sigma1=sigma1, sigma2=sigma2, N1=N1, N2=N2,
        d1=d1, d2=d2,
        resid=resid,
    )


# --------------------------------------------------------------------------- #
# Analytic reference and reporting (same as membrane_stress_fd.py)
# --------------------------------------------------------------------------- #

def analytic_axisym(pts, dp, t, a, b):
    """Closed-form stresses for a spheroid (axis=x, semi-axes a along x, b equatorial)."""
    x   = pts[:, 0]
    rho = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2)
    beta = np.arctan2(rho / b, x / a)
    sb, cb = np.sin(beta), np.cos(beta)
    J    = np.sqrt((a * sb) ** 2 + (b * cb) ** 2)
    r1   = J ** 3 / (a * b)
    r2   = b * J / a
    N_phi   = dp * r2 / 2.0
    N_theta = dp * r2 * (1.0 - r2 / (2.0 * r1))
    return N_phi / t, N_theta / t   # meridional, hoop


def report(tag, res, dp, t, a, b):
    print(f"\n[{tag}]  residual ||L S - rhs|| / ||rhs|| = {res['resid']:.3e}")
    pts = res["pts"]
    # Exclude long-axis poles (x is the stretch axis for the spheroid)
    radial = pts - pts.mean(axis=0)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-15
    pole = np.abs(radial[:, 0]) < 0.92

    num_max = np.maximum(res["sigma1"], res["sigma2"])
    num_min = np.minimum(res["sigma1"], res["sigma2"])

    sm, sh  = analytic_axisym(pts, dp, t, a, b)
    an_max  = np.maximum(sm, sh)
    an_min  = np.minimum(sm, sh)

    def cmp(name, num, an):
        e = np.abs(num[pole] - an[pole]) / np.maximum(np.abs(an[pole]), 1e-12)
        print(f"    {name}: numeric mean={num[pole].mean():+.3f}  "
              f"analytic mean={an[pole].mean():+.3f}  "
              f"rel-err mean={e.mean():.3%}  median={np.median(e):.3%}")

    cmp("sigma_max", num_max, an_max)
    cmp("sigma_min", num_min, an_min)

    # Shear in the curvature frame — should be ~0 for axisymmetric surfaces
    r_rel = np.abs(res["r"][pole]) / (np.abs(res["p"][pole]) + np.abs(res["q"][pole]) + 1e-15)
    print(f"    |r| / (|p|+|q|) in curvature frame: mean={r_rel.mean():.3%}  "
          f"max={r_rel.max():.3%}  (0% = perfectly aligned with curvature axes)")


# --------------------------------------------------------------------------- #
# Capsule mesh builder (duplicated here to avoid importing show_capsule module-
# level code; keep in sync with show_capsule.make_capsule if that changes)
# --------------------------------------------------------------------------- #

def make_capsule(R: float = 1.0, H: float = 2.0,
                 ntheta: int = 40, nphi: int = 14) -> vedo.Mesh:
    """Watertight capsule: cylinder radius R, half-height H, hemisphere caps."""
    verts: list = []; faces: list = []

    def push_ring(r, z):
        idx = len(verts)
        for j in range(ntheta):
            t = 2.0 * np.pi * j / ntheta
            verts.append([r * np.cos(t), r * np.sin(t), z])
        return idx

    def push_pole(z):
        idx = len(verts); verts.append([0.0, 0.0, z]); return idx

    def fan(pole, ring, flip):
        for j in range(ntheta):
            a = ring + j; b = ring + (j + 1) % ntheta
            faces.append([pole, b, a] if flip else [pole, a, b])

    def strip(r0, r1):
        for j in range(ntheta):
            a0 = r0 + j; a1 = r0 + (j + 1) % ntheta
            b0 = r1 + j; b1 = r1 + (j + 1) % ntheta
            faces.append([a0, b0, a1]); faces.append([b0, b1, a1])

    np_idx = push_pole(H + R)
    prev = None
    for i in range(1, nphi + 1):
        phi = (np.pi / 2.0) * i / nphi
        idx = push_ring(R * np.sin(phi), H + R * np.cos(phi))
        fan(np_idx, idx, flip=False) if i == 1 else strip(prev, idx)
        prev = idx
    n_cyl = max(1, round(4.0 * H * nphi / (R * np.pi)))
    for ic in range(1, n_cyl + 1):
        z = H - 2.0 * H * ic / n_cyl
        idx = push_ring(R, z); strip(prev, idx); prev = idx
    for i in range(1, nphi):
        phi = np.pi / 2.0 + (np.pi / 2.0) * i / nphi
        idx = push_ring(R * np.sin(phi), -H + R * np.cos(phi))
        strip(prev, idx); prev = idx
    sp_idx = push_pole(-H - R)
    fan(sp_idx, prev, flip=True)

    mesh = vedo.Mesh([np.array(verts, dtype=float), np.array(faces, dtype=int)])
    mesh.compute_normals()
    return mesh


# --------------------------------------------------------------------------- #
# 3-panel visualisation
# --------------------------------------------------------------------------- #

def plot_stress_frame(cases, out=None, show=False):
    """N-panel plot: mesh coloured by sigma1, white d1 arrows for stress direction.

    cases : list of (vedo.Mesh, result_dict, title_str)
    """
    all_s1 = np.concatenate([res["sigma1"] for _, res, _ in cases])
    vmin = float(np.percentile(all_s1, 2))
    vmax = float(np.percentile(all_s1, 98))

    N = len(cases)
    plt = vedo.Plotter(N=N, size=(900 * N, 820), offscreen=not show)

    for k, (mesh, res, title) in enumerate(cases):
        mc = mesh.clone()
        mc.pointdata["sigma1"] = res["sigma1"]
        mc.cmap("plasma", "sigma1", vmin=vmin, vmax=vmax).alpha(0.55)
        if k == N - 1:
            mc.add_scalarbar(title="sigma1 (Pa)")

        pts   = res["pts"]
        diag  = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        scale = diag * 0.055

        # Suppress arrows at umbilic regions (disc ≈ 0 → principal directions
        # are arbitrary there and the BFS-propagated d1 is noisy).
        disc = np.abs(res["kappa1"] - res["kappa2"])
        if disc.max() > 0.1:          # non-trivial (spheroid, capsule)
            well_def = disc > 0.25 * disc.max()
        else:                          # sphere: all umbilic, show all
            well_def = np.ones(len(pts), dtype=bool)
        idx_pool = np.where(well_def)[0]
        idx      = idx_pool[::max(1, len(idx_pool) // 150)]

        # Symmetric line segments: stress direction is a LINE FIELD (±d1 equivalent).
        # A directed arrow would imply a sign that doesn't exist physically.
        half = scale / 2
        segs = vedo.Lines(
            pts[idx] - half * res["d1"][idx],
            pts[idx] + half * res["d1"][idx],
            c="white", alpha=0.9, lw=2,
        )
        txt = vedo.Text2D(title, pos="top-left", font="Calco", s=0.78)
        plt.at(k).show(mc, segs, txt, axes=1, resetcam=True)

    if out:
        os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
        plt.screenshot(out)
        print(f"Saved {out}")
    if show:
        plt.interactive()
    plt.close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius",  type=float, default=1.0)
    ap.add_argument("--subdiv",  type=int,   default=4,
                    help="IcoSphere subdivisions")
    ap.add_argument("--depth",   type=int,   default=3,
                    help="k-ring neighbourhood depth (same for curvature frame and stress)")
    ap.add_argument("--dp",      type=float, default=20.0,  help="pressure jump (Pa)")
    ap.add_argument("--t",       type=float, default=0.05,  help="wall thickness")
    ap.add_argument("--lam",     type=float, default=0.05,  help="Tikhonov weight")
    ap.add_argument("--stretch", type=float, default=2.0,   help="spheroid long-axis scale")
    ap.add_argument("--out",     default="out/membrane_stress_v2.png")
    ap.add_argument("--show",    action="store_true")
    args = ap.parse_args()

    R = args.radius

    # ---- sphere -------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print(f"Sphere  R={R}  subdiv={args.subdiv}  depth={args.depth}")
    sphere = vedo.IcoSphere(r=R, subdivisions=args.subdiv)
    rs = solve_membrane(sphere, args.dp, args.t, depth=args.depth, lam=args.lam)
    report("sphere", rs, args.dp, args.t, a=R, b=R)

    # ---- prolate spheroid ---------------------------------------------------- #
    print("\n" + "=" * 60)
    a_ax = args.stretch * R
    print(f"Spheroid  a={a_ax}  b={R}  subdiv={args.subdiv}  depth={args.depth}")
    ell = vedo.IcoSphere(r=R, subdivisions=args.subdiv).scale([args.stretch, 1.0, 1.0])
    re  = solve_membrane(ell, args.dp, args.t, depth=args.depth, lam=args.lam)
    report(f"spheroid a={a_ax}", re, args.dp, args.t, a=a_ax, b=R)

    # ---- capsule ------------------------------------------------------------- #
    print("\n" + "=" * 60)
    CAP_R, CAP_H = 1.0, 2.0
    print(f"Capsule  R={CAP_R}  H={CAP_H}")
    cap = make_capsule(R=CAP_R, H=CAP_H)
    cap.rotate_y(90)   # long axis z → x, consistent with spheroid for default view
    print(f"  {cap.npoints} vertices, {cap.ncells} faces")
    rc  = solve_membrane(cap, args.dp, args.t, depth=args.depth, lam=args.lam)
    print(f"  residual = {rc['resid']:.3e}")
    # Analytic reference for cylinder body: sigma_hoop=dp*R/t, sigma_axial=dp*R/(2t)
    cyl_mask = np.abs(rc["pts"][:, 2]) < CAP_H * 0.8
    s1c = rc["sigma1"][cyl_mask]; s2c = rc["sigma2"][cyl_mask]
    print(f"  Cylinder body: sigma_max mean={s1c.mean():.1f}  analytic={args.dp*CAP_R/args.t:.1f}")
    print(f"                 sigma_min mean={s2c.mean():.1f}  analytic={args.dp*CAP_R/(2*args.t):.1f}")

    # ---- 3-panel plot -------------------------------------------------------- #
    cases = [
        (sphere, rs, f"Sphere  R={R}"),
        (ell,    re, f"Spheroid  a={a_ax}  b={R}"),
        (cap,    rc, f"Capsule  R={CAP_R}  H={CAP_H}"),
    ]
    plot_stress_frame(cases, out=args.out, show=args.show)


if __name__ == "__main__":
    main()
