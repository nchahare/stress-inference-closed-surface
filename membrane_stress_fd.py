"""Membrane stress on an arbitrary surface by generalized finite differences.

Solves the membrane equilibrium PDE
        div_s(N) + dp * n = 0
for the tangential symmetric stress-resultant tensor N on a triangulated surface, using
the GFDM surface-derivative operators from ``surface_fd``. This single ambient vector
equation encodes BOTH tangential equilibrium and the normal Laplace law (the curvature
enters through the surface divergence of the tangential tensor field), so no material
constants and no axisymmetry assumption are needed -- it works for arbitrary surfaces.

Unknowns: 3 DOF per vertex describing N in the local tangent basis (t1, t2):
        N = p (t1 x t1) + q (t2 x t2) + r (t1 x t2 + t2 x t1).
Principal membrane stresses are the eigenvalues of [[p, r], [r, q]] divided by the
wall thickness t:  sigma_1, sigma_2 = eig / t.

Validation:
  * sphere      -> sigma_1 = sigma_2 = dp*R/(2t)  (isotropic)
  * prolate spheroid (stretched sphere) -> sigma_1 != sigma_2, anisotropy growing
    toward the equator (hoop > meridional).

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe membrane_stress_fd.py
    & ... membrane_stress_fd.py --show
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


def _component_operator(t1, t2, a, b):
    """Sparse (n x 3n) map S -> nodal ambient component N_{ab} of the stress tensor."""
    n = len(t1)
    cp = t1[:, a] * t1[:, b]
    cq = t2[:, a] * t2[:, b]
    cr = t1[:, a] * t2[:, b] + t2[:, a] * t1[:, b]
    rows = np.repeat(np.arange(n), 3)
    cols = np.column_stack([3 * np.arange(n), 3 * np.arange(n) + 1,
                            3 * np.arange(n) + 2]).ravel()
    vals = np.column_stack([cp, cq, cr]).ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, 3 * n))


def assemble_equilibrium(pts, t1, t2, normals, G_xi, G_eta):
    """Assemble L (3n x 3n) for div_s(N) and rhs operator pieces. Returns L."""
    n = len(pts)
    blocks = []
    for a in range(3):                       # ambient row / equation component
        La = sp.csr_matrix((n, 3 * n))
        for b in range(3):
            Dt1 = sp.diags(t1[:, b])
            Dt2 = sp.diags(t2[:, b])
            Pab = _component_operator(t1, t2, a, b)
            La = La + (Dt1 @ G_xi + Dt2 @ G_eta) @ Pab
        blocks.append(La)
    return sp.vstack(blocks).tocsr()


def _roughness_operator(t1, t2, G_xi, G_eta):
    """Tikhonov roughness operator R: penalizes spatial variation of the 6 independent
    ambient stress components. ||R S||^2 = sum_ab ||grad_s N_ab||^2, which suppresses the
    oscillatory closed-surface null modes while leaving the smooth physical field."""
    blocks = []
    for a in range(3):
        for b in range(a, 3):
            Cab = _component_operator(t1, t2, a, b)
            blocks.append(G_xi @ Cab)
            blocks.append(G_eta @ Cab)
    return sp.vstack(blocks).tocsr()


def solve_membrane(mesh: vedo.Mesh, dp: float, t: float, depth: int = 2, lam: float = 0.05,
                   solver: str = "auto", lsqr_thresh: int = 60000, lsqr_iters: int = 4000):
    """Solve the regularized membrane-equilibrium least squares for sigma_1, sigma_2.

    solver: "direct" forms the normal equations and factors them with a sparse LU
    (exact, used for the sphere/spheroid validation); "lsqr" runs an iterative least
    squares on the augmented system [L; lam R] (memory-light, no dense LU fill-in,
    needed for large meshes where the LU runs out of memory); "auto" picks lsqr once
    the number of DOFs (3*npoints) exceeds lsqr_thresh. Both minimize the same
    objective  ||L S - rhs||^2 + lam^2 ||R S||^2, so they agree to solver tolerance."""
    f = compute_vertex_frames(mesh, depth=depth)
    pts, t1, t2, n = f["pts"], f["v1"], f["v2"], f["normals"]
    # orient normals outward
    c = pts.mean(axis=0)
    radial = pts - c
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    flip = np.sum(n * radial, axis=1) < 0
    n = n.copy(); n[flip] *= -1
    # frames must stay orthonormal & right-handed after the flip
    t1 = t1.copy(); t2 = t2.copy()
    t2[flip] *= -1   # keep (t1, t2, n) right-handed when n is flipped

    neigh = get_neighborhoods(mesh, depth=depth)
    G_xi, G_eta = build_grad_operators(pts, t1, t2, n, neigh)

    L = assemble_equilibrium(pts, t1, t2, n, G_xi, G_eta)
    rhs = -dp * np.concatenate([n[:, 0], n[:, 1], n[:, 2]])

    # Tikhonov-smoothed least squares: minimize ||L S - rhs||^2 + lam^2 ||R S||^2
    R = _roughness_operator(t1, t2, G_xi, G_eta)
    ndof = L.shape[1]
    use_lsqr = solver == "lsqr" or (solver == "auto" and ndof > lsqr_thresh)
    if use_lsqr:
        # iterative least squares on the augmented system [L; lam R] S = [rhs; 0]
        Aug = sp.vstack([L, lam * R]).tocsr()
        baug = np.concatenate([rhs, np.zeros(R.shape[0])])
        out = spla.lsqr(Aug, baug, atol=1e-8, btol=1e-8, iter_lim=lsqr_iters)
        S = out[0]
        print(f"    [lsqr] istop={out[1]} iters={out[2]} "
              f"normr={out[3]:.3e} (ndof={ndof})")
    else:
        A = (L.T @ L + (lam ** 2) * (R.T @ R)).tocsc()
        bvec = L.T @ rhs
        S = spla.spsolve(A, bvec)
    resid = np.linalg.norm(L @ S - rhs) / np.linalg.norm(rhs)

    p = S[0::3]; q = S[1::3]; r = S[2::3]
    # principal resultants = eigenvalues of [[p, r], [r, q]]
    tr = p + q
    det = p * q - r * r
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0.0))
    N1 = tr / 2 + disc
    N2 = tr / 2 - disc
    sigma1 = N1 / t
    sigma2 = N2 / t
    return dict(pts=pts, normals=n, radial=radial, sigma1=sigma1, sigma2=sigma2,
                N1=N1, N2=N2, resid=resid)


def analytic_axisym(pts, dp, t, a, b):
    """Closed-form axisymmetric membrane stresses for a spheroid (axis = x), semi-axis
    `a` along x and equatorial radius `b` (sphere: a = b = R). Returns (sigma_merid,
    sigma_hoop) per vertex via N_phi = dp*r2/2, N_theta = dp*r2*(1 - r2/(2 r1))."""
    x = pts[:, 0]
    rho = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2)
    beta = np.arctan2(rho / b, x / a)
    sb, cb = np.sin(beta), np.cos(beta)
    J = np.sqrt((a * sb) ** 2 + (b * cb) ** 2)
    r1 = J ** 3 / (a * b)
    r2 = b * J / a
    N_phi = dp * r2 / 2.0
    N_theta = dp * r2 * (1.0 - r2 / (2.0 * r1))
    return N_phi / t, N_theta / t   # meridional, hoop


def report(tag, res, dp, t, a, b):
    print(f"\n[{tag}]  relative equilibrium residual ||L S - rhs|| / ||rhs|| = {res['resid']:.3e}")
    pole = np.abs(res["radial"][:, 0]) < 0.92   # exclude long-axis poles (x is stretch axis)
    num_max = np.maximum(res["sigma1"], res["sigma2"])
    num_min = np.minimum(res["sigma1"], res["sigma2"])

    sm, sh = analytic_axisym(res["pts"], dp, t, a, b)
    an_max = np.maximum(sm, sh); an_min = np.minimum(sm, sh)

    def cmp(name, num, an):
        e = np.abs(num[pole] - an[pole]) / np.maximum(np.abs(an[pole]), 1e-12)
        print(f"    {name}: numeric mean={num[pole].mean():+.3f}  "
              f"analytic mean={an[pole].mean():+.3f}  "
              f"rel-err mean={e.mean():.3%}  median={np.median(e):.3%}")

    cmp("sigma_max", num_max, an_max)
    cmp("sigma_min", num_min, an_min)
    print(f"    analytic equator anisotropy hoop/merid = "
          f"{1.0 - b**2/(2*a**2):.3f}/0.5 = {(1.0 - b**2/(2*a**2))/0.5:.3f}")


def show(meshes_fields, out):
    plt = vedo.Plotter(N=len(meshes_fields), size=(900 * len(meshes_fields), 800),
                       offscreen=True)
    alls = np.concatenate([np.maximum(r["sigma1"], r["sigma2"]) for _, r in meshes_fields])
    clim = (float(np.percentile(alls, 2)), float(np.percentile(alls, 98)))
    for k, (m, r) in enumerate(meshes_fields):
        m.pointdata["sigma_max"] = np.maximum(r["sigma1"], r["sigma2"])
        m.cmap("plasma", "sigma_max", vmin=clim[0], vmax=clim[1]).add_scalarbar(title="sigma_max")
        plt.at(k).show(m, vedo.Text2D(r["title"], pos="top-left"), axes=1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.screenshot(out); plt.close()
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--subdiv", type=int, default=4, help="icosphere subdivisions (uniform, pole-free)")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dp", type=float, default=1.0, help="pressure jump")
    ap.add_argument("--t", type=float, default=0.05, help="wall thickness")
    ap.add_argument("--lam", type=float, default=0.05, help="Tikhonov smoothing weight")
    ap.add_argument("--stretch", type=float, default=2.0)
    ap.add_argument("--out", default="out/membrane_stress.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    R = args.radius
    sphere = vedo.IcoSphere(r=R, subdivisions=args.subdiv)
    rs = solve_membrane(sphere, args.dp, args.t, depth=args.depth, lam=args.lam)
    rs["title"] = f"Sphere R={R}"
    report("sphere", rs, args.dp, args.t, a=R, b=R)

    ell = vedo.IcoSphere(r=R, subdivisions=args.subdiv).scale([args.stretch, 1.0, 1.0])
    re = solve_membrane(ell, args.dp, args.t, depth=args.depth, lam=args.lam)
    re["title"] = f"Stretched x{args.stretch}"
    report(f"stretched x{args.stretch}", re, args.dp, args.t, a=args.stretch * R, b=R)

    show([(sphere, rs), (ell, re)], args.out)


if __name__ == "__main__":
    main()
