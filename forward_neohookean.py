"""Forward hyperelastic membrane inflation (method M3) -- sphere prototype.

Inflates a stress-free reference surface under internal pressure with a P1 constant-strain
-triangle (CST) membrane neo-Hookean model, by minimising the total potential energy

    Pi(x) = sum_T A0_T * W(I, J)  -  dp * V(x),

with W the STRESS-FREE incompressible neo-Hookean membrane energy per unit reference area

    W(I, J) = mu_s * (I + J^-2 - 3),      I = tr(C),  J = sqrt(det C),  C = F^T F.

(We use this instead of cMSM's compressible mu(I-2)+lambda(J-1)^2 precisely because the
latter carries a residual pre-tension 2*mu at the reference, which under a stiff modulus
would force a large deformation; the incompressible form is genuinely stress-free at the
reference, so a stiff mu_s gives a small strain -- the deformed shape stays ~the target.)

Because dp is constant, Pi is a genuine potential (pressure work = dp*V, a state function),
so we minimise it with L-BFGS using the analytic gradient (assembled internal forces minus
the pressure follower force). No Hessian / autodiff needed for this first cut.

Sphere validation (project rule): a neo-Hookean sphere inflates self-similarly to a stretch
lambda* solving 4*mu_s*(1 - lambda^-6)/lambda = dp, at which the membrane resultant is the
Laplace value sigma1 = sigma2 = dp*r/2 (r = deformed radius). We check the solver reproduces
lambda*, the isotropy sigma1 == sigma2, and the Laplace tension, and that the shape drift is
small (the stiff-reference strategy).

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe forward_neohookean.py --subdiv 4
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from scipy.optimize import minimize, brentq
import vedo

from membrane_stress_fd import analytic_axisym
from membrane_stress_fd_v2 import make_capsule

# shape -> spheroid semi-axes (axis = x); sphere is A=B=1, prolate A>B, oblate A<B
SPHEROID = {"sphere": (1.0, 1.0), "prolate": (2.0, 1.0), "oblate": (0.5, 1.0)}
SHAPES = list(SPHEROID) + ["capsule"]
CAP_R, CAP_H, CAP_NTHETA, CAP_NPHI = 1.0, 2.0, 40, 14     # capsule (axis = z, body |z|<=H)


def build_reference(shape: str, subdiv: int, R: float = 1.0) -> vedo.Mesh:
    if shape == "capsule":
        return make_capsule(R=CAP_R, H=CAP_H, ntheta=CAP_NTHETA, nphi=CAP_NPHI)
    A, B = SPHEROID[shape]
    return vedo.IcoSphere(r=R, subdivisions=subdiv).scale([A * R, B * R, B * R])


def reference_geometry(X: np.ndarray, faces: np.ndarray):
    """Per-triangle reference frame, inverse edge map Minv (2x2) and area A0."""
    X0, X1, X2 = X[faces[:, 0]], X[faces[:, 1]], X[faces[:, 2]]
    E1, E2 = X1 - X0, X2 - X0
    nrm = np.cross(E1, E2)
    A0 = 0.5 * np.linalg.norm(nrm, axis=1)
    t1 = E1 / np.linalg.norm(E1, axis=1, keepdims=True)
    nhat = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    t2 = np.cross(nhat, t1)                                   # in-plane, orthonormal to t1
    # reference 2D edge coords in (t1, t2): columns are E1, E2
    DXhat = np.empty((len(faces), 2, 2))
    DXhat[:, 0, 0] = np.einsum("ij,ij->i", E1, t1); DXhat[:, 1, 0] = np.einsum("ij,ij->i", E1, t2)
    DXhat[:, 0, 1] = np.einsum("ij,ij->i", E2, t1); DXhat[:, 1, 1] = np.einsum("ij,ij->i", E2, t2)
    Minv = np.linalg.inv(DXhat)
    return Minv, A0


def _kinematics(x, faces, Minv):
    """Per-triangle F (3x2), C (2x2), invariants I, J."""
    x0, x1, x2 = x[faces[:, 0]], x[faces[:, 1]], x[faces[:, 2]]
    Dx = np.stack([x1 - x0, x2 - x0], axis=2)                 # (M,3,2)
    F = Dx @ Minv                                            # (M,3,2)
    C = np.einsum("mki,mkj->mij", F, F)                      # F^T F -> (M,2,2)
    I = C[:, 0, 0] + C[:, 1, 1]
    detC = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0]
    J = np.sqrt(np.maximum(detC, 1e-12))
    return x0, x1, x2, Dx, F, C, I, J


def energy_and_grad(xflat, X, faces, Minv, A0, mu_s, dp):
    n = X.shape[0]
    x = xflat.reshape(n, 3)
    x0, x1, x2, Dx, F, C, I, J = _kinematics(x, faces, Minv)

    # --- elastic energy  W = mu_s (I + J^-2 - 3) ---
    W = mu_s * (I + J ** -2 - 3.0)
    E_elastic = float(np.sum(A0 * W))

    # --- enclosed volume V = (1/6) sum x0 . (x1 x x2) ---
    V = float(np.sum(np.einsum("ij,ij->i", x0, np.cross(x1, x2))) / 6.0)
    Pi = E_elastic - dp * V

    # --- elastic internal force: S = 2 mu_s (I2 - J^-2 C^-1);  G = A0 F S Minv^T ---
    detC = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0]
    Cinv = np.empty_like(C)
    Cinv[:, 0, 0] = C[:, 1, 1]; Cinv[:, 1, 1] = C[:, 0, 0]
    Cinv[:, 0, 1] = -C[:, 0, 1]; Cinv[:, 1, 0] = -C[:, 1, 0]
    Cinv /= detC[:, None, None]
    eye = np.broadcast_to(np.eye(2), C.shape)
    S = 2.0 * mu_s * (eye - (J ** -2)[:, None, None] * Cinv)  # (M,2,2)
    P = F @ S                                                # (M,3,2)
    G = A0[:, None, None] * (P @ Minv.transpose(0, 2, 1))     # (M,3,2)

    grad = np.zeros((n, 3))
    np.add.at(grad, faces[:, 1], G[:, :, 0])
    np.add.at(grad, faces[:, 2], G[:, :, 1])
    np.add.at(grad, faces[:, 0], -(G[:, :, 0] + G[:, :, 1]))

    # --- pressure follower force: dV/dx0 = (1/6) x1 x x2, cyclic ---
    gV = np.zeros((n, 3))
    np.add.at(gV, faces[:, 0], np.cross(x1, x2) / 6.0)
    np.add.at(gV, faces[:, 1], np.cross(x2, x0) / 6.0)
    np.add.at(gV, faces[:, 2], np.cross(x0, x1) / 6.0)

    grad -= dp * gV
    return Pi, grad.ravel()


def recover_stress(x, faces, Minv, mu_s):
    """Per-triangle Cauchy membrane resultant sigma = (1/J) F S F^T, principal sigma1>=sigma2."""
    _, _, _, _, F, C, I, J = _kinematics(x, faces, Minv)
    detC = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0]
    Cinv = np.empty_like(C)
    Cinv[:, 0, 0] = C[:, 1, 1]; Cinv[:, 1, 1] = C[:, 0, 0]
    Cinv[:, 0, 1] = -C[:, 0, 1]; Cinv[:, 1, 0] = -C[:, 1, 0]
    Cinv /= detC[:, None, None]
    eye = np.broadcast_to(np.eye(2), C.shape)
    S = 2.0 * mu_s * (eye - (J ** -2)[:, None, None] * Cinv)
    sig = np.einsum("m,mik,mkl,mjl->mij", 1.0 / J, F, S, F)   # (1/J) F S F^T  (M,3,3)
    # two non-zero eigenvalues of the symmetric rank-2 tensor
    w = np.linalg.eigvalsh(sig)                              # ascending (M,3)
    s2, s1 = w[:, 1], w[:, 2]                                # smallest non-zero, largest
    return s1, s2


def analytic_sphere_stretch(mu_s, dp, R=1.0):
    """lambda solving 4 mu_s (1 - lambda^-6)/lambda = dp/R  (equilibrium of a NH sphere)."""
    f = lambda lam: 4.0 * mu_s * (1.0 - lam ** -6) / lam - dp / R
    return brentq(f, 1.0 + 1e-9, 5.0)


def report_sphere(x, faces, Minv, s1, s2, mu_s, dp, R):
    r = np.linalg.norm(x - x.mean(0), axis=1)
    lam = r.mean() / R
    lam_star = analytic_sphere_stretch(mu_s, dp, R)
    drift = (r.max() - r.min()) / r.mean()
    N_laplace = dp * (lam * R) / 2.0
    print(f"\n  stretch lambda : solver {lam:.6f}   analytic {lam_star:.6f}   "
          f"err {abs(lam-lam_star)/lam_star:.2%}")
    print(f"  shape drift    : (rmax-rmin)/rmean = {drift:.2e}   (stiff-reference -> small)")
    print(f"  sigma1 (mean)  : {s1.mean():.4f}  std {s1.std():.4f}")
    print(f"  sigma2 (mean)  : {s2.mean():.4f}  std {s2.std():.4f}")
    print(f"  Laplace target : dp*r/2 = {N_laplace:.4f}")
    print(f"  isotropy |s1-s2|/s1 (mean) : {np.mean(np.abs(s1-s2)/np.abs(s1)):.2%}")
    print(f"  tension err vs Laplace     : "
          f"{abs(0.5*(s1.mean()+s2.mean()) - N_laplace)/N_laplace:.2%}")


def report_spheroid(shape, X, x, faces, s1, s2, dp):
    """Compare per-triangle forward resultants to the analytic axisymmetric solution
    (resultants = analytic_axisym with t=1) over an equatorial belt."""
    A, B = SPHEROID[shape]
    cen = x[faces].mean(axis=1)                                   # deformed triangle centroids
    an_merid, an_hoop = analytic_axisym(cen, dp, 1.0, a=A, b=B)   # resultants N/m (t=1)
    an_max, an_min = np.maximum(an_merid, an_hoop), np.minimum(an_merid, an_hoop)
    smax, smin = np.maximum(s1, s2), np.minimum(s1, s2)
    rad = cen / np.linalg.norm(cen, axis=1, keepdims=True)
    belt = np.abs(rad[:, 0]) < 0.9                                # exclude umbilic x-poles
    disp = np.linalg.norm(x - X, axis=1)
    drift = disp.max() / np.linalg.norm(X, axis=1).mean()
    e_max = np.median(np.abs(smax[belt] - an_max[belt]) / np.abs(an_max[belt]))
    e_min = np.median(np.abs(smin[belt] - an_min[belt]) / np.abs(an_min[belt]))
    # equatorial values (near the y/z great circle, |x|~0)
    eq = np.abs(cen[:, 0]) < 0.1 * A
    print(f"\n  shape drift    : max|x-X| / mean|X| = {drift:.2e}   (stiff-reference -> small)")
    print(f"  belt (n={belt.sum()}) median rel err:  sigma_max {e_max:.2%}   sigma_min {e_min:.2%}")
    print(f"  equator (n={eq.sum()}):  sigma_max {smax[eq].mean():.3f} (analytic "
          f"{an_max[eq].mean():.3f})   sigma_min {smin[eq].mean():.3f} (analytic "
          f"{an_min[eq].mean():.3f})")
    print(f"  equator anisotropy sigma_max/sigma_min: solver "
          f"{smax[eq].mean()/smin[eq].mean():.3f}   analytic "
          f"{an_max[eq].mean()/an_min[eq].mean():.3f}")


def report_capsule(X, x, faces, s1, s2, dp):
    """Compare per-triangle forward resultants to the pressure-vessel solution:
    cylinder body (|z|<=H) hoop=dp*R, axial=dp*R/2; hemispherical caps isotropic dp*R/2."""
    cen = x[faces].mean(axis=1)
    z = cen[:, 2]
    s_hoop, s_axial = dp * CAP_R, dp * CAP_R / 2.0           # resultants (N/m)
    on_cyl = np.abs(z) <= CAP_H
    an_max = np.where(on_cyl, s_hoop, s_axial)
    an_min = np.full(len(faces), s_axial)
    smax, smin = np.maximum(s1, s2), np.minimum(s1, s2)
    band = 0.12
    belt = (np.abs(z) < CAP_H + CAP_R - band) & (np.abs(np.abs(z) - CAP_H) > band)
    cyl = on_cyl & belt
    cap = (~on_cyl) & belt
    drift = np.linalg.norm(x - X, axis=1).max() / np.linalg.norm(X, axis=1).mean()
    e_max = np.median(np.abs(smax[belt] - an_max[belt]) / np.abs(an_max[belt]))
    e_min = np.median(np.abs(smin[belt] - an_min[belt]) / np.abs(an_min[belt]))
    print(f"\n  shape drift    : max|x-X| / mean|X| = {drift:.2e}   (stiff-reference -> small)")
    print(f"  belt (n={belt.sum()}) median rel err:  sigma_max {e_max:.2%}   sigma_min {e_min:.2%}")
    print(f"  cylinder (n={cyl.sum()}): hoop {smax[cyl].mean():.3f} (target {s_hoop:.3f})   "
          f"axial {smin[cyl].mean():.3f} (target {s_axial:.3f})")
    print(f"  caps     (n={cap.sum()}): sigma {0.5*(smax[cap]+smin[cap]).mean():.3f} "
          f"(target {s_axial:.3f})   anisotropy {np.mean(np.abs(smax[cap]-smin[cap])/smax[cap]):.2%}")


def load_normalized_mesh(path, decimate=None):
    """Load an arbitrary closed surface mesh, triangulate, centre at the origin and scale
    to mean radius 1 (the forward stress PATTERN is scale-invariant), and orient faces
    outward so the enclosed-volume / pressure term inflates. `decimate` (target vertex
    count) coarsens the mesh first -- e.g. HH17 down to HH20's element size."""
    m = vedo.load(path).clean().triangulate()
    if decimate and m.npoints > decimate:
        m = m.decimate(n=decimate).clean().triangulate()
        print(f"  decimated to {m.npoints} verts")
    X = m.coordinates.astype(float)
    X = X - X.mean(0)
    X = X / np.linalg.norm(X, axis=1).mean()
    faces = np.asarray(m.cells, dtype=int)
    # drop degenerate triangles (repeated index or zero area = collinear sliver); they carry
    # no area/volume/energy, so removing them changes nothing geometrically and avoids NaNs.
    nondeg = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[nondeg]
    A = 0.5 * np.linalg.norm(np.cross(X[faces[:, 1]] - X[faces[:, 0]],
                                      X[faces[:, 2]] - X[faces[:, 0]]), axis=1)
    faces = faces[A > 1e-10 * A.max()]
    ndrop = nondeg.size - len(faces)
    if ndrop:
        print(f"  dropped {ndrop} degenerate triangle(s)")
    x0, x1, x2 = X[faces[:, 0]], X[faces[:, 1]], X[faces[:, 2]]
    V = np.sum(np.einsum("ij,ij->i", x0, np.cross(x1, x2))) / 6.0
    if V < 0:                                  # inward-oriented -> flip so pressure inflates
        faces = faces[:, [0, 2, 1]]
    return vedo.Mesh([X, faces])


def run_mesh_file(args):
    mesh = load_normalized_mesh(args.mesh, decimate=args.decimate)
    X = mesh.coordinates.astype(float)
    faces = np.asarray(mesh.cells, dtype=int)
    Minv, A0 = reference_geometry(X, faces)
    print(f"{os.path.basename(args.mesh)}: n={len(X)} verts, {len(faces)} tris (normalised), "
          f"mu_s={args.mu}, dp={args.dp}")
    res = minimize(energy_and_grad, X.ravel(), args=(X, faces, Minv, A0, args.mu, args.dp),
                   jac=True, method="L-BFGS-B",
                   options=dict(maxiter=20000, maxfun=40000, ftol=1e-13, gtol=1e-10))
    x = res.x.reshape(len(X), 3)
    print(f"  L-BFGS: {res.nit} iters, ||grad||={np.linalg.norm(res.jac):.2e}, {res.message}")
    s1, s2 = recover_stress(x, faces, Minv, args.mu)
    drift = np.linalg.norm(x - X, axis=1).max() / np.linalg.norm(X, axis=1).mean()
    print(f"  shape drift: {drift:.2e}")
    for name, v in [("sigma_max", np.maximum(s1, s2)), ("sigma_min", np.minimum(s1, s2)),
                    ("trace", s1 + s2)]:
        print(f"  {name:10s}: min {v.min():8.3f}  median {np.median(v):8.3f}  max {v.max():8.3f}")
    render(args, x, faces, s1, s2, tag=os.path.splitext(os.path.basename(args.mesh))[0])


def render(args, x, faces, s1, s2, tag="forward"):
    if not (args.show or args.save):
        return
    fields = {"trace": s1 + s2, "sigma1": s1, "sigma2": s2, "shear": 0.5 * (s1 - s2),
              "sigma_max": np.maximum(s1, s2), "sigma_min": np.minimum(s1, s2)}
    vals = fields[args.field]
    dm = vedo.Mesh([x, faces]); dm.celldata[args.field] = vals
    vmin = float(np.percentile(vals, 2)) if args.vmin is None else args.vmin
    vmax = float(np.percentile(vals, 98)) if args.vmax is None else args.vmax
    dm.cmap("viridis", args.field, on="cells", vmin=vmin, vmax=vmax)
    dm.add_scalarbar(title=f"{args.field} (N/m)")
    txt = vedo.Text2D(f"Forward NH {tag}  |  {args.field}  [{vmin:.2f}, {vmax:.2f}]", pos="top-left")
    plt = vedo.Plotter(offscreen=not args.show, size=(1000, 850))
    plt.show(dm, txt, axes=1, azimuth=30, elevation=15)
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.screenshot(args.save); print(f"  saved {args.save}")
    if args.show:
        plt.interactive()
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=None, help="run on an arbitrary closed surface mesh (e.g. HH20.vtk)")
    ap.add_argument("--decimate", type=int, default=None, help="coarsen --mesh to this vertex count first")
    ap.add_argument("--shape", default="sphere", choices=SHAPES)
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--dp", type=float, default=20.0)
    ap.add_argument("--mu", type=float, default=500.0, help="surface shear modulus mu_s (N/m)")
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--show", action="store_true", help="open a vedo window of the result")
    ap.add_argument("--save", default=None, help="render offscreen and save a PNG to this path")
    ap.add_argument("--field", default="trace",
                    choices=["trace", "sigma1", "sigma2", "shear", "sigma_max", "sigma_min"])
    ap.add_argument("--vmin", type=float, default=None, help="colorbar lower limit (default: data min)")
    ap.add_argument("--vmax", type=float, default=None, help="colorbar upper limit (default: data max)")
    args = ap.parse_args()

    if args.mesh:                       # arbitrary closed surface (e.g. the HH20 embryo)
        run_mesh_file(args)
        return

    mesh = build_reference(args.shape, args.subdiv, args.R)
    X = mesh.coordinates.astype(float)
    faces = np.asarray(mesh.cells, dtype=int)
    Minv, A0 = reference_geometry(X, faces)
    print(f"{args.shape} reference: n={len(X)} verts, {len(faces)} tris, "
          f"mu_s={args.mu}, dp={args.dp}")

    res = minimize(energy_and_grad, X.ravel(), args=(X, faces, Minv, A0, args.mu, args.dp),
                   jac=True, method="L-BFGS-B",
                   options=dict(maxiter=4000, ftol=1e-12, gtol=1e-9))
    x = res.x.reshape(len(X), 3)
    print(f"  L-BFGS: {res.nit} iters, Pi={res.fun:.6e}, "
          f"||grad||={np.linalg.norm(res.jac):.2e}, {res.message}")

    s1, s2 = recover_stress(x, faces, Minv, args.mu)
    if args.shape == "sphere":
        report_sphere(x, faces, Minv, s1, s2, args.mu, args.dp, args.R)
    elif args.shape == "capsule":
        report_capsule(X, x, faces, s1, s2, args.dp)
    else:
        report_spheroid(args.shape, X, x, faces, s1, s2, args.dp)

    if args.show or args.save:
        fields = {"trace": s1 + s2, "sigma1": s1, "sigma2": s2, "shear": 0.5 * (s1 - s2),
                  "sigma_max": np.maximum(s1, s2), "sigma_min": np.minimum(s1, s2)}
        vals = fields[args.field]
        dm = vedo.Mesh([x, faces])
        dm.celldata[args.field] = vals
        vmin = float(vals.min()) if args.vmin is None else args.vmin
        vmax = float(vals.max()) if args.vmax is None else args.vmax
        dm.cmap("viridis", args.field, on="cells", vmin=vmin, vmax=vmax)
        dm.add_scalarbar(title=f"{args.field} (N/m)")
        txt = vedo.Text2D(f"Forward NH {args.shape}  |  {args.field}  "
                          f"[{vmin:.1f}, {vmax:.1f}] N/m", pos="top-left")
        plt = vedo.Plotter(offscreen=not args.show, size=(1000, 850),
                           title="Forward neo-Hookean inflation")
        plt.show(dm, txt, axes=1, azimuth=30, elevation=15)
        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            plt.screenshot(args.save)
            print(f"  saved {args.save}")
        if args.show:
            plt.interactive()
        plt.close()


if __name__ == "__main__":
    main()
