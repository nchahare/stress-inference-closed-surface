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
import numpy as np
from scipy.optimize import minimize, brentq
import vedo


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--dp", type=float, default=20.0)
    ap.add_argument("--mu", type=float, default=500.0, help="surface shear modulus mu_s (N/m)")
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--show", action="store_true", help="open a vedo window of the result")
    ap.add_argument("--field", default="trace",
                    choices=["trace", "sigma1", "sigma2", "shear"])
    args = ap.parse_args()

    mesh = vedo.IcoSphere(r=args.R, subdivisions=args.subdiv)
    X = mesh.coordinates.astype(float)
    faces = np.asarray(mesh.cells, dtype=int)
    Minv, A0 = reference_geometry(X, faces)
    print(f"sphere reference: n={len(X)} verts, {len(faces)} tris, mu_s={args.mu}, dp={args.dp}")

    res = minimize(energy_and_grad, X.ravel(), args=(X, faces, Minv, A0, args.mu, args.dp),
                   jac=True, method="L-BFGS-B",
                   options=dict(maxiter=2000, ftol=1e-12, gtol=1e-9))
    x = res.x.reshape(len(X), 3)
    gnorm = np.linalg.norm(res.jac)
    print(f"  L-BFGS: {res.nit} iters, Pi={res.fun:.6e}, ||grad||={gnorm:.2e}, {res.message}")

    # deformed radius / stretch
    r = np.linalg.norm(x - x.mean(0), axis=1)
    lam = r.mean() / args.R
    lam_star = analytic_sphere_stretch(args.mu, args.dp, args.R)
    drift = (r.max() - r.min()) / r.mean()

    s1, s2 = recover_stress(x, faces, Minv, args.mu)
    N_laplace = args.dp * (lam * args.R) / 2.0               # equilibrium Laplace resultant

    print(f"\n  stretch lambda : solver {lam:.6f}   analytic {lam_star:.6f}   "
          f"err {abs(lam-lam_star)/lam_star:.2%}")
    print(f"  shape drift    : (rmax-rmin)/rmean = {drift:.2e}   (stiff-reference -> small)")
    print(f"  sigma1 (mean)  : {s1.mean():.4f}  std {s1.std():.4f}")
    print(f"  sigma2 (mean)  : {s2.mean():.4f}  std {s2.std():.4f}")
    print(f"  Laplace target : dp*r/2 = {N_laplace:.4f}")
    print(f"  isotropy |s1-s2|/s1 (mean) : {np.mean(np.abs(s1-s2)/np.abs(s1)):.2%}")
    print(f"  tension error vs Laplace   : "
          f"{abs(0.5*(s1.mean()+s2.mean()) - N_laplace)/N_laplace:.2%}")

    if args.show:
        fields = {"trace": s1 + s2, "sigma1": s1, "sigma2": s2, "shear": 0.5 * (s1 - s2)}
        vals = fields[args.field]
        dm = vedo.Mesh([x, faces])
        dm.celldata[args.field] = vals
        clim = (float(np.percentile(vals, 2)), float(np.percentile(vals, 98)))
        dm.cmap("viridis", args.field, on="cells", vmin=clim[0], vmax=clim[1])
        dm.add_scalarbar(title=f"{args.field} (N/m)")
        txt = vedo.Text2D(f"Forward NH sphere  dp={args.dp}  mu_s={args.mu}  "
                          f"lambda={lam:.4f}  |  {args.field}", pos="top-left")
        vedo.show(dm, txt, axes=1, title="Forward neo-Hookean inflation").close()


if __name__ == "__main__":
    main()
