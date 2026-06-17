"""Membrane stress via an Airy / Beltrami stress function on an arbitrary surface.

Motivation: the direct GFDM solve (``membrane_stress_fd.py``) solves a first-order system
for the 3 stress components and so carries spurious near-null ("hourglass") modes -> the
streaky 'lines'. Here the unknown is a single smooth scalar stress function Phi, which has
no such tensor null space, so the lines should largely vanish.

Subtleties handled (the honest curved-surface version):
  * The Airy construction is NOT divergence-free on a curved surface (curvature coupling),
    and it can only represent self-equilibrated stress -- it cannot carry the pressure on a
    closed surface. So we split  N = N_p + N_Airy(Phi):
       - particular  N_p = (dp / tr B) * g     (isotropic, satisfies N_p : B = dp exactly,
         smooth, line-free; exact on the sphere),
       - Airy         N_Airy = cof(Hess_s Phi)  (local components from the WLS Hessian:
         N11 = d2Phi/deta2,  N22 = d2Phi/dxi2,  N12 = -d2Phi/dxi deta).
  * Phi is found by least squares so the TOTAL N satisfies tangential equilibrium
    (tangential part of div_s N = 0) while N_Airy : B = 0 keeps the normal balance = dp.
  * B (second fundamental form in the local frame) is obtained from the Weingarten map
    B = -grad_s n using the GFDM first-derivative operators.

Everything is sparse scipy; no FEM framework, no axisymmetry, no material constants.

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe membrane_stress_beltrami.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import vedo

from sphere_curvature import compute_vertex_frames
from surface_fd import get_neighborhoods, build_derivative_operators
from membrane_stress_fd import analytic_axisym


def _diag(v):
    return sp.diags(v)


def solve_beltrami(mesh, dp, t, depth=3, wn=1.0, eps=1e-8):
    f = compute_vertex_frames(mesh, depth=depth)
    pts, t1, t2, n = f["pts"], f["v1"], f["v2"], f["normals"]
    # outward, right-handed frame
    c = pts.mean(0); radial = pts - c
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    flip = np.sum(n * radial, axis=1) < 0
    n = n.copy(); t2 = t2.copy(); t1 = t1.copy()
    n[flip] *= -1; t2[flip] *= -1

    neigh = get_neighborhoods(mesh, depth=depth)
    D = build_derivative_operators(pts, t1, t2, n, neigh)
    gx, gy = D["g_xi"], D["g_eta"]
    hxx, hxy, hyy = D["h_xixi"], D["h_xieta"], D["h_etaeta"]
    N = len(pts)

    # --- second fundamental form B in the local frame via Weingarten B = -grad_s n -----
    dn_x = np.column_stack([gx @ n[:, 0], gx @ n[:, 1], gx @ n[:, 2]])   # dn/dxi
    dn_y = np.column_stack([gy @ n[:, 0], gy @ n[:, 1], gy @ n[:, 2]])   # dn/deta
    # second fundamental form: B_ij = +d_i n . e_j (so principal curvatures are +1/R on a
    # sphere with outward normal -> tr B = 2H > 0, tension under positive pressure)
    B11 = np.sum(dn_x * t1, axis=1)
    B22 = np.sum(dn_y * t2, axis=1)
    B12 = 0.5 * (np.sum(dn_x * t2, axis=1) + np.sum(dn_y * t1, axis=1))
    trB = B11 + B22                                   # = 2H
    s = dp / np.where(np.abs(trB) > 1e-12, trB, np.nan)   # particular isotropic magnitude

    # --- ambient divergence operator helper -------------------------------------------
    # for a tensor with ambient components N_ab (nodal arrays), (div_s N)_a =
    #   sum_b [ t1[:,b]*(gx @ N_ab) + t2[:,b]*(gy @ N_ab) ]
    def amb_comp_operator_airy(a, b):
        """n x n operator: Phi -> ambient component (a,b) of N_Airy = cof(Hess Phi)."""
        ca = t1[:, a] * t1[:, b]      # multiplies N11 = hyy Phi
        cb = t2[:, a] * t2[:, b]      # multiplies N22 = hxx Phi
        cc = t1[:, a] * t2[:, b] + t2[:, a] * t1[:, b]   # multiplies N12 = -hxy Phi
        return _diag(ca) @ hyy + _diag(cb) @ hxx + _diag(cc) @ (-hxy)

    # divergence of N_Airy as 3 blocks (each n x n) acting on Phi
    Dblocks = []
    for a in range(3):
        Da = sp.csr_matrix((N, N))
        for b in range(3):
            Pab = amb_comp_operator_airy(a, b)
            Da = Da + (_diag(t1[:, b]) @ gx + _diag(t2[:, b]) @ gy) @ Pab
        Dblocks.append(Da.tocsr())

    # tangential projection of the divergence: A_a = D_a - n_a * sum_b n_b D_b
    Dn = sp.csr_matrix((N, N))
    for b in range(3):
        Dn = Dn + _diag(n[:, b]) @ Dblocks[b]
    A = sp.vstack([Dblocks[a] - _diag(n[:, a]) @ Dn for a in range(3)]).tocsr()

    # --- divergence of the particular part N_p = s*(I - n⊗n), ambient -------------------
    divNp = np.zeros((3, N))
    Npab = {}
    for a in range(3):
        for b in range(3):
            Npab[(a, b)] = s * ((1.0 if a == b else 0.0) - n[:, a] * n[:, b])
    for a in range(3):
        v = np.zeros(N)
        for b in range(3):
            v += t1[:, b] * (gx @ Npab[(a, b)]) + t2[:, b] * (gy @ Npab[(a, b)])
        divNp[a] = v
    # tangential part of div N_p
    dn_proj = sum(n[:, b] * divNp[b] for b in range(3))
    divNp_tan = np.concatenate([divNp[a] - n[:, a] * dn_proj for a in range(3)])

    # --- normal-preservation operator: N_Airy : B = 0 ---------------------------------
    Cn = (_diag(B11) @ hyy + _diag(B22) @ hxx + _diag(2.0 * B12) @ (-hxy)).tocsr()

    # --- least squares: min ||A Phi + divNp_tan||^2 + wn||Cn Phi||^2 + eps||Phi||^2 ----
    LHS = (A.T @ A + wn * (Cn.T @ Cn) + eps * sp.identity(N)).tocsc()
    rhs = -(A.T @ divNp_tan)
    Phi = spla.spsolve(LHS, rhs)

    tan_resid = np.linalg.norm(A @ Phi + divNp_tan) / max(np.linalg.norm(divNp_tan), 1e-30)

    # --- assemble total local stress and principal stresses ----------------------------
    A11 = hyy @ Phi; A22 = hxx @ Phi; A12 = -(hxy @ Phi)   # N_Airy local
    n11 = s + A11; n22 = s + A22; n12 = A12                # total local 2x2
    tr = n11 + n22
    disc = np.sqrt(np.maximum((n11 - n22) ** 2 / 4 + n12 ** 2, 0.0))
    sigma1 = (tr / 2 + disc) / t
    sigma2 = (tr / 2 - disc) / t
    return dict(pts=pts, radial=radial, normals=n, sigma1=sigma1, sigma2=sigma2,
                Phi=Phi, tan_resid=tan_resid, s=s / t, trB=trB)


def report(tag, res, dp, t, a, b):
    print(f"\n[{tag}]  tangential-equilibrium residual = {res['tan_resid']:.3e}")
    pole = np.abs(res["radial"][:, 0]) < 0.92
    num_max = np.maximum(res["sigma1"], res["sigma2"])
    num_min = np.minimum(res["sigma1"], res["sigma2"])
    sm, sh = analytic_axisym(res["pts"], dp, t, a, b)
    an_max = np.maximum(sm, sh); an_min = np.minimum(sm, sh)

    def cmp(name, num, an):
        e = np.abs(num[pole] - an[pole]) / np.maximum(np.abs(an[pole]), 1e-12)
        print(f"    {name}: numeric mean={num[pole].mean():+.3f} std={num[pole].std():.3f}  "
              f"analytic mean={an[pole].mean():+.3f}  rel-err med={np.median(e):.2%}")
    cmp("sigma_max", num_max, an_max)
    cmp("sigma_min", num_min, an_min)
    dev = (num_max - num_min) / 2.0
    print(f"    spurious/real deviatoric (sigma_max-sigma_min)/2: "
          f"mean={dev[pole].mean():.3f} std={dev[pole].std():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dp", type=float, default=1.0)
    ap.add_argument("--t", type=float, default=0.05)
    ap.add_argument("--wn", type=float, default=1.0, help="normal-preservation weight")
    ap.add_argument("--stretch", type=float, default=2.0)
    ap.add_argument("--vmin", type=float, default=6.0)
    ap.add_argument("--vmax", type=float, default=23.0)
    ap.add_argument("--out", default="out/membrane_stress_beltrami.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    R = args.radius
    sphere = vedo.IcoSphere(r=R, subdivisions=args.subdiv)
    rs = solve_beltrami(sphere, args.dp, args.t, depth=args.depth, wn=args.wn)
    report("sphere", rs, args.dp, args.t, a=R, b=R)

    ell = vedo.IcoSphere(r=R, subdivisions=args.subdiv).scale([args.stretch, 1, 1])
    re = solve_beltrami(ell, args.dp, args.t, depth=args.depth, wn=args.wn)
    report(f"stretched x{args.stretch}", re, args.dp, args.t, a=args.stretch * R, b=R)

    plt = vedo.Plotter(N=2, size=(1500, 750), offscreen=not args.show,
                       title="Beltrami stress-function: sigma_max")
    for k, (m, r, title) in enumerate([(sphere, rs, f"Sphere R={R}"),
                                       (ell, re, f"Stretched x{args.stretch}")]):
        m.pointdata["sigma_max"] = np.maximum(r["sigma1"], r["sigma2"])
        m.cmap("plasma", "sigma_max", vmin=args.vmin, vmax=args.vmax).add_scalarbar(title="sigma_max")
        plt.at(k).show(m, vedo.Text2D(title + " (Beltrami)", pos="top-left"), axes=1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.screenshot(args.out); print(f"\nSaved {args.out}")
    if args.show:
        plt.interactive()
    plt.close()


if __name__ == "__main__":
    main()
