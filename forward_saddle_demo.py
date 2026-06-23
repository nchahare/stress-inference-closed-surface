"""Forward neo-Hookean inflation on a PEANUT (dumbbell) shape -- does the negative
Gaussian-curvature neck go compressive / wrinkle?

The peanut is a surface of revolution with two lobes and a waist. At the waist the
meridian curves toward the axis (one principal curvature) while the azimuthal circle
curves the other way -> the two principal curvatures have opposite signs -> negative
Gaussian curvature K<0 (a saddle / anticlastic patch). The lobes are convex (K>0).

We inflate a stress-free peanut reference under internal pressure with the same stress-free
incompressible neo-Hookean membrane as forward_neohookean.py, recover the principal
resultants, and ask: where does sigma_min go negative (compression), and how does that
correlate with the sign of K? A passive membrane cannot hold compression -- it wrinkles
(sigma_min -> 0) -- so the neck is the place to watch.

-> out/forward_peanut_sigmin.png  (deformed shape coloured by sigma_min, diverging)
-> out/forward_peanut_gauss.png    (coloured by sign of Gaussian curvature)

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe forward_saddle_demo.py --show
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from scipy.optimize import minimize
import vedo

from forward_neohookean import reference_geometry, energy_and_grad, recover_stress


def build_peanut(nv: int, nphi: int, d: float, Lz: float) -> vedo.Mesh:
    """Surface of revolution: rho(v)=sin(v)(1-d sin^2 v), z(v)=Lz cos(v), v in [0,pi].
    d controls the waist depth (bigger d -> thinner neck); Lz elongates the body."""
    verts: list = []
    faces: list = []

    def profile(v):
        s = np.sin(v)
        return s * (1.0 - d * s * s), Lz * np.cos(v)

    def ring(v):
        idx = len(verts)
        rho, z = profile(v)
        for j in range(nphi):
            a = 2.0 * np.pi * j / nphi
            verts.append([rho * np.cos(a), rho * np.sin(a), z])
        return idx

    def strip(r0, r1):
        for j in range(nphi):
            a0, a1 = r0 + j, r0 + (j + 1) % nphi
            b0, b1 = r1 + j, r1 + (j + 1) % nphi
            faces.append([a0, b0, a1]); faces.append([b0, b1, a1])

    vs = np.linspace(0.0, np.pi, nv + 1)
    ptop = len(verts); verts.append([0.0, 0.0, Lz])         # top pole
    prev = None
    for i in range(1, nv):
        idx = ring(vs[i])
        if i == 1:
            for j in range(nphi):
                faces.append([ptop, idx + j, idx + (j + 1) % nphi])
        else:
            strip(prev, idx)
        prev = idx
    pbot = len(verts); verts.append([0.0, 0.0, -Lz])        # bottom pole
    for j in range(nphi):
        faces.append([pbot, prev + (j + 1) % nphi, prev + j])
    return vedo.Mesh([np.array(verts, float), faces])


def solve_forward(mesh, dp, mu):
    X = mesh.coordinates.astype(float)
    faces = np.asarray(mesh.cells, dtype=int)
    Minv, A0 = reference_geometry(X, faces)
    res = minimize(energy_and_grad, X.ravel(), args=(X, faces, Minv, A0, mu, dp),
                   jac=True, method="L-BFGS-B",
                   options=dict(maxiter=6000, ftol=1e-12, gtol=1e-9))
    x = res.x.reshape(len(X), 3)
    s1, s2 = recover_stress(x, faces, Minv, mu)
    return X, x, faces, s1, s2, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nv", type=int, default=70)
    ap.add_argument("--nphi", type=int, default=70)
    ap.add_argument("--d", type=float, default=0.8, help="waist depth (0..1)")
    ap.add_argument("--Lz", type=float, default=1.6, help="axial elongation")
    ap.add_argument("--dp", type=float, default=20.0)
    ap.add_argument("--mu", type=float, default=1000.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    mesh = build_peanut(args.nv, args.nphi, args.d, args.Lz)
    mesh.compute_curvature(method=0)                          # Gaussian -> pointdata
    Kv = mesh.pointdata["Gauss_Curvature"]
    faces = np.asarray(mesh.cells, dtype=int)
    Ktri = Kv[faces].mean(axis=1)                            # per-triangle Gaussian curvature
    print(f"peanut: n={mesh.npoints} verts, {len(faces)} tris, d={args.d}, Lz={args.Lz}, "
          f"mu_s={args.mu}, dp={args.dp}")

    X, x, faces, s1, s2, res = solve_forward(mesh, args.dp, args.mu)
    print(f"  L-BFGS: {res.nit} iters, ||grad||={np.linalg.norm(res.jac):.2e}")
    smin = np.minimum(s1, s2)
    drift = np.linalg.norm(x - X, axis=1).max() / np.linalg.norm(X, axis=1).mean()

    Kneg, Kpos = Ktri < 0, Ktri > 0
    print(f"  shape drift: {drift:.2e}")
    print(f"  K<0 (saddle) tris: {Kneg.mean():.1%}   K>0 (convex) tris: {Kpos.mean():.1%}")
    print(f"  sigma_min < 0 (compression) overall:        {np.mean(smin < 0):.1%}")
    print(f"  sigma_min < 0  WITHIN the K<0 saddle band:  {np.mean(smin[Kneg] < 0):.1%}")
    print(f"  sigma_min < 0  WITHIN the K>0 convex lobes: {np.mean(smin[Kpos] < 0):.1%}")
    print(f"  mean sigma_min  in K<0 band: {smin[Kneg].mean():.3f}   "
          f"in K>0 lobes: {smin[Kpos].mean():.3f}   (N/m)")

    os.makedirs("out", exist_ok=True)
    lim = float(np.percentile(np.abs(smin), 98))
    for field, vals, cmap, vlim, tag in [
        ("sigma_min", smin, "coolwarm", (-lim, lim), "sigmin"),
        ("Gauss_K_sign", np.sign(Ktri), "PiYG", (-1, 1), "gauss"),
    ]:
        dm = vedo.Mesh([x, faces]); dm.celldata[field] = vals
        dm.cmap(cmap, field, on="cells", vmin=vlim[0], vmax=vlim[1])
        dm.add_scalarbar(title=field)
        txt = vedo.Text2D(f"Forward NH peanut  |  {field}", pos="top-left")
        plt = vedo.Plotter(offscreen=not args.show, size=(950, 800))
        plt.show(dm, txt, axes=1, azimuth=20, elevation=10)
        out = f"out/forward_peanut_{tag}.png"
        plt.screenshot(out); print(f"  saved {out}")
        if args.show:
            plt.interactive()
        plt.close()


if __name__ == "__main__":
    main()
