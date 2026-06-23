"""Side-by-side forward neo-Hookean comparison of HH17 (decimated) and HH20 on a COMMON
scale, with a diverging colorbar that shows negative (compressive) values.

Unlike running each mesh through forward_neohookean.py separately (which normalises each to
its own unit radius and so hides the real size difference), here BOTH meshes are scaled by a
single common length so their relative physical size is preserved -- HH20 stays ~1.7x bigger,
and its larger radius shows up as higher Laplace tension. Both panels share one diverging
colour scale (coolwarm centred at 0: red = tension, blue = compression).

-> out/forward_hh_compare.png   (and --show for an interactive 2-panel window)

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe forward_compare_hh.py --show
"""

from __future__ import annotations

import argparse
import numpy as np
from scipy.optimize import minimize
import vedo

from forward_neohookean import reference_geometry, energy_and_grad, recover_stress

HH17, HH20 = "2025-09-18-16-46-HH17.vtk", "2025-10-23-13-06-HH20.vtk"


def load_common(path, L, decimate=None):
    """Load, optionally decimate, centre, and scale by the COMMON length L (preserving
    relative size); drop degenerate triangles and orient faces outward."""
    m = vedo.load(path).clean().triangulate()
    if decimate and m.npoints > decimate:
        m = m.decimate(n=decimate).clean().triangulate()
    X = m.coordinates.astype(float)
    X = (X - X.mean(0)) / L
    F = np.asarray(m.cells, dtype=int)
    nd = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[nd]
    A = 0.5 * np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
    F = F[A > 1e-10 * A.max()]
    x0, x1, x2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    if np.sum(np.einsum("ij,ij->i", x0, np.cross(x1, x2))) < 0:
        F = F[:, [0, 2, 1]]
    return X, F


def solve(X, F, mu, dp, maxiter):
    Minv, A0 = reference_geometry(X, F)
    r = minimize(energy_and_grad, X.ravel(), args=(X, F, Minv, A0, mu, dp), jac=True,
                 method="L-BFGS-B", options=dict(maxiter=maxiter, maxfun=2 * maxiter,
                                                 ftol=1e-13, gtol=1e-10))
    x = r.x.reshape(len(X), 3)
    s1, s2 = recover_stress(x, F, Minv, mu)
    drift = np.linalg.norm(x - X, axis=1).max() / np.linalg.norm(X, axis=1).mean()
    return x, s1, s2, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", type=float, default=20.0)
    ap.add_argument("--mu", type=float, default=5000.0)
    ap.add_argument("--decimate", type=int, default=3766)
    ap.add_argument("--maxiter", type=int, default=15000)
    ap.add_argument("--field", default="trace", choices=["trace", "sigma_max", "sigma_min"])
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # common scale = mean of the two raw mean-radii (keeps both O(1), preserves relative size)
    R = {}
    for tag, f in [("HH17", HH17), ("HH20", HH20)]:
        Xc = vedo.load(f).coordinates.astype(float)
        R[tag] = np.linalg.norm(Xc - Xc.mean(0), axis=1).mean()
    L = 0.5 * (R["HH17"] + R["HH20"])
    print(f"raw mean radii: HH17 {R['HH17']:.1f}  HH20 {R['HH20']:.1f}  (HH20/HH17={R['HH20']/R['HH17']:.2f})")
    print(f"common scale L = {L:.1f}  -> normalised radii HH17 {R['HH17']/L:.3f}  HH20 {R['HH20']/L:.3f}")

    panels = []
    for tag, f, dec in [("HH17", HH17, args.decimate), ("HH20", HH20, None)]:
        X, F = load_common(f, L, decimate=dec)
        x, s1, s2, drift = solve(X, F, args.mu, args.dp, args.maxiter)
        fld = {"trace": s1 + s2, "sigma_max": np.maximum(s1, s2),
               "sigma_min": np.minimum(s1, s2)}[args.field]
        print(f"{tag}: n={len(X)}  drift={drift:.2e}  {args.field} "
              f"median {np.median(fld):.3f}  [{np.percentile(fld,2):.2f}, {np.percentile(fld,98):.2f}]")
        panels.append((tag, vedo.Mesh([x, F]), fld))

    allv = np.concatenate([p[2] for p in panels])
    lim = float(min(np.percentile(np.abs(allv), 97), 15.0))      # symmetric, spike-robust
    print(f"shared diverging colour scale: [-{lim:.2f}, {lim:.2f}] (coolwarm, 0 = white)")

    plt = vedo.Plotter(N=2, size=(1700, 850), sharecam=False, offscreen=not args.show,
                       title="Forward NH: HH17 vs HH20 (common scale)")
    for k, (tag, m, fld) in enumerate(panels):
        m.celldata[args.field] = fld
        m.cmap("coolwarm", args.field, on="cells", vmin=-lim, vmax=lim)
        m.add_scalarbar(title=f"{args.field} (N/m)")
        plt.at(k).show(m, vedo.Text2D(f"{tag}  |  {args.field}", pos="top-left"),
                       axes=1, azimuth=30, elevation=15)
    plt.screenshot("out/forward_hh_compare.png")
    print("saved out/forward_hh_compare.png")
    if args.show:
        plt.interactive()
    plt.close()


if __name__ == "__main__":
    main()
