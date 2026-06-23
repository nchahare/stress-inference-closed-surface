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
import os
import numpy as np
from scipy.optimize import minimize
import vedo

from forward_neohookean import reference_geometry, energy_and_grad, recover_stress

HH17, HH20 = "2025-09-18-16-46-HH17.vtk", "2025-10-23-13-06-HH20.vtk"
FIELDS = ["trace", "sigma_max", "sigma_min", "sigma1", "sigma2", "shear"]


def all_fields(s1, s2):
    """All per-triangle stress scalars from the two principal resultants."""
    smax, smin = np.maximum(s1, s2), np.minimum(s1, s2)
    return {"sigma1": s1, "sigma2": s2, "sigma_max": smax, "sigma_min": smin,
            "trace": s1 + s2, "shear": 0.5 * (smax - smin)}


def panel_mesh(x, F, s1, s2):
    """Deformed mesh carrying every stress field as cell data (for save / colour)."""
    m = vedo.Mesh([x, F])
    for k, v in all_fields(s1, s2).items():
        m.celldata[k] = v
    return m


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
    ap.add_argument("--field", default="trace", choices=FIELDS)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--load", action="store_true",
                    help="skip the solve and reload saved out/forward_<tag>.vtp")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    panels = []
    if args.load:                                   # reuse saved results -- no solving
        for tag in ("HH17", "HH20"):
            path = os.path.join(args.outdir, f"forward_{tag}.vtp")
            m = vedo.load(path)
            print(f"loaded {path}  ({m.ncells} tris, fields: {list(m.celldata.keys())})")
            panels.append((tag, m))
    else:
        # common scale = mean of the two raw mean-radii (keeps both O(1), preserves relative size)
        R = {}
        for tag, f in [("HH17", HH17), ("HH20", HH20)]:
            Xc = vedo.load(f).coordinates.astype(float)
            R[tag] = np.linalg.norm(Xc - Xc.mean(0), axis=1).mean()
        L = 0.5 * (R["HH17"] + R["HH20"])
        print(f"raw mean radii: HH17 {R['HH17']:.1f}  HH20 {R['HH20']:.1f}  "
              f"(HH20/HH17={R['HH20']/R['HH17']:.2f}); common scale L={L:.1f}")
        for tag, f, dec in [("HH17", HH17, args.decimate), ("HH20", HH20, None)]:
            X, F = load_common(f, L, decimate=dec)
            x, s1, s2, drift = solve(X, F, args.mu, args.dp, args.maxiter)
            tr = s1 + s2
            print(f"{tag}: n={len(X)}  drift={drift:.2e}  trace median {np.median(tr):.3f}  "
                  f"[{np.percentile(tr,2):.2f}, {np.percentile(tr,98):.2f}]")
            m = panel_mesh(x, F, s1, s2)
            path = os.path.join(args.outdir, f"forward_{tag}.vtp")
            m.write(path)
            print(f"  saved {path}")
            panels.append((tag, m))

    render(panels, args.field, args.show, args.outdir)


def render(panels, field, show, outdir):
    allv = np.concatenate([p[1].celldata[field] for p in panels])
    lim = float(min(np.percentile(np.abs(allv), 97), 15.0))      # symmetric, spike-robust
    print(f"shared diverging colour scale ({field}): [-{lim:.2f}, {lim:.2f}] (coolwarm, 0=white)")
    plt = vedo.Plotter(N=2, size=(1700, 850), sharecam=False, offscreen=not show,
                       title="Forward NH: HH17 vs HH20 (common scale)")
    for k, (tag, m) in enumerate(panels):
        m.cmap("coolwarm", field, on="cells", vmin=-lim, vmax=lim)
        m.add_scalarbar(title=f"{field} (N/m)")
        plt.at(k).show(m, vedo.Text2D(f"{tag}  |  {field}", pos="top-left"),
                       axes=1, azimuth=30, elevation=15)
    out = os.path.join(outdir, "forward_hh_compare.png")
    plt.screenshot(out); print(f"saved {out}")
    if show:
        plt.interactive()
    plt.close()


if __name__ == "__main__":
    main()
