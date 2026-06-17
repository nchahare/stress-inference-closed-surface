"""Compare raw sigma_max, Laplacian-smoothed sigma_max, and mean stress.

The streaky 'lines' in the raw principal stress are the residual spurious deviatoric modes
of the GFDM operator, organized along the icosphere triangulation. They live in the
DEVIATORIC part, so:
  * a light Laplacian (umbrella) smoothing pass cleans them cosmetically, and
  * the MEAN stress (sigma_1+sigma_2)/2 is essentially free of them already.

Renders a 2x3 grid (rows: sphere, stretched spheroid):
    [ raw sigma_max | Laplacian-smoothed sigma_max | mean stress (sigma_1+sigma_2)/2 ]

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe stress_smoothing_compare.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.sparse as sp
import vedo

from membrane_stress_fd import solve_membrane


def one_ring_average_matrix(mesh: vedo.Mesh):
    """Row-normalized 1-ring adjacency matrix A (umbrella operator)."""
    adlist = mesh.compute_adjacency()
    n = mesh.npoints
    rows, cols, vals = [], [], []
    for i in range(n):
        nb = np.asarray(mesh.find_adjacent_vertices(i, depth=1, adjacency_list=adlist), dtype=int)
        nb = nb[nb != i]
        if len(nb) == 0:
            rows.append(i); cols.append(i); vals.append(1.0); continue
        w = 1.0 / len(nb)
        rows += [i] * len(nb); cols += list(nb); vals += [w] * len(nb)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def laplacian_smooth(f, A, iters=10, alpha=0.5):
    """Umbrella Laplacian smoothing: f <- (1-alpha) f + alpha A f, repeated."""
    out = f.copy()
    for _ in range(iters):
        out = (1.0 - alpha) * out + alpha * (A @ out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dp", type=float, default=1.0)
    ap.add_argument("--t", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--stretch", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=12, help="Laplacian smoothing iterations")
    ap.add_argument("--alpha", type=float, default=0.5, help="Laplacian smoothing step")
    ap.add_argument("--vmin", type=float, default=6.0, help="shared colour-scale minimum")
    ap.add_argument("--vmax", type=float, default=23.0, help="shared colour-scale maximum")
    ap.add_argument("--out", default="out/stress_smoothing_compare.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    R = args.radius
    cases = [
        ("Sphere", vedo.IcoSphere(r=R, subdivisions=args.subdiv)),
        (f"Stretched x{args.stretch}",
         vedo.IcoSphere(r=R, subdivisions=args.subdiv).scale([args.stretch, 1.0, 1.0])),
    ]

    plt = vedo.Plotter(N=6, shape=(2, 3), size=(1500, 1000), offscreen=not args.show,
                       title="Raw vs Laplacian-smoothed sigma_max vs mean stress")
    col_titles = ["raw sigma_max", f"Laplacian-smoothed (x{args.iters})", "mean (s1+s2)/2"]

    for row, (name, mesh) in enumerate(cases):
        res = solve_membrane(mesh, args.dp, args.t, depth=args.depth, lam=args.lam)
        smax = np.maximum(res["sigma1"], res["sigma2"])
        smean = 0.5 * (res["sigma1"] + res["sigma2"])
        A = one_ring_average_matrix(mesh)
        smax_s = laplacian_smooth(smax, A, iters=args.iters, alpha=args.alpha)

        # report how much the deviatoric noise drops
        print(f"[{name}] resid={res['resid']:.2e}  "
              f"sigma_max std: raw={smax.std():.3f} -> smoothed={smax_s.std():.3f}; "
              f"mean-stress std={smean.std():.3f}")

        fields = [("smax_raw", smax), ("smax_smooth", smax_s), ("smean", smean)]
        # fixed shared colour scale across ALL panels for direct comparison
        for col, (key, fld) in enumerate(fields):
            m = mesh.clone()
            m.pointdata[key] = fld
            m.cmap("plasma", key, vmin=args.vmin, vmax=args.vmax).add_scalarbar(title=key)
            idx = row * 3 + col
            label = f"{name}\n{col_titles[col]}"
            plt.at(idx).show(m, vedo.Text2D(label, pos="top-left"), axes=0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.screenshot(args.out)
    print(f"\nSaved {args.out}")
    if args.show:
        plt.interactive()
    plt.close()


if __name__ == "__main__":
    main()
