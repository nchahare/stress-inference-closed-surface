"""Show mean curvature (H) + normal arrows on a sphere, then stretch it and compare.

Reuses ``compute_vertex_frames`` from ``sphere_curvature.py``. Builds a vedo sphere,
colours it by mean curvature with normal-vector arrows, then applies ``mesh.scale`` to
stretch it along one axis (-> ellipsoid), re-runs the same curvature calculation, and
shows the two side by side. A sphere has constant H = 1/R; the stretched ellipsoid has
spatially varying H, which is the difference we want to see.

Run (saves a PNG comparison, headless):
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe curvature_compare.py
Interactive window instead of / in addition to the PNG:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe curvature_compare.py --show
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import vedo

from sphere_curvature import compute_vertex_frames


def build_actors(mesh: vedo.Mesh, fields: dict, every: int, arrow_scale: float, clim=None):
    """Return [coloured mesh, normal arrows] for one viewport."""
    H = fields["H"].copy()
    mesh.pointdata["H"] = np.nan_to_num(H, nan=0.0)
    mesh.cmap("coolwarm", "H", vmin=clim[0] if clim else None,
              vmax=clim[1] if clim else None)
    mesh.add_scalarbar(title="Mean curvature H")

    valid = ~np.isnan(fields["H"])
    idx = np.where(valid)[0][::every]
    p = fields["pts"][idx]
    # orient arrows outward for a consistent look
    nrm = fields["normals"][idx].copy()
    radial = p - p.mean(axis=0)
    flip = np.sum(nrm * radial, axis=1) < 0
    nrm[flip] *= -1
    arrows = vedo.Arrows(p, p + arrow_scale * nrm, c="black", alpha=0.6)
    return [mesh, arrows]


def report(tag: str, fields: dict):
    H = fields["H"][~np.isnan(fields["H"])]
    print(f"\n[{tag}] mean curvature H over {H.size} vertices:")
    print(f"    mean={H.mean():+.4f}  std={H.std():.4f}  "
          f"min={H.min():+.4f}  max={H.max():+.4f}  range={H.max()-H.min():.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--res", type=int, default=50)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--stretch", type=float, default=2.0,
                    help="scale factor along x to stretch the sphere into an ellipsoid")
    ap.add_argument("--every", type=int, default=60, help="show one normal arrow per N vertices")
    ap.add_argument("--out", default="out/curvature_compare.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    R = args.radius
    arrow_scale = 0.3 * R

    # --- 1) sphere -------------------------------------------------------------
    sphere = vedo.Sphere(r=R, res=args.res)
    f_sphere = compute_vertex_frames(sphere, depth=args.depth)
    report(f"sphere r={R}", f_sphere)

    # --- 2) stretched sphere (ellipsoid) via vedo scale ------------------------
    ellipsoid = vedo.Sphere(r=R, res=args.res).scale([args.stretch, 1.0, 1.0])
    f_ell = compute_vertex_frames(ellipsoid, depth=args.depth)
    report(f"stretched x{args.stretch} (ellipsoid)", f_ell)

    # shared colour limits so the two panels are directly comparable
    allH = np.concatenate([f_sphere["H"][~np.isnan(f_sphere["H"])],
                           f_ell["H"][~np.isnan(f_ell["H"])]])
    clim = (float(np.percentile(allH, 2)), float(np.percentile(allH, 98)))

    a_sphere = build_actors(sphere, f_sphere, args.every, arrow_scale, clim)
    a_ell = build_actors(ellipsoid, f_ell, args.every, arrow_scale, clim)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt = vedo.Plotter(N=2, size=(1600, 800), offscreen=not args.show,
                       title="Mean curvature + normals: sphere vs stretched")
    plt.at(0).show(a_sphere, vedo.Text2D(f"Sphere r={R}\nH = 1/R (constant)",
                                         pos="top-left"), axes=1)
    plt.at(1).show(a_ell, vedo.Text2D(f"Stretched x{args.stretch} (ellipsoid)\nH varies",
                                      pos="top-left"), axes=1)
    plt.screenshot(args.out)
    print(f"\nSaved comparison image: {args.out}")
    if args.show:
        plt.interactive()
    plt.close()


if __name__ == "__main__":
    main()
