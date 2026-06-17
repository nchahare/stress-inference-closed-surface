"""Final-results runs on the real meshes HH17 / HH20 (M1 local + M2 cMSM).

HH17 is decimated to HH20's vertex count so the two are compared at matched resolution.
Config: dp=20, t=0.05, depth3, lam0.02, Laplacian smoothing 12 iters @ alpha 0.5.

M1 here is the ISOTROPIC mean-curvature estimate sigma=dp/(2tH) (valid on any surface);
the axisymmetric two-curvature M1 used for the sphere/ellipsoid has no axis of revolution
on the neural tube. M2 is the GFDM inference + Laplacian smoothing.

Saves per-vertex CSV/NPZ/VTP to out/final/ (no figure -- use view_final.py to visualize).

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe final_real.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import vedo

from membrane_stress_fd import solve_membrane
from reg_compare import avg_operator, laplacian_smooth
from local_stress import local_stress_isotropic

MESHES = [("HH17", "2025-09-18-16-46-HH17.vtk"),
          ("HH20", "2025-10-23-13-06-HH20.vtk")]


def save_fields(mesh, pts, s1, s2, tag, outdir):
    os.makedirs(outdir, exist_ok=True)
    smax = np.maximum(s1, s2); smin = np.minimum(s1, s2)
    np.savetxt(os.path.join(outdir, f"{tag}.csv"),
               np.column_stack([pts, s1, s2, smax, smin]), delimiter=",",
               header="X,Y,Z,sigma1,sigma2,sigma_max,sigma_min", comments="")
    np.savez(os.path.join(outdir, f"{tag}.npz"), pts=pts, faces=np.asarray(mesh.cells),
             sigma1=s1, sigma2=s2, sigma_max=smax, sigma_min=smin)
    m = mesh.clone()
    m.pointdata["sigma1"] = s1; m.pointdata["sigma2"] = s2
    m.pointdata["sigma_max"] = smax; m.pointdata["sigma_min"] = smin
    m.write(os.path.join(outdir, f"{tag}.vtp"))
    print(f"    saved {tag}.{{csv,npz,vtp}}  (sigma_max mean={smax.mean():.4g}, "
          f"std={smax.std():.4g})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dp", type=float, default=20.0)
    ap.add_argument("--t", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--outdir", default="out/final")
    args = ap.parse_args()

    # match HH17 resolution to HH20
    target = vedo.Mesh(MESHES[1][1]).clean().npoints
    print(f"HH20 vertex count = {target}; HH17 will be decimated to ~that.")

    for tag, path in MESHES:
        mesh = vedo.Mesh(path).clean()
        print(f"\n=== {tag} ({path})  npoints={mesh.npoints} ===")
        if mesh.npoints > target * 1.05:
            mesh = mesh.decimate(fraction=target / mesh.npoints).clean()
            print(f"    decimated -> npoints={mesh.npoints}")

        # M1 -- isotropic mean-curvature local estimate
        m1 = local_stress_isotropic(mesh, args.dp, args.t, depth=args.depth)
        save_fields(mesh, m1["pts"], m1["sigma1"], m1["sigma2"],
                    f"{tag.lower()}_local", args.outdir)

        # M2 -- GFDM inference + Laplacian smoothing
        raw = solve_membrane(mesh, args.dp, args.t, depth=args.depth, lam=args.lam)
        avg = avg_operator(mesh)
        s1 = laplacian_smooth(avg, raw["sigma1"], args.iters, args.alpha)
        s2 = laplacian_smooth(avg, raw["sigma2"], args.iters, args.alpha)
        print(f"    M2 equilibrium residual {raw['resid']:.2e}")
        save_fields(mesh, raw["pts"], s1, s2, f"{tag.lower()}_cmsm", args.outdir)


if __name__ == "__main__":
    main()
