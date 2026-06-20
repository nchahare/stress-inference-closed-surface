"""Export the stress-based FEM solution to VTK (.vtp) files for sphere, prolate/oblate
spheroids and the capsule -- open in ParaView for 3D inspection / glyphing.

Each file carries, per vertex: the principal stresses sigma1/sigma2 (and sigma_max/min),
the trace (sigma1+sigma2), mean, von Mises and shear scalars, the resultants N1/N2, the
principal-stress direction vectors d1/d2 and the surface normal, and -- where an analytic
solution exists -- sigma_max/sigma_min/trace from the closed form for direct comparison.
The lambda used (auto by default) is stored as field data and printed.

Run (PowerShell):
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe export_fem_vtk.py
    & .../python.exe export_fem_vtk.py --shapes sphere prolate oblate capsule --lam auto
    & .../python.exe export_fem_vtk.py --shapes capsule --subdiv 5 --lam 0.05 --outdir out/fem_vtk
"""

from __future__ import annotations

import argparse
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np

from membrane_stress_fem import solve_membrane_fem, stress_scalar
from fem_smoothing_sweep import build_mesh, analytic_coord_mask, SHAPES

SCALARS = ["sigma_max", "sigma_min", "trace", "mean", "vonmises", "shear"]


def export(shape: str, subdiv: int, dp: float, t: float, lam, outdir: str):
    mesh = build_mesh(shape, subdiv)
    r = solve_membrane_fem(mesh, dp, t, depth=3, lam=lam, raw=False)
    pts = r["pts"]

    # per-vertex scalars + resultants
    for name in SCALARS:
        mesh.pointdata[name] = stress_scalar(r, name)
    mesh.pointdata["sigma1"] = r["sigma1"]
    mesh.pointdata["sigma2"] = r["sigma2"]
    mesh.pointdata["N1"] = r["N1"]
    mesh.pointdata["N2"] = r["N2"]
    # per-vertex direction / normal vectors
    mesh.pointdata["d1"] = r["d1"]
    mesh.pointdata["d2"] = r["d2"]
    mesh.pointdata["normal"] = r["normals"]

    # analytic comparison (all our shapes are axisymmetric / piecewise-analytic)
    an_max, an_min, an_tr, coord, _, mask = analytic_coord_mask(shape, pts)
    mesh.pointdata["sigma_max_analytic"] = an_max
    mesh.pointdata["sigma_min_analytic"] = an_min
    mesh.pointdata["trace_analytic"] = an_tr
    mesh.pointdata["belt_mask"] = mask.astype(np.int8)        # 1 where error is measured

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"fem_{shape}.vtp")
    mesh.write(path)
    print(f"  [{shape}] n={mesh.npoints}  lam={r['lam']:.4g}  resid={r['resid']:.2e}  -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", nargs="+", default=["sphere", "prolate", "oblate", "capsule"],
                    choices=list(SHAPES))
    ap.add_argument("--subdiv", type=int, default=4, help="icosphere subdivisions (spheroids)")
    ap.add_argument("--dp", type=float, default=20.0, help="pressure jump (project default 20 Pa)")
    ap.add_argument("--t", type=float, default=0.05, help="wall thickness")
    ap.add_argument("--lam", default="auto",
                    help="Tikhonov weight: a float, or 'auto' (L-curve corner per mesh)")
    ap.add_argument("--outdir", default="out/fem_vtk")
    args = ap.parse_args()
    lam = "auto" if str(args.lam).lower() == "auto" else float(args.lam)

    print(f"Exporting FEM stress to VTK (dp={args.dp}, t={args.t}, lam={args.lam}) -> {args.outdir}")
    for shape in args.shapes:
        export(shape, args.subdiv, args.dp, args.t, lam, args.outdir)


if __name__ == "__main__":
    main()
