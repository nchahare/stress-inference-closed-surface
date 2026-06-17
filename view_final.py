"""Interactive viewer for the final-results stress fields -- rotate the view yourself.

Shows the two meshes of a group side by side with a SHARED colour scale (2-98 percentile
over BOTH meshes), so colour limits match within a group as requested:
    sphere + ellipsoid share limits ;  HH17 + HH20 share limits.

Examples:
    & $py view_final.py --group analytic --method m2          # sphere + ellipsoid, cMSM
    & $py view_final.py --group analytic --method m1          # local (axisym two-curvature)
    & $py view_final.py --group real     --method m2          # HH17 + HH20, cMSM
    & $py view_final.py --group real     --method m2 --field sigma_min
The window is interactive: drag to rotate, scroll to zoom. Pass --save PATH to dump a PNG
instead (uses the default camera).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import vedo

FILES = {
    ("analytic", "m1"): ["sim01_sphere_uniform_local.vtp", "sim07_ellipsoid_uniform_local.vtp"],
    ("analytic", "m2"): ["sim02_sphere_uniform_cmsm.vtp", "sim08_ellipsoid_uniform_cmsm.vtp"],
    ("real", "m1"): ["hh17_local.vtp", "hh20_local.vtp"],
    ("real", "m2"): ["hh17_cmsm.vtp", "hh20_cmsm.vtp"],
}
LABELS = {
    ("analytic", "m1"): ["Sphere  -  M1 local", "Ellipsoid  -  M1 local"],
    ("analytic", "m2"): ["Sphere  -  M2 cMSM", "Ellipsoid  -  M2 cMSM"],
    ("real", "m1"): ["HH17  -  M1 local", "HH20  -  M1 local"],
    ("real", "m2"): ["HH17  -  M2 cMSM", "HH20  -  M2 cMSM"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", choices=["analytic", "real"], default="analytic")
    ap.add_argument("--method", choices=["m1", "m2"], default="m2")
    ap.add_argument("--field", default="sigma_max",
                    help="sigma_max | sigma_min | sigma1 | sigma2")
    ap.add_argument("--dir", default="out/final")
    ap.add_argument("--cmap", default="plasma")
    ap.add_argument("--save", default="", help="optional screenshot path (offscreen)")
    args = ap.parse_args()

    files = [os.path.join(args.dir, f) for f in FILES[(args.group, args.method)]]
    labels = LABELS[(args.group, args.method)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"missing data: {missing}\nRun final_sims.py (analytic) / "
                         f"final_real.py (real) first.")

    meshes = [vedo.Mesh(f) for f in files]
    vals = np.concatenate([np.asarray(m.pointdata[args.field]) for m in meshes])
    lo, hi = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
    print(f"{args.group}/{args.method}  field={args.field}  "
          f"SHARED clim=[{lo:.4g}, {hi:.4g}]")

    plt = vedo.Plotter(shape=(1, 2), size=(1500, 760), sharecam=False,
                       title=f"{args.group} {args.method} {args.field}",
                       offscreen=bool(args.save))
    for i, (m, lab) in enumerate(zip(meshes, labels)):
        m.cmap(args.cmap, args.field, vmin=lo, vmax=hi)
        if i == len(meshes) - 1:
            m.add_scalarbar(title=args.field)
        plt.at(i).show(m, vedo.Text2D(f"{lab}\nclim [{lo:.3g}, {hi:.3g}]",
                                      pos="top-left"), axes=1)
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.screenshot(args.save); print(f"saved {args.save}"); plt.close()
    else:
        plt.interactive(); plt.close()


if __name__ == "__main__":
    main()
