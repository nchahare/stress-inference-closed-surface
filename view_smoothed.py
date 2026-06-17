"""Interactive view of the Laplacian-smoothed sigma_max only, for the real meshes.

Loads the saved out/<tag>_stress.vtp (which carry every field as point data) and opens a
vedo window showing the smoothed sigma_max field for HH17 (decimated) and HH20 side by
side. No re-solve -- just visualizes the stored results.

Run (opens a window):
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe view_smoothed.py
Options: --field sigma_max_smooth|sigma_min_smooth|sigma_mean_smooth  --save out.png
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import vedo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="sigma_max_smooth",
                    help="point-data field to show (default: sigma_max_smooth)")
    ap.add_argument("--vtps", nargs="*",
                    default=["out/HH17_dec_stress.vtp", "out/HH20_stress.vtp"])
    ap.add_argument("--save", default="", help="optional screenshot path")
    args = ap.parse_args()

    items = []
    for path in args.vtps:
        if not os.path.exists(path):
            print(f"  (missing {path}, skipping)")
            continue
        m = vedo.Mesh(path)
        tag = os.path.basename(path).replace("_stress.vtp", "")
        fld = m.pointdata[args.field]
        lo, hi = float(np.percentile(fld, 2)), float(np.percentile(fld, 98))
        m.cmap("plasma", args.field, vmin=lo, vmax=hi).add_scalarbar(title=args.field)
        items.append((tag, m, lo, hi))

    if not items:
        print("No .vtp files found -- run real_mesh_stress.py first.")
        return

    plt = vedo.Plotter(shape=(1, len(items)), size=(800 * len(items), 800),
                       sharecam=False, offscreen=bool(args.save),
                       title=f"Laplacian-smoothed {args.field}")
    for k, (tag, m, lo, hi) in enumerate(items):
        label = f"{tag}\nLaplacian-smoothed {args.field}\nscale [{lo:.2g}, {hi:.2g}]"
        plt.at(k).show(m, vedo.Text2D(label, pos="top-left"), axes=1)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.screenshot(args.save)
        print(f"Saved {args.save}")
    else:
        plt.interactive()
    plt.close()


if __name__ == "__main__":
    main()
