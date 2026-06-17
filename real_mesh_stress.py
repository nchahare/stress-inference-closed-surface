"""Direct GFDM membrane stress + Laplacian smoothing on real surface meshes.

Runs the validated pipeline (``membrane_stress_fd.solve_membrane`` -> Laplacian/umbrella
smoothing, the combination confirmed as the best *accurate* line-removal on the sphere /
spheroid) on the real chick-embryo meshes (HH17, HH20 ``.vtk``), and stores the per-vertex
results in several formats so the fields can be re-plotted any way later:

  out/<tag>_stress.csv   - per-vertex table (X,Y,Z, normal, raw & smoothed sigma fields)
  out/<tag>_stress.npz   - same arrays as a compressed numpy archive (+ faces for 3D plots)
  out/<tag>_stress.vtp   - the mesh with all fields as point data (open in ParaView / vedo)
  out/real_mesh_stress_compare.png - side-by-side raw vs smoothed sigma_max render

These meshes are closed surfaces (0 boundary edges), so the same centroid-outward normal
orientation and pressure load used for the sphere applies unchanged - no axisymmetry, no
boundary conditions. dp (pressure) and t (wall thickness) are placeholders: the stress
scales linearly as dp/t, so the *pattern* is independent of them; pass measured values
when available.

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe real_mesh_stress.py
    & ... real_mesh_stress.py --show
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import vedo

from membrane_stress_fd import solve_membrane
from stress_smoothing_compare import one_ring_average_matrix, laplacian_smooth

MESHES = [
    ("HH17", "2025-09-18-16-46-HH17.vtk"),
    ("HH20", "2025-10-23-13-06-HH20.vtk"),
]


def process(tag, path, dp, t, depth, lam, iters, alpha, outdir, decimate_to=0):
    mesh = vedo.Mesh(path).clean()
    print(f"\n=== {tag}  ({path}) ===")
    print(f"    npoints={mesh.npoints}  ncells={mesh.ncells}")
    if decimate_to and mesh.npoints > decimate_to:
        mesh = mesh.decimate(fraction=decimate_to / mesh.npoints).clean()
        tag = f"{tag}_dec"
        print(f"    decimated -> npoints={mesh.npoints}  ncells={mesh.ncells}")

    res = solve_membrane(mesh, dp, t, depth=depth, lam=lam)
    s1, s2 = res["sigma1"], res["sigma2"]
    smax = np.maximum(s1, s2)
    smin = np.minimum(s1, s2)
    smean = 0.5 * (s1 + s2)

    # Laplacian (umbrella) smoothing of each principal-stress field on the 1-ring graph
    A = one_ring_average_matrix(mesh)
    s1_s = laplacian_smooth(s1, A, iters=iters, alpha=alpha)
    s2_s = laplacian_smooth(s2, A, iters=iters, alpha=alpha)
    smax_s = laplacian_smooth(smax, A, iters=iters, alpha=alpha)
    smin_s = laplacian_smooth(smin, A, iters=iters, alpha=alpha)
    smean_s = laplacian_smooth(smean, A, iters=iters, alpha=alpha)

    print(f"    equilibrium residual ||LS-b||/||b|| = {res['resid']:.3e}")
    print(f"    sigma_max: raw mean={smax.mean():.4g} std={smax.std():.4g} "
          f"-> smoothed std={smax_s.std():.4g}")
    print(f"    sigma_min: raw mean={smin.mean():.4g} std={smin.std():.4g} "
          f"-> smoothed std={smin_s.std():.4g}")
    print(f"    mean stress: mean={smean.mean():.4g} std={smean.std():.4g} "
          f"-> smoothed std={smean_s.std():.4g}")

    pts = res["pts"]
    nrm = res["normals"]
    faces = np.asarray(mesh.cells, dtype=np.int64)

    os.makedirs(outdir, exist_ok=True)

    # --- structured arrays for re-plotting ------------------------------------------
    fields = dict(
        X=pts[:, 0], Y=pts[:, 1], Z=pts[:, 2],
        nx=nrm[:, 0], ny=nrm[:, 1], nz=nrm[:, 2],
        sigma1=s1, sigma2=s2,
        sigma_max=smax, sigma_min=smin, sigma_mean=smean,
        sigma1_smooth=s1_s, sigma2_smooth=s2_s,
        sigma_max_smooth=smax_s, sigma_min_smooth=smin_s, sigma_mean_smooth=smean_s,
    )

    # CSV
    cols = list(fields.keys())
    header = ",".join(cols)
    data = np.column_stack([fields[c] for c in cols])
    csv_path = os.path.join(outdir, f"{tag}_stress.csv")
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    # NPZ (+ faces so the surface can be reconstructed for 3D plots)
    npz_path = os.path.join(outdir, f"{tag}_stress.npz")
    np.savez_compressed(npz_path, faces=faces, dp=dp, t=t,
                        depth=depth, lam=lam, iters=iters, alpha=alpha, **fields)

    # VTP with all fields as point data (ParaView / vedo 3D viewing)
    out_mesh = mesh.clone()
    for k, v in fields.items():
        out_mesh.pointdata[k] = v
    vtp_path = os.path.join(outdir, f"{tag}_stress.vtp")
    out_mesh.write(vtp_path)

    print(f"    saved {csv_path}")
    print(f"    saved {npz_path}")
    print(f"    saved {vtp_path}")

    return dict(tag=tag, mesh=mesh, smax=smax, smax_s=smax_s, smean_s=smean_s)


def render_compare(results, out, show):
    """2 rows (meshes) x 3 cols: raw sigma_max | smoothed sigma_max | smoothed mean.
    Each mesh uses its own 2-98 percentile colour scale (the two embryos differ in
    absolute stress), printed in the panel label."""
    nrow = len(results)
    plt = vedo.Plotter(shape=(nrow, 3), size=(1500, 500 * nrow),
                       offscreen=not show, title="Real meshes: raw vs Laplacian-smoothed")
    col_titles = ["raw sigma_max", "Laplacian-smoothed sigma_max", "smoothed mean (s1+s2)/2"]

    for row, r in enumerate(results):
        lo = float(np.percentile(r["smax_s"], 2))
        hi = float(np.percentile(r["smax_s"], 98))
        panels = [("smax_raw", r["smax"]), ("smax_smooth", r["smax_s"]),
                  ("smean_smooth", r["smean_s"])]
        for col, (key, fld) in enumerate(panels):
            m = r["mesh"].clone()
            m.pointdata[key] = fld
            m.cmap("plasma", key, vmin=lo, vmax=hi).add_scalarbar(title=key)
            idx = row * 3 + col
            label = f"{r['tag']}\n{col_titles[col]}\nscale [{lo:.2g},{hi:.2g}]"
            plt.at(idx).show(m, vedo.Text2D(label, pos="top-left"), axes=1)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.screenshot(out)
    print(f"\nSaved {out}")
    if show:
        plt.interactive()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dp", type=float, default=1.0, help="pressure jump (placeholder)")
    ap.add_argument("--t", type=float, default=0.05, help="wall thickness (placeholder)")
    ap.add_argument("--depth", type=int, default=3, help="GFDM neighbourhood ring depth")
    ap.add_argument("--lam", type=float, default=0.02, help="Tikhonov smoothing weight")
    ap.add_argument("--iters", type=int, default=12, help="Laplacian smoothing iterations")
    ap.add_argument("--alpha", type=float, default=0.5, help="Laplacian smoothing step")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--out", default="out/real_mesh_stress_compare.png")
    ap.add_argument("--decimate", type=int, default=4000,
                    help="decimate meshes larger than this to ~this many points "
                         "(0 = no decimation; default matches HH17 to HH20 size)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    results = []
    for tag, path in MESHES:
        results.append(process(tag, path, args.dp, args.t, args.depth,
                               args.lam, args.iters, args.alpha, args.outdir,
                               decimate_to=args.decimate))
    render_compare(results, args.out, args.show)


if __name__ == "__main__":
    main()
