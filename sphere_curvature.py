"""Per-vertex curvature, normals and local axes on a surface mesh (vedo).

Adapted from the spatchcocking curvature routine
(https://github.com/nchahare/spatchcocking, src/spatchcocking/spatchcocking_utils.py:
``getProperCurvature`` / ``compute_and_save_curvatures``), which is built around
``vedo.project_point_on_variety``: a local degree-2 polynomial surface is fit to the
k-ring neighbourhood of each vertex, yielding the Gaussian (K) and mean (H) curvature
plus the 3x3 rotation matrix R that defines the *local frame* in which the fit is done
(rows = tangent v1, tangent v2, normal).

We start with a sphere because it is an exact analytical validation case:
    K = 1/R^2,  H = 1/R,  k1 = k2 = 1/R,  principal radii = R,  normal = radial.

Run (headless validation report):
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe sphere_curvature.py
Interactive 3D view (curvature colouring + local-axis glyphs):
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe sphere_curvature.py --show
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import vedo


def compute_vertex_frames(mesh: vedo.Mesh, depth: int = 3, degree: int = 2):
    """Fit a local polynomial surface at every vertex and return per-vertex fields.

    For each vertex we gather its k-ring neighbourhood (``find_adjacent_vertices`` with
    the given ``depth``) and call ``vedo.project_point_on_variety``. From the returned
    ``(coeffs, R, centroid, K, H)`` we keep the local frame (R) and the curvatures, and
    derive principal curvatures / radii.

    Returns a dict of arrays, each of length ``mesh.npoints``.
    """
    mesh.compute_normals()
    adlist = mesh.compute_adjacency()

    pts = np.asarray(mesh.coordinates)
    vnormals = np.asarray(mesh.vertex_normals)
    n = len(pts)

    normals = np.zeros((n, 3))  # normal from the fit (R row 2)
    v1 = np.zeros((n, 3))       # tangent axis 1 (R row 0)
    v2 = np.zeros((n, 3))       # tangent axis 2 (R row 1)
    K = np.full(n, np.nan)      # Gaussian curvature
    H = np.full(n, np.nan)      # mean curvature

    for i in range(n):
        neigh = mesh.find_adjacent_vertices(i, depth=depth, adjacency_list=adlist)
        neigh = np.unique(np.append(neigh, i))
        bpts = pts[neigh]
        # need enough points to fit a degree-2 surface (6 coeffs)
        if len(bpts) < (degree + 1) * (degree + 2) // 2:
            continue
        try:
            out = vedo.project_point_on_variety(
                pts[i], bpts, degree=degree, normal=vnormals[i]
            )
        except Exception:
            continue
        poly = out[1]  # (coeffs, R, centroid, gauss_curv, mean_curv)
        _coeffs, Rmat, _centroid, gauss, mean = poly
        Rmat = np.asarray(Rmat)
        v1[i] = Rmat[0]
        v2[i] = Rmat[1]
        normals[i] = Rmat[2]
        K[i] = gauss
        H[i] = mean

    # principal curvatures and radii (same formulation as spatchcocking)
    disc = np.maximum(H ** 2 - K, 0.0)
    sq = np.sqrt(disc)
    k1 = H + sq
    k2 = H - sq
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(np.abs(k1) > 1e-12, 1.0 / k1, np.inf)
        r2 = np.where(np.abs(k2) > 1e-12, 1.0 / k2, np.inf)

    return dict(
        pts=pts, normals=normals, v1=v1, v2=v2,
        K=K, H=H, k1=k1, k2=k2, r1=r1, r2=r2,
    )


def validate_sphere(fields: dict, R: float, center=(0.0, 0.0, 0.0)):
    """Print how well the computed fields match the analytical sphere of radius R."""
    pts = fields["pts"]
    center = np.asarray(center, dtype=float)
    valid = ~np.isnan(fields["H"])
    nv = int(valid.sum())

    radial = pts - center
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    # orient computed normals outward for a fair comparison
    nrm = fields["normals"].copy()
    flip = np.sum(nrm * radial, axis=1) < 0
    nrm[flip] *= -1
    cos = np.clip(np.sum(nrm[valid] * radial[valid], axis=1), -1, 1)
    angle_deg = np.degrees(np.arccos(cos))

    def stat(name, arr, expected):
        a = arr[valid]
        a = a[np.isfinite(a)]
        print(f"  {name:<14} mean={a.mean():+.5f}  std={a.std():.5f}  "
              f"expected={expected:+.5f}  |mean-exp|={abs(a.mean()-expected):.5f}")

    print(f"\n=== Sphere validation (R={R}, vertices fitted: {nv}/{len(pts)}) ===")
    stat("K (1/R^2)", fields["K"], 1.0 / R ** 2)
    stat("|H| (1/R)", np.abs(fields["H"]), 1.0 / R)
    stat("|k1| (1/R)", np.abs(fields["k1"]), 1.0 / R)
    stat("|k2| (1/R)", np.abs(fields["k2"]), 1.0 / R)
    stat("|r1| (R)", np.abs(fields["r1"]), R)
    stat("|r2| (R)", np.abs(fields["r2"]), R)
    print(f"  normal-vs-radial angle: mean={angle_deg.mean():.3f} deg  "
          f"max={angle_deg.max():.3f} deg")


def save_results(fields: dict, out_dir: str, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    cols = np.hstack([
        fields["pts"], fields["normals"], fields["v1"], fields["v2"],
        fields["K"][:, None], fields["H"][:, None],
        fields["k1"][:, None], fields["k2"][:, None],
        fields["r1"][:, None], fields["r2"][:, None],
    ])
    header = ("X,Y,Z,nx,ny,nz,v1x,v1y,v1z,v2x,v2y,v2z,K,H,k1,k2,r1,r2")
    npy_path = os.path.join(out_dir, stem + ".npy")
    csv_path = os.path.join(out_dir, stem + ".csv")
    np.save(npy_path, cols)
    np.savetxt(csv_path, cols, delimiter=",", header=header, comments="")
    print(f"\nSaved: {npy_path}\n       {csv_path}")


def show(mesh: vedo.Mesh, fields: dict, R: float, every: int = 40):
    """Interactive view: colour by principal radius r1 + local-axis / normal glyphs."""
    valid = ~np.isnan(fields["H"])
    r1 = np.where(np.isfinite(fields["r1"]), np.abs(fields["r1"]), np.nan)
    mesh.pointdata["r1"] = np.nan_to_num(r1, nan=0.0)
    mesh.cmap("viridis", "r1").add_scalarbar(title="|r1| (radius of curvature)")

    idx = np.where(valid)[0][::every]
    pts = fields["pts"][idx]
    scale = 0.25 * R
    n_arr = vedo.Arrows(pts, pts + scale * fields["normals"][idx], c="red")
    v1_arr = vedo.Arrows(pts, pts + scale * fields["v1"][idx], c="green")
    v2_arr = vedo.Arrows(pts, pts + scale * fields["v2"][idx], c="blue")
    txt = vedo.Text2D("normal (red), v1 tangent (green), v2 tangent (blue)", pos="top-left")
    vedo.show(mesh, n_arr, v1_arr, v2_arr, txt, axes=1, title="Sphere curvature & local frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--res", type=int, default=40, help="vedo Sphere triangulation resolution")
    ap.add_argument("--depth", type=int, default=3, help="k-ring neighbourhood depth for the fit")
    ap.add_argument("--degree", type=int, default=2, help="local polynomial degree")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--show", action="store_true", help="open interactive 3D view")
    args = ap.parse_args()

    mesh = vedo.Sphere(r=args.radius, res=args.res)
    print(f"Sphere: r={args.radius}, res={args.res}, "
          f"{mesh.npoints} vertices, {mesh.ncells} faces")

    fields = compute_vertex_frames(mesh, depth=args.depth, degree=args.degree)
    validate_sphere(fields, R=args.radius)
    save_results(fields, args.out_dir, stem=f"sphere_curvature_R{args.radius}_res{args.res}")

    if args.show:
        show(mesh, fields, R=args.radius)


if __name__ == "__main__":
    main()
