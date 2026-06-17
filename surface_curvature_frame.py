"""Per-vertex principal curvature frame on a triangulated surface mesh.

For each vertex the script computes:
  kappa1, kappa2  — principal curvatures (kappa1 >= kappa2)
  r1, r2          — principal radii 1/kappa
  H, K            — mean and Gaussian curvature
  e1, e2          — principal curvature direction unit vectors in world R3
  n               — outward unit vertex normal

Strategy: call compute_vertex_frames() (sphere_curvature.py) to get the local fit
frame (v1, v2, n) and the scalar curvatures H, K from the polynomial fit.  Then build
GFDM first-order derivative operators on the normal field (surface_fd.py) to evaluate
the Weingarten map B_ij = (d_i n) . e_j.  Diagonalise the 2x2 symmetric shape
operator analytically -- vectorised, no per-vertex loop -- and rotate the eigenvectors
back to world R3.

Validation:
  sphere   R=1   -> kappa1 = kappa2 = 1 everywhere (umbilic; direction arbitrary)
  spheroid a=2   -> kappa1 (hoop) = 1/b = 1, kappa2 (meridional) = b/a^2 = 0.25 at equator

Run (headless validation):
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe surface_curvature_frame.py
With interactive view:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe surface_curvature_frame.py --show
Save CSV:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe surface_curvature_frame.py --csv
"""

from __future__ import annotations

import argparse
import os
from collections import deque

import numpy as np
import vedo

from sphere_curvature import compute_vertex_frames
from surface_fd import get_neighborhoods, build_derivative_operators


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _build_adjacency(mesh: vedo.Mesh) -> list[list[int]]:
    """Edge-adjacency list from triangle faces: adj[i] = list of vertex indices."""
    n = mesh.npoints
    adj: list[list[int]] = [[] for _ in range(n)]
    for face in np.asarray(mesh.cells):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        adj[a].append(b); adj[b].append(a)
        adj[b].append(c); adj[c].append(b)
        adj[a].append(c); adj[c].append(a)
    return adj


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #

def compute_curvature_frame(mesh: vedo.Mesh, depth: int = 3):
    """Per-vertex principal curvature frame and curvature scalars.

    Parameters
    ----------
    mesh  : vedo.Mesh  — triangulated surface (closed or open).
    depth : int        — k-ring neighbourhood depth for the GFDM/polynomial fit.

    Returns
    -------
    dict with arrays of shape (n,) or (n,3):
      pts     (n,3)  vertex positions
      normals (n,3)  outward unit normals (from polynomial fit, flipped outward)
      e1      (n,3)  principal direction for kappa1 (world R3, unit)
      e2      (n,3)  principal direction for kappa2 (world R3, unit) = n x e1
      kappa1  (n,)   larger principal curvature
      kappa2  (n,)   smaller principal curvature
      r1      (n,)   principal radius 1/kappa1
      r2      (n,)   principal radius 1/kappa2
      H       (n,)   mean curvature (kappa1+kappa2)/2
      K       (n,)   Gaussian curvature kappa1*kappa2
    """
    # ---- 1. Polynomial fit: local frame (v1,v2,n) and scalar curvatures ---- #
    frm = compute_vertex_frames(mesh, depth=depth)
    pts = frm["pts"]
    v1  = frm["v1"].copy()
    v2  = frm["v2"].copy()
    n   = frm["normals"].copy()

    # Orient normals outward: compare with radial direction from mesh centroid.
    centroid = pts.mean(axis=0)
    radial   = pts - centroid
    radial  /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-15
    flip     = np.sum(n * radial, axis=1) < 0
    # Flip n and v2 together to preserve right-handedness (v1 x v2 = n).
    n[flip]  *= -1
    v2[flip] *= -1

    # ---- 2. GFDM first-order derivative operators on the normal field ------- #
    neigh = get_neighborhoods(mesh, depth=depth)
    D     = build_derivative_operators(pts, v1, v2, n, neigh)
    gx, gy = D["g_xi"], D["g_eta"]

    # ---- 3. Shape operator B via Weingarten map: B_ij = (d_i n) . e_j ------- #
    # Sign convention: B = +grad_s n -> tr B = 2H > 0 (tensile under positive dp).
    dn_xi  = np.column_stack([gx @ n[:, c] for c in range(3)])   # dn/dxi  (n,3)
    dn_eta = np.column_stack([gy @ n[:, c] for c in range(3)])   # dn/deta (n,3)

    B11 = np.sum(dn_xi  * v1, axis=1)                                   # dn/dxi  . v1
    B22 = np.sum(dn_eta * v2, axis=1)                                   # dn/deta . v2
    B12 = 0.5 * (np.sum(dn_xi * v2, axis=1) + np.sum(dn_eta * v1, axis=1))

    # ---- 4. Analytic 2x2 symmetric eigenproblem (vectorised) ---------------- #
    # Mean value and half-range of the shape-operator matrix.
    Hmean = 0.5 * (B11 + B22)
    disc  = np.sqrt(np.maximum(0.25 * (B11 - B22)**2 + B12**2, 0.0))

    kappa1 = Hmean + disc   # larger principal curvature
    kappa2 = Hmean - disc   # smaller principal curvature

    # Eigenvector angle for kappa1 in the (v1, v2) tangent plane.
    # theta = 0.5 * atan2(2*B12, B11-B22)
    # Derivation: standard rotation-diagonalisation of 2x2 symmetric matrix;
    # cos(theta)*v1 + sin(theta)*v2 is the eigenvector for the LARGER eigenvalue.
    theta = 0.5 * np.arctan2(2.0 * B12, B11 - B22)
    e1 = np.cos(theta)[:, None] * v1 + np.sin(theta)[:, None] * v2
    e2 = np.cross(n, e1)          # right-handed: e1 x e2 = n  =>  e2 = n x e1

    # Re-normalise against floating-point drift.
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-15
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-15

    # ---- 5. BFS sign propagation: make e1/e2 globally consistent ------------ #
    # Principal directions are only defined up to sign (±e1 equally valid).
    # The arctan2 formula picks the sign independently at each vertex, causing
    # flips at umbilic/near-umbilic points (κ1≈κ2, discriminant → 0).
    # Fix: walk the mesh once (BFS from vertex 0) and flip e1[j], e2[j] whenever
    # e1[j] disagrees with its already-visited parent e1[i].
    # Flipping e1 → -e1 also flips e2 → -e2 (since e2 = n × e1), which keeps
    # the frame right-handed: (-e1) × (-e2) = e1 × e2 = n.
    adj = _build_adjacency(mesh)
    visited = np.zeros(len(pts), dtype=bool)
    queue   = deque([0])
    visited[0] = True
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            if not visited[j]:
                visited[j] = True
                if e1[i] @ e1[j] < 0:
                    e1[j] *= -1
                    e2[j] *= -1
                queue.append(j)

    # ---- 7. Derived scalars -------------------------------------------------- #
    K_out = kappa1 * kappa2
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(np.abs(kappa1) > 1e-12, 1.0 / kappa1, np.inf)
        r2 = np.where(np.abs(kappa2) > 1e-12, 1.0 / kappa2, np.inf)

    return dict(
        pts=pts, normals=n, e1=e1, e2=e2,
        kappa1=kappa1, kappa2=kappa2, r1=r1, r2=r2,
        H=Hmean, K=K_out,
    )


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def validate_orthonormality(f: dict, label: str = ""):
    """Check that {e1, e2, n} form an orthonormal right-handed frame."""
    e1, e2, n = f["e1"], f["e2"], f["normals"]
    d12  = np.abs(np.sum(e1 * e2, axis=1))
    d1n  = np.abs(np.sum(e1 * n,  axis=1))
    d2n  = np.abs(np.sum(e2 * n,  axis=1))
    n1   = np.abs(np.linalg.norm(e1, axis=1) - 1)
    n2   = np.abs(np.linalg.norm(e2, axis=1) - 1)
    # right-handedness: e1 x e2 should equal n
    cross_err = np.linalg.norm(np.cross(e1, e2) - n, axis=1)
    tag = f" [{label}]" if label else ""
    print(f"\n=== Frame orthonormality{tag} ===")
    print(f"  |e1·e2|   max={d12.max():.2e}  mean={d12.mean():.2e}")
    print(f"  |e1·n|    max={d1n.max():.2e}  mean={d1n.mean():.2e}")
    print(f"  |e2·n|    max={d2n.max():.2e}  mean={d2n.mean():.2e}")
    print(f"  ||e1|-1|  max={n1.max():.2e}   mean={n1.mean():.2e}")
    print(f"  ||e2|-1|  max={n2.max():.2e}   mean={n2.mean():.2e}")
    print(f"  |e1xe2-n| max={cross_err.max():.2e}  mean={cross_err.mean():.2e}")


def validate_sphere(f: dict, R: float):
    """On a sphere of radius R all principal curvatures should equal 1/R."""
    k1, k2 = f["kappa1"], f["kappa2"]
    H,  K  = f["H"],      f["K"]
    expected_k = 1.0 / R
    print(f"\n=== Sphere validation (R={R}) ===")

    def _stat(name, arr, exp):
        a = arr[np.isfinite(arr)]
        print(f"  {name:<12} mean={a.mean():+.5f}  std={a.std():.5f}  "
              f"expected={exp:+.5f}  |mean-exp|={abs(a.mean()-exp):.5f}")

    _stat("|kappa1|", np.abs(k1), expected_k)
    _stat("|kappa2|", np.abs(k2), expected_k)
    _stat("|H|",      np.abs(H),  expected_k)
    _stat("|K|",      np.abs(K),  expected_k**2)


def validate_spheroid(f: dict, a: float = 2.0, b: float = 1.0, axis: int = 0):
    """Compare computed curvatures with the analytic axisymmetric formula.

    For a prolate spheroid with long semi-axis a along `axis` and short semi-axis b:
      kappa_meridional = b / a^2  (at the equator, where the curvature is smallest)
      kappa_hoop       = 1 / b    (at the equator, where the curvature is largest)
    We identify points near the equator (coordinate along `axis` ~ 0) and report stats.
    """
    pts = f["pts"]
    # Equatorial band: |x_axis| < 0.1*a
    eq = np.abs(pts[:, axis]) < 0.1 * a
    if eq.sum() < 5:
        print("  (no equatorial points found for spheroid validation)")
        return
    k1 = f["kappa1"][eq]
    k2 = f["kappa2"][eq]
    k_hoop   = 1.0 / b
    k_merid  = b / (a * a)
    print(f"\n=== Spheroid validation (a={a}, b={b}, equatorial band {eq.sum()} pts) ===")
    print(f"  kappa1 (hoop):   mean={k1.mean():.4f}  std={k1.std():.4f}  analytic={k_hoop:.4f}")
    print(f"  kappa2 (merid):  mean={k2.mean():.4f}  std={k2.std():.4f}  analytic={k_merid:.4f}")


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def save_csv(f: dict, path: str):
    """Write per-vertex frame data to a CSV file."""
    cols = np.hstack([
        f["pts"],
        f["kappa1"][:, None], f["kappa2"][:, None],
        f["r1"][:, None],     f["r2"][:, None],
        f["H"][:, None],      f["K"][:, None],
        f["e1"], f["e2"], f["normals"],
    ])
    header = ("X,Y,Z,"
              "kappa1,kappa2,r1,r2,H,K,"
              "e1x,e1y,e1z,e2x,e2y,e2z,nx,ny,nz")
    np.savetxt(path, cols, delimiter=",", header=header, comments="")
    print(f"Saved: {path}")


# --------------------------------------------------------------------------- #
# Interactive viewer
# --------------------------------------------------------------------------- #

def show_frame(mesh: vedo.Mesh, f: dict, every: int = 40):
    """Display the mesh coloured by kappa1 with principal direction glyphs."""
    k1 = f["kappa1"]
    mesh.pointdata["kappa1"] = np.nan_to_num(k1)
    mesh.cmap("coolwarm", "kappa1").add_scalarbar(title="kappa1")

    valid = np.isfinite(k1)
    idx   = np.where(valid)[0][::every]
    pts   = f["pts"][idx]
    scale = 0.15 * np.abs(np.nanmedian(f["r1"][np.isfinite(f["r1"])]))

    n_arr  = vedo.Arrows(pts, pts + scale * f["normals"][idx], c="red3",    alpha=0.8)
    e1_arr = vedo.Arrows(pts, pts + scale * f["e1"][idx],      c="green5",  alpha=0.8)
    e2_arr = vedo.Arrows(pts, pts + scale * f["e2"][idx],      c="blue5",   alpha=0.8)
    txt    = vedo.Text2D(
        "n (red) | e1 principal-max (green) | e2 principal-min (blue)",
        pos="top-left", font="Calco",
    )
    vedo.show(mesh, n_arr, e1_arr, e2_arr, txt, axes=1,
              title="Principal curvature frame", sharecam=False)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file",   default=None,  help="mesh file to load (VTK/OBJ/…)")
    ap.add_argument("--subdiv", type=int, default=5, help="IcoSphere subdivisions")
    ap.add_argument("--depth",  type=int, default=3, help="k-ring neighbourhood depth")
    ap.add_argument("--show",   action="store_true", help="open interactive 3D view")
    ap.add_argument("--csv",    action="store_true", help="save CSV to out/")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- sphere (unit) ------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("Sphere (IcoSphere, subdiv=%d)" % args.subdiv)
    sphere = vedo.IcoSphere(r=1.0, subdivisions=args.subdiv)
    print(f"  {sphere.npoints} vertices, {sphere.ncells} faces")
    fs = compute_curvature_frame(sphere, depth=args.depth)
    validate_sphere(fs, R=1.0)
    validate_orthonormality(fs, label="sphere")
    if args.csv:
        save_csv(fs, os.path.join(args.out_dir,
                                  f"curvature_frame_sphere_s{args.subdiv}.csv"))
    if args.show:
        show_frame(sphere, fs)

    # ---- prolate spheroid (a=2, b=1) ---------------------------------------- #
    print("\n" + "=" * 60)
    print("Prolate spheroid (a=2, b=1, IcoSphere subdiv=%d)" % args.subdiv)
    sph2 = vedo.IcoSphere(r=1.0, subdivisions=args.subdiv)
    sph2.scale([2.0, 1.0, 1.0])
    print(f"  {sph2.npoints} vertices, {sph2.ncells} faces")
    fe = compute_curvature_frame(sph2, depth=args.depth)
    validate_spheroid(fe, a=2.0, b=1.0, axis=0)
    validate_orthonormality(fe, label="spheroid")
    if args.csv:
        save_csv(fe, os.path.join(args.out_dir,
                                   f"curvature_frame_spheroid_s{args.subdiv}.csv"))
    if args.show:
        show_frame(sph2, fe, every=20)

    # ---- custom file --------------------------------------------------------- #
    if args.file:
        print("\n" + "=" * 60)
        print(f"Custom mesh: {args.file}")
        m = vedo.Mesh(args.file)
        print(f"  {m.npoints} vertices, {m.ncells} faces")
        fc = compute_curvature_frame(m, depth=args.depth)
        validate_orthonormality(fc, label="custom")
        stem = os.path.splitext(os.path.basename(args.file))[0]
        if args.csv:
            save_csv(fc, os.path.join(args.out_dir,
                                       f"curvature_frame_{stem}.csv"))
        if args.show:
            show_frame(m, fc)


if __name__ == "__main__":
    main()
