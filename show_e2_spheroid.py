"""Show e2 (meridional principal direction) on a prolate spheroid and check sign consistency.

e2 = n x e1 is only defined up to sign at each vertex; neighbouring vertices can flip.
We colour arrows red/blue based on whether e2 agrees with the majority of 1-ring
neighbours, so flips are immediately visible.

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe show_e2_spheroid.py
"""

import numpy as np
import vedo
from surface_curvature_frame import compute_curvature_frame

# ── build mesh ────────────────────────────────────────────────────────────────
mesh = vedo.IcoSphere(r=1.0, subdivisions=4).scale([2.0, 1.0, 1.0])
mesh.compute_normals()
print(f"Spheroid: {mesh.npoints} vertices, {mesh.ncells} faces")

f = compute_curvature_frame(mesh, depth=3)
e2  = f["e2"]      # (n,3) — meridional principal direction, sign arbitrary per vertex
pts = f["pts"]

# ── consistency check via 1-ring dot products ─────────────────────────────────
adlist = mesh.compute_adjacency()
n_verts = mesh.npoints

flip_fraction = np.zeros(n_verts)   # fraction of 1-ring neighbours with opposite sign
for i in range(n_verts):
    neigh = np.array(mesh.find_adjacent_vertices(i, depth=1, adjacency_list=adlist))
    neigh = neigh[neigh != i]
    if len(neigh) == 0:
        continue
    dots = e2[neigh] @ e2[i]           # dot with each 1-ring neighbour
    flip_fraction[i] = (dots < 0).mean()

# A vertex is "inconsistent" if more than half its neighbours disagree with it.
inconsistent = flip_fraction > 0.5

n_bad = inconsistent.sum()
print(f"\nSign-consistency check (1-ring dot products):")
print(f"  Vertices with >50% neighbours flipped: {n_bad} / {n_verts}  "
      f"({100*n_bad/n_verts:.1f}%)")
print(f"  Max flip fraction at any vertex: {flip_fraction.max():.2f}")
print(f"  Mean flip fraction:              {flip_fraction.mean():.4f}")

# ── build arrow glyphs ────────────────────────────────────────────────────────
scale = 0.10   # arrow length

# Show every vertex (dense to make pattern visible)
idx_ok  = np.where(~inconsistent)[0]
idx_bad = np.where( inconsistent)[0]

arrows_ok  = vedo.Arrows(
    pts[idx_ok],  pts[idx_ok]  + scale * e2[idx_ok],  c="cyan5",  alpha=0.9)
arrows_bad = vedo.Arrows(
    pts[idx_bad], pts[idx_bad] + scale * e2[idx_bad], c="red5",   alpha=0.9)

# ── colour mesh by kappa2 ─────────────────────────────────────────────────────
mesh.pointdata["kappa2"] = f["kappa2"]
mesh.cmap("Greens_r", "kappa2", vmin=0.0, vmax=0.5).alpha(0.35)
mesh.add_scalarbar(title="kappa2 (meridional)")

txt = vedo.Text2D(
    "e2 arrows: cyan = consistent | red = sign-flipped\n"
    "Prolate spheroid a=2 b=1  (e2 = meridional direction)",
    pos="top-left", font="Calco", s=0.8,
)

vedo.show(mesh, arrows_ok, arrows_bad, txt, axes=1,
          title="e2 sign consistency on prolate spheroid", sharecam=False)
