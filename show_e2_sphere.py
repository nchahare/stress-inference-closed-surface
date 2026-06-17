"""Show e1 arrows on a sphere with sign-consistency colouring after BFS.

The sphere is totally umbilic (kappa1=kappa2=1 everywhere), so principal
directions are undefined at every vertex.  BFS propagates one arbitrary choice
outward from vertex 0; the result is a smooth vector field on the sphere that
MUST have singularities by the Poincare-Hopf theorem (Euler characteristic = 2).
Cyan = arrows that agree with the BFS-propagated field.
Red  = singularity neighbourhood where the field cannot be made consistent.

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe show_e2_sphere.py
"""

import numpy as np
import vedo
from surface_curvature_frame import compute_curvature_frame

mesh = vedo.IcoSphere(r=1.0, subdivisions=4)
mesh.compute_normals()
print(f"Sphere: {mesh.npoints} vertices, {mesh.ncells} faces")

f = compute_curvature_frame(mesh, depth=3)
e1  = f["e1"]
pts = f["pts"]

# 1-ring consistency check (same as show_e2_spheroid.py)
adlist = mesh.compute_adjacency()
flip_frac = np.zeros(mesh.npoints)
for i in range(mesh.npoints):
    neigh = np.array(mesh.find_adjacent_vertices(i, depth=1, adjacency_list=adlist))
    neigh = neigh[neigh != i]
    if len(neigh):
        flip_frac[i] = (e1[neigh] @ e1[i] < 0).mean()

inconsistent = flip_frac > 0.4   # lower threshold: sphere singularities are broader

n_bad = inconsistent.sum()
print(f"\nSign-consistency check (after BFS):")
print(f"  Vertices with >40% neighbours flipped: {n_bad} / {mesh.npoints}")
print(f"  Max flip fraction: {flip_frac.max():.3f}")

# colour mesh by flip_frac so the singularity location is visible on the surface
mesh.pointdata["flip_frac"] = flip_frac
mesh.cmap("coolwarm", "flip_frac", vmin=0, vmax=0.5).alpha(0.4)
mesh.add_scalarbar(title="flip fraction (0=consistent, 0.5=singular)")

# arrows
scale = 0.12
idx_ok  = np.where(~inconsistent)[0]
idx_bad = np.where( inconsistent)[0]
arrows_ok  = vedo.Arrows(pts[idx_ok],  pts[idx_ok]  + scale * e1[idx_ok],  c="cyan5", alpha=0.9)
arrows_bad = vedo.Arrows(pts[idx_bad], pts[idx_bad] + scale * e1[idx_bad], c="red5",  alpha=0.9)

txt = vedo.Text2D(
    "e1 arrows on sphere (all umbilic: directions arbitrary)\n"
    "cyan = BFS-consistent | red = singularity region\n"
    "Poincare-Hopf: sum of indices must equal chi=2",
    pos="top-left", font="Calco", s=0.75,
)

vedo.show(mesh, arrows_ok, arrows_bad, txt, axes=1,
          title="e1 sign consistency on sphere", sharecam=False)
