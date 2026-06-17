"""Capsule (cylinder + hemispherical caps) curvature-frame demo.

Analytic curvatures:
  Cylindrical body (|z| < H):  kappa1 = 1/R (hoop), kappa2 = 0 (axial)
  Spherical caps   (|z| > H):  kappa1 = kappa2 = 1/R  (umbilic)

This gives three regions with different sign-propagation behaviour:
  - Cylinder:  non-umbilic, well-defined e1/e2 -> mostly cyan after BFS
  - Caps:      totally umbilic -> sign flips unavoidable (hairy-ball patches)
  - Junction:  smooth transition; d = (kappa1-kappa2)/2 ramps between 0 and 1/(2R)

Mesh coloured by discriminant  d = sqrt(((B11-B22)/2)^2 + B12^2),
which equals |(kappa1-kappa2)|/2 and is zero at umbilic points.

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe show_capsule.py
"""

import numpy as np
import vedo
from surface_curvature_frame import compute_curvature_frame


# --------------------------------------------------------------------------- #
# Capsule mesh builder
# --------------------------------------------------------------------------- #

def make_capsule(R: float = 1.0, H: float = 2.0,
                 ntheta: int = 40, nphi: int = 12) -> vedo.Mesh:
    """Watertight capsule: cylinder radius R, cylinder half-height H.

    Total height = 2H + 2R.  Ring spacing is kept uniform (matched between
    cap arc and cylinder height) so the triangulation has no large jumps.

    ntheta : vertices around circumference
    nphi   : rings per hemisphere (controls cap resolution)
    """
    verts: list = []
    faces: list = []

    def push_ring(r: float, z: float) -> int:
        idx = len(verts)
        for j in range(ntheta):
            t = 2.0 * np.pi * j / ntheta
            verts.append([r * np.cos(t), r * np.sin(t), z])
        return idx

    def push_pole(z: float) -> int:
        idx = len(verts)
        verts.append([0.0, 0.0, z])
        return idx

    def fan(pole: int, ring: int, flip: bool):
        for j in range(ntheta):
            a = ring + j
            b = ring + (j + 1) % ntheta
            faces.append([pole, b, a] if flip else [pole, a, b])

    def strip(r0: int, r1: int):
        for j in range(ntheta):
            a0 = r0 + j;         a1 = r0 + (j + 1) % ntheta
            b0 = r1 + j;         b1 = r1 + (j + 1) % ntheta
            faces.append([a0, b0, a1])
            faces.append([b0, b1, a1])

    # north pole
    np_idx = push_pole(H + R)

    # north hemisphere: nphi rings, phi from pi/nphi to pi/2
    prev = None
    for i in range(1, nphi + 1):
        phi = (np.pi / 2.0) * i / nphi
        idx = push_ring(R * np.sin(phi), H + R * np.cos(phi))
        if i == 1:
            fan(np_idx, idx, flip=False)
        else:
            strip(prev, idx)
        prev = idx
    # prev is now at z=H, r=R  (phi = pi/2)

    # cylinder: n_cyl rings from z=H-dz to z=-H (inclusive)
    # ring spacing matched to cap: dz = R*(pi/2)/nphi
    n_cyl = max(1, round(4.0 * H * nphi / (R * np.pi)))
    for ic in range(1, n_cyl + 1):
        z = H - 2.0 * H * ic / n_cyl      # ends at z = -H when ic = n_cyl
        idx = push_ring(R, z)
        strip(prev, idx)
        prev = idx
    # prev is now at z=-H, r=R

    # south hemisphere: nphi-1 rings, phi from pi/2+dphi to pi-dphi
    for i in range(1, nphi):
        phi = np.pi / 2.0 + (np.pi / 2.0) * i / nphi
        idx = push_ring(R * np.sin(phi), -H + R * np.cos(phi))
        strip(prev, idx)
        prev = idx

    # south pole
    sp_idx = push_pole(-H - R)
    fan(sp_idx, prev, flip=True)

    mesh = vedo.Mesh([np.array(verts, dtype=float),
                      np.array(faces, dtype=int)])
    mesh.compute_normals()
    return mesh


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

R, H = 1.0, 2.0
mesh = make_capsule(R=R, H=H, ntheta=40, nphi=14)
print(f"Capsule (R={R}, H={H}): {mesh.npoints} vertices, {mesh.ncells} faces")

f = compute_curvature_frame(mesh, depth=3)
pts    = f["pts"]
e1     = f["e1"]
kappa1 = f["kappa1"]
kappa2 = f["kappa2"]
H_mean = f["H"]
K_val  = f["K"]

# discriminant: zero at umbilic points, 1/(2R) on cylinder
disc = 0.5 * np.abs(kappa1 - kappa2)

# --------------------------------------------------------------------------- #
# Sign-consistency check (1-ring dot products)
# --------------------------------------------------------------------------- #
adlist = mesh.compute_adjacency()
flip_frac = np.zeros(mesh.npoints)
for i in range(mesh.npoints):
    neigh = np.array(mesh.find_adjacent_vertices(i, depth=1, adjacency_list=adlist))
    neigh = neigh[neigh != i]
    if len(neigh):
        flip_frac[i] = (e1[neigh] @ e1[i] < 0).mean()

inconsistent = flip_frac > 0.4
n_bad = inconsistent.sum()

print(f"\nSign-consistency check (after BFS):")
print(f"  Inconsistent vertices (>40% flipped): {n_bad} / {mesh.npoints}  "
      f"({100*n_bad/mesh.npoints:.1f}%)")
print(f"  Max flip fraction: {flip_frac.max():.3f}")

# --------------------------------------------------------------------------- #
# Analytic validation by region
# --------------------------------------------------------------------------- #
tol = 0.05 * H          # thin band around junction
cyl  = np.abs(pts[:, 2]) < H - tol
cap  = np.abs(pts[:, 2]) > H + tol

print(f"\nCurvature validation:")
print(f"  Region               n     kappa1 mean±std        kappa2 mean±std"
      f"        analytic")
for mask, label, ak1, ak2 in [
        (cyl,  "cylinder (body)",  1.0/R,      0.0   ),
        (cap,  "spherical caps",   1.0/R,      1.0/R ),
]:
    k1 = kappa1[mask]; k2 = kappa2[mask]
    print(f"  {label:<22} {mask.sum():4d}  "
          f"{k1.mean():+.3f}±{k1.std():.3f}    "
          f"{k2.mean():+.3f}±{k2.std():.3f}    "
          f"({ak1:.3f}, {ak2:.3f})")

# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
# Mesh coloured by discriminant d = |kappa1-kappa2|/2
#   d = 0       on caps (umbilic)      -> blue
#   d = 1/(2R)  on cylinder body       -> red
mesh.pointdata["disc"] = disc
mesh.cmap("RdBu_r", "disc", vmin=0, vmax=0.6/R).alpha(0.4)
mesh.add_scalarbar(title="|kappa1-kappa2|/2\n(0=umbilic, 1/2R=cylinder)")

# e1 arrows: cyan = BFS-consistent, red = sign-flipped
scale = 0.12
idx_ok  = np.where(~inconsistent)[0]
idx_bad = np.where( inconsistent)[0]
arr_ok  = vedo.Arrows(pts[idx_ok],  pts[idx_ok]  + scale * e1[idx_ok],
                      c="cyan5", alpha=0.9)
arr_bad = vedo.Arrows(pts[idx_bad], pts[idx_bad] + scale * e1[idx_bad],
                      c="red5",  alpha=0.9)

txt = vedo.Text2D(
    f"Capsule  R={R}  H={H}\n"
    f"Mesh colour: discriminant |k1-k2|/2  (0=umbilic blue, 1/(2R) red)\n"
    f"e1 arrows: cyan=consistent  red=sign-flipped after BFS\n"
    f"Cylinder: k1=1/R k2=0 (hoop/axial)   Caps: k1=k2=1/R (umbilic)",
    pos="top-left", font="Calco", s=0.75,
)

vedo.show(mesh, arr_ok, arr_bad, txt, axes=1,
          title="Capsule curvature frame", sharecam=False)
