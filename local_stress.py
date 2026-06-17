"""Local membrane-stress estimate from curvature -- Method M1 (axisymmetric two-curvature).

For a thin pressurized shell of revolution the membrane stresses follow from the two
principal radii of curvature with NO equilibrium PDE solve:
    sigma_merid = dp * r_hoop / (2 t)
    sigma_hoop  = dp * r_hoop / t * (1 - r_hoop / (2 r_merid))
(meridional from the vertical force balance on the cap above a parallel circle; hoop from
the normal Laplace law). This is the classic pressure-vessel result.

We obtain the two curvatures locally and exactly for a surface of revolution about a chosen
axis:
    kappa_hoop  = (n . rho_hat) / rho      (rho = distance to the axis; purely geometric)
    kappa_merid = 2 H - kappa_hoop         (H = measured mean curvature from the fit)
At the poles (rho -> 0) the surface is locally umbilic, so kappa_hoop = H there.

Exact (up to curvature-fit error) on the sphere and prolate spheroid. NOT valid on a
non-axisymmetric surface -- there the meridional/hoop split is undefined and one must use
the general GFDM inference (membrane_stress_fd.py). This is the M1 baseline in the
final-results matrix; the contrast M1 (assumes axisymmetry) vs M2 (general equilibrium) vs
M3 (FEM) is the point of the comparison.
"""

from __future__ import annotations

import numpy as np

from sphere_curvature import compute_vertex_frames


def local_stress_axisym(mesh, dp, t, depth=3, axis=0):
    """Per-vertex axisymmetric membrane stress (sigma_hoop, sigma_merid) from curvature.

    axis: index (0=x, 1=y, 2=z) of the axis of revolution. Our sphere/prolate are
    revolutions about x (the stretch axis)."""
    f = compute_vertex_frames(mesh, depth=depth)
    pts = f["pts"]
    n = f["normals"].copy()
    H = f["H"]

    c = pts.mean(axis=0)
    radial = pts - c
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    flip = np.sum(n * radial, axis=1) < 0
    n[flip] *= -1                                   # outward normals

    # distance to the axis of revolution and the radial direction perpendicular to it
    perp = np.delete(np.arange(3), axis)
    rho_vec = np.zeros_like(pts)
    rho_vec[:, perp] = pts[:, perp] - c[perp]
    rho = np.linalg.norm(rho_vec, axis=1)
    small = rho < 1e-6 * np.maximum(rho.max(), 1.0)
    rho_hat = rho_vec / np.maximum(rho, 1e-12)[:, None]

    kappa_hoop = np.sum(n * rho_hat, axis=1) / np.maximum(rho, 1e-12)
    kappa_hoop[small] = H[small]                    # umbilic at the pole
    kappa_merid = 2.0 * H - kappa_hoop

    with np.errstate(divide="ignore", invalid="ignore"):
        r_hoop = 1.0 / kappa_hoop
        r_merid = 1.0 / kappa_merid
        N_merid = dp * r_hoop / 2.0
        N_hoop = dp * r_hoop * (1.0 - r_hoop / (2.0 * r_merid))

    sig_merid = N_merid / t
    sig_hoop = N_hoop / t
    sigma1 = np.maximum(sig_hoop, sig_merid)
    sigma2 = np.minimum(sig_hoop, sig_merid)
    return dict(pts=pts, normals=n, radial=radial,
                sigma1=sigma1, sigma2=sigma2,
                sig_hoop=sig_hoop, sig_merid=sig_merid,
                kappa_hoop=kappa_hoop, kappa_merid=kappa_merid, H=H,
                resid=0.0)


def local_stress_isotropic(mesh, dp, t, depth=3):
    """Isotropic mean-curvature (Young-Laplace) local estimate, valid on ANY surface:
        sigma = dp / (2 t H),   sigma1 = sigma2 = sigma.

    This is the M1 baseline for non-axisymmetric meshes (the chick neural tube), where the
    meridional/hoop split of the axisymmetric formula is undefined. H is the measured mean
    curvature, sign-oriented so the outward (centroid-facing) normal gives 2H>0 on a convex
    region (tension under dp>0); concave pockets give 2H<0 -> local compression."""
    f = compute_vertex_frames(mesh, depth=depth)
    pts = f["pts"]
    n_fit = f["normals"]
    H = f["H"]

    c = pts.mean(axis=0)
    radial = pts - c
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    # mean curvature wrt the OUTWARD normal: flip H where the fit normal points inward
    s = np.sign(np.sum(n_fit * radial, axis=1))
    s[s == 0] = 1.0
    H_out = H * s
    n = n_fit * s[:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = dp / (2.0 * t * H_out)
    return dict(pts=pts, normals=n, radial=radial,
                sigma1=sigma, sigma2=sigma.copy(),
                sig_hoop=sigma, sig_merid=sigma.copy(),
                H=H_out, resid=0.0)
