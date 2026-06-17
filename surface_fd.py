"""Generalized finite differences (GFDM) on an arbitrary triangulated surface.

No FEM framework: at each vertex we build a local tangent frame, project the k-ring
neighbours into the tangent plane, and fit a weighted-least-squares quadratic Taylor
expansion. The first-order coefficients give finite-difference weights for the surface
derivatives d/dxi and d/deta. These assemble into two sparse operators G_xi, G_eta that
act on nodal scalar fields, from which surface gradient, divergence and the
Laplace-Beltrami operator follow.

Works on any surface mesh (not just surfaces of revolution); the only inputs are the
vertex positions, a per-vertex orthonormal tangent frame (t1, t2, n), and neighbour
lists. We reuse the frames/curvature from ``sphere_curvature.compute_vertex_frames``.

Run the self-test (validates the operators on a sphere against analytic results):
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe surface_fd.py
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import vedo


# --------------------------------------------------------------------------- #
# Neighbourhoods
# --------------------------------------------------------------------------- #
def get_neighborhoods(mesh: vedo.Mesh, depth: int = 2):
    """Return a list: for each vertex, the array of k-ring neighbour indices."""
    adlist = mesh.compute_adjacency()
    n = mesh.npoints
    neigh = []
    for i in range(n):
        idx = mesh.find_adjacent_vertices(i, depth=depth, adjacency_list=adlist)
        idx = np.unique(np.asarray(idx, dtype=int))
        idx = idx[idx != i]
        neigh.append(idx)
    return neigh


# --------------------------------------------------------------------------- #
# GFDM derivative operators
# --------------------------------------------------------------------------- #
def build_grad_operators(pts, t1, t2, normals, neighborhoods):
    """Build sparse operators G_xi, G_eta (n x n) for tangential derivatives.

    For a nodal scalar field f, (G_xi @ f)[i] ~= df/dxi and (G_eta @ f)[i] ~= df/deta
    at vertex i, where (xi, eta) are coordinates along the local tangent axes (t1, t2).

    Local quadratic Taylor fit:
        f_j - f_i ~= a*xi + b*eta + 0.5*c*xi^2 + d*xi*eta + 0.5*e*eta^2
    solved by weighted least squares; (a, b) are the first derivatives. The normal
    offset of each neighbour is folded into the quadratic terms, which keeps the
    gradient accurate on curved surfaces.
    """
    n = len(pts)
    rows_xi, cols_xi, val_xi = [], [], []
    rows_eta, cols_eta, val_eta = [], [], []

    for i in range(n):
        nb = neighborhoods[i]
        if len(nb) < 5:
            continue
        d = pts[nb] - pts[i]                # (m, 3)
        xi = d @ t1[i]
        eta = d @ t2[i]
        # quadratic design matrix (no constant term: it cancels via f_j - f_i)
        M = np.column_stack([xi, eta, 0.5 * xi**2, xi * eta, 0.5 * eta**2])  # (m,5)
        w = 1.0 / np.maximum(xi**2 + eta**2, 1e-14)   # inverse-distance^2 weights
        W = M * w[:, None]
        # normal equations: (M^T W M) coeff = (M^T W) (f_j - f_i)
        A = M.T @ W                          # (5,5)
        try:
            Ainv = np.linalg.solve(A, np.eye(5))
        except np.linalg.LinAlgError:
            Ainv = np.linalg.pinv(A)
        C = Ainv @ W.T                       # (5, m): coeff_k = C @ (f_nb - f_i)
        ca = C[0]                            # weights for df/dxi
        cb = C[1]                            # weights for df/deta
        # df/dxi = sum_j ca_j (f_j - f_i) -> off-diag ca_j, diag -sum(ca)
        rows_xi += [i] * len(nb) + [i]
        cols_xi += list(nb) + [i]
        val_xi += list(ca) + [-ca.sum()]
        rows_eta += [i] * len(nb) + [i]
        cols_eta += list(nb) + [i]
        val_eta += list(cb) + [-cb.sum()]

    G_xi = sp.csr_matrix((val_xi, (rows_xi, cols_xi)), shape=(n, n))
    G_eta = sp.csr_matrix((val_eta, (rows_eta, cols_eta)), shape=(n, n))
    return G_xi, G_eta


def build_derivative_operators(pts, t1, t2, normals, neighborhoods):
    """Build first- AND second-order tangential derivative operators from one WLS fit.

    Returns a dict of sparse (n x n) operators acting on nodal scalar fields:
      g_xi, g_eta            -> d/dxi, d/deta
      h_xixi, h_xieta, h_etaeta -> d2/dxi2, d2/dxi deta, d2/deta2
    At the base vertex (normal coordinates) the surface covariant Hessian equals these
    plain second partials, so they give Hess_s directly. Used by the Beltrami solver.
    """
    n = len(pts)
    ops = {k: ([], [], []) for k in
           ("g_xi", "g_eta", "h_xixi", "h_xieta", "h_etaeta")}
    rowmap = {"g_xi": 0, "g_eta": 1, "h_xixi": 2, "h_xieta": 3, "h_etaeta": 4}

    for i in range(n):
        nb = neighborhoods[i]
        if len(nb) < 6:
            continue
        d = pts[nb] - pts[i]
        xi = d @ t1[i]
        eta = d @ t2[i]
        M = np.column_stack([xi, eta, 0.5 * xi**2, xi * eta, 0.5 * eta**2])
        w = 1.0 / np.maximum(xi**2 + eta**2, 1e-14)
        W = M * w[:, None]
        A = M.T @ W
        try:
            Ainv = np.linalg.solve(A, np.eye(5))
        except np.linalg.LinAlgError:
            Ainv = np.linalg.pinv(A)
        C = Ainv @ W.T                       # (5, m)
        for key, ridx in rowmap.items():
            c = C[ridx]
            r, cc, v = ops[key]
            r += [i] * len(nb) + [i]
            cc += list(nb) + [i]
            v += list(c) + [-c.sum()]

    return {k: sp.csr_matrix((v, (r, cc)), shape=(n, n))
            for k, (r, cc, v) in ops.items()}


def surface_gradient(G_xi, G_eta, t1, t2, f):
    """Surface gradient of scalar field f -> (n, 3) ambient tangential vectors."""
    a = G_xi @ f
    b = G_eta @ f
    return a[:, None] * t1 + b[:, None] * t2


def surface_divergence_vec(G_xi, G_eta, t1, t2, V):
    """Surface divergence of an ambient (tangential) vector field V (n, 3) -> (n,)."""
    # div = t1 . dV/dxi + t2 . dV/deta, componentwise derivatives of each ambient comp
    dV_dxi = np.column_stack([G_xi @ V[:, 0], G_xi @ V[:, 1], G_xi @ V[:, 2]])
    dV_deta = np.column_stack([G_eta @ V[:, 0], G_eta @ V[:, 1], G_eta @ V[:, 2]])
    return np.sum(t1 * dV_dxi, axis=1) + np.sum(t2 * dV_deta, axis=1)


def laplace_beltrami(G_xi, G_eta, t1, t2, f):
    """Laplace-Beltrami of scalar f = surface divergence of surface gradient."""
    grad = surface_gradient(G_xi, G_eta, t1, t2, f)
    return surface_divergence_vec(G_xi, G_eta, t1, t2, grad)


# --------------------------------------------------------------------------- #
# Self-test on a sphere
# --------------------------------------------------------------------------- #
def _interior_mask(normals, axis=2, cap=0.9):
    """Exclude vertices near the +/- poles of the chosen axis (triangulation
    singularities) so the error report reflects the method, not the poles."""
    return np.abs(normals[:, axis]) < cap


def selftest(R=1.0, res=60, depth=2):
    from sphere_curvature import compute_vertex_frames

    mesh = vedo.Sphere(r=R, res=res)
    f = compute_vertex_frames(mesh, depth=depth)
    pts, t1, t2, n = f["pts"], f["v1"], f["v2"], f["normals"]
    # orient normals outward (radial) for a clean analytic comparison
    radial = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    flip = np.sum(n * radial, axis=1) < 0
    n = n.copy(); n[flip] *= -1

    neigh = get_neighborhoods(mesh, depth=depth)
    G_xi, G_eta = build_grad_operators(pts, t1, t2, n, neigh)

    valid = np.array([len(nb) >= 5 for nb in neigh]) & _interior_mask(radial, axis=2)

    print(f"Sphere R={R}, res={res}, depth={depth}: {mesh.npoints} vertices, "
          f"{valid.sum()} used in error report (poles excluded)\n")

    # --- test 1: surface gradient of f = z ---------------------------------
    # analytic: grad_s z = zhat - (zhat.n) n
    fz = pts[:, 2]
    grad_num = surface_gradient(G_xi, G_eta, t1, t2, fz)
    zhat = np.array([0.0, 0.0, 1.0])
    grad_exact = zhat[None, :] - (radial @ zhat)[:, None] * radial
    err1 = np.linalg.norm(grad_num - grad_exact, axis=1)
    print(f"[grad_s z]            max err={err1[valid].max():.4e}  "
          f"mean err={err1[valid].mean():.4e}")

    # --- test 2: Laplace-Beltrami of f = z ---------------------------------
    # analytic on sphere radius R: lap_s z = -2 z / R^2  (z is an l=1 harmonic)
    lap_num = laplace_beltrami(G_xi, G_eta, t1, t2, fz)
    lap_exact = -2.0 * fz / R**2
    err2 = np.abs(lap_num - lap_exact)
    print(f"[lap_s z = -2z/R^2]   max err={err2[valid].max():.4e}  "
          f"mean err={err2[valid].mean():.4e}")

    # --- test 3: Laplace-Beltrami of f = x*y (l=2 harmonic) ----------------
    # lap_s (xy) = -6 xy / R^2
    fxy = pts[:, 0] * pts[:, 1]
    lap_num3 = laplace_beltrami(G_xi, G_eta, t1, t2, fxy)
    lap_exact3 = -6.0 * fxy / R**2
    err3 = np.abs(lap_num3 - lap_exact3)
    print(f"[lap_s xy = -6xy/R^2] max err={err3[valid].max():.4e}  "
          f"mean err={err3[valid].mean():.4e}")


if __name__ == "__main__":
    for res in (40, 80):
        selftest(res=res)
        print("-" * 60)
