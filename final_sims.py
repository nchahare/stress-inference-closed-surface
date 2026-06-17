"""Final-results simulations -- runnable subset (uniform thickness, inference methods).

Runs the four cases of the 12-sim matrix that need neither the FEM solver (M3) nor the
DV-thickness field (both pending):
  Sim 1 : sphere,            uniform t, M1 local (axisymmetric two-curvature)
  Sim 2 : sphere,            uniform t, M2 cMSM  (GFDM solve + Laplacian smoothing)
  Sim 7 : prolate ellipsoid, uniform t, M1 local
  Sim 8 : prolate ellipsoid, uniform t, M2 cMSM

Config (final): IcoSphere subdiv5, depth3, dp=20, t=0.05; prolate aspect ratio 2:1.
Saves per-vertex CSV + NPZ + VTP for each sim (so the FEM/DV columns can be added later
and everything re-plotted) and a M1-vs-M2 sigma_max comparison figure. Reports accuracy
against the analytic axisymmetric reference.

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe final_sims.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import vedo

from membrane_stress_fd import solve_membrane, analytic_axisym
from reg_compare import avg_operator, laplacian_smooth
from local_stress import local_stress_axisym


def save_fields(mesh, res, tag, outdir):
    os.makedirs(outdir, exist_ok=True)
    pts = res["pts"]
    s1, s2 = res["sigma1"], res["sigma2"]
    smax = np.maximum(s1, s2); smin = np.minimum(s1, s2)
    cols = np.column_stack([pts, s1, s2, smax, smin])
    header = "X,Y,Z,sigma1,sigma2,sigma_max,sigma_min"
    np.savetxt(os.path.join(outdir, f"{tag}.csv"), cols, delimiter=",",
               header=header, comments="")
    np.savez(os.path.join(outdir, f"{tag}.npz"), pts=pts, faces=np.asarray(mesh.cells),
             sigma1=s1, sigma2=s2, sigma_max=smax, sigma_min=smin)
    m = mesh.clone()
    for k in ("sigma1", "sigma2"):
        m.pointdata[k] = res[k]
    m.pointdata["sigma_max"] = smax
    m.pointdata["sigma_min"] = smin
    m.write(os.path.join(outdir, f"{tag}.vtp"))


def metrics(res, dp, t, a, b):
    s1, s2 = res["sigma1"], res["sigma2"]
    smax = np.maximum(s1, s2); smin = np.minimum(s1, s2)
    mask = np.abs(res["radial"][:, 0]) < 0.92            # exclude long-axis poles
    sm, sh = analytic_axisym(res["pts"], dp, t, a, b)
    an_max = np.maximum(sm, sh); an_min = np.minimum(sm, sh)

    def relerr(num, an):
        e = np.abs(num[mask] - an[mask]) / np.maximum(np.abs(an[mask]), 1e-12)
        return np.median(e)
    return dict(mean_max=smax[mask].mean(), mean_min=smin[mask].mean(),
                err_max=relerr(smax, an_max), err_min=relerr(smin, an_min),
                std_max=smax[mask].std(),
                anis=(smax[mask] - smin[mask]).mean() / max(abs(smax[mask].mean()), 1e-12),
                an_max=an_max[mask].mean(), an_min=an_min[mask].mean())


def fmt(tag, m):
    return (f"  {tag:<16s} sig_max={m['mean_max']:7.2f} (err {m['err_max']:5.1%})  "
            f"sig_min={m['mean_min']:7.2f} (err {m['err_min']:5.1%})  "
            f"std_max={m['std_max']:6.3f}  anis={m['anis']:6.1%}  "
            f"[analytic {m['an_max']:.2f}/{m['an_min']:.2f}]")


def run_geom(mesh, name, dp, t, depth, lam, iters, alpha, a, b, outdir, sims):
    print(f"\n================ {name}  (analytic sig_max/min anisotropy ratio "
          f"-> {(1.0 - b**2/(2*a**2))/0.5:.3f}) ================")

    # M1 -- local axisymmetric two-curvature
    m1 = local_stress_axisym(mesh, dp, t, depth=depth, axis=0)
    save_fields(mesh, m1, f"{sims[0]}", outdir)
    print(fmt("M1 local", metrics(m1, dp, t, a, b)))

    # M2 -- cMSM GFDM solve + Laplacian smoothing
    raw = solve_membrane(mesh, dp, t, depth=depth, lam=lam, solver="direct")
    avg = avg_operator(mesh)
    m2 = dict(pts=raw["pts"], radial=raw["radial"],
              sigma1=laplacian_smooth(avg, raw["sigma1"], iters, alpha),
              sigma2=laplacian_smooth(avg, raw["sigma2"], iters, alpha))
    save_fields(mesh, m2, f"{sims[1]}", outdir)
    print(f"  (M2 equilibrium residual {raw['resid']:.2e})")
    print(fmt("M2 cMSM", metrics(m2, dp, t, a, b)))

    return (mesh, np.maximum(m1["sigma1"], m1["sigma2"]),
            np.maximum(m2["sigma1"], m2["sigma2"]))


def render(rows, out):
    """rows: list of (mesh, smax_M1, smax_M2). Columns: M1 | M2 | (FEM pending)."""
    titles = ["M1 local (curvature)", "M2 cMSM (GFDM)", "M3 FEM (pending)"]
    plt = vedo.Plotter(shape=(len(rows), 3), size=(1650, 540 * len(rows)),
                       sharecam=False, offscreen=True)
    for ri, (m, s1, s2) in enumerate(rows):
        allv = np.concatenate([s1, s2])
        clim = (float(np.percentile(allv, 2)), float(np.percentile(allv, 98)))
        for ci, field in enumerate([s1, s2, None]):
            cell = plt.at(ri * 3 + ci)
            if field is None:
                cell.show(vedo.Text2D(titles[ci], pos="center"), axes=0)
                continue
            mm = m.clone(); mm.pointdata["s"] = field
            mm.cmap("plasma", "s", vmin=clim[0], vmax=clim[1])
            if ci == 1:
                mm.add_scalarbar(title="sigma_max")
            cell.show(mm, vedo.Text2D(titles[ci], pos="top-left"), axes=0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.screenshot(out); plt.close()
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--subdiv", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dp", type=float, default=20.0)
    ap.add_argument("--t", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--stretch", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--outdir", default="out/final")
    ap.add_argument("--out", default="out/final_sims_compare.png")
    args = ap.parse_args()

    R = args.radius
    print(f"Final sims: IcoSphere subdiv{args.subdiv}, depth{args.depth}, "
          f"dp={args.dp}, t={args.t}, prolate x{args.stretch}")

    sphere = vedo.IcoSphere(r=R, subdivisions=args.subdiv)
    row_s = run_geom(sphere, "Sphere (Sim 1=M1, Sim 2=M2)", args.dp, args.t, args.depth,
                     args.lam, args.iters, args.alpha, a=R, b=R, outdir=args.outdir,
                     sims=["sim01_sphere_uniform_local", "sim02_sphere_uniform_cmsm"])

    ell = vedo.IcoSphere(r=R, subdivisions=args.subdiv).scale([args.stretch, 1.0, 1.0])
    row_e = run_geom(ell, f"Prolate ellipsoid x{args.stretch} (Sim 7=M1, Sim 8=M2)",
                     args.dp, args.t, args.depth, args.lam, args.iters, args.alpha,
                     a=args.stretch * R, b=R, outdir=args.outdir,
                     sims=["sim07_ellipsoid_uniform_local", "sim08_ellipsoid_uniform_cmsm"])

    render([row_s, row_e], args.out)


if __name__ == "__main__":
    main()
