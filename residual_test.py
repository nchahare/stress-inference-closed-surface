"""Equilibrium-residual test (§9.4 of tension_inference.tex).

Verifies the relative equilibrium residual  ε = ||L s − b|| / ||b||  is below the
expected thresholds (≲10⁻² on smooth analytic meshes) for the sphere, prolate
spheroid, and capsule, and maps the *per-vertex* residual onto each surface to
show WHERE equilibrium is least satisfied.

Expectation: residual concentrates at
  - umbilic poles (curvature-frame singularities: sphere everywhere, spheroid
    and capsule at the poles), and
  - the capsule cylinder/cap junction (curvature discontinuity → stencils
    straddle two curvature regimes).

Saves: out/residual_map.png

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe residual_test.py
    & ... residual_test.py --show
"""

import argparse
import os

import numpy as np
import vedo

from membrane_stress_fd_v2 import solve_membrane, make_capsule

DP    = 20.0
T     = 0.05
DEPTH = 3
LAM   = 0.05
THRESH_ANALYTIC = 1e-2
THRESH_REAL     = 0.2


def panel_map(cases, out=None, show=False):
    """3-panel vedo plot: each surface coloured by per-vertex residual (log scale)."""
    # shared colour limits across all three (log10 of per-vertex residual)
    all_rpv = np.concatenate([np.log10(res["resid_pv"] + 1e-12) for _, res, _ in cases])
    vmin = float(np.percentile(all_rpv, 2))
    vmax = float(np.percentile(all_rpv, 98))

    N   = len(cases)
    plt = vedo.Plotter(N=N, size=(960 * N, 820), offscreen=not show)

    for k, (mesh, res, title) in enumerate(cases):
        mc = mesh.clone()
        mc.pointdata["log_resid"] = np.log10(res["resid_pv"] + 1e-12)
        mc.cmap("hot_r", "log_resid", vmin=vmin, vmax=vmax).alpha(1.0)
        if k == N - 1:
            mc.add_scalarbar(title="log10(residual/dp)")
        eps = res["resid"]
        txt = vedo.Text2D(f"{title}\nglobal eps = {eps:.2e}",
                          pos="top-left", font="Calco", s=0.8)
        plt.at(k).show(mc, txt, axes=1, resetcam=True)

    if out:
        os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
        plt.screenshot(out)
        print(f"Saved {out}")
    if show:
        plt.interactive()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--show",   action="store_true")
    ap.add_argument("--out",    default="out/residual_map.png")
    args = ap.parse_args()

    R    = 1.0
    A    = 2.0
    CAP_R, CAP_H = 1.0, 2.0

    sphere = vedo.IcoSphere(r=R, subdivisions=args.subdiv)
    ell    = vedo.IcoSphere(r=R, subdivisions=args.subdiv).scale([A, 1.0, 1.0])
    cap    = make_capsule(R=CAP_R, H=CAP_H)
    cap.rotate_y(90)

    cases = []
    print(f"{'Geometry':<22} {'n':>6} {'global eps':>12} {'pv median':>12} "
          f"{'pv p95':>10} {'pv max':>10} {'threshold':>10}")
    print("-" * 86)
    for mesh, title, geom in [
        (sphere, "Sphere R=1",            "sphere"),
        (ell,    "Spheroid a=2 b=1",      "spheroid"),
        (cap,    "Capsule R=1 H=2",       "capsule"),
    ]:
        res = solve_membrane(mesh, DP, T, depth=DEPTH, lam=LAM)
        rpv = res["resid_pv"]
        flag = "OK" if res["resid"] < THRESH_ANALYTIC else "ABOVE"
        print(f"{title:<22} {mesh.npoints:>6} {res['resid']:>12.3e} "
              f"{np.median(rpv):>12.3e} {np.percentile(rpv,95):>10.3e} "
              f"{rpv.max():>10.3e} {THRESH_ANALYTIC:>10.0e}  {flag}")
        cases.append((mesh, res, title))

    print(f"\nThresholds: analytic meshes eps <= {THRESH_ANALYTIC:.0e}, "
          f"real meshes eps <= {THRESH_REAL:.1f}")

    panel_map(cases, out=args.out, show=args.show)


if __name__ == "__main__":
    main()
