"""FEM validation suite -- reproduce the GFDM Section-10 tests for the stress-based FEM.

Runs, for membrane_stress_fem.solve_membrane_fem, the same battery used to validate the
GFDM solve, head-to-head with GFDM where cheap:

  1. Convergence   (S10.2): sphere mean error + spurious deviatoric vs mesh spacing h
                            (IcoSphere subdiv 3-5), FEM vs GFDM.
  2. Linearity     (S10.3): sigma proportional to dp/t across six (dp, t) combos.
  3. Benchmark     (S10.1): sphere / prolate spheroid / capsule vs analytic, subdiv-4.
  4. Lambda curve  (S10.7): spheroid sigma_max error vs the Tikhonov weight (U-curve).

Writes out/fem_validation.png and prints all tables.

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe fem_validation.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vedo

from membrane_stress_fem import solve_membrane_fem
from membrane_stress_fd import solve_membrane as solve_gfdm, analytic_axisym
from convergence_study import one_ring_avg, laplacian_smooth
from show_capsule import make_capsule

R, DP, T = 1.0, 20.0, 0.05
SIGMA_REF = DP * R / (2 * T)          # 200 Pa, sphere analytic
STRETCH = 2.0                          # prolate spheroid a:b = 2:1


# ------------------------------------------------------------------ helpers
def _dev_std(s1, s2):
    return float(np.std((s1 - s2) / 2.0))


def _spheroid_err(res, a, b):
    """Mean relative error of sigma_max, sigma_min vs the axisymmetric analytic,
    over the equatorial belt (exclude the long-axis poles), like report()."""
    pole = np.abs(res["radial"][:, 0]) < 0.92
    num_max = np.maximum(res["sigma1"], res["sigma2"])[pole]
    num_min = np.minimum(res["sigma1"], res["sigma2"])[pole]
    sm, sh = analytic_axisym(res["pts"], DP, T, a, b)
    an_max = np.maximum(sm, sh)[pole]
    an_min = np.minimum(sm, sh)[pole]
    e_max = np.abs(num_max - an_max) / np.maximum(np.abs(an_max), 1e-12)
    e_min = np.abs(num_min - an_min) / np.maximum(np.abs(an_min), 1e-12)
    return e_max.mean(), e_min.mean()


# ------------------------------------------------------------------ 1. convergence
def convergence(subdivs=(3, 4, 5)):
    print("\n" + "=" * 64 + "\n1. CONVERGENCE (sphere)  FEM vs GFDM\n" + "=" * 64)
    print(f"{'sd':>3} {'n':>6} {'h':>7} | {'FEM err':>8} {'FEMdevTik':>9} {'FEMdevSm':>9} "
          f"{'FEMdevRaw':>9} | {'GFDM err':>8} {'GFDMdev':>8}")
    rows = []
    for sd in subdivs:
        m = vedo.IcoSphere(r=R, subdivisions=sd)
        n = m.npoints
        h = np.sqrt(4 * np.pi * R ** 2 / n)
        rf = solve_membrane_fem(m, DP, T, depth=3, lam=0.05, raw=True)
        A = one_ring_avg(m)
        s1s, s2s = laplacian_smooth(rf["sigma1"], A), laplacian_smooth(rf["sigma2"], A)
        f_err = np.abs((rf["sigma1"] + rf["sigma2"]) / 2 - SIGMA_REF).mean() / SIGMA_REF
        f_dev_tik = _dev_std(rf["sigma1"], rf["sigma2"])
        f_dev_sm = _dev_std(s1s, s2s)
        f_dev_raw = _dev_std(rf["sigma1_raw"], rf["sigma2_raw"])
        rg = solve_gfdm(m, DP, T, depth=3, lam=0.05, solver=("lsqr" if sd >= 5 else "auto"))
        g_err = np.abs((rg["sigma1"] + rg["sigma2"]) / 2 - SIGMA_REF).mean() / SIGMA_REF
        g_dev = _dev_std(rg["sigma1"], rg["sigma2"])
        rows.append(dict(sd=sd, n=n, h=h, f_err=f_err, f_dev_tik=f_dev_tik,
                         f_dev_sm=f_dev_sm, f_dev_raw=f_dev_raw, g_err=g_err, g_dev=g_dev))
        print(f"{sd:>3} {n:>6} {h:>7.4f} | {f_err:>8.2%} {f_dev_tik:>9.2f} {f_dev_sm:>9.2f} "
              f"{f_dev_raw:>9.1f} | {g_err:>8.2%} {g_dev:>8.2f}")
    return rows


# ------------------------------------------------------------------ 2. linearity
def linearity(subdiv=4):
    print("\n" + "=" * 64 + "\n2. LINEARITY  sigma / (dp/t) should be constant\n" + "=" * 64)
    combos = [(10, 0.05), (20, 0.10), (20, 0.05), (40, 0.10), (40, 0.05), (20, 0.025)]
    sph = vedo.IcoSphere(r=R, subdivisions=subdiv)
    ell = vedo.IcoSphere(r=R, subdivisions=subdiv).scale([STRETCH, 1.0, 1.0])
    print(f"{'dp':>5} {'t':>6} {'dp/t':>6} | {'sphere mean':>11} {'ratio':>7} | "
          f"{'ell sig_max':>11} {'ratio':>7}")
    rows = []
    for dp, t in combos:
        rs = solve_membrane_fem(sph, dp, t, depth=3, lam=0.05, raw=False)
        re = solve_membrane_fem(ell, dp, t, depth=3, lam=0.05, raw=False)
        sm = ((rs["sigma1"] + rs["sigma2"]) / 2).mean()
        emax = np.maximum(re["sigma1"], re["sigma2"]).mean()
        rows.append(dict(dp=dp, t=t, dpt=dp / t, sph_ratio=sm / (dp / t),
                         ell_ratio=emax / (dp / t)))
        print(f"{dp:>5.0f} {t:>6.3f} {dp/t:>6.0f} | {sm:>11.2f} {sm/(dp/t):>7.4f} | "
              f"{emax:>11.2f} {emax/(dp/t):>7.4f}")
    return rows


# ------------------------------------------------------------------ 3. benchmark
def benchmark(subdiv=4):
    print("\n" + "=" * 64 + "\n3. ANALYTIC BENCHMARK (FEM, subdiv-4)\n" + "=" * 64)
    out = {}
    # sphere
    sph = vedo.IcoSphere(r=R, subdivisions=subdiv)
    rs = solve_membrane_fem(sph, DP, T, depth=3, lam=0.05, raw=False)
    s_mean = ((rs["sigma1"] + rs["sigma2"]) / 2).mean()
    out["sphere"] = (abs(s_mean - SIGMA_REF) / SIGMA_REF, _dev_std(rs["sigma1"], rs["sigma2"]))
    print(f"  sphere   : mean {s_mean:7.2f} (err {out['sphere'][0]:.2%})  "
          f"dev-std {out['sphere'][1]:.2f}  [analytic {SIGMA_REF:.0f}]")
    # spheroid
    ell = vedo.IcoSphere(r=R, subdivisions=subdiv).scale([STRETCH, 1.0, 1.0])
    re = solve_membrane_fem(ell, DP, T, depth=3, lam=0.05, raw=False)
    em_max, em_min = _spheroid_err(re, a=STRETCH * R, b=R)
    out["spheroid"] = (em_max, em_min)
    print(f"  spheroid : sigma_max err {em_max:.2%}   sigma_min err {em_min:.2%}  "
          f"[equator ratio 1.75]")
    # capsule (R=1, H=2): cylinder hoop=dp R/t, axial=dp R/2t; caps=dp R/2t
    cap = make_capsule(R=1.0, H=2.0, ntheta=40, nphi=14)
    rc = solve_membrane_fem(cap, DP, T, depth=3, lam=0.05, raw=False)
    z = rc["pts"][:, 2]
    body = np.abs(z) < 0.85 * 2.0
    hoop = np.maximum(rc["sigma1"], rc["sigma2"])[body]
    axial = np.minimum(rc["sigma1"], rc["sigma2"])[body]
    e_hoop = abs(hoop.mean() - DP * R / T) / (DP * R / T)
    e_axial = abs(axial.mean() - DP * R / (2 * T)) / (DP * R / (2 * T))
    out["capsule"] = (e_hoop, e_axial)
    print(f"  capsule  : hoop {hoop.mean():7.2f} (err {e_hoop:.2%})  "
          f"axial {axial.mean():7.2f} (err {e_axial:.2%})  [analytic 400 / 200]")
    return out


# ------------------------------------------------------------------ 4. lambda curve
def lambda_curve(subdiv=4):
    print("\n" + "=" * 64 + "\n4. LAMBDA TRADEOFF (spheroid sigma_max error)\n" + "=" * 64)
    ell = vedo.IcoSphere(r=R, subdivisions=subdiv).scale([STRETCH, 1.0, 1.0])
    lams = np.geomspace(0.005, 0.3, 8)
    rows = []
    for lam in lams:
        re = solve_membrane_fem(ell, DP, T, depth=3, lam=lam, raw=False)
        em_max, em_min = _spheroid_err(re, a=STRETCH * R, b=R)
        rows.append(dict(lam=lam, e_max=em_max, e_min=em_min, resid=re["resid"]))
        print(f"  lam={lam:6.3f}  sigma_max err {em_max:6.2%}  sigma_min err {em_min:6.2%}  "
              f"resid {re['resid']:.2e}")
    return rows


# ------------------------------------------------------------------ figure
def make_figure(conv, lin, lam, out="out/fem_validation.png"):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Stress-based FEM validation suite (sphere/spheroid, dp=20, t=0.05)",
                 fontsize=13)

    # (A) convergence: mean error vs h
    hs = np.array([r["h"] for r in conv])
    ax[0, 0].loglog(hs, [r["f_err"] for r in conv], "o-", label="FEM (Tikhonov)")
    ax[0, 0].loglog(hs, [r["g_err"] for r in conv], "s--", label="GFDM (Tikhonov)")
    ax[0, 0].set_xlabel("mesh spacing h"); ax[0, 0].set_ylabel("sphere mean rel. error")
    ax[0, 0].set_title("(A) Convergence: mean stress error"); ax[0, 0].legend(); ax[0, 0].grid(True, which="both", alpha=0.3)

    # (B) convergence: spurious deviatoric vs h
    ax[0, 1].loglog(hs, [r["f_dev_raw"] for r in conv], "^:", color="gray", label="FEM raw (min-norm)")
    ax[0, 1].loglog(hs, [r["f_dev_tik"] for r in conv], "o-", label="FEM Tikhonov")
    ax[0, 1].loglog(hs, [r["f_dev_sm"] for r in conv], "o-", label="FEM Tik+Laplacian")
    ax[0, 1].loglog(hs, [r["g_dev"] for r in conv], "s--", label="GFDM Tikhonov")
    ax[0, 1].set_xlabel("mesh spacing h"); ax[0, 1].set_ylabel("std of (sigma1-sigma2)/2 [Pa]")
    ax[0, 1].set_title("(B) Spurious deviatoric (=0 exact)"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(True, which="both", alpha=0.3)

    # (C) linearity
    dpt = np.array([r["dpt"] for r in lin])
    order = np.argsort(dpt)
    ax[1, 0].plot(dpt[order], np.array([r["sph_ratio"] for r in lin])[order], "o-", label="sphere mean / (dp/t)")
    ax[1, 0].plot(dpt[order], np.array([r["ell_ratio"] for r in lin])[order], "s-", label="spheroid sig_max / (dp/t)")
    ax[1, 0].axhline(0.5, color="gray", ls=":", label="sphere analytic 0.5")
    ax[1, 0].set_xlabel("dp / t"); ax[1, 0].set_ylabel("sigma / (dp/t)")
    ax[1, 0].set_title("(C) Linearity: ratio constant"); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, alpha=0.3)

    # (D) lambda tradeoff
    lv = np.array([r["lam"] for r in lam])
    ax[1, 1].semilogx(lv, [100 * r["e_max"] for r in lam], "o-", label="sigma_max error")
    ax[1, 1].semilogx(lv, [100 * r["e_min"] for r in lam], "s-", label="sigma_min error")
    ax[1, 1].set_xlabel("Tikhonov weight lambda"); ax[1, 1].set_ylabel("spheroid error [%]")
    ax[1, 1].set_title("(D) Lambda tradeoff (U-curve)"); ax[1, 1].legend(); ax[1, 1].grid(True, which="both", alpha=0.3)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


def main():
    conv = convergence()
    lin = linearity()
    bench = benchmark()
    lam = lambda_curve()
    make_figure(conv, lin, lam)


if __name__ == "__main__":
    main()
