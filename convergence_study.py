"""Convergence study: sphere σ₁ error vs mesh refinement (IcoSphere subdiv 3–6).

Plots relative error and spurious-deviatoric std vs mesh spacing h,
both raw (Tikhonov only) and after 12-iteration Laplacian smoothing.
Saves to out/convergence.png.

Run:
    & C:/Users/nimes/miniforge3/envs/fem_env/python.exe convergence_study.py
"""
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vedo

from membrane_stress_fd import solve_membrane

R   = 1.0
DP  = 20.0
T   = 0.05
SIGMA_REF = DP * R / (2 * T)   # 200 Pa  (sphere analytic)


def one_ring_avg(mesh: vedo.Mesh) -> sp.csr_matrix:
    adlist = mesh.compute_adjacency()
    n = mesh.npoints
    rows, cols, vals = [], [], []
    for i in range(n):
        nb = np.asarray(
            mesh.find_adjacent_vertices(i, depth=1, adjacency_list=adlist), dtype=int
        )
        nb = nb[nb != i]
        if len(nb) == 0:
            rows.append(i); cols.append(i); vals.append(1.0)
            continue
        w = 1.0 / len(nb)
        rows += [i] * len(nb)
        cols += list(nb)
        vals += [w] * len(nb)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def laplacian_smooth(f: np.ndarray, A: sp.csr_matrix,
                     iters: int = 12, alpha: float = 0.5) -> np.ndarray:
    out = f.copy()
    for _ in range(iters):
        out = (1 - alpha) * out + alpha * (A @ out)
    return out


def main():
    subdivs = [3, 4, 5, 6]
    rows = []

    for sd in subdivs:
        print(f"\n=== IcoSphere subdiv={sd} ===")
        mesh = vedo.IcoSphere(r=R, subdivisions=sd)
        n = mesh.npoints
        h = np.sqrt(4 * np.pi * R**2 / n)
        print(f"  vertices={n}  h={h:.4f}")

        res = solve_membrane(mesh, dp=DP, t=T, depth=3, lam=0.05)
        A = one_ring_avg(mesh)
        s1_raw = res["sigma1"]
        s2_raw = res["sigma2"]
        s1_sm  = laplacian_smooth(s1_raw, A)
        s2_sm  = laplacian_smooth(s2_raw, A)

        # relative error of σ₁ vs analytic
        err_raw = np.abs(s1_raw - SIGMA_REF) / SIGMA_REF
        err_sm  = np.abs(s1_sm  - SIGMA_REF) / SIGMA_REF

        # spurious deviatoric: (σ₁ − σ₂)/2 should be 0 on sphere
        dev_raw = (s1_raw - s2_raw) / 2
        dev_sm  = (s1_sm  - s2_sm)  / 2

        row = dict(
            subdiv=sd, n=n, h=h,
            err_mean_raw=err_raw.mean(),  err_med_raw=np.median(err_raw),
            err_mean_sm=err_sm.mean(),    err_med_sm=np.median(err_sm),
            dev_std_raw=dev_raw.std(),    dev_std_sm=dev_sm.std(),
            resid=res["resid"],
        )
        rows.append(row)
        print(f"  resid={row['resid']:.3e}  "
              f"err_mean(raw)={row['err_mean_raw']:.3f}  "
              f"err_mean(sm)={row['err_mean_sm']:.3f}  "
              f"dev_std(raw)={row['dev_std_raw']:.2f} Pa")

    hs     = np.array([r["h"]           for r in rows])
    ns     = np.array([r["n"]           for r in rows])
    subdivs_arr = np.array([r["subdiv"] for r in rows])

    err_mean_raw = np.array([r["err_mean_raw"] for r in rows])
    err_med_raw  = np.array([r["err_med_raw"]  for r in rows])
    err_mean_sm  = np.array([r["err_mean_sm"]  for r in rows])
    err_med_sm   = np.array([r["err_med_sm"]   for r in rows])
    dev_raw      = np.array([r["dev_std_raw"]  for r in rows])
    dev_sm       = np.array([r["dev_std_sm"]   for r in rows])
    resids       = np.array([r["resid"]        for r in rows])

    # h² reference anchored to raw mean at coarsest mesh
    h_ref = np.logspace(np.log10(hs.min() * 0.8), np.log10(hs.max() * 1.2), 50)
    ref2  = err_mean_raw[0] * (h_ref / hs[0]) ** 2
    ref1  = err_mean_raw[0] * (h_ref / hs[0]) ** 1

    labels = [f"sd{s}\n(n={n})" for s, n in zip(subdivs_arr, ns)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("GFDM membrane stress: convergence on IcoSphere (sphere, $R=1$, "
                 r"$\Delta p=20$ Pa, $t=0.05$)", fontsize=11)

    # ---- left: relative error in σ₁ vs h ----
    ax = axes[0]
    ax.loglog(hs, err_mean_raw, "o-",  color="tab:blue",  label="raw — mean error")
    ax.loglog(hs, err_med_raw,  "s--", color="tab:blue",  label="raw — median error", alpha=0.6)
    ax.loglog(hs, err_mean_sm,  "o-",  color="tab:orange", label="smoothed — mean error")
    ax.loglog(hs, err_med_sm,   "s--", color="tab:orange", label="smoothed — median error", alpha=0.6)
    ax.loglog(h_ref, ref2, "k:",  linewidth=1.2, label=r"$\propto h^2$ reference")
    ax.loglog(h_ref, ref1, "k--", linewidth=0.8, label=r"$\propto h^1$ reference", alpha=0.5)
    ax.set_xlabel("Mesh spacing $h = \\sqrt{4\\pi R^2/n}$")
    ax.set_ylabel(r"Relative error $|\sigma_1 - \sigma_{\rm ref}|/\sigma_{\rm ref}$")
    ax.set_title(r"$\sigma_1$ error vs mesh spacing")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    for hi, label in zip(hs, labels):
        ax.annotate(label, (hi, err_mean_raw[list(hs).index(hi)]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7, color="tab:blue")

    # ---- right: spurious deviatoric std vs h ----
    ax2 = axes[1]
    ax2.loglog(hs, dev_raw, "o-",  color="tab:blue",   label="raw")
    ax2.loglog(hs, dev_sm,  "o-",  color="tab:orange", label="smoothed (12 iters, α=0.5)")
    ax2.set_xlabel("Mesh spacing $h$")
    ax2.set_ylabel(r"Std of $(σ_1 - σ_2)/2$ [Pa]  (should → 0)")
    ax2.set_title("Spurious deviatoric (null-mode bias)")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    for hi, label in zip(hs, labels):
        ax2.annotate(label, (hi, dev_raw[list(hs).index(hi)]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7, color="tab:blue")

    plt.tight_layout()
    import os; os.makedirs("out", exist_ok=True)
    out_path = "out/convergence.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # plain text summary table
    print(f"\n{'subdiv':>7} {'n':>6} {'h':>7} "
          f"{'err_mean(raw)':>15} {'err_mean(sm)':>14} "
          f"{'dev_std(raw)':>14} {'dev_std(sm)':>13} {'resid':>10}")
    for r in rows:
        print(f"{r['subdiv']:>7} {r['n']:>6} {r['h']:>7.4f} "
              f"{r['err_mean_raw']:>15.4f} {r['err_mean_sm']:>14.4f} "
              f"{r['dev_std_raw']:>14.2f} {r['dev_std_sm']:>13.2f} "
              f"{r['resid']:>10.3e}")


if __name__ == "__main__":
    main()
