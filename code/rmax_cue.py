"""
rmax_cue.py
-----------
Compute CUE via the maximum-growth-rate method:

  1. Run each species in monoculture.
  2. Find t* = argmax dC/dt along the trajectory.
  3. At t*, the per-capita net growth rate is
         rmax = Σ_α u_{iα} R_α(t*)(1−λ) − m
    4. growth_CUE = rmax / (rmax + m)

Output: results/rmax_cue.csv  +  figure comparison with intrinsic CUE
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
import param

# ── parameters (mirror single_cue.py / monoculture_cue.py) ───────────────────
BASE_SEED        = 37
N_SIMULATIONS    = 50

N_POOL, M_POOL   = 1000, 100
N_MODULES        = 1
S_RATIO          = 1
LEAKAGE_RATE     = 0.2

N1, M1           = 100, 50

MAINTENANCE_COST = 0.2
RHO_VALUE        = 0.6
OMEGA_VALUE      = 0.1
T_SPAN           = (0, 100000)
C0_VALUE         = 0.01
R0_VALUE         = 1.0

SURVIVAL_THRESHOLD = 1e-5
T_EVAL_DT          = 0.1
# ─────────────────────────────────────────────────────────────────────────────


def _compute_eta(l):
    """Retention fraction: η_{iα} = 1 − Σ_β l_{iαβ}."""
    return 1.0 - np.sum(l, axis=2)


def intrinsic_cue(u, l, R0, m):
    """CUE evaluated at the initial resource level R0."""
    eta = _compute_eta(l)                        # (N, M)
    Ui0 = np.sum(u * R0[None, :], axis=1)
    Gi0 = np.sum(u * eta * R0[None, :], axis=1)
    return np.where(np.abs(Ui0) > 1e-12, (Gi0 - m) / Ui0, np.nan)


def rmax_cue_monoculture(sol, u_i, l_i, m):
    """
    Compute CUE via the rmax method for a single-species trajectory.

    Steps
    -----
    1. Compute μ(t) = Σ_α u_{iα} η_{iα} R_α(t) − m at every saved time point.
    2. Find t* = argmax μ(t).
    3. rmax = μ(t*) — the maximum per-capita net growth rate.
    4. Return rmax / (rmax + m).

    Returns
    -------
    cue   : float   growth_CUE (nan if trajectory is too short or rmax <= 0)
    rmax  : float   maximum per-capita net growth rate μ(t*)
    t_star: float   time of maximum per-capita growth rate μ(t)
    """
    if sol.t is None or len(sol.t) < 2:
        return np.nan, np.nan, np.nan

    t   = sol.t                                   # (T,)
    R_t = np.maximum(sol.y[1:, :], 0.0)          # (M, T)

    eta = _compute_eta(l_i)                       # (1, M)

    # per-capita gross anabolism at each time point: Σ_α u η R_α(t)
    anab_gross_t = (u_i[0] * eta[0]) @ R_t       # (T,)  dot over M

    # per-capita net growth rate: μ(t) = Σ u η R(t) − m
    mu_t = anab_gross_t - float(m)               # (T,)

    idx_star = int(np.argmax(mu_t))
    t_star   = float(t[idx_star])
    rmax     = float(mu_t[idx_star])

    if rmax <= 0.0 or np.isnan(rmax):
        return np.nan, rmax, t_star

    cue = rmax / (rmax + float(m))
    return cue, rmax, t_star


def run_monoculture(u_i, l_i, R0, rho, omega):
    """Integrate a single species to equilibrium."""
    N_mono  = 1
    M_mono  = u_i.shape[1]
    C0_mono = np.array([C0_VALUE])
    lambda_mono = np.full(M_mono, LEAKAGE_RATE)
    rho_mono    = np.full(M_mono, rho)
    omega_mono  = np.full(M_mono, omega)

    t_eval = np.arange(T_SPAN[0], T_SPAN[1] + T_EVAL_DT, T_EVAL_DT)
    sol = param.solve_micrm(
        N_mono, M_mono,
        u_i, l_i,
        MAINTENANCE_COST,
        lambda_mono, rho_mono, omega_mono,
        C0_mono, R0,
        T_SPAN,
        t_eval=t_eval,
    )
    return sol


def simulate(seed):
    rng = np.random.default_rng(seed)

    u_pool = param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = param.generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    species_indices  = rng.choice(N_POOL, N1, replace=False)
    resource_indices = rng.choice(M_POOL, M1, replace=False)

    u1 = u_pool[np.ix_(species_indices, resource_indices)]
    l1 = l_pool[np.ix_(species_indices, resource_indices, resource_indices)]

    R0_1    = np.full(M1, R0_VALUE)
    intr_cue = intrinsic_cue(u1, l1, R0_1, MAINTENANCE_COST)   # (N1,)

    rows = []
    for i in range(N1):
        u_i = u1[i:i+1, :]
        l_i = l1[i:i+1, :, :]

        sol = run_monoculture(u_i, l_i, R0_1.copy(), RHO_VALUE, OMEGA_VALUE)

        C_final  = float(sol.y[0, -1]) if sol.y.shape[1] > 0 else 0.0
        survived = bool(C_final > SURVIVAL_THRESHOLD)
        t_eq     = float(sol.t[-1]) if len(sol.t) > 0 else float(T_SPAN[1])

        growth_cue, rmax_val, t_star = rmax_cue_monoculture(
            sol, u_i, l_i, MAINTENANCE_COST
        )

        rows.append({
            "Seed":          seed,
            "Species_ID":    i + 1,
            "survived":      survived,
            "C_final":       C_final,
            "t_eq":          t_eq,
            "intrinsic_CUE": float(intr_cue[i]),
            "rmax":          rmax_val,
            "t_star":        t_star,
            "growth_CUE":    growth_cue,
        })

    return rows


def plot(df, out_path):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"],
        "mathtext.fontset":   "custom",
        "mathtext.rm":        "Times New Roman",
        "mathtext.it":        "Times New Roman:italic",
        "mathtext.bf":        "Times New Roman:bold",
        "font.size":          12,
        "axes.labelsize":     12,
        "axes.titlesize":     12,
        "xtick.labelsize":    12,
        "ytick.labelsize":    12,
        "axes.linewidth":     0.4,
        "xtick.major.width":  0.4,
        "ytick.major.width":  0.4,
        "xtick.major.size":   4.5,
        "ytick.major.size":   4.5,
        "legend.frameon":     False,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })

    df_plot = df.dropna(subset=["growth_CUE", "intrinsic_CUE"]).copy()

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(
        df_plot["intrinsic_CUE"], df_plot["growth_CUE"],
        s=5, alpha=0.45,
        facecolors="#9FB7CC", edgecolors="black", linewidths=0.4,
        zorder=3,
    )

    # style_ax equivalent
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.4)
    ax.tick_params(axis="both", width=0.4, colors="black", pad=4)
    ax.grid(False)

    x_lo = df_plot["intrinsic_CUE"].min() - 0.03
    x_hi = df_plot["intrinsic_CUE"].max() + 0.03
    y_lo = df_plot["growth_CUE"].min()     - 0.03
    y_hi = df_plot["growth_CUE"].max()     + 0.03
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("Intrinsic CUE", labelpad=6)
    ax.set_ylabel("Growth CUE", labelpad=8)
    ax.set_title("", pad=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Figure saved → {out_path}")


if __name__ == "__main__":
    seeds = [BASE_SEED + i for i in range(N_SIMULATIONS)]

    n_workers = min(cpu_count(), N_SIMULATIONS)
    print(f"Running {N_SIMULATIONS} seeds with {n_workers} workers …")

    with Pool(n_workers) as pool:
        nested = pool.map(simulate, seeds)

    rows = [r for batch in nested for r in batch]
    df   = pd.DataFrame(rows)

    out_csv = os.path.join(os.path.dirname(__file__), "../results/rmax_cue.csv")
    df.to_csv(out_csv, index=False)
    print(f"Results saved → {out_csv}")
    print(df[["intrinsic_CUE", "rmax", "growth_CUE"]].describe().round(4))

    out_fig = os.path.join(os.path.dirname(__file__), "../results/monoculture_figures/rmax_cue.pdf")
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plot(df, out_fig)


