"""
monoculture_cue.py
------------------
For each seed, run every species from one community in monoculture and compute:
  - intrinsic_CUE : CUE evaluated at reference resource level R0 (before any depletion)
  - actual_CUE    : biomass-weighted time-integral CUE over the monoculture trajectory

Output: results/monoculture_cue.csv
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import param

# ── reproduce main.py parameters exactly ──────────────────────────────────────
BASE_SEED        = 37
N_SIMULATIONS    = 10

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

SURVIVAL_THRESHOLD  = 1e-5
T_EVAL_DT           = 1.0   # fixed t_eval spacing (independent of T_SPAN)
# ──────────────────────────────────────────────────────────────────────────────


def _compute_eta(l):
    """Retention fraction: η_{iα} = 1 - Σ_β l_{iαβ}."""
    return 1.0 - np.sum(l, axis=2)


def intrinsic_cue(u, l, R0, m):
    """
    CUE at reference resource levels (no depletion).

    CUE_i = (Σ_α u_{iα} η_{iα} R0_α  −  m_i) / (Σ_α u_{iα} R0_α)
    """
    eta   = _compute_eta(l)                        # (N, M)
    Ui0   = np.sum(u * R0[None, :], axis=1)        # total uptake
    Gi0   = np.sum(u * eta * R0[None, :], axis=1)  # gross anabolism
    return np.where(np.abs(Ui0) > 1e-12, (Gi0 - m) / Ui0, np.nan)


def actual_cue_monoculture(sol, u1, l1, m):
    """
    Species-level actual CUE matching compute_actual_cue() in main.py.

    CUE = ∫ C(t)[Σ_α u_α η_α R_α(t) − m] dt  /  ∫ C(t) Σ_α u_α R_α(t) dt

    Numerator  : biomass-weighted time-integral of net anabolism flux
    Denominator: biomass-weighted time-integral of gross uptake flux
    """
    if sol.t is None or len(sol.t) < 2:
        return np.nan

    t   = sol.t
    C_t = np.maximum(sol.y[0, :], 0.0)   # (T,)
    R_t = np.maximum(sol.y[1:, :], 0.0)  # (M, T)

    eta          = 1.0 - np.sum(l1, axis=2)        # (1, M)
    uptake_t     = u1[0] @ R_t                      # (T,)
    anab_gross_t = (u1[0] * eta[0]) @ R_t           # (T,)
    anab_t       = anab_gross_t - float(m)          # (T,)

    total_uptake = np.trapezoid(C_t * uptake_t, x=t)
    total_anab   = np.trapezoid(C_t * anab_t,   x=t)

    if np.abs(total_uptake) < 1e-12:
        return np.nan
    return total_anab / total_uptake


def run_monoculture(u_i, l_i, R0, rho, omega):
    """Run a single species to equilibrium and return the ODE solution."""
    N_mono = 1
    M_mono = u_i.shape[1]
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

    # ── generate pool (identical to main.py) ──────────────────────────────────
    u_pool = param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = param.generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    # ── Community ─────────────────────────────────────────────────────────────
    species_indices  = rng.choice(N_POOL, N1, replace=False)
    resource_indices = rng.choice(M_POOL, M1, replace=False)

    u1 = u_pool[np.ix_(species_indices, resource_indices)]
    l1 = l_pool[np.ix_(species_indices, resource_indices, resource_indices)]

    R0_1 = np.full(M1, R0_VALUE)

    # ── Monocultures ──────────────────────────────────────────────────────────
    rows = []
    intr_cue = intrinsic_cue(u1, l1, R0_1, MAINTENANCE_COST)  # (N1,)

    for i in range(N1):
        u_i = u1[i:i+1, :]    # (1, M)
        l_i = l1[i:i+1, :, :]

        sol = run_monoculture(u_i, l_i, R0_1.copy(), RHO_VALUE, OMEGA_VALUE)

        C_final = sol.y[0, -1] if sol.y.shape[1] > 0 else 0.0
        survived = bool(C_final > SURVIVAL_THRESHOLD)
        t_eq = float(sol.t[-1]) if len(sol.t) > 0 else float(T_SPAN[1])

        act_cue = actual_cue_monoculture(sol, u_i, l_i, MAINTENANCE_COST)

        rows.append({
            "Seed":          seed,
            "Species_ID":    i + 1,
            "survived":      survived,
            "C_final":       float(C_final),
            "t_eq":          t_eq,
            "intrinsic_CUE": float(intr_cue[i]),
            "actual_CUE":    float(act_cue),
        })

    return rows


def plot(df, out_path):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,
    })

    df_surv = df[df["actual_CUE"].notna()].copy()

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(df_surv["intrinsic_CUE"], df_surv["actual_CUE"],
                    c=df_surv["t_eq"], cmap="plasma_r",
                    alpha=0.6, s=15, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("$t_{eq}$", fontsize=14, fontfamily="Times New Roman")

    xlo = df_surv["intrinsic_CUE"].min() - 0.02
    xhi = df_surv["intrinsic_CUE"].max() + 0.02
    ylo = df_surv["actual_CUE"].min()    - 0.02
    yhi = df_surv["actual_CUE"].max()    + 0.02
    ref_lo, ref_hi = min(xlo, ylo), max(xhi, yhi)
    ax.plot([ref_lo, ref_hi], [ref_lo, ref_hi], "k--", lw=1, zorder=0)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    ax.set_xlabel("Intrinsic CUE")
    ax.set_ylabel("Actual CUE (monoculture)")
    ax.set_title("Intrinsic vs Actual CUE")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved → {out_path}")


def plot_biomass_cue(df, out_path):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,
    })

    df_surv = df[df["actual_CUE"].notna()].copy()

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(df_surv["C_final"], df_surv["actual_CUE"],
                    c=df_surv["t_eq"], cmap="plasma_r",
                    alpha=0.6, s=15, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("$t_{eq}$", fontsize=14, fontfamily="Times New Roman")

    ax.set_xlabel("Biomass at equilibrium ($C^*$)")
    ax.set_ylabel("Actual CUE (monoculture)")
    ax.set_title("Biomass vs Actual CUE")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved → {out_path}")


def main():
    seeds = [BASE_SEED + i for i in range(N_SIMULATIONS)]

    n_workers = min(cpu_count(), len(seeds))
    print(f"Running {N_SIMULATIONS} seeds with {n_workers} workers …")

    with Pool(n_workers) as pool:
        results = pool.map(simulate, seeds)

    all_rows = [row for seed_rows in results if seed_rows for row in seed_rows]
    df = pd.DataFrame(all_rows)

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "monoculture_cue.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
    print(df[["intrinsic_CUE", "actual_CUE"]].describe())

    fig_path = os.path.join(os.path.dirname(__file__), "..", "results", "mono_intrinsic_vs_actual_cue.pdf")
    plot(df, fig_path)
    fig_path2 = os.path.join(os.path.dirname(__file__), "..", "results", "mono_biomass_vs_actual_cue.pdf")
    plot_biomass_cue(df, fig_path2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation; load existing CSV and only produce figures.")
    args = parser.parse_args()

    if args.plot_only:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "monoculture_cue.csv")
        df = pd.read_csv(csv_path)
        fig_path = os.path.join(os.path.dirname(__file__), "..", "results", "mono_intrinsic_vs_actual_cue.pdf")
        plot(df, fig_path)
        fig_path2 = os.path.join(os.path.dirname(__file__), "..", "results", "mono_biomass_vs_actual_cue.pdf")
        plot_biomass_cue(df, fig_path2)
    else:
        main()
