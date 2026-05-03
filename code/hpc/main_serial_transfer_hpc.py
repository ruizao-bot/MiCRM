"""
main_serial_transfer_hpc.py
============================
HPC version of main_serial_transfer.py for SLURM array jobs.

Each array task handles a subset of seeds:
    sbatch --array=0-9 serial_transfer.sh   (10 tasks, each runs N_SIMULATIONS/10 seeds)

Results are saved per-task and merged by the submit script.

Usage
-----
    python main_serial_transfer_hpc.py --task-id 0 --n-tasks 10
"""

import argparse
import os
import sys

# Allow import of param.py from the code/ directory regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import param

# ─── Simulation parameters ────────────────────────────────────────────────────
BASE_SEED         = 37
N_SIMULATIONS     = 500          # total seeds across all tasks

# ─── Output ───────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_DIR   = os.path.join(DATA_DIR, "serial_transfer_hpc")

# ─── Species / resource pool ──────────────────────────────────────────────────
N_POOL, M_POOL    = 1000, 100
N_MODULES         = 1
S_RATIO           = 1.0
LEAKAGE_RATE      = 0.2

# ─── Community size ───────────────────────────────────────────────────────────
N1, M1            = 100, 50
N2, M2            = 100, 50

# ─── Physiology ───────────────────────────────────────────────────────────────
MAINTENANCE_COST  = 0.2

# ─── Serial-transfer protocol ─────────────────────────────────────────────────
T_CYCLE           = 24.0
DILUTION_FACTOR   = 0.01
R_FEED_VALUE      = 1.0
N_CYCLES_MAX      = 1000
CONVERGENCE_TOL   = 1e-3
N_CHECK_CYCLES    = 5
N_T_EVAL_DENSE    = 200

# ─── Initial conditions ───────────────────────────────────────────────────────
C0_VALUE          = 0.01
R0_VALUE          = 1.0
SURVIVAL_THRESHOLD = 1e-5


# =============================================================================
# Batch-cycle ODE
# =============================================================================

def _solve_batch_cycle(N, M, u, l, m, lambda_alpha, C0, R0, T_CYCLE,
                       tol_ode=1e-8, method="BDF", n_t_eval=None):
    def ode(t, y):
        C = np.maximum(y[:N], 0.0)
        R = np.maximum(y[N:], 0.0)
        anab_flux   = u * (R * (1.0 - lambda_alpha))
        dCdt        = C * (np.sum(anab_flux, axis=1) - m)
        consumption = np.sum(C[:, None] * u * R, axis=0)
        leakage     = np.einsum("i,j,ij,ijk->k", C, R, u, l)
        dRdt        = -consumption + leakage
        return np.concatenate([dCdt, dRdt])

    t_eval = np.linspace(0.0, T_CYCLE, int(n_t_eval)) if n_t_eval else None
    sol = solve_ivp(
        ode, (0.0, T_CYCLE), np.concatenate([C0, R0]),
        t_eval=t_eval, method=method,
        rtol=tol_ode, atol=tol_ode * 1e-3, dense_output=False,
    )
    return sol


def _transfer_step(C_end, R_end, R_feed, D):
    return D * C_end, D * R_end + (1.0 - D) * R_feed


def run_serial_transfer(N, M, u, l, m, C0, R_feed):
    lambda_alpha = np.full(M, LEAKAGE_RATE)
    C, R = C0.copy(), np.full(M, R_feed)
    converged_streak, C_prev = 0, C.copy()
    C_end = C.copy()
    R_end = R.copy()

    for cycle in range(N_CYCLES_MAX):
        sol  = _solve_batch_cycle(N, M, u, l, m, lambda_alpha, C, R, T_CYCLE)
        C_end = np.maximum(sol.y[:N, -1], 0.0)
        R_end = np.maximum(sol.y[N:,  -1], 0.0)

        norm_prev  = np.linalg.norm(C_prev)
        rel_change = (np.linalg.norm(C_end - C_prev) / norm_prev
                      if norm_prev > 0 else np.inf)

        converged_streak = converged_streak + 1 if rel_change < CONVERGENCE_TOL else 0
        C_prev = C_end.copy()

        if converged_streak >= N_CHECK_CYCLES:
            C_s, R_s = _transfer_step(C_end, R_end, R_feed, DILUTION_FACTOR)
            last_sol = _solve_batch_cycle(
                N, M, u, l, m, lambda_alpha, C_s, R_s, T_CYCLE,
                n_t_eval=N_T_EVAL_DENSE
            )
            return C_end, last_sol, cycle + 1

        C, R = _transfer_step(C_end, R_end, R_feed, DILUTION_FACTOR)

    last_sol = _solve_batch_cycle(
        N, M, u, l, m, lambda_alpha, C, R, T_CYCLE, n_t_eval=N_T_EVAL_DENSE
    )
    return C_end, last_sol, N_CYCLES_MAX


# =============================================================================
# CUE helpers
# =============================================================================

def _compute_eta(l):
    return 1.0 - np.sum(l, axis=2)


def intrinsic_cue(u, l, R0, m):
    eta = _compute_eta(l)
    Ui0 = np.sum(u * R0[None, :],       axis=1)
    Gi0 = np.sum(u * eta * R0[None, :], axis=1)
    return (Gi0 - m) / (Ui0 + 1e-12)


def _flux_rates(u, l, R_t, m, N):
    eta          = 1.0 - np.sum(l, axis=2)
    uptake_pb_t  = u @ R_t
    anab_gross_t = (u * eta) @ R_t
    m_vec = np.full(N, float(m)) if np.ndim(m) == 0 else np.asarray(m, dtype=float)
    return uptake_pb_t, anab_gross_t - m_vec[:, None]


def compute_actual_cue_cycle(u, l, sol, N, m):
    if sol is None or sol.t is None or len(sol.t) < 2:
        return np.full(N, np.nan)
    t   = sol.t
    C_t = np.maximum(sol.y[:N, :], 0.0)
    R_t = np.maximum(sol.y[N:, :], 0.0)
    up, an = _flux_rates(u, l, R_t, m, N)
    return np.divide(
        np.trapezoid(C_t * an, x=t, axis=1),
        np.trapezoid(C_t * up, x=t, axis=1),
        out=np.full(N, np.nan, dtype=float),
        where=np.abs(np.trapezoid(C_t * up, x=t, axis=1)) > 1e-12
    )


def compute_actual_community_cue_cycle(u, l, sol, N, m, survivor_idx=None):
    if sol is None or sol.t is None or len(sol.t) < 2:
        return np.nan
    t   = sol.t
    C_t = np.maximum(sol.y[:N, :], 0.0)
    R_t = np.maximum(sol.y[N:, :], 0.0)
    if survivor_idx is not None:
        idx = np.asarray(survivor_idx, dtype=int)
        if idx.size == 0:
            return np.nan
        C_t, u, l = C_t[idx], u[idx], l[idx]
        m    = np.asarray(m)[idx] if np.ndim(m) > 0 else m
        N    = idx.size
    up, an = _flux_rates(u, l, R_t, m, N)
    tot_up = np.sum(np.trapezoid(C_t * up, x=t, axis=1))
    tot_an = np.sum(np.trapezoid(C_t * an, x=t, axis=1))
    return np.nan if np.abs(tot_up) < 1e-12 else tot_an / tot_up


# =============================================================================
# Per-seed simulation
# =============================================================================

def simulate(seed):
    rng    = np.random.default_rng(seed)
    u_pool = param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = param.generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    species_indices1  = rng.choice(N_POOL, N1, replace=False)
    resource_indices1 = rng.choice(M_POOL, M1, replace=False)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]

    resource_indices2 = param.choose_resources_for_second_community(
        M_POOL, M1, M2, resource_indices1, rng)
    remaining = np.setdiff1d(np.arange(N_POOL), species_indices1)
    species_indices2  = rng.choice(remaining, N2, replace=False)
    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]

    C_ss1, sol1, cyc1 = run_serial_transfer(N1, M1, u1, l1, MAINTENANCE_COST,
                                             np.full(N1, C0_VALUE), R_FEED_VALUE)
    C_ss2, sol2, cyc2 = run_serial_transfer(N2, M2, u2, l2, MAINTENANCE_COST,
                                             np.full(N2, C0_VALUE), R_FEED_VALUE)

    C_final1 = np.maximum(C_ss1, 0.0)
    C_final2 = np.maximum(C_ss2, 0.0)

    species_indices3  = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    N3, M3 = N1 + N2, len(resource_indices3)

    C_ss3, sol3, cyc3 = run_serial_transfer(N3, M3, u3, l3, MAINTENANCE_COST,
                                             np.concatenate([C_final1, C_final2]), R_FEED_VALUE)
    C_final3 = np.maximum(C_ss3, 0.0)

    surv1 = np.where(C_final1 > SURVIVAL_THRESHOLD)[0]
    surv2 = np.where(C_final2 > SURVIVAL_THRESHOLD)[0]
    surv3 = np.where(C_final3 > SURVIVAL_THRESHOLD)[0]

    R0_1 = np.full(M1, R0_VALUE)
    R0_2 = np.full(M2, R0_VALUE)
    R0_3 = np.full(M3, R0_VALUE)
    sp_cue1 = intrinsic_cue(u1, l1, R0_1, MAINTENANCE_COST)
    sp_cue2 = intrinsic_cue(u2, l2, R0_2, MAINTENANCE_COST)
    sp_cue3 = intrinsic_cue(u3, l3, R0_3, MAINTENANCE_COST)

    act_cue1 = compute_actual_cue_cycle(u1, l1, sol1, N1, MAINTENANCE_COST)
    act_cue2 = compute_actual_cue_cycle(u2, l2, sol2, N2, MAINTENANCE_COST)
    act_cue3 = compute_actual_cue_cycle(u3, l3, sol3, N3, MAINTENANCE_COST)

    act_comm1 = compute_actual_community_cue_cycle(u1, l1, sol1, N1, MAINTENANCE_COST, surv1)
    act_comm2 = compute_actual_community_cue_cycle(u2, l2, sol2, N2, MAINTENANCE_COST, surv2)
    act_comm3 = compute_actual_community_cue_cycle(u3, l3, sol3, N3, MAINTENANCE_COST, surv3)

    comm_cue1 = param.safe_weighted_average(sp_cue1[surv1], C_final1[surv1])
    comm_cue2 = param.safe_weighted_average(sp_cue2[surv2], C_final2[surv2])
    comm_cue3 = param.safe_weighted_average(sp_cue3[surv3], C_final3[surv3])

    comp1 = param.community_level_competition(u1)
    comp2 = param.community_level_competition(u2)
    comp3 = param.community_level_competition(u3)

    comp_sp1 = param.species_level_competition(u1)
    comp_sp2 = param.species_level_competition(u2)
    comp_sp3 = param.species_level_competition(u3)

    comp_dot1 = param.species_level_competition_dot(u1)
    comp_dot2 = param.species_level_competition_dot(u2)
    comp_dot3 = param.species_level_competition_dot(u3)

    L_eff1 = param.calculate_effective_leakage(u1, l1)
    L_eff2 = param.calculate_effective_leakage(u2, l2)
    L_eff3 = param.calculate_effective_leakage(u3, l3)

    dep1 = float(np.sum(np.maximum(sol1.y[N1:, -1], 0.0))) if sol1 else np.nan
    dep2 = float(np.sum(np.maximum(sol2.y[N2:, -1], 0.0))) if sol2 else np.nan
    dep3 = float(np.sum(np.maximum(sol3.y[N3:, -1], 0.0))) if sol3 else np.nan

    dominant = "Community 1" if np.sum(C_final3[:N1]) > np.sum(C_final3[N1:]) else "Community 2"

    rows = []
    for comm_id, (N_c, u_c, act_c, sp_c, comm_c, act_comm_c,
                  C_c, comp_c, comp_sp_c, comp_dot_c,
                  fac_c, dep_c, surv_c, cyc_c) in enumerate([
        (N1, u1, act_cue1, sp_cue1, comm_cue1, act_comm1,
         C_final1, comp1, comp_sp1, comp_dot1,
         np.mean(L_eff1, axis=1), dep1, surv1, cyc1),
        (N2, u2, act_cue2, sp_cue2, comm_cue2, act_comm2,
         C_final2, comp2, comp_sp2, comp_dot2,
         np.mean(L_eff2, axis=1), dep2, surv2, cyc2),
        (N3, u3, act_cue3, sp_cue3, comm_cue3, act_comm3,
         C_final3, comp3, comp_sp3, comp_dot3,
         np.mean(L_eff3, axis=1), dep3, surv3, cyc3),
    ], start=1):
        total_ab = float(np.sum(C_c))
        for i in range(N_c):
            rows.append({
                "Seed": seed, "Community": comm_id, "Species_ID": i + 1,
                "Species_CUE": sp_c[i], "actual_CUE": act_c[i],
                "actual_community_CUE": act_comm_c,
                "Community_CUE": comm_c,
                "Abundance": C_c[i], "Total_Abundance": total_ab,
                "Dominant_Community": dominant,
                "Competition": comp_c,
                "Species_Competition": comp_sp_c[i],
                "Species_Competition_Dot": comp_dot_c[i],
                "Facilitation": fac_c[i],
                "Depletion": dep_c,
                "N_Survivors": len(surv_c),
                "N_Cycles": cyc_c,
            })
    return rows


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id",  type=int, default=0,
                        help="SLURM_ARRAY_TASK_ID (0-based)")
    parser.add_argument("--n-tasks",  type=int, default=1,
                        help="Total number of array tasks")
    parser.add_argument("--n-cores",  type=int, default=None,
                        help="CPU cores per task (default: all available)")
    args = parser.parse_args()

    # Distribute seeds across tasks
    rng_master = np.random.default_rng(BASE_SEED)
    all_seeds  = rng_master.integers(0, 2**32 - 1,
                                     size=N_SIMULATIONS, dtype=np.uint32).tolist()
    task_seeds = all_seeds[args.task_id::args.n_tasks]

    n_cores = args.n_cores or os.cpu_count()
    print(f"Task {args.task_id}/{args.n_tasks}: "
          f"{len(task_seeds)} seeds, {n_cores} cores", flush=True)

    with Pool(n_cores) as pool:
        results = pool.map(simulate, task_seeds)

    rows = [row for result in results if result for row in result]
    df   = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(OUT_DIR, f"coal_serial_task{args.task_id:04d}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} rows → {out_file}", flush=True)


if __name__ == "__main__":
    main()
