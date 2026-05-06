"""
main_hpc.py
===========
HPC version of main.py for SLURM array jobs.

Each array task handles a subset of seeds:
    sbatch --array=0-9 main_hpc.sh   (10 tasks, each runs N_SIMULATIONS/10 seeds)

Results are saved per-task and merged after all tasks complete.

Usage
-----
    python main_hpc.py --task-id 0 --n-tasks 10
    python main_hpc.py --task-id 0 --n-tasks 10 --n-cores 8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
import param

# ─── Simulation parameters ────────────────────────────────────────────────────
BASE_SEED     = 37
N_SIMULATIONS = 100          # total seeds across all tasks

# ─── Output ───────────────────────────────────────────────────────────────────
# Use env variable PROJECT_DIR if set (PBS copies script to spool, __file__ is unreliable)
_default_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.environ.get("PROJECT_DIR", _default_project)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUT_DIR  = os.path.join(DATA_DIR, "main_hpc")

# ─── Species / resource pool ──────────────────────────────────────────────────
N_POOL, M_POOL = 1000, 100
N_MODULES      = 1
S_RATIO        = 1
LEAKAGE_RATE   = 0.2

# ─── Community size ───────────────────────────────────────────────────────────
N1, M1 = 100, 50
N2, M2 = 100, 50

# ─── Physiology ───────────────────────────────────────────────────────────────
MAINTENANCE_COST = 0.2
RHO_VALUE        = 0.6
OMEGA_VALUE      = 0.1
T_SPAN           = (0, 100000)

# ─── Initial conditions ───────────────────────────────────────────────────────
C0_VALUE           = 0.01
R0_VALUE           = 1.0
SURVIVAL_THRESHOLD = 1e-5
EV_THRESHOLD       = 0.01

INTE_CUE_N_SAVE_POINTS = 40

# ─── Mechanistic theory curve settings ────────────────────────────────────────
THEORY_LOCAL_Q       = 0.35
MIN_POINTS_FOR_THEORY = 5


# =============================================================================
# Helper functions (identical to main.py)
# =============================================================================

def _compute_eta(l):
    return 1.0 - np.sum(l, axis=2)


def _ensure_m_vec(m, N):
    if np.ndim(m) == 0:
        return np.full(N, float(m))
    m_vec = np.asarray(m, dtype=float)
    assert m_vec.shape[0] == N
    return m_vec


def compute_Gi0_Ui0_eps(u, l, R0, m):
    N = u.shape[0]
    eta = _compute_eta(l)
    m_vec = _ensure_m_vec(m, N)
    Ui0 = np.sum(u * R0[None, :], axis=1)
    Gi0 = np.sum(u * eta * R0[None, :], axis=1)
    eps = (Gi0 - m_vec) / (Ui0 + 1e-12)
    return eta, Gi0, Ui0, eps


def _flux_rates(u, l, R_t, m, N):
    eta = 1.0 - np.sum(l, axis=2)
    uptake_pb_t  = u @ R_t
    anab_gross_t = (u * eta) @ R_t
    m_vec = np.full(N, float(m)) if np.ndim(m) == 0 else np.asarray(m, dtype=float)
    return uptake_pb_t, anab_gross_t - m_vec[:, None]


def compute_actual_cue(u, l, sol, N, m, C0):
    if sol.t is None or len(sol.t) < 2:
        return np.full(N, np.nan)
    t   = sol.t
    C_t = np.maximum(sol.y[:N, :], 0.0)
    R_t = np.maximum(sol.y[N:, :], 0.0)
    uptake_pb_t, anab_pb_t = _flux_rates(u, l, R_t, m, N)
    total_uptake = np.trapezoid(C_t * uptake_pb_t, x=t, axis=1)
    total_anab   = np.trapezoid(C_t * anab_pb_t,   x=t, axis=1)
    return np.divide(total_anab, total_uptake,
                     out=np.full(N, np.nan, dtype=float),
                     where=np.abs(total_uptake) > 1e-12)


def compute_actual_community_cue(u, l, sol, N, m, C0, survivor_idx=None):
    if sol.t is None or len(sol.t) < 2:
        return np.nan
    t   = sol.t
    C_t = np.maximum(sol.y[:N, :], 0.0)
    R_t = np.maximum(sol.y[N:, :], 0.0)
    if survivor_idx is not None:
        survivor_idx = np.asarray(survivor_idx, dtype=int)
        if survivor_idx.size == 0:
            return np.nan
        C_t = C_t[survivor_idx, :]
        u   = u[survivor_idx, :]
        l   = l[survivor_idx, :, :]
        if np.ndim(m) > 0:
            m = np.asarray(m)[survivor_idx]
        N = survivor_idx.size
    uptake_pb_t, anab_pb_t = _flux_rates(u, l, R_t, m, N)
    tot_up  = np.sum(np.trapezoid(C_t * uptake_pb_t, x=t, axis=1))
    tot_an  = np.sum(np.trapezoid(C_t * anab_pb_t,   x=t, axis=1))
    return np.nan if np.abs(tot_up) < 1e-12 else tot_an / tot_up


def cue_abundance_theory(eps, eps_c, H, Cmax):
    eps = np.asarray(eps, dtype=float)
    if not np.isfinite(eps_c) or not np.isfinite(H) or not np.isfinite(Cmax):
        return np.full_like(eps, np.nan, dtype=float)
    H    = max(float(H),    1e-12)
    Cmax = max(float(Cmax), 1e-12)
    return Cmax * (1.0 - np.exp(-np.maximum(eps - eps_c, 0.0) / H))


def _gi_of_R(u_i, eta_i, R):
    return np.sum(u_i * eta_i * R)


def _solve_resident_env(species_idx, N, M, u, l, m, lambda_alpha, rho, omega, C_full, R_full, t_span):
    keep  = np.arange(N) != species_idx
    N_res = int(np.sum(keep))
    if N_res == 0:
        return np.asarray(rho, dtype=float) / np.maximum(np.asarray(omega, dtype=float), 1e-12)
    sol_res = param.solve_micrm(
        N_res, M, u[keep], l[keep], _ensure_m_vec(m, N)[keep],
        lambda_alpha, rho, omega,
        np.maximum(C_full[keep], 1e-12), np.maximum(R_full, 1e-12),
        t_span, t_eval=None, n_save_points=2
    )
    return np.maximum(sol_res.y[N_res:, -1], 0.0)


def compute_mechanistic_curve_params(N, M, u, l, m, lambda_alpha, rho, omega,
                                     R0_ref, C_full, R_full, t_span,
                                     survival_threshold=1e-5, local_q=0.35):
    eta, Gi0, Ui0, eps = compute_Gi0_Ui0_eps(u, l, R0_ref, m)
    eps_c   = np.full(N, np.nan)
    gi_res  = np.full(N, np.nan)
    gi_full = np.full(N, np.nan)
    D_obs   = np.full(N, np.nan)
    chi_obs = np.full(N, np.nan)

    # Only run leave-one-out ODE for survivors (speed optimisation)
    survivor_indices = np.where(C_full > survival_threshold)[0]
    for i in survivor_indices:
        R_res_i    = _solve_resident_env(i, N, M, u, l, m, lambda_alpha, rho, omega, C_full, R_full, t_span)
        gi_res[i]  = _gi_of_R(u[i], eta[i], R_res_i)
        gi_full[i] = _gi_of_R(u[i], eta[i], R_full)
        eps_c[i]   = (Gi0[i] - gi_res[i]) / (Ui0[i] + 1e-12)
        D_obs[i]   = gi_res[i] - gi_full[i]
        if np.isfinite(D_obs[i]) and D_obs[i] > 0:
            chi_obs[i] = D_obs[i] / max(C_full[i], 1e-12)

    delta_eps = np.maximum(eps - eps_c, 0.0)
    valid = (
        np.isfinite(chi_obs) & np.isfinite(delta_eps) & np.isfinite(C_full) &
        (delta_eps > 0) & (C_full > survival_threshold) & (D_obs > 0)
    )
    near_mask = valid.copy()
    if np.sum(valid) >= MIN_POINTS_FOR_THEORY:
        delta_cut = np.quantile(delta_eps[valid], local_q)
        abund_cut = np.quantile(C_full[valid],    local_q)
        near_mask = valid & (delta_eps <= delta_cut) & (C_full <= abund_cut)
        if np.sum(near_mask) < 3:
            near_mask = valid

    chi_bar  = np.nanmedian(chi_obs[near_mask]) if np.any(near_mask) else np.nan
    U_bar    = np.nanmedian(Ui0[near_mask])      if np.any(near_mask) else np.nanmedian(Ui0[np.isfinite(Ui0)])
    eps_c_bar = np.nanmedian(eps_c[np.isfinite(eps_c)])
    Cmax     = np.nanmax(C_full[np.isfinite(C_full)]) if np.any(np.isfinite(C_full)) else np.nan

    H = np.nan
    if all(np.isfinite(v) and v > 0 for v in [chi_bar, U_bar, Cmax]):
        H = chi_bar * Cmax / U_bar

    surv_mask = np.isfinite(C_full) & (C_full > survival_threshold)
    y_pred    = cue_abundance_theory(eps, eps_c_bar, H, Cmax)
    if np.any(surv_mask) and np.all(np.isfinite(y_pred[surv_mask])):
        log_obs  = np.log10(np.maximum(C_full[surv_mask], survival_threshold))
        log_pred = np.log10(np.maximum(y_pred[surv_mask], survival_threshold))
        ss_res   = np.sum((log_obs - log_pred) ** 2)
        ss_tot   = np.sum((log_obs - np.mean(log_obs)) ** 2)
        theory_R2_log = np.nan if ss_tot <= 0 else 1 - ss_res / ss_tot
    else:
        theory_R2_log = np.nan

    species_df = pd.DataFrame({
        "Gi0": Gi0, "Ui0": Ui0, "eps_c_i": eps_c, "Delta_eps_i": delta_eps,
        "gi_res_i": gi_res, "gi_full_i": gi_full, "D_obs_i": D_obs, "chi_i_obs": chi_obs
    })
    params = {
        "eps_c": eps_c_bar, "chi_bar": chi_bar, "U_bar": U_bar,
        "Cmax": Cmax, "H": H, "Theory_R2_log": theory_R2_log,
        "NearThresholdUsed": int(np.sum(near_mask)),
        "N_survivors": int(np.sum(surv_mask))
    }
    return species_df, params


def estimate_theory_params_mechanistic(df_comm, survival_threshold=1e-5):
    dat = df_comm.copy()
    dat = dat[np.isfinite(dat["Species_CUE"]) & np.isfinite(dat["Abundance"])]
    if len(dat) < MIN_POINTS_FOR_THEORY:
        return None
    required_cols = ["Theory_eps_c_seed", "Theory_chi_bar_seed", "Theory_U_bar_seed",
                     "Theory_Cmax_seed", "Theory_H_seed"]
    if not all(col in dat.columns for col in required_cols):
        return None

    seed_params = (
        dat.groupby("Seed", as_index=False)
        .agg(
            eps_c=("Theory_eps_c_seed", "first"),
            chi_bar=("Theory_chi_bar_seed", "first"),
            U_bar=("Theory_U_bar_seed", "first"),
            Cmax=("Theory_Cmax_seed", "first"),
            H_seed=("Theory_H_seed", "first"),
            Theory_R2_log_seed=("Theory_R2_log_seed", "first"),
            NearThresholdUsed=("Theory_NearThresholdUsed_seed", "first")
        )
    )
    eps_c   = np.nanmedian(seed_params["eps_c"])
    chi_bar = np.nanmedian(seed_params["chi_bar"])
    U_bar   = np.nanmedian(seed_params["U_bar"])
    Cmax    = np.nanmedian(seed_params["Cmax"])
    if not all(np.isfinite(v) and v > 0 for v in [chi_bar, U_bar, Cmax]):
        return None
    if not np.isfinite(eps_c):
        return None
    H = chi_bar * Cmax / U_bar

    x      = dat["Species_CUE"].to_numpy()
    y      = dat["Abundance"].to_numpy()
    y_pred = cue_abundance_theory(x, eps_c, H, Cmax)
    surv_mask = y > survival_threshold
    if np.any(surv_mask):
        log_obs  = np.log10(np.maximum(y[surv_mask], survival_threshold))
        log_pred = np.log10(np.maximum(y_pred[surv_mask], survival_threshold))
        ss_res   = np.sum((log_obs - log_pred) ** 2)
        ss_tot   = np.sum((log_obs - np.mean(log_obs)) ** 2)
        theory_R2_log = np.nan if ss_tot <= 0 else 1 - ss_res / ss_tot
    else:
        theory_R2_log = np.nan

    return {
        "eps_c": eps_c, "chi_bar": chi_bar, "U_bar": U_bar, "H": H, "Cmax": Cmax,
        "Theory_R2_log": theory_R2_log,
        "N_total": len(dat),
        "N_survivors": int(np.sum(surv_mask)),
        "N_seeds": len(seed_params),
        "NearThresholdUsed_median": np.nanmedian(seed_params["NearThresholdUsed"])
    }


# =============================================================================
# Per-seed simulation (identical logic to main.py)
# =============================================================================

def simulate(seed):
    rng    = np.random.default_rng(seed)
    u_pool = param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = param.generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    species_indices1  = rng.choice(N_POOL, N1, replace=False)
    resource_indices1 = rng.choice(M_POOL, M1, replace=False)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
    lambda_alpha1 = np.full(M1, LEAKAGE_RATE)
    rho1   = np.full(M1, RHO_VALUE)
    omega1 = np.full(M1, OMEGA_VALUE)
    C0_1   = np.full(N1, C0_VALUE)
    R0_1   = np.full(M1, R0_VALUE)

    resource_indices2 = param.choose_resources_for_second_community(M_POOL, M1, M2, resource_indices1, rng)
    remaining_species = np.setdiff1d(np.arange(N_POOL), species_indices1)
    species_indices2  = rng.choice(remaining_species, N2, replace=False)
    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]
    lambda_alpha2 = np.full(M2, LEAKAGE_RATE)
    rho2   = np.full(M2, RHO_VALUE)
    omega2 = np.full(M2, OMEGA_VALUE)
    C0_2   = np.full(N2, C0_VALUE)
    R0_2   = np.full(M2, R0_VALUE)

    sol1 = param.solve_micrm(N1, M1, u1, l1, MAINTENANCE_COST, lambda_alpha1, rho1, omega1,
                              C0_1, R0_1, T_SPAN, n_save_points=INTE_CUE_N_SAVE_POINTS)
    sol2 = param.solve_micrm(N2, M2, u2, l2, MAINTENANCE_COST, lambda_alpha2, rho2, omega2,
                              C0_2, R0_2, T_SPAN, n_save_points=INTE_CUE_N_SAVE_POINTS)

    # Early stability check
    lambda_vec1 = np.full(N1, LEAKAGE_RATE)
    lambda_vec2 = np.full(N2, LEAKAGE_RATE)
    ev1 = param.leading_eigenvalue(param.MiCRM_jac(N1, M1, u1, l1, MAINTENANCE_COST, rho1, omega1, lambda_vec1, sol1))
    ev2 = param.leading_eigenvalue(param.MiCRM_jac(N2, M2, u2, l2, MAINTENANCE_COST, rho2, omega2, lambda_vec2, sol2))
    if not (np.isfinite(ev1) and ev1 < EV_THRESHOLD and np.isfinite(ev2) and ev2 < EV_THRESHOLD):
        return None

    C_final1 = np.maximum(sol1.y[:N1, -1], 0.0)
    C_final2 = np.maximum(sol2.y[:N2, -1], 0.0)
    R_final1 = sol1.y[N1:, -1]
    R_final2 = sol2.y[N2:, -1]

    species_indices3  = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    N3, M3        = N1 + N2, len(resource_indices3)
    lambda_alpha3 = np.full(M3, LEAKAGE_RATE)
    rho3   = np.full(M3, 2 * RHO_VALUE)
    omega3 = np.full(M3, OMEGA_VALUE)
    C0_3   = np.concatenate([C_final1, C_final2])
    R0_3   = np.full(M3, R0_VALUE)

    sol3 = param.solve_micrm(N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3, rho3, omega3,
                              C0_3, R0_3, T_SPAN, n_save_points=INTE_CUE_N_SAVE_POINTS)

    C_final3 = np.maximum(sol3.y[:N3, -1], 0.0)
    R_final3 = np.maximum(sol3.y[N3:, -1], 0.0)

    _, _, _, species_CUE1 = compute_Gi0_Ui0_eps(u1, l1, R0_1, MAINTENANCE_COST)
    _, _, _, species_CUE2 = compute_Gi0_Ui0_eps(u2, l2, R0_2, MAINTENANCE_COST)
    _, _, _, species_CUE3 = compute_Gi0_Ui0_eps(u3, l3, R0_3, MAINTENANCE_COST)

    survivors1_t = np.where(C_final1 > SURVIVAL_THRESHOLD)[0]
    survivors2_t = np.where(C_final2 > SURVIVAL_THRESHOLD)[0]
    survivors3_t = np.where(C_final3 > SURVIVAL_THRESHOLD)[0]

    actual_CUE1 = compute_actual_cue(u1, l1, sol1, N1, MAINTENANCE_COST, C0_1)
    actual_CUE2 = compute_actual_cue(u2, l2, sol2, N2, MAINTENANCE_COST, C0_2)
    actual_CUE3 = compute_actual_cue(u3, l3, sol3, N3, MAINTENANCE_COST, C0_3)
    actual_community_CUE1 = compute_actual_community_cue(u1, l1, sol1, N1, MAINTENANCE_COST, C0_1, survivor_idx=survivors1_t)
    actual_community_CUE2 = compute_actual_community_cue(u2, l2, sol2, N2, MAINTENANCE_COST, C0_2, survivor_idx=survivors2_t)
    actual_community_CUE3 = compute_actual_community_cue(u3, l3, sol3, N3, MAINTENANCE_COST, C0_3, survivor_idx=survivors3_t)

    community_CUE1     = param.safe_weighted_average(species_CUE1[survivors1_t], C_final1[survivors1_t])
    community_CUE2     = param.safe_weighted_average(species_CUE2[survivors2_t], C_final2[survivors2_t])
    community_CUE3     = param.safe_weighted_average(species_CUE3[survivors3_t], C_final3[survivors3_t])
    community_CUE1_surv = community_CUE1
    community_CUE2_surv = community_CUE2
    community_CUE3_surv = community_CUE3

    facilitation1 = np.mean(param.calculate_effective_leakage(u1, l1), axis=1)
    facilitation2 = np.mean(param.calculate_effective_leakage(u2, l2), axis=1)
    facilitation3 = np.mean(param.calculate_effective_leakage(u3, l3), axis=1)

    competition_comm1   = param.community_level_competition(u1)
    competition_comm2   = param.community_level_competition(u2)
    competition_comm3   = param.community_level_competition(u3)
    competition_species1 = param.species_level_competition(u1)
    competition_species2 = param.species_level_competition(u2)
    competition_species3 = param.species_level_competition(u3)
    competition_dot1    = param.species_level_competition_dot(u1)
    competition_dot2    = param.species_level_competition_dot(u2)
    competition_dot3    = param.species_level_competition_dot(u3)
    uptake_var1 = param.compute_uptake_variance(u1)
    uptake_var2 = param.compute_uptake_variance(u2)
    uptake_var3 = param.compute_uptake_variance(u3)

    depletion1 = np.sum(sol1.y[N1:, -1])
    depletion2 = np.sum(sol2.y[N2:, -1])
    depletion3 = np.sum(sol3.y[N3:, -1])

    dominant = ("Community 1" if np.sum(C_final3[:N1]) > np.sum(C_final3[N1:]) else "Community 2")
    lambda_vec3 = np.full(N3, LEAKAGE_RATE)

    mech1, tparams1 = compute_mechanistic_curve_params(
        N1, M1, u1, l1, MAINTENANCE_COST, lambda_alpha1, rho1, omega1,
        R0_1, C_final1, np.maximum(R_final1, 0.0), T_SPAN, SURVIVAL_THRESHOLD, THEORY_LOCAL_Q)
    mech2, tparams2 = compute_mechanistic_curve_params(
        N2, M2, u2, l2, MAINTENANCE_COST, lambda_alpha2, rho2, omega2,
        R0_2, C_final2, np.maximum(R_final2, 0.0), T_SPAN, SURVIVAL_THRESHOLD, THEORY_LOCAL_Q)
    mech3, tparams3 = compute_mechanistic_curve_params(
        N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3, rho3, omega3,
        R0_3, C_final3, np.maximum(R_final3, 0.0), T_SPAN, SURVIVAL_THRESHOLD, THEORY_LOCAL_Q)

    pred1 = cue_abundance_theory(species_CUE1, tparams1["eps_c"], tparams1["H"], tparams1["Cmax"])
    pred2 = cue_abundance_theory(species_CUE2, tparams2["eps_c"], tparams2["H"], tparams2["Cmax"])
    pred3 = cue_abundance_theory(species_CUE3, tparams3["eps_c"], tparams3["H"], tparams3["Cmax"])

    alpha1, r1 = param.calculate_elv_params(C_final1, R_final1, N1, M1, u1, l1, MAINTENANCE_COST, rho1, omega1, lambda_vec1)
    alpha2, r2 = param.calculate_elv_params(C_final2, R_final2, N2, M2, u2, l2, MAINTENANCE_COST, rho2, omega2, lambda_vec2)
    alpha3, r3 = param.calculate_elv_params(C_final3, R_final3, N3, M3, u3, l3, MAINTENANCE_COST, rho3, omega3, lambda_vec3)

    mask1 = C_final1 > SURVIVAL_THRESHOLD
    mask2 = C_final2 > SURVIVAL_THRESHOLD
    mask3 = C_final3 > SURVIVAL_THRESHOLD
    m1 = param.map_log_volume_to_feasible_proxies(
        param.compute_proxies_from_A(param.adaptive_ridge(alpha1[np.ix_(mask1, mask1)])["A_reg"])["log_volume"],
        int(np.sum(mask1)))
    m2 = param.map_log_volume_to_feasible_proxies(
        param.compute_proxies_from_A(param.adaptive_ridge(alpha2[np.ix_(mask2, mask2)])["A_reg"])["log_volume"],
        int(np.sum(mask2)))
    m3 = param.map_log_volume_to_feasible_proxies(
        param.compute_proxies_from_A(param.adaptive_ridge(alpha3[np.ix_(mask3, mask3)])["A_reg"])["log_volume"],
        int(np.sum(mask3)))

    feas1_val = m1["log10_feasible_scale_per_dim"]
    feas2_val = m2["log10_feasible_scale_per_dim"]
    feas3_val = m3["log10_feasible_scale_per_dim"]
    ev3 = param.leading_eigenvalue(param.MiCRM_jac(N3, M3, u3, l3, MAINTENANCE_COST, rho3, omega3, lambda_vec3, sol3))

    def _make_row(seed, comm_id, i, sp_cue, act_cue, act_comm_cue, comm_cue, comm_cue_surv,
                  C_c, total_ab, comp_comm, comp_sp, comp_dot, fac, dep, uptake_var,
                  n_surv, feas_val, ev, gr, pred, mech_df, tparams):
        return {
            "Seed": seed, "Community": comm_id, "Species_ID": i + 1,
            "Species_CUE": sp_cue[i], "actual_CUE": act_cue[i],
            "actual_community_CUE": act_comm_cue,
            "Community_CUE": comm_cue, "Community_CUE_surv": comm_cue_surv,
            "Abundance": C_c[i], "Total_Abundance": total_ab,
            "Dominant_Community": dominant,
            "Competition": comp_comm,
            "Species_Competition": comp_sp[i],
            "Species_Competition_Dot": comp_dot[i],
            "Facilitation": fac[i], "Depletion": dep,
            "UptakeVar": uptake_var[i], "N_Survivors": n_surv,
            "feasibility": feas_val, "Leading_Eigenvalue": float(ev),
            "Growth_Rate": float(gr[i]),
            "Theory_Abundance": pred[i],
            "Theory_DeltaEps": max(sp_cue[i] - tparams["eps_c"], 0.0),
            "Gi0": mech_df.loc[i, "Gi0"], "Ui0": mech_df.loc[i, "Ui0"],
            "eps_c_i": mech_df.loc[i, "eps_c_i"], "Delta_eps_i": mech_df.loc[i, "Delta_eps_i"],
            "gi_res_i": mech_df.loc[i, "gi_res_i"], "gi_full_i": mech_df.loc[i, "gi_full_i"],
            "D_obs_i": mech_df.loc[i, "D_obs_i"], "chi_i_obs": mech_df.loc[i, "chi_i_obs"],
            "Theory_eps_c_seed": tparams["eps_c"], "Theory_chi_bar_seed": tparams["chi_bar"],
            "Theory_U_bar_seed": tparams["U_bar"], "Theory_Cmax_seed": tparams["Cmax"],
            "Theory_H_seed": tparams["H"], "Theory_R2_log_seed": tparams["Theory_R2_log"],
            "Theory_NearThresholdUsed_seed": tparams["NearThresholdUsed"],
        }

    species_data = (
        [_make_row(seed, 1, i, species_CUE1, actual_CUE1, actual_community_CUE1,
                   community_CUE1, community_CUE1_surv, C_final1, float(np.sum(C_final1)),
                   competition_comm1, competition_species1, competition_dot1,
                   facilitation1, depletion1, uptake_var1, len(survivors1_t),
                   feas1_val, ev1, r1, pred1, mech1, tparams1)
         for i in range(N1)] +
        [_make_row(seed, 2, i, species_CUE2, actual_CUE2, actual_community_CUE2,
                   community_CUE2, community_CUE2_surv, C_final2, float(np.sum(C_final2)),
                   competition_comm2, competition_species2, competition_dot2,
                   facilitation2, depletion2, uptake_var2, len(survivors2_t),
                   feas2_val, ev2, r2, pred2, mech2, tparams2)
         for i in range(N2)] +
        [_make_row(seed, 3, i, species_CUE3, actual_CUE3, actual_community_CUE3,
                   community_CUE3, community_CUE3_surv, C_final3, float(np.sum(C_final3)),
                   competition_comm3, competition_species3, competition_dot3,
                   facilitation3, depletion3, uptake_var3, len(survivors3_t),
                   feas3_val, ev3, r3, pred3, mech3, tparams3)
         for i in range(N3)]
    )
    return species_data


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cores",  type=int, default=None,
                        help="CPU cores (default: all available)")
    args = parser.parse_args()

    rng_master = np.random.default_rng(BASE_SEED)
    all_seeds  = rng_master.integers(0, 2**32 - 1, size=N_SIMULATIONS, dtype=np.uint32).tolist()

    n_cores = args.n_cores or os.cpu_count()
    print(f"Running {len(all_seeds)} seeds on {n_cores} cores", flush=True)

    with Pool(n_cores) as pool:
        results = pool.map(simulate, all_seeds)

    rows = [row for result in results if result for row in result]
    df   = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(OUT_DIR, "coal_task0000.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} rows → {out_file}", flush=True)


if __name__ == "__main__":
    import argparse
    main()
