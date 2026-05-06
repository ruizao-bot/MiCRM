from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import os
from scipy.integrate import cumulative_trapezoid
import param

# Random seed and simulation parameters
BASE_SEED = 37
N_SIMULATIONS = 100

# Exported file names
COAL_FILE = "data/coal_rho2.csv"

# Species pooland resource pool parameters
N_POOL = 1000
M_POOL = 100
N_MODULES = 1
S_RATIO = 1
LEAKAGE_RATE = 0.2

# Community parameters
N1, M1 = 100, 50
N2, M2 = 100, 50

# Physiological parameters
MAINTENANCE_COST = 0.2
RHO_VALUE = 0.6
OMEGA_VALUE = 0.1
T_SPAN = (0, 100000)
# Initial conditions
C0_VALUE = 0.01
R0_VALUE = 1

# Survival threshold
SURVIVAL_THRESHOLD = 1e-5
EV_THRESHOLD = 0.00

INTE_CUE_N_SAVE_POINTS = 40

# Mechanistic theory curve settings
THEORY_LOCAL_Q = 0.35
MIN_POINTS_FOR_THEORY = 5
COAL_THEORY_PARAMS_FILE = "data/cue_abundance_theory_params_rho2.csv"


# =============================================================================
# Theory curve helper functions (from main_for_plot)
# =============================================================================

def _compute_eta(l):
    """Per-species per-resource retention fraction from 3-D leakage tensor."""
    return 1.0 - np.sum(l, axis=2) 


def _ensure_m_vec(m, N):
    if np.ndim(m) == 0:
        return np.full(N, float(m))
    m_vec = np.asarray(m, dtype=float)
    assert m_vec.shape[0] == N
    return m_vec


def compute_Gi0_Ui0_eps(u, l, R0, m):
    """Compute per-species Gi0, Ui0, and CUE (eps) at reference resource levels."""
    N = u.shape[0]
    eta = _compute_eta(l)          
    m_vec = _ensure_m_vec(m, N)
    Ui0 = np.sum(u * R0[None, :], axis=1)                    # total potential uptake
    Gi0 = np.sum(u * eta * R0[None, :], axis=1)              # net growth flux
    eps = (Gi0 - m_vec) / (Ui0 + 1e-12)                      # species CUE
    return eta, Gi0, Ui0, eps

def _flux_rates(u, l, R_t, m, N):
    eta = 1.0 - np.sum(l, axis=2)  # (N, M)
    uptake_pb_t = u @ R_t           # (N, T)
    anab_gross_t = (u * eta) @ R_t  # (N, T)
 
    if np.ndim(m) == 0:
        m_vec = np.full(N, float(m))
    else:
        m_vec = np.asarray(m, dtype=float)
 
    anab_pb_t = anab_gross_t - m_vec[:, None]
    return uptake_pb_t, anab_pb_t
 
 
def compute_actual_cue(u, l, sol, N, m, C0):
    if sol.t is None or len(sol.t) < 2:
        return np.full(N, np.nan)
 
    t = sol.t
    C_t = np.maximum(sol.y[:N, :], 0.0)
    R_t = np.maximum(sol.y[N:, :], 0.0)
 
    uptake_pb_t, anab_pb_t = _flux_rates(u, l, R_t, m, N)
 
    total_uptake = np.trapezoid(C_t * uptake_pb_t, x=t, axis=1)  # (N,) biomass-weighted uptake
    total_anab = np.trapezoid(C_t * anab_pb_t, x=t, axis=1)      # (N,) biomass-weighted anabolism
 
    actual_cue = np.divide(
        total_anab,
        total_uptake,
        out=np.full(N, np.nan, dtype=float),
        where=np.abs(total_uptake) > 1e-12
    )
    return actual_cue
 
 
def compute_actual_community_cue(u, l, sol, N, m, C0, survivor_idx=None):
    if sol.t is None or len(sol.t) < 2:
        return np.nan
 
    t = sol.t
    C_t = np.maximum(sol.y[:N, :], 0.0)
    R_t = np.maximum(sol.y[N:, :], 0.0)
 
    if survivor_idx is not None:
        survivor_idx = np.asarray(survivor_idx, dtype=int)
        if survivor_idx.size == 0:
            return np.nan
        C_t = C_t[survivor_idx, :]
        u = u[survivor_idx, :]
        l = l[survivor_idx, :, :]
        if np.ndim(m) > 0:
            m = np.asarray(m)[survivor_idx]
        N_eff = survivor_idx.size
    else:
        N_eff = N

    uptake_pb_t, anab_pb_t = _flux_rates(u, l, R_t, m, N_eff)

    total_uptake = np.sum(np.trapezoid(C_t * uptake_pb_t, x=t, axis=1))  # Σ_i ∫ C_i(t)·uptake_pb_i dt
    total_anab = np.sum(np.trapezoid(C_t * anab_pb_t, x=t, axis=1))      # Σ_i ∫ C_i(t)·anab_pb_i dt

    if np.abs(total_uptake) < 1e-12:
        return np.nan
    return total_anab / total_uptake
 

# =============================================================================
# Mechanistic theory curve functions (migrated from main_for_plot)
# =============================================================================

def cue_abundance_theory(eps, eps_c, H, Cmax):
    eps = np.asarray(eps, dtype=float)
    if not np.isfinite(eps_c) or not np.isfinite(H) or not np.isfinite(Cmax):
        return np.full_like(eps, np.nan, dtype=float)
    H = max(float(H), 1e-12)
    Cmax = max(float(Cmax), 1e-12)
    delta = np.maximum(eps - eps_c, 0.0)
    return Cmax * (1.0 - np.exp(-delta / H))


def _gi_of_R(u_i, eta_i, R):
    return np.sum(u_i * eta_i * R)


def _solve_resident_env(species_idx, N, M, u, l, m, lambda_alpha, rho, omega, C_full, R_full, t_span):
    keep = np.arange(N) != species_idx
    N_res = int(np.sum(keep))
    if N_res == 0:
        return np.asarray(rho, dtype=float) / np.maximum(np.asarray(omega, dtype=float), 1e-12)
    u_res = u[keep]
    l_res = l[keep]
    m_res = _ensure_m_vec(m, N)[keep]
    C0_res = np.maximum(C_full[keep], 1e-12)
    R0_res = np.maximum(R_full, 1e-12)
    sol_res = param.solve_micrm(
        N_res, M, u_res, l_res, m_res, lambda_alpha, rho, omega,
        C0_res, R0_res, t_span, t_eval=None, n_save_points=2
    )
    return np.maximum(sol_res.y[N_res:, -1], 0.0)


def compute_mechanistic_curve_params(N, M, u, l, m, lambda_alpha, rho, omega,
                                     R0_ref, C_full, R_full, t_span,
                                     survival_threshold=1e-5, local_q=0.35):
    eta, Gi0, Ui0, eps = compute_Gi0_Ui0_eps(u, l, R0_ref, m)
    eps_c = np.full(N, np.nan)
    gi_res = np.full(N, np.nan)
    gi_full = np.full(N, np.nan)
    D_obs = np.full(N, np.nan)
    chi_obs = np.full(N, np.nan)

    survivor_indices = np.where(C_full > survival_threshold)[0]
    for i in survivor_indices:
        R_res_i = _solve_resident_env(i, N, M, u, l, m, lambda_alpha, rho, omega, C_full, R_full, t_span)
        gi_res[i] = _gi_of_R(u[i], eta[i], R_res_i)
        gi_full[i] = _gi_of_R(u[i], eta[i], R_full)
        eps_c[i] = (Gi0[i] - gi_res[i]) / (Ui0[i] + 1e-12)
        D_obs[i] = gi_res[i] - gi_full[i]
        if np.isfinite(D_obs[i]) and (D_obs[i] > 0):
            chi_obs[i] = D_obs[i] / max(C_full[i], 1e-12)

    delta_eps = np.maximum(eps - eps_c, 0.0)
    valid = (
        np.isfinite(chi_obs) & np.isfinite(delta_eps) & np.isfinite(C_full) &
        (delta_eps > 0) & (C_full > survival_threshold) & (D_obs > 0)
    )
    near_mask = valid.copy()
    if np.sum(valid) >= MIN_POINTS_FOR_THEORY:
        delta_cut = np.quantile(delta_eps[valid], local_q)
        abund_cut = np.quantile(C_full[valid], local_q)
        near_mask = valid & (delta_eps <= delta_cut) & (C_full <= abund_cut)
        if np.sum(near_mask) < 3:
            near_mask = valid

    chi_bar = np.nanmedian(chi_obs[near_mask]) if np.any(near_mask) else np.nan
    U_bar = np.nanmedian(Ui0[near_mask]) if np.any(near_mask) else np.nanmedian(Ui0[np.isfinite(Ui0)])
    eps_c_bar = np.nanmedian(eps_c[np.isfinite(eps_c)])
    Cmax = np.nanmax(C_full[np.isfinite(C_full)]) if np.any(np.isfinite(C_full)) else np.nan

    H = np.nan
    if all(np.isfinite(v) and v > 0 for v in [chi_bar, U_bar, Cmax]):
        H = chi_bar * Cmax / U_bar

    surv_mask = np.isfinite(C_full) & (C_full > survival_threshold)
    y_pred = cue_abundance_theory(eps, eps_c_bar, H, Cmax)
    if np.any(surv_mask) and np.all(np.isfinite(y_pred[surv_mask])):
        log_obs = np.log10(np.maximum(C_full[surv_mask], survival_threshold))
        log_pred = np.log10(np.maximum(y_pred[surv_mask], survival_threshold))
        ss_res = np.sum((log_obs - log_pred) ** 2)
        ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
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
    eps_c = np.nanmedian(seed_params["eps_c"])
    chi_bar = np.nanmedian(seed_params["chi_bar"])
    U_bar = np.nanmedian(seed_params["U_bar"])
    Cmax = np.nanmedian(seed_params["Cmax"])
    if not all(np.isfinite(v) and v > 0 for v in [chi_bar, U_bar, Cmax]):
        return None
    if not np.isfinite(eps_c):
        return None
    H = chi_bar * Cmax / U_bar

    x = dat["Species_CUE"].to_numpy()
    y = dat["Abundance"].to_numpy()
    y_pred = cue_abundance_theory(x, eps_c, H, Cmax)
    surv_mask = y > survival_threshold
    if np.any(surv_mask):
        log_obs = np.log10(np.maximum(y[surv_mask], survival_threshold))
        log_pred = np.log10(np.maximum(y_pred[surv_mask], survival_threshold))
        ss_res = np.sum((log_obs - log_pred) ** 2)
        ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
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


def simulate(seed):
    rng = np.random.default_rng(seed)

    u_pool = param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = param.generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    species_indices1 = rng.choice(N_POOL, N1, replace=False)
    resource_indices1 = rng.choice(M_POOL, M1, replace=False)

    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]

    lambda_alpha1 = np.full(M1, LEAKAGE_RATE)
    rho1 = np.full(M1, RHO_VALUE)
    omega1 = np.full(M1, OMEGA_VALUE)
    C0_1 = np.full(N1, C0_VALUE)
    R0_1 = np.full(M1, R0_VALUE)

    resource_indices2 = param.choose_resources_for_second_community(M_POOL, M1, M2, resource_indices1, rng)
    remaining_species = np.setdiff1d(np.arange(N_POOL), species_indices1)
    species_indices2 = rng.choice(remaining_species, N2, replace=False)

    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]

    lambda_alpha2 = np.full(M2, LEAKAGE_RATE)
    rho2 = np.full(M2, RHO_VALUE)
    omega2 = np.full(M2, OMEGA_VALUE)
    C0_2 = np.full(N2, C0_VALUE)
    R0_2 = np.full(M2, R0_VALUE)

    sol1 = param.solve_micrm(
        N1, M1, u1, l1, MAINTENANCE_COST, lambda_alpha1, rho1, omega1, C0_1, R0_1, T_SPAN,
        n_save_points=INTE_CUE_N_SAVE_POINTS
    )
    sol2 = param.solve_micrm(
        N2, M2, u2, l2, MAINTENANCE_COST, lambda_alpha2, rho2, omega2, C0_2, R0_2, T_SPAN,
        n_save_points=INTE_CUE_N_SAVE_POINTS
    )

    # Early stability check: skip seed if either parent community is unstable
    C_final1 = sol1.y[:N1, -1]
    C_final2 = sol2.y[:N2, -1]
    R_final1 = sol1.y[N1:, -1]
    R_final2 = sol2.y[N2:, -1]
    lambda_vec1 = np.full(N1, LEAKAGE_RATE)
    lambda_vec2 = np.full(N2, LEAKAGE_RATE)
    J1 = param.MiCRM_jac(N1, M1, u1, l1, MAINTENANCE_COST, rho1, omega1, lambda_vec1, sol1)
    ev1 = param.leading_eigenvalue(J1)
    J2 = param.MiCRM_jac(N2, M2, u2, l2, MAINTENANCE_COST, rho2, omega2, lambda_vec2, sol2)
    ev2 = param.leading_eigenvalue(J2)
    if not (np.isfinite(ev1) and ev1 < EV_THRESHOLD and np.isfinite(ev2) and ev2 < EV_THRESHOLD):
        return None

    C_final1 = sol1.y[:N1, -1]
    C_final2 = sol2.y[:N2, -1]

    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2

    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]

    N3 = N1 + N2
    M3 = len(resource_indices3)
    lambda_alpha3 = np.full(M3, LEAKAGE_RATE)
    rho3 = np.full(M3, 2 * RHO_VALUE)
    omega3 = np.full(M3, OMEGA_VALUE)
    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])
    R0_3 = np.full(M3, R0_VALUE)

    sol3 = param.solve_micrm(
        N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3, rho3, omega3, C0_3, R0_3, T_SPAN,
        n_save_points=INTE_CUE_N_SAVE_POINTS
    )

    C_final1 = np.maximum(sol1.y[:N1, -1], 0.0)
    C_final2 = np.maximum(sol2.y[:N2, -1], 0.0)
    C_final3 = np.maximum(sol3.y[:N3, -1], 0.0)
    R_final3 = np.maximum(sol3.y[N3:, -1], 0.0)

    # Per-species CUE (using eta from l tensor for mechanistic consistency)
    _, _, _, species_CUE1 = compute_Gi0_Ui0_eps(u1, l1, R0_1, MAINTENANCE_COST)
    _, _, _, species_CUE2 = compute_Gi0_Ui0_eps(u2, l2, R0_2, MAINTENANCE_COST)
    _, _, _, species_CUE3 = compute_Gi0_Ui0_eps(u3, l3, R0_3, MAINTENANCE_COST)
    survivors1_t = np.where(C_final1 > SURVIVAL_THRESHOLD)[0]
    survivors2_t = np.where(C_final2 > SURVIVAL_THRESHOLD)[0]
    survivors3_t = np.where(C_final3 > SURVIVAL_THRESHOLD)[0]

    actual_CUE1 = compute_actual_cue(u1, l1, sol1, N1, MAINTENANCE_COST, C0_1)
    actual_CUE2 = compute_actual_cue(u2, l2, sol2, N2, MAINTENANCE_COST, C0_2)
    actual_CUE3 = compute_actual_cue(u3, l3, sol3, N3, MAINTENANCE_COST, C0_3)
    actual_community_CUE1 = compute_actual_community_cue(
        u1, l1, sol1, N1, MAINTENANCE_COST, C0_1, survivor_idx=survivors1_t
    )
    actual_community_CUE2 = compute_actual_community_cue(
        u2, l2, sol2, N2, MAINTENANCE_COST, C0_2, survivor_idx=survivors2_t
    )
    actual_community_CUE3 = compute_actual_community_cue(
        u3, l3, sol3, N3, MAINTENANCE_COST, C0_3, survivor_idx=survivors3_t
    )


    survivors1_count = len(survivors1_t)
    survivors2_count = len(survivors2_t)
    survivors3_count = len(survivors3_t)

    # Community CUE is computed on surviving species only.
    community_CUE1 = param.safe_weighted_average(species_CUE1[survivors1_t], C_final1[survivors1_t])
    community_CUE2 = param.safe_weighted_average(species_CUE2[survivors2_t], C_final2[survivors2_t])
    community_CUE3 = param.safe_weighted_average(species_CUE3[survivors3_t], C_final3[survivors3_t])
    community_CUE1_surv = param.safe_weighted_average(species_CUE1[survivors1_t], C_final1[survivors1_t])
    community_CUE2_surv = param.safe_weighted_average(species_CUE2[survivors2_t], C_final2[survivors2_t])
    community_CUE3_surv = param.safe_weighted_average(species_CUE3[survivors3_t], C_final3[survivors3_t])

    L_eff1 = param.calculate_effective_leakage(u1, l1)
    L_eff2 = param.calculate_effective_leakage(u2, l2)
    L_eff3 = param.calculate_effective_leakage(u3, l3)

    facilitation1 = np.mean(L_eff1, axis=1)
    facilitation2 = np.mean(L_eff2, axis=1)
    facilitation3 = np.mean(L_eff3, axis=1)

    competition_comm1 = param.community_level_competition(u1)
    competition_comm2 = param.community_level_competition(u2)
    competition_comm3 = param.community_level_competition(u3)

    competition_species1 = param.species_level_competition(u1)
    competition_species2 = param.species_level_competition(u2)
    competition_species3 = param.species_level_competition(u3)

    competition_dot1 = param.species_level_competition_dot(u1)
    competition_dot2 = param.species_level_competition_dot(u2)
    competition_dot3 = param.species_level_competition_dot(u3)

    uptake_var1 = param.compute_uptake_variance(u1)
    uptake_var2 = param.compute_uptake_variance(u2)
    uptake_var3 = param.compute_uptake_variance(u3)

    depletion1 = np.sum(sol1.y[N1:, -1])
    depletion2 = np.sum(sol2.y[N2:, -1])
    depletion3 = np.sum(sol3.y[N3:, -1])

    total_abundance1 = np.sum(C_final1)
    total_abundance2 = np.sum(C_final2)
    total_abundance3 = np.sum(C_final3)

    origin1_in_coalesced = np.sum(C_final3[:N1])
    origin2_in_coalesced = np.sum(C_final3[N1:])
    dominant = "Community 1" if origin1_in_coalesced > origin2_in_coalesced else "Community 2"
    lambda_vec3 = np.full(N3, LEAKAGE_RATE)

    # Mechanistic theory curve params (leave-one-out ODE per species)
    mech1, tparams1 = compute_mechanistic_curve_params(
        N1, M1, u1, l1, MAINTENANCE_COST, lambda_alpha1, rho1, omega1,
        R0_1, C_final1, np.maximum(R_final1, 0.0), T_SPAN, SURVIVAL_THRESHOLD, THEORY_LOCAL_Q
    )
    mech2, tparams2 = compute_mechanistic_curve_params(
        N2, M2, u2, l2, MAINTENANCE_COST, lambda_alpha2, rho2, omega2,
        R0_2, C_final2, np.maximum(R_final2, 0.0), T_SPAN, SURVIVAL_THRESHOLD, THEORY_LOCAL_Q
    )
    mech3, tparams3 = compute_mechanistic_curve_params(
        N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3, rho3, omega3,
        R0_3, C_final3, np.maximum(R_final3, 0.0), T_SPAN, SURVIVAL_THRESHOLD, THEORY_LOCAL_Q
    )
    pred1 = cue_abundance_theory(species_CUE1, tparams1["eps_c"], tparams1["H"], tparams1["Cmax"])
    pred2 = cue_abundance_theory(species_CUE2, tparams2["eps_c"], tparams2["H"], tparams2["Cmax"])
    pred3 = cue_abundance_theory(species_CUE3, tparams3["eps_c"], tparams3["H"], tparams3["Cmax"])

    alpha1, r1 = param.calculate_elv_params(C_final1, R_final1, N1, M1, u1, l1, MAINTENANCE_COST, rho1, omega1, lambda_vec1)
    alpha2, r2 = param.calculate_elv_params(C_final2, R_final2, N2, M2, u2, l2, MAINTENANCE_COST, rho2, omega2, lambda_vec2)
    alpha3, r3 = param.calculate_elv_params(C_final3, R_final3, N3, M3, u3, l3, MAINTENANCE_COST, rho3, omega3, lambda_vec3)

    # Feasibility via spectral proxy (adaptive ridge + SVD log-volume)
    mask1 = C_final1 > SURVIVAL_THRESHOLD
    mask2 = C_final2 > SURVIVAL_THRESHOLD
    mask3 = C_final3 > SURVIVAL_THRESHOLD
    m1 = param.map_log_volume_to_feasible_proxies(
    param.compute_proxies_from_A(param.adaptive_ridge(alpha1[np.ix_(mask1, mask1)])["A_reg"])["log_volume"],
    int(np.sum(mask1))
)
    m2 = param.map_log_volume_to_feasible_proxies(
    param.compute_proxies_from_A(param.adaptive_ridge(alpha2[np.ix_(mask2, mask2)])["A_reg"])["log_volume"],
    int(np.sum(mask2))
)
    m3 = param.map_log_volume_to_feasible_proxies(
    param.compute_proxies_from_A(param.adaptive_ridge(alpha3[np.ix_(mask3, mask3)])["A_reg"])["log_volume"],
    int(np.sum(mask3))
)

    feas1_val, log10_feas1, feas1_raw = m1["log10_feasible_scale_per_dim"], m1["log10_feasible_volume_proxy"], m1["feasible_volume_proxy"]
    feas2_val, log10_feas2, feas2_raw = m2["log10_feasible_scale_per_dim"], m2["log10_feasible_volume_proxy"], m2["feasible_volume_proxy"]
    feas3_val, log10_feas3, feas3_raw = m3["log10_feasible_scale_per_dim"], m3["log10_feasible_volume_proxy"], m3["feasible_volume_proxy"]
    J3 = param.MiCRM_jac(N3, M3, u3, l3, MAINTENANCE_COST, rho3, omega3, lambda_vec3, sol3)
    ev3 = param.leading_eigenvalue(J3)

    species_data = []

    for i in range(N1):
        row = {
            "Seed": seed, "Community": 1, "Species_ID": i + 1,
            "Species_CUE": species_CUE1[i],
            "actual_CUE": actual_CUE1[i],
            "actual_community_CUE": actual_community_CUE1,
            "Community_CUE": community_CUE1, "Community_CUE_surv": community_CUE1_surv,
            "Abundance": C_final1[i], "Total_Abundance": total_abundance1,
            "Dominant_Community": dominant,
            "Competition": competition_comm1,
            "Species_Competition": competition_species1[i],
            "Species_Competition_Dot": competition_dot1[i],
            "Facilitation": facilitation1[i], "Depletion": depletion1,
            "UptakeVar": uptake_var1[i], "N_Survivors": survivors1_count,
            "feasibility": feas1_val, "Leading_Eigenvalue": float(ev1),
            "Growth_Rate": float(r1[i]),
            "Theory_Abundance": pred1[i],
            "Theory_DeltaEps": max(species_CUE1[i] - tparams1["eps_c"], 0.0),
            "Gi0": mech1.loc[i, "Gi0"], "Ui0": mech1.loc[i, "Ui0"],
            "eps_c_i": mech1.loc[i, "eps_c_i"], "Delta_eps_i": mech1.loc[i, "Delta_eps_i"],
            "gi_res_i": mech1.loc[i, "gi_res_i"], "gi_full_i": mech1.loc[i, "gi_full_i"],
            "D_obs_i": mech1.loc[i, "D_obs_i"], "chi_i_obs": mech1.loc[i, "chi_i_obs"],
            "Theory_eps_c_seed": tparams1["eps_c"], "Theory_chi_bar_seed": tparams1["chi_bar"],
            "Theory_U_bar_seed": tparams1["U_bar"], "Theory_Cmax_seed": tparams1["Cmax"],
            "Theory_H_seed": tparams1["H"], "Theory_R2_log_seed": tparams1["Theory_R2_log"],
            "Theory_NearThresholdUsed_seed": tparams1["NearThresholdUsed"],
        }
        species_data.append(row)

    for i in range(N2):
        row = {
            "Seed": seed, "Community": 2, "Species_ID": i + 1,
            "Species_CUE": species_CUE2[i],
            "actual_CUE": actual_CUE2[i],
            "actual_community_CUE": actual_community_CUE2,
            "Community_CUE": community_CUE2, "Community_CUE_surv": community_CUE2_surv,
            "Abundance": C_final2[i], "Total_Abundance": total_abundance2,
            "Dominant_Community": dominant,
            "Competition": competition_comm2,
            "Species_Competition": competition_species2[i],
            "Species_Competition_Dot": competition_dot2[i],
            "Facilitation": facilitation2[i], "Depletion": depletion2,
            "UptakeVar": uptake_var2[i], "N_Survivors": survivors2_count,
            "feasibility": feas2_val, "Leading_Eigenvalue": float(ev2),
            "Growth_Rate": float(r2[i]),
            "Theory_Abundance": pred2[i],
            "Theory_DeltaEps": max(species_CUE2[i] - tparams2["eps_c"], 0.0),
            "Gi0": mech2.loc[i, "Gi0"], "Ui0": mech2.loc[i, "Ui0"],
            "eps_c_i": mech2.loc[i, "eps_c_i"], "Delta_eps_i": mech2.loc[i, "Delta_eps_i"],
            "gi_res_i": mech2.loc[i, "gi_res_i"], "gi_full_i": mech2.loc[i, "gi_full_i"],
            "D_obs_i": mech2.loc[i, "D_obs_i"], "chi_i_obs": mech2.loc[i, "chi_i_obs"],
            "Theory_eps_c_seed": tparams2["eps_c"], "Theory_chi_bar_seed": tparams2["chi_bar"],
            "Theory_U_bar_seed": tparams2["U_bar"], "Theory_Cmax_seed": tparams2["Cmax"],
            "Theory_H_seed": tparams2["H"], "Theory_R2_log_seed": tparams2["Theory_R2_log"],
            "Theory_NearThresholdUsed_seed": tparams2["NearThresholdUsed"],
        }
        species_data.append(row)

    for i in range(N3):
        row = {
            "Seed": seed, "Community": 3, "Species_ID": i + 1,
            "Species_CUE": species_CUE3[i],
            "actual_CUE": actual_CUE3[i],
            "actual_community_CUE": actual_community_CUE3,
            "Community_CUE": community_CUE3, "Community_CUE_surv": community_CUE3_surv,
            "Abundance": C_final3[i], "Total_Abundance": total_abundance3,
            "Dominant_Community": dominant,
            "Competition": competition_comm3,
            "Species_Competition": competition_species3[i],
            "Species_Competition_Dot": competition_dot3[i],
            "Facilitation": facilitation3[i], "Depletion": depletion3,
            "UptakeVar": uptake_var3[i], "N_Survivors": survivors3_count,
            "feasibility": feas3_val, "Leading_Eigenvalue": float(ev3),
            "Growth_Rate": float(r3[i]),
            "Theory_Abundance": pred3[i],
            "Theory_DeltaEps": max(species_CUE3[i] - tparams3["eps_c"], 0.0),
            "Gi0": mech3.loc[i, "Gi0"], "Ui0": mech3.loc[i, "Ui0"],
            "eps_c_i": mech3.loc[i, "eps_c_i"], "Delta_eps_i": mech3.loc[i, "Delta_eps_i"],
            "gi_res_i": mech3.loc[i, "gi_res_i"], "gi_full_i": mech3.loc[i, "gi_full_i"],
            "D_obs_i": mech3.loc[i, "D_obs_i"], "chi_i_obs": mech3.loc[i, "chi_i_obs"],
            "Theory_eps_c_seed": tparams3["eps_c"], "Theory_chi_bar_seed": tparams3["chi_bar"],
            "Theory_U_bar_seed": tparams3["U_bar"], "Theory_Cmax_seed": tparams3["Cmax"],
            "Theory_H_seed": tparams3["H"], "Theory_R2_log_seed": tparams3["Theory_R2_log"],
            "Theory_NearThresholdUsed_seed": tparams3["NearThresholdUsed"],
        }
        species_data.append(row)

    # Collect time series data
    t1, t2, t3 = sol1.t, sol2.t, sol3.t

    def _instantaneous_community_cue(u, l, sol, N, m, survivor_idx):
        C_t = np.maximum(sol.y[:N, :], 0.0)
        R_t = np.maximum(sol.y[N:, :], 0.0)
        idx = np.asarray(survivor_idx, dtype=int)
        C_s = C_t[idx, :]
        uptake_pb_t, anab_pb_t = _flux_rates(u[idx], l[idx], R_t, m, idx.size)
        uptake_comm = np.sum(C_s * uptake_pb_t, axis=0)
        anab_comm = np.sum(C_s * anab_pb_t, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cue_ts = np.where(np.abs(uptake_comm) > 1e-12, anab_comm / uptake_comm, np.nan)
        return cue_ts

    cue_ts1 = _instantaneous_community_cue(u1, l1, sol1, N1, MAINTENANCE_COST, survivors1_t if len(survivors1_t) else np.arange(N1))
    cue_ts2 = _instantaneous_community_cue(u2, l2, sol2, N2, MAINTENANCE_COST, survivors2_t if len(survivors2_t) else np.arange(N2))
    cue_ts3 = _instantaneous_community_cue(u3, l3, sol3, N3, MAINTENANCE_COST, survivors3_t if len(survivors3_t) else np.arange(N3))

    timeseries_data = []
    for idx, time_val in enumerate(t1):
        timeseries_data.append({
            "Seed": seed,
            "Community": 1,
            "Time": time_val,
            "actual_community_CUE": cue_ts1[idx]
        })
    for idx, time_val in enumerate(t2):
        timeseries_data.append({
            "Seed": seed,
            "Community": 2,
            "Time": time_val,
            "actual_community_CUE": cue_ts2[idx]
        })
    for idx, time_val in enumerate(t3):
        timeseries_data.append({
            "Seed": seed,
            "Community": 3,
            "Time": time_val,
            "actual_community_CUE": cue_ts3[idx]
        })

    return species_data, timeseries_data


def main():
    seed_generator = np.random.default_rng(BASE_SEED)
    seeds = seed_generator.integers(0, 2**32 - 1, size=N_SIMULATIONS, dtype=np.uint32).tolist()

    with Pool(cpu_count()) as pool:
        all_results_nested = pool.map(simulate, seeds)

    # Separate species data and timeseries data
    all_species_data = [
        row
        for one_seed_result in all_results_nested
        if one_seed_result
        for row in one_seed_result[0]
    ]
    
    all_timeseries_data = [
        row
        for one_seed_result in all_results_nested
        if one_seed_result
        for row in one_seed_result[1]
    ]

    os.makedirs("data", exist_ok=True)
    
    # Save species data
    df = pd.DataFrame(all_species_data)
    df.to_csv(COAL_FILE, index=False)
    print(f"Saved: {COAL_FILE}")

    # Estimate cross-seed theory params per community and save
    params_rows = []
    for comm in sorted(df["Community"].astype(str).unique()):
        dat_comm = df[df["Community"].astype(str) == comm].copy()
        params = estimate_theory_params_mechanistic(dat_comm, survival_threshold=SURVIVAL_THRESHOLD)
        if params is None:
            params_rows.append({
                "Community": comm, "eps_c": np.nan, "chi_bar": np.nan, "U_bar": np.nan,
                "H": np.nan, "Cmax": np.nan, "Theory_R2_log": np.nan,
                "N_total": len(dat_comm),
                "N_survivors": int(np.sum(dat_comm["Abundance"] > SURVIVAL_THRESHOLD)),
                "N_seeds": dat_comm["Seed"].nunique(), "NearThresholdUsed_median": np.nan
            })
        else:
            params_rows.append({"Community": comm, **params})
    pd.DataFrame(params_rows).to_csv(COAL_THEORY_PARAMS_FILE, index=False)
    print(f"Saved: {COAL_THEORY_PARAMS_FILE}")


if __name__ == "__main__":
    main()