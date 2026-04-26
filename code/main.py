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
COAL_FILE = "data/coal.csv"
COAL_SUMMARY_FILE = "data/coal_summary.csv"
COAL_CUE_TIMESERIES_FILE = "data/coal_cue_timeseries.csv"

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
 
    # Absolute fluxes (multiply per-biomass rate by C_i(t))
    uptake_abs_t = C_t * uptake_pb_t  # (N, T)
    anab_abs_t = C_t * anab_pb_t      # (N, T)
 
    total_uptake = np.trapezoid(uptake_abs_t, x=t, axis=1)  # (N,)
    total_anab = np.trapezoid(anab_abs_t, x=t, axis=1)      # (N,)
 
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
 
    # Sum absolute fluxes over species at each time point
    uptake_comm_t = np.sum(C_t * uptake_pb_t, axis=0)  # (T,)
    anab_comm_t = np.sum(C_t * anab_pb_t, axis=0)      # (T,)
 
    total_uptake = np.trapezoid(uptake_comm_t, x=t)
    total_anab = np.trapezoid(anab_comm_t, x=t)
 
    if np.abs(total_uptake) < 1e-12:
        return np.nan
    return total_anab / total_uptake
 

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
    rho3 = np.full(M3, RHO_VALUE)
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
        }
        species_data.append(row)

    # Collect time series data
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


if __name__ == "__main__":
    main()