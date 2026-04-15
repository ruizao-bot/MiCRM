from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import param
from overlap_species import (adaptive_ridge, compute_proxies_from_A,
                             map_log_volume_to_feasible_proxies)

# Random seed and simulation parameters
BASE_SEED = 37
N_SIMULATIONS = 50

# Exported file names
COAL_FILE = "data/coal.csv"

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

    sol1 = param.solve_micrm(N1, M1, u1, l1, MAINTENANCE_COST, lambda_alpha1, rho1, omega1, C0_1, R0_1, T_SPAN)
    sol2 = param.solve_micrm(N2, M2, u2, l2, MAINTENANCE_COST, lambda_alpha2, rho2, omega2, C0_2, R0_2, T_SPAN)

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

    sol3 = param.solve_micrm(N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3, rho3, omega3, C0_3, R0_3, T_SPAN)
    C_final3 = sol3.y[:N3, -1]

    species_CUE1 = param.compute_species_CUE(u1, R0_1, lambda_alpha1, MAINTENANCE_COST)
    species_CUE2 = param.compute_species_CUE(u2, R0_2, lambda_alpha2, MAINTENANCE_COST)
    species_CUE3 = param.compute_species_CUE(u3, R0_3, lambda_alpha3, MAINTENANCE_COST)

    C_final3 = sol3.y[:N3, -1]
    R_final3 = sol3.y[N3:, -1]

    survivors1_t = np.where(C_final1 > SURVIVAL_THRESHOLD)[0]
    survivors2_t = np.where(C_final2 > SURVIVAL_THRESHOLD)[0]
    survivors3_t = np.where(C_final3 > SURVIVAL_THRESHOLD)[0]

    survivors1_count = len(survivors1_t)
    survivors2_count = len(survivors2_t)
    survivors3_count = len(survivors3_t)

    community_CUE1 = param.safe_weighted_average(species_CUE1[survivors1_t], C_final1[survivors1_t])
    community_CUE2 = param.safe_weighted_average(species_CUE2[survivors2_t], C_final2[survivors2_t])
    community_CUE3 = param.safe_weighted_average(species_CUE3[survivors3_t], C_final3[survivors3_t])

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

    def _feas_proxy(alpha, mask):
        if not np.any(mask):
            return 0.0, np.nan, np.nan, "no_survivors"
        A_surv = alpha[np.ix_(mask, mask)]
        reg = adaptive_ridge(A_surv)
        if reg is None:
            return 0.0, np.nan, np.nan, "ridge_failed"
        proxies = compute_proxies_from_A(reg["A_reg"])
        mapped = map_log_volume_to_feasible_proxies(proxies["log_volume"], int(np.sum(mask)))
        return (float(mapped["feasible_volume_proxy"]),
                float(mapped["log10_feasible_volume_proxy"]),
                float(mapped["feasible_scale_per_dim"]),
                "ok")

    feas1_val, log10_feas1, feas1_scale, feas1_status = _feas_proxy(alpha1, mask1)
    feas2_val, log10_feas2, feas2_scale, feas2_status = _feas_proxy(alpha2, mask2)
    feas3_val, log10_feas3, feas3_scale, feas3_status = _feas_proxy(alpha3, mask3)
    J3 = param.MiCRM_jac(N3, M3, u3, l3, MAINTENANCE_COST, rho3, omega3, lambda_vec3, sol3)
    ev3 = param.leading_eigenvalue(J3)


    species_data = []

    for i in range(N1):
        species_data.append({
            "Seed": seed,
            "Community": 1,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE1[i],
            "Community_CUE": community_CUE1,
            "Abundance": C_final1[i],
            "Total_Abundance": total_abundance1,
            "Dominant_Community": dominant,
            "Competition": competition_comm1,
            "Species_Competition": competition_species1[i],
            "Species_Competition_Dot": competition_dot1[i],
            "Facilitation": facilitation1[i],
            "Depletion": depletion1,
            "N_Survivors": survivors1_count,
            "feasibility": feas1_val,
            "log10_feasibility": log10_feas1,
            "feasible_scale_per_dim": feas1_scale,
            "feasibility_status": feas1_status,
            "Leading_Eigenvalue": float(ev1)
        })

    for i in range(N2):
        species_data.append({
            "Seed": seed,
            "Community": 2,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE2[i],
            "Community_CUE": community_CUE2,
            "Abundance": C_final2[i],
            "Total_Abundance": total_abundance2,
            "Dominant_Community": dominant,
            "Competition": competition_comm2,
            "Species_Competition": competition_species2[i],
            "Species_Competition_Dot": competition_dot2[i],
            "Facilitation": facilitation2[i],
            "Depletion": depletion2,
            "UptakeVar": uptake_var2[i],
            "N_Survivors": survivors2_count,
            "feasibility": feas2_val,
            "log10_feasibility": log10_feas2,
            "feasible_scale_per_dim": feas2_scale,
            "feasibility_status": feas2_status,
            "Leading_Eigenvalue": float(ev2)
        })

    for i in range(N3):
        species_data.append({
            "Seed": seed,
            "Community": 3,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE3[i],
            "Community_CUE": community_CUE3,
            "Abundance": C_final3[i],
            "Total_Abundance": total_abundance3,
            "Dominant_Community": dominant,
            "Competition": competition_comm3,
            "Species_Competition": competition_species3[i],
            "Species_Competition_Dot": competition_dot3[i],
            "Facilitation": facilitation3[i],
            "Depletion": depletion3,
            "UptakeVar": uptake_var3[i],
            "N_Survivors": survivors3_count,
            "feasibility": feasible_volume_proxy,
            "log10_feasibility": log10_feas3,
            "feasible_scale_per_dim": feas3_scale,
            "feasibility_status": feas3_status,
            "Leading_Eigenvalue": float(ev3)
        })

    return species_data


def main():
    seed_generator = np.random.default_rng(BASE_SEED)
    seeds = seed_generator.integers(0, 2**32 - 1, size=N_SIMULATIONS, dtype=np.uint32).tolist()

    with Pool(cpu_count()) as pool:
        all_species_data_nested = pool.map(simulate, seeds)

    all_species_data = [
        row
        for one_seed_result in all_species_data_nested
        if one_seed_result
        for row in one_seed_result
    ]

    df = pd.DataFrame(all_species_data)
    df.to_csv(COAL_FILE, index=False)


if __name__ == "__main__":
    main()