from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import os, sys
from scipy.integrate import solve_ivp

code_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(code_path)

# Project root and data directory (absolute paths)
project_root = os.path.abspath(os.path.join(code_path, os.pardir))
data_dir = os.path.join(project_root, "data")

def modular_uptake(N, M, N_modules, s_ratio, rng):
    """生成模块化资源摄取矩阵 u。"""
    assert N_modules <= M and N_modules <= N, "N_modules must be less than or equal to both M and N"

    sR = M // N_modules
    dR = M - (N_modules * sR)

    sC = N // N_modules
    dC = N - (N_modules * sC)

    diffR = np.full(N_modules, sR, dtype=int)
    if dR > 0:
        diffR[rng.choice(N_modules, dR, replace=False)] += 1
    mR = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffR) - diffR + 1), np.cumsum(diffR))]

    diffC = np.full(N_modules, sC, dtype=int)
    if dC > 0:
        diffC[rng.choice(N_modules, dC, replace=False)] += 1
    mC = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffC) - diffC + 1), np.cumsum(diffC))]

    u = rng.random((N, M))

    for x, y in zip(mC, mR):
        u[np.ix_(x, y)] *= s_ratio

    row_sums = np.sum(u, axis=1, keepdims=True)
    u = u / row_sums
    return u


def modular_leakage(M, N_modules, s_ratio, lam, rng):
    assert N_modules <= M, "N_modules must be less than or equal to M"
    sR = M // N_modules
    dR = M - (N_modules * sR)

    diffR = np.full(N_modules, sR, dtype=int)
    if dR > 0:
        diffR[rng.choice(N_modules, dR, replace=False)] += 1
    mR = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffR) - diffR + 1), np.cumsum(diffR))]

    l = rng.random((M, M))

    for i, x in enumerate(mR):
        for j, y in enumerate(mR):
            if i == j or i + 1 == j:
                l[np.ix_(x, y)] *= s_ratio

    row_sums = np.sum(l, axis=1, keepdims=True)
    l = lam * l / row_sums
    return l


def generate_l_tensor(N, M, N_modules, s_ratio, lam, u, rng):
    l_tensor = np.zeros((N, M, M))
    for i in range(N):
        l_tensor[i] = modular_leakage(M, N_modules, s_ratio, lam, rng)
    return l_tensor


def safe_weighted_average(values, weights):
    """安全计算加权平均，避免分母为 0。"""
    total_weight = np.sum(weights)
    if total_weight <= 0:
        return np.nan
    return np.sum(values * weights) / total_weight


def compute_species_CUE(u, R_ref, lam, m):
    """在给定资源状态 R_ref 下，计算每个物种的瞬时净增长效率指标（species CUE proxy）。"""
    total_uptake = np.sum(u * R_ref, axis=1)
    net_uptake = np.sum(u * R_ref * (1 - lam), axis=1) - m
    species_CUE = net_uptake / (total_uptake + 1e-12)
    return species_CUE


def compute_community_CUE(species_CUE, C_ref):
    """用同一时刻的消费者丰度 C_ref 对 species_CUE 做加权平均，得到瞬时群落 CUE。"""
    return safe_weighted_average(species_CUE, C_ref)


def solve_micrm(
    N, M, u, l, m, lambda_alpha, rho, omega, C0, R0,
    t_span, t_eval=None, tol=1e-5, method='BDF'
):
    """
    Integrate the MiCRM ODEs until equilibrium or t_span is reached.

    Parameters:
        N, M: int
            Number of consumers and resources.
        u, l, m, lambda_alpha, rho, omega: model parameters.
        C0, R0: initial conditions for consumers and resources.
        t_span: tuple
            Time span for integration (default: (0, 1000)).
        t_eval: array or None
            Time points to evaluate solution (default: 300 points in t_span).
        tol: float
            Tolerance for equilibrium detection.
        method: str
            Integration method for solve_ivp.

    Returns:
        sol: OdeResult
            Solution object from scipy.integrate.solve_ivp.
    """
    def dCdt_Rdt(t, y):
        C = y[:N]
        R = y[N:]
        uptake = u * (R * (1 - lambda_alpha))  # (N, M)
        dCdt = C * (np.sum(uptake, axis=1) - m)
        dRdt = rho - omega * R
        consumption = np.sum(C[:, None] * u * R, axis=0)  # (M,)
        dRdt -= consumption
        leakage = np.einsum('i,j,ij,ijk->k', C, R, u, l)
        dRdt += leakage
        return np.concatenate([dCdt, dRdt])

    def equilibrium_event(t, y):
        deriv = dCdt_Rdt(t, y)
        return np.max(np.abs(deriv)) - tol
    equilibrium_event.terminal = True
    equilibrium_event.direction = -1

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 100)
    Y0 = np.concatenate([C0, R0])

    sol = solve_ivp(
        dCdt_Rdt, t_span, Y0, t_eval=t_eval, method=method,
        events=equilibrium_event
    )
    return sol


def extract_state_at_target_time(sol, N, omega):
    """提取 t = 2 / omega 时刻的消费者丰度与资源浓度。"""
    t_eff = 2
    t_target = t_eff / np.mean(omega)
    idx = np.argmin(np.abs(sol.t - t_target))
    C_at_t = sol.y[:N, idx]
    R_at_t = sol.y[N:, idx]
    return C_at_t, R_at_t


def calculate_effective_leakage(u, l):
    """
    Calculate effective leakage vector for each consumer.
    
    L_i^eff = Σ_{α=1}^{M} u_i_α * l_i_α
    
    Parameters:
    u: uptake matrix (N x M)
    l: leakage tensor (N x M x M)
    
    Returns:
    L_eff: effective leakage matrix (N x M)
    """
    N, M = u.shape
    L_eff = np.zeros((N, M))
    
    for i in range(N):
        for alpha in range(M):
            # L_i^eff = Σ_{α=1}^{M} u_i_α * l_i_α
            L_eff[i] += u[i, alpha] * l[i, alpha, :]
    
    return L_eff


def community_level_competition(u):
    """
    Compute community-level average competition based on cosine similarity.
    
    Average cosine similarity across all pairs of species:
        C_comp = 2/(N(N-1)) * Σ_{1≤i<j≤N} cos_sim(u_i, u_j)
    where cos_sim(u_i, u_j) = (u_i · u_j) / (||u_i|| ||u_j||)
    
    Returns:
        scalar: average cosine similarity across all species pairs
    """
    N, M = u.shape
    if N < 2:
        return np.nan
    
    # Normalize each species vector
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_normalized = u / (norms + 1e-10)
    
    # Compute cosine similarity matrix
    similarity = u_normalized @ u_normalized.T
    
    # Sum over upper triangle (i < j) and multiply by 2
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            total += similarity[i, j]
    
    return 2 * total / (N * (N - 1))


def species_level_competition(u):
    """
    Compute species-level competition based on cosine similarity of uptake vectors.

    For each species i, compute average cosine similarity with all other species:
        comp_i = (1/(N-1)) * Σ_{j≠i} cos_sim(u_i, u_j)
    where cos_sim(u_i, u_j) = (u_i · u_j) / (||u_i|| ||u_j||)

    Returns:
        comp: (N,) array with species-level competition values.
    """
    N, M = u.shape
    if N < 2:
        return np.full(N, np.nan)

    # Normalize each species vector
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_normalized = u / (norms + 1e-10)  # Add small value to avoid division by zero
    
    # Compute cosine similarity matrix
    similarity = u_normalized @ u_normalized.T
    np.fill_diagonal(similarity, 0.0)
    
    # Average similarity with all other species
    comp = np.sum(similarity, axis=1) / (N - 1)
    return comp


def species_level_competition_dot(u):
    """
    Compute species-level competition based on the dot product (CCT).
    For each species i, compute the average dot product with all other species:
        comp_i = (1/(N-1)) * sum_{j!=i} dot(C_i, C_j)
    where dot(C_i, C_j) = sum_k C[i, k] * C[j, k]

    Args:
        u: (N, M) array, uptake matrix for N species and M resources.

    Returns:
        comp: (N,) array, average competition pressure for each species.
    """
    N, M = u.shape
    if N < 2:
        return np.full(N, np.nan)

    comp_matrix = u @ u.T
    np.fill_diagonal(comp_matrix, 0.0)
    comp = np.sum(comp_matrix, axis=1) / (N - 1)
    return comp


def compute_uptake_variance(u):
    """计算每个物种摄取向量的方差。"""
    return np.var(u, axis=1)

def simulate(seed):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    
    N_pool, M_pool, λ, N_modules, s_ratio = 1000, 100, 0.2, 1, 1.0
    N1, M1, N2, M2 = 100, 50, 100, 50

    u_pool = modular_uptake(N_pool, M_pool, N_modules, s_ratio, rng)
    l_pool = generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ, u_pool, rng)
    t_span = (0, 100000)  # Simulation time span
    species_indices1 = np.random.choice(N_pool, N1, replace=False)
    resource_indices1 = np.random.choice(M_pool, M1, replace=False)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
    m1 = 0.2
    lambda_alpha1 = np.full(M1, λ)
    rho1, omega1 = np.full(M1, 0.6), np.full(M1, 0.1)

    if M1 > M2:
        resource_indices2 = np.random.choice(resource_indices1, M2, replace=False)
    elif M1 < M2:
        remaining_resources = np.setdiff1d(np.arange(M_pool), resource_indices1)
        additional_resources = np.random.choice(remaining_resources, M2 - M1, replace=False)
        resource_indices2 = np.concatenate([resource_indices1, additional_resources])
    else:
        resource_indices2 = resource_indices1.copy()
    remaining_species = np.setdiff1d(np.arange(N_pool), species_indices1)
    species_indices2 = np.random.choice(remaining_species, N2, replace=False)
    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]

    m2 = 0.2
    lambda_alpha2 = np.full(M2, λ)
    rho2, omega2 = np.full(M2, 0.6), np.full(M2, 0.1)
    C0_1 = np.full(N1, 0.01) 
    C0_2 = np.full(N2, 0.01) 
    R0_1, R0_2 = np.full(M1, 1.0), np.full(M2, 1.0)
    # R0_1 = np.random.lognormal(mean=0.0, sigma=1.0, size=M1)
    # R0_2 = np.random.lognormal(mean=0.0, sigma=1.0, size=M2)
    sol1 = solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)
    sol2 = solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2, R0_2, t_span)
    
    # Extract resource concentrations at t = 20/omega
    t_target1 = 20.0 / np.mean(omega1)
    idx1 = np.argmin(np.abs(sol1.t - t_target1))
    R_at_t1 = sol1.y[N1:, idx1]
    
    t_target2 = 20.0 / np.mean(omega2)
    idx2 = np.argmin(np.abs(sol2.t - t_target2))
    R_at_t2 = sol2.y[N2:, idx2]
    
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    N3, M3 = N1 + N2, len(resource_indices3)
    omega3 = np.full(M3, 0.1)
    rho3 = np.full(M3, 0.6)

    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])
    R0_3 = np.full(M1, 1)#sol1.y[N1:, -1] + sol2.y[N2:, -1]
    m3 = 0.2
    sol3 = solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)
    
    # Extract resource concentration at t = 20/omega for community 3
    t_target3 = 20.0 / np.mean(omega3)
    idx3 = np.argmin(np.abs(sol3.t - t_target3))
    R_at_t3 = sol3.y[N3:, idx3]
    

    # Calculate facilitation metrics (use L_eff mean per species)
    L_eff1 = calculate_effective_leakage(u1, l1)
    Facilitation1 = np.mean(L_eff1, axis=1)

    L_eff2 = calculate_effective_leakage(u2, l2)
    Facilitation2 = np.mean(L_eff2, axis=1)

    L_eff3 = calculate_effective_leakage(u3, l3)
    Facilitation3 = np.mean(L_eff3, axis=1)

    C_final1 = sol1.y[:N1, -1]
    C_final2 = sol2.y[:N2, -1]
    C_final3 = sol3.y[:N3, -1]

    # Compute CUE using the new functions
    species_CUE1 = compute_species_CUE(u1, R0_1, lambda_alpha1, m1)
    species_CUE2 = compute_species_CUE(u2, R0_2, lambda_alpha2, m2)
    species_CUE3 = compute_species_CUE(u3, R0_3, lambda_alpha3, m3)

    community_CUE1 = compute_community_CUE(species_CUE1, C_final1)
    community_CUE2 = compute_community_CUE(species_CUE2, C_final2)
    community_CUE3 = compute_community_CUE(species_CUE3, C_final3)

    # Get survivors and calculate survivor CUE
    survivors1 = np.where(C_final1 > 1e-5)[0]
    survivors2 = np.where(C_final2 > 1e-5)[0]
    survivors3 = np.where(C_final3 > 1e-5)[0]

    community_CUE1_surv = safe_weighted_average(species_CUE1[survivors1], C_final1[survivors1])
    community_CUE2_surv = safe_weighted_average(species_CUE2[survivors2], C_final2[survivors2])
    community_CUE3_surv = safe_weighted_average(species_CUE3[survivors3], C_final3[survivors3])

    # Calculate competition metrics
    competition1 = community_level_competition(u1)
    competition2 = community_level_competition(u2)
    competition3 = community_level_competition(u3)

    species_competition1 = species_level_competition(u1)
    species_competition2 = species_level_competition(u2)
    species_competition3 = species_level_competition(u3)

    species_competition_dot1 = species_level_competition_dot(u1)
    species_competition_dot2 = species_level_competition_dot(u2)
    species_competition_dot3 = species_level_competition_dot(u3)

    # Calculate uptake variance
    uptake_var1 = compute_uptake_variance(u1)
    uptake_var2 = compute_uptake_variance(u2)
    uptake_var3 = compute_uptake_variance(u3)

    # Calculate resource depletion
    depletion1 = np.sum(sol1.y[N1:, -1])
    depletion2 = np.sum(sol2.y[N2:, -1])
    depletion3 = np.sum(sol3.y[N3:, -1])

    # Calculate total abundances
    total_abundance1 = np.sum(C_final1)
    total_abundance2 = np.sum(C_final2)
    total_abundance3 = np.sum(C_final3)

    origin1_in_coalesced = np.sum(C_final3[:N1])
    origin2_in_coalesced = np.sum(C_final3[N1:])
    dominant = "Community 1" if origin1_in_coalesced > origin2_in_coalesced else "Community 2"

    # Build species data
    species_data = []
    for i in range(len(species_CUE1)):
        species_data.append({
            "Seed": seed,
            "Community": 1,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE1[i],
            "Community_CUE": community_CUE1_surv,
            "Abundance": C_final1[i],
            "Total_Abundance": total_abundance1,
            "Dominant_Community": dominant,
            "Competition": competition1,
            "Species_Competition": species_competition1[i],
            "Species_Competition_Dot": species_competition_dot1[i],
            "Facilitation": Facilitation1[i],
            "Depletion": depletion1,
            "UptakeVar": uptake_var1[i]
        })

    for i in range(len(species_CUE2)):
        species_data.append({
            "Seed": seed,
            "Community": 2,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE2[i],
            "Community_CUE": community_CUE2_surv,
            "Abundance": C_final2[i],
            "Total_Abundance": total_abundance2,
            "Dominant_Community": dominant,
            "Competition": competition2,
            "Species_Competition": species_competition2[i],
            "Species_Competition_Dot": species_competition_dot2[i],
            "Facilitation": Facilitation2[i],
            "Depletion": depletion2,
            "UptakeVar": uptake_var2[i]
        })

    for i in range(len(species_CUE3)):
        species_data.append({
            "Seed": seed,
            "Community": 3,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE3[i],
            "Community_CUE": community_CUE3_surv,
            "Abundance": C_final3[i],
            "Total_Abundance": total_abundance3,
            "Dominant_Community": dominant,
            "Competition": competition3,
            "Species_Competition": species_competition3[i],
            "Species_Competition_Dot": species_competition_dot3[i],
            "Facilitation": Facilitation3[i],
            "Depletion": depletion3,
            "UptakeVar": uptake_var3[i]
        })

    return species_data

if __name__ == "__main__":
    seeds_file = os.path.join(code_path, 'seeds.txt')
    with open(seeds_file, 'r') as f:
        seeds = [int(line.strip()) for line in f]

    with Pool(cpu_count()) as pool:
        all_species_data_nested = pool.map(simulate, seeds)
    all_species_data = [row for seed_result in all_species_data_nested if seed_result for row in seed_result]
    os.makedirs(data_dir, exist_ok=True)
    df = pd.DataFrame(all_species_data)
    df.to_csv(os.path.join(data_dir, "coal.csv"), index=False)
