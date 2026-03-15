from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# 随机种子设置
BASE_SEED = 4000
N_SIMULATIONS = 20

# 输出文件名
COAL_FILE = "coal.csv"
SUMMARY_FILE = "coal_summary.csv"

# 物种池与资源池参数
N_POOL = 100
M_POOL = 100
N_MODULES = 1
S_RATIO = 1
LEAKAGE_RATE = 0.25

# 群落参数
N1, M1 = 40, 20
N2, M2 = 40, 20

# 动力学参数
MAINTENANCE_COST = 0.2
RHO_VALUE = 5
OMEGA_VALUE = 0.2
T_SPAN = (0, 100000)
t_eff = 2
# 初始条件
C0_VALUE = 1
R0_VALUE = 1

# 存活阈值
SURVIVAL_THRESHOLD = 1e-5


def modular_uptake(N, M, N_modules, s_ratio, rng):
    """生成模块化资源摄取矩阵 u。"""
    assert N_modules <= M and N_modules <= N, "N_modules 必须同时不大于 M 和 N"

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
    """生成模块化泄漏矩阵 l。"""
    assert N_modules <= M, "N_modules 必须不大于 M"

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
    """为每个物种独立生成泄漏矩阵，构成三维泄漏张量。"""
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
    """数值积分 MiCRM，直到达到平衡或到达最大时间。"""
    def dCdt_Rdt(t, y):
        C = y[:N]
        R = y[N:]

        uptake = u * (R * (1 - lambda_alpha))
        dCdt = C * (np.sum(uptake, axis=1) - m)

        dRdt = rho - omega * R
        consumption = np.sum(C[:, None] * u * R, axis=0)
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
        dCdt_Rdt,
        t_span,
        Y0,
        t_eval=t_eval,
        method=method,
        events=equilibrium_event
    )
    return sol


def calculate_effective_leakage(u, l):
    """计算每个物种的有效泄漏向量。"""
    return np.einsum('ia,iab->ib', u, l)


def community_level_competition(u):
    """计算群落层面的平均竞争强度（余弦相似度）。"""
    N, _ = u.shape
    if N < 2:
        return np.nan

    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_normalized = u / (norms + 1e-10)
    similarity = u_normalized @ u_normalized.T

    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            total += similarity[i, j]

    return 2 * total / (N * (N - 1))


def species_level_competition(u):
    """计算物种层面的竞争强度（余弦相似度）。"""
    N, _ = u.shape
    if N < 2:
        return np.full(N, np.nan)

    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_normalized = u / (norms + 1e-10)
    similarity = u_normalized @ u_normalized.T
    np.fill_diagonal(similarity, 0.0)

    comp = np.sum(similarity, axis=1) / (N - 1)
    return comp


def species_level_competition_dot(u):
    """计算物种层面的竞争强度（点积形式）。"""
    N, _ = u.shape
    if N < 2:
        return np.full(N, np.nan)

    comp_matrix = u @ u.T
    np.fill_diagonal(comp_matrix, 0.0)
    comp = np.sum(comp_matrix, axis=1) / (N - 1)
    return comp


def compute_uptake_variance(u):
    """计算每个物种摄取向量的方差。"""
    return np.var(u, axis=1)


def extract_state_at_target_time(sol, N, omega):
    """提取 t = 1 / omega 时刻的消费者丰度与资源浓度。"""
    t_target = t_eff / np.mean(omega)
    idx = np.argmin(np.abs(sol.t - t_target))
    C_at_t = sol.y[:N, idx]
    R_at_t = sol.y[N:, idx]
    return C_at_t, R_at_t


def choose_resources_for_second_community(M_pool, M1, M2, resource_indices1, rng):
    """为第二个群落选择资源索引。"""
    if M1 > M2:
        return rng.choice(resource_indices1, M2, replace=False)
    if M1 < M2:
        remaining_resources = np.setdiff1d(np.arange(M_pool), resource_indices1)
        additional_resources = rng.choice(remaining_resources, M2 - M1, replace=False)
        return np.concatenate([resource_indices1, additional_resources])
    return resource_indices1.copy()


def simulate(seed):
    rng = np.random.default_rng(seed)

    u_pool = modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    species_indices1 = rng.choice(N_POOL, N1, replace=False)
    resource_indices1 = rng.choice(M_POOL, M1, replace=False)

    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]

    lambda_alpha1 = np.full(M1, LEAKAGE_RATE)
    rho1 = np.full(M1, RHO_VALUE)
    omega1 = np.full(M1, OMEGA_VALUE)
    C0_1 = np.full(N1, C0_VALUE)
    R0_1 = np.full(M1, R0_VALUE)

    resource_indices2 = choose_resources_for_second_community(M_POOL, M1, M2, resource_indices1, rng)
    remaining_species = np.setdiff1d(np.arange(N_POOL), species_indices1)
    species_indices2 = rng.choice(remaining_species, N2, replace=False)

    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]

    lambda_alpha2 = np.full(M2, LEAKAGE_RATE)
    rho2 = np.full(M2, RHO_VALUE)
    omega2 = np.full(M2, OMEGA_VALUE)
    C0_2 = np.full(N2, C0_VALUE)
    R0_2 = np.full(M2, R0_VALUE)

    sol1 = solve_micrm(N1, M1, u1, l1, MAINTENANCE_COST, lambda_alpha1, rho1, omega1, C0_1, R0_1, T_SPAN)
    sol2 = solve_micrm(N2, M2, u2, l2, MAINTENANCE_COST, lambda_alpha2, rho2, omega2, C0_2, R0_2, T_SPAN)
    C_final1 = sol1.y[:N1, -1]
    C_final2 = sol2.y[:N2, -1]
    C_at_t1, R_at_t1 = extract_state_at_target_time(sol1, N1, omega1)
    C_at_t2, R_at_t2 = extract_state_at_target_time(sol2, N2, omega2)

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

    sol3 = solve_micrm(N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3, rho3, omega3, C0_3, R0_3, T_SPAN)
    C_at_t3, R_at_t3 = extract_state_at_target_time(sol3, N3, omega3)
    C_final3 = sol3.y[:N3, -1]
    species_CUE1 = compute_species_CUE(u1, R0_1, lambda_alpha1, MAINTENANCE_COST)
    species_CUE2 = compute_species_CUE(u2, R0_2, lambda_alpha2, MAINTENANCE_COST)
    species_CUE3 = compute_species_CUE(u3, R0_3, lambda_alpha3, MAINTENANCE_COST)

    community_CUE1 = compute_community_CUE(species_CUE1, C_final1)
    community_CUE2 = compute_community_CUE(species_CUE2, C_final2)
    community_CUE3 = compute_community_CUE(species_CUE3, C_final3)

    C_final1 = sol1.y[:N1, -1]
    C_final2 = sol2.y[:N2, -1]
    C_final3 = sol3.y[:N3, -1]

    survivors1_t = np.where(C_final1 > SURVIVAL_THRESHOLD)[0]
    survivors2_t = np.where(C_final2 > SURVIVAL_THRESHOLD)[0]
    survivors3_t = np.where(C_final3 > SURVIVAL_THRESHOLD)[0]

    community_CUE1_surv = safe_weighted_average(species_CUE1[survivors1_t], C_final1[survivors1_t])
    community_CUE2_surv = safe_weighted_average(species_CUE2[survivors2_t], C_final2[survivors2_t])
    community_CUE3_surv = safe_weighted_average(species_CUE3[survivors3_t], C_final3[survivors3_t])

    L_eff1 = calculate_effective_leakage(u1, l1)
    L_eff2 = calculate_effective_leakage(u2, l2)
    L_eff3 = calculate_effective_leakage(u3, l3)

    facilitation1 = np.mean(L_eff1, axis=1)
    facilitation2 = np.mean(L_eff2, axis=1)
    facilitation3 = np.mean(L_eff3, axis=1)

    competition_comm1 = community_level_competition(u1)
    competition_comm2 = community_level_competition(u2)
    competition_comm3 = community_level_competition(u3)

    competition_species1 = species_level_competition(u1)
    competition_species2 = species_level_competition(u2)
    competition_species3 = species_level_competition(u3)

    competition_dot1 = species_level_competition_dot(u1)
    competition_dot2 = species_level_competition_dot(u2)
    competition_dot3 = species_level_competition_dot(u3)

    uptake_var1 = compute_uptake_variance(u1)
    uptake_var2 = compute_uptake_variance(u2)
    uptake_var3 = compute_uptake_variance(u3)

    depletion1 = np.sum(sol1.y[N1:, -1])
    depletion2 = np.sum(sol2.y[N2:, -1])
    depletion3 = np.sum(sol3.y[N3:, -1])

    total_abundance1 = np.sum(C_final1)
    total_abundance2 = np.sum(C_final2)
    total_abundance3 = np.sum(C_final3)

    origin1_in_coalesced = np.sum(C_final3[:N1])
    origin2_in_coalesced = np.sum(C_final3[N1:])
    dominant = "Community 1" if origin1_in_coalesced > origin2_in_coalesced else "Community 2"

    species_data = []

    for i in range(N1):
        species_data.append({
            "Seed": seed,
            "Community": 1,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE1[i],
            "Community_CUE": community_CUE1,
            "Community_CUE_surv": community_CUE1_surv,
            "Abundance": C_final1[i],
            "Total_Abundance": total_abundance1,
            "Dominant_Community": dominant,
            "Competition": competition_comm1,
            "Species_Competition": competition_species1[i],
            "Species_Competition_Dot": competition_dot1[i],
            "Facilitation": facilitation1[i],
            "Depletion": depletion1,
            "UptakeVar": uptake_var1[i]
        })

    for i in range(N2):
        species_data.append({
            "Seed": seed,
            "Community": 2,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE2[i],
            "Community_CUE": community_CUE2,
            "Community_CUE_surv": community_CUE2_surv,
            "Abundance": C_final2[i],
            "Total_Abundance": total_abundance2,
            "Dominant_Community": dominant,
            "Competition": competition_comm2,
            "Species_Competition": competition_species2[i],
            "Species_Competition_Dot": competition_dot2[i],
            "Facilitation": facilitation2[i],
            "Depletion": depletion2,
            "UptakeVar": uptake_var2[i]
        })

    for i in range(N3):
        species_data.append({
            "Seed": seed,
            "Community": 3,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE3[i],
            "Community_CUE": community_CUE3,
            "Community_CUE_surv": community_CUE3_surv,
            "Abundance": C_final3[i],
            "Total_Abundance": total_abundance3,
            "Dominant_Community": dominant,
            "Competition": competition_comm3,
            "Species_Competition": competition_species3[i],
            "Species_Competition_Dot": competition_dot3[i],
            "Facilitation": facilitation3[i],
            "Depletion": depletion3,
            "UptakeVar": uptake_var3[i]
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

    summary_df = (
        df.groupby("Community")
        .agg(
            Mean_Community_CUE=("Community_CUE", "mean"),
            Mean_Community_CUE_surv=("Community_CUE_surv", "mean"),
            Mean_Abundance=("Abundance", "mean"),
            Mean_Competition=("Competition", "mean"),
            Mean_Facilitation=("Facilitation", "mean"),
            Mean_Depletion=("Depletion", "mean"),
            Mean_UptakeVar=("UptakeVar", "mean")
        )
        .reset_index()
    )

    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("模拟完成，结果已保存：")
    print(COAL_FILE)
    print(SUMMARY_FILE)
    print("\n各群落汇总结果：")
    print(summary_df)


if __name__ == "__main__":
    main()