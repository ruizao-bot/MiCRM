import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
import os
import sys
sys.path.append(os.path.expanduser("~/Documents/MiCRM/code"))
import param
import CUE
import pandas as pd
from scipy.optimize import minimize

results = []
with open('seeds.txt', 'r') as f:
    seeds = [int(line.strip()) for line in f]

# Loop over the 50 seeds
for seed in seeds:
    np.random.seed(seed)
    N_pool = 1000  # Species pool size
    M_pool = 20     # Resource pool size
    λ = 0.2        # Total leakage rate
    N_modules = 5  # Number of modules
    s_ratio = 10.0 # Modularity ratio
    N1 = 10
    M1 = 5
    m1 = np.full(N1, 0.2)  # maintaining cost rate
    N2 = 10
    M2 = 5
    m2 = np.full(N2, 0.2)
    # Generate uptake matrix and leakage tensor for the species pool
    u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
    l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ)                                                                                                                                                           
    # Set rho and omega for the resource pool
    rho_pool = np.full(M_pool, 0.6)
    omega_pool = np.full(M_pool, 0.1)
    # Community 1
    species_indices1 = np.random.choice(N_pool, N1, replace=False)
    resource_indices1 = np.random.choice(M_pool, M1, replace=False)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
    lambda_alpha1 = np.full(M1, λ)
    rho1 = rho_pool[resource_indices1]
    omega1 = omega_pool[resource_indices1]
    # Community 2
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
    lambda_alpha2 = np.full(M2, λ)
    rho2 = rho_pool[resource_indices2]
    omega2 = omega_pool[resource_indices2]
    # Define the steady-state equation
    def steady_state_eq(y, n_species, m_resources, u, l, m, lambda_alpha, rho, omega):
        C = y[:n_species]
        R = y[n_species:]
        eq_C = C * (np.sum(R * u * (1 - lambda_alpha), axis=1) - m)
        eq_R = rho - omega * R - np.sum(C[:, np.newaxis] * u * R, axis=0)
        eq_R += np.einsum('i,ib,iba->a', C, u * R, l)
        return np.concatenate([eq_C, eq_R])
    # Solve for the steady state of Community 1
    def objective(y):
        C = y[:N1]
        R = y[N1:]
        eq = steady_state_eq(y, N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1)
        return np.sum(eq**2)  

    # Constrain：C_i >= 0, R_alpha >= 0
    bounds = [(0, None)] * (N1 + M1) 
    R_guess1 = np.full(M1, 1.0)
    C_guess1 = np.full(N1, 0.1)
    y_guess1 = np.concatenate([C_guess1,R_guess1])
    sol_steady1 = minimize(objective, y_guess1, bounds=bounds, method='L-BFGS-B')

    if sol_steady1.success:
        C_hat1 = sol_steady1.x[:N1]
        R_hat1 = sol_steady1.x[N1:]
    else:
        raise ValueError("nan")
    # D for Community 1
    D1 = np.diag(omega1 + np.sum(C_hat1[:, np.newaxis] * u1, axis=0))
    D1 -= np.einsum('i,ig,iag->ag', C_hat1, u1 * R_hat1, l1)
    # Calculate ∂R_α/∂C_j
    partial_R_C1 = np.zeros((M1, N1))
    for j in range(N1):
        v_j = -R_hat1 * u1[j] + np.einsum('b,b,ba->a', R_hat1, u1[j], l1[j])
        partial_R_C1[:, j] = np.linalg.solve(D1, v_j)
    # Calculate α_ij and r_i
    alpha1 = np.einsum('ia,a,aj->ij', u1, 1 - lambda_alpha1, partial_R_C1)
    r1 = np.sum(u1 * (1 - lambda_alpha1) * R_hat1, axis=1) - m1 - np.sum(alpha1 * C_hat1, axis=1)
    # Define the eLV model for Community 1
    def dCdt_elv1(t, C):
        return C * (r1 + alpha1 @ C)
    # Solve the dynamics of Community 1
    C0_1 = np.full(N1, 0.01)
    t_span = (0, 500)
    t_eval = np.linspace(*t_span, 300)
    sol1 = solve_ivp(dCdt_elv1, t_span, C0_1, t_eval=t_eval)
    def objective(y):
        C = y[:N2]
        R = y[N2:]
        eq = steady_state_eq(y, N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2)
        return np.sum(eq**2)  

    # constrains
    bounds = [(0, None)] * (N2 + M2)
    R_guess2 = np.full(M2, 1.0)
    C_guess2 = np.full(N2, 0.1)
    y_guess2 = np.concatenate([C_guess2, R_guess2])
    sol_steady2 = minimize(objective, y_guess2, bounds=bounds, method='L-BFGS-B')

    if sol_steady2.success:
        C_hat2 = sol_steady2.x[:N2]
        R_hat2 = sol_steady2.x[N2:]
    else:
        raise ValueError("nan")
    # D for Community 2
    D2 = np.diag(omega2 + np.sum(C_hat2[:, np.newaxis] * u2, axis=0))
    D2 -= np.einsum('i,ig,iag->ag', C_hat2, u2 * R_hat2, l2)
    # Calculate ∂R_α/∂C_j
    partial_R_C2 = np.zeros((M2, N2))
    for j in range(N2):
        v_j = -R_hat2 * u2[j] + np.einsum('b,b,ba->a', R_hat2, u2[j], l2[j])
        partial_R_C2[:, j] = np.linalg.solve(D2, v_j)
    # Calculate α_ij and r_i
    alpha2 = np.einsum('ia,a,aj->ij', u2, 1 - lambda_alpha2, partial_R_C2)
    r2 = np.sum(u2 * (1 - lambda_alpha2) * R_hat2, axis=1) - m2 - np.sum(alpha2 * C_hat2, axis=1)
    # Define the eLV model for Community 2
    def dCdt_elv2(t, C):
        return C * (r2 + alpha2 @ C)
    # Solve the dynamics of Community 2
    C0_2 = np.full(N2, 0.01)
    sol2 = solve_ivp(dCdt_elv2, t_span, C0_2, t_eval=t_eval)
    # Merge into Community 3
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    m3 = np.concatenate([m1, m2])
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    rho3 = rho_pool[resource_indices3]
    omega3 = omega_pool[resource_indices3]
    N3 = N1 + N2
    M3 = len(resource_indices3)
    # initial conditions for Community 3
    C0_3 = np.zeros(N1 + N2)
    C0_3[:N1] = sol1.y[:, -1]
    C0_3[N1:] = sol2.y[:, -1]
    def objective(y):
        C = y[:N3]
        R = y[N3:]
        eq = steady_state_eq(y, N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3)
        return np.sum(eq**2)  

    # constrains
    bounds = [(0, None)] * (N3 + M3)
    R_guess3 = np.full(M3, 1.0)
    C_guess3 = np.full(N3, 0.1)
    y_guess3 = np.concatenate([C_guess3, R_guess3])
    sol_steady3 = minimize(objective, y_guess3, bounds=bounds, method='L-BFGS-B')

    if sol_steady3.success:
        C_hat3 = sol_steady3.x[:N3]
        R_hat3 = sol_steady3.x[N3:]
    else:
        raise ValueError("nan")
    if not sol_steady3.success:
        print("Warning: Failed to find steady-state solution for Community 3")
        C_hat3 = np.ones(N3) * 0.1
        R_hat3 = np.ones(M3) * 0.5
    else:
        C_hat3 = sol_steady3.x[:N3]
        R_hat3 = sol_steady3.x[N3:]
    # D3
    D3 = np.zeros((M3, M3))
    for a in range(M3):
        for gamma in range(M3):
            if a == gamma:
               D3[a, a] = omega3[a] + sum(C_hat3[i] * u3[i, a] for i in range(N3))
            else:
                D3[a, gamma] = -sum(C_hat3[i] * u3[i, gamma] * l3[i, gamma, a] for i in range(N3))
    # Define the v_j function
    def compute_v_j3(j):
        v = np.zeros(M3)
        for alpha in range(M3):
            v[alpha] = -R_hat3[alpha] * u3[j, alpha] + sum(R_hat3[beta] * u3[j, beta] * l3[j, beta, alpha] for beta in range(M3))
        return v
    # Calculate ∂R_α/∂C_j
    partial_R_C3 = np.zeros((M3, N3))
    for j in range(N3):
        v_j = compute_v_j3(j)
        partial_R_C3[:, j] = np.linalg.solve(D3, v_j)
    # Calculate α_ij and r_i
    alpha3 = np.zeros((N3, N3))
    for i in range(N3):
        for j in range(N3):
            alpha3[i, j] = sum(u3[i, a] * (1 - lambda_alpha3[a]) * partial_R_C3[a, j] for a in range(M3))
    r3 = np.zeros(N3)
    for i in range(N3):
        growth_term = sum(u3[i, a] * (1 - lambda_alpha3[a]) * R_hat3[a] for a in range(M3))
        interaction_term = sum(alpha3[i, j] * C_hat3[j] for j in range(N3))
        r3[i] = growth_term - m3[i] - interaction_term
    # Define the eLV model for Community 3
    def dCdt_elv3(t, C):
        dCdt = np.zeros(N3)
        for i in range(N3):
            interaction_term = sum(alpha3[i, j] * C[j] for j in range(N3))
            dCdt[i] = C[i] * (r3[i] + interaction_term)
        return dCdt
    # Solve the dynamics of Community 3
    sol3 = solve_ivp(dCdt_elv3, t_span, C0_3, t_eval=t_eval)

    community_CUE1, species_CUE1 = CUE.compute_community_CUE2(sol1, N1, u1, R_guess1, l1, m1)
    community_CUE2, species_CUE2 = CUE.compute_community_CUE2(sol2, N2, u2, R_guess2, l2, m2)
    community_CUE3, species_CUE3 = CUE.compute_community_CUE2(sol3, N3, u3, R_guess3, l3, m3)
    
    result_entry = {"Seed": seed}
    sol_list = [sol1, sol2, sol3] 
    N_list = [N1, N2, N3] 
    u_list = [u1, u2, u3]
    R0_list = [R_guess1, R_guess2, R_guess3]
    l_list = [l1, l2, l3]
    m_list = [m1, m2, m3]
    M_list = [M1, M2, M3]
    r_list = [r1, r2, r3]
    C_final_list = []
    num_communities = len(sol_list) 
    for i in range(num_communities):

        C_final = np.array(sol_list[i].y[:, -1])
        C_final_list.append(C_final)

        _, species_CUE = CUE.compute_community_CUE2(
            sol_list[i], N_list[i], u_list[i], R0_list[i], l_list[i], m_list[i]
        )

        for j, (r_val, cue_val, c_val) in enumerate(zip(r_list[i], species_CUE, C_final)):
            result_entry[f"r_Comm{i+1}_Sp{j+1}"] = r_val
            result_entry[f"CUE_Comm{i+1}_Sp{j+1}"] = cue_val
            result_entry[f"Cfinal_Comm{i+1}_Sp{j+1}"] = c_val

    results.append(result_entry)

df_results = pd.DataFrame(results)
df_results.to_csv("data/elv_recursive.csv", index=False)


# plt.figure(figsize=(6, 4))
# plt.scatter(df_results["Length r1"], df_results["Community CUE 1"], label="CUE 1", color='blue')
# plt.scatter(df_results["Length r2"], df_results["Community CUE 2"], label="CUE 2", color='red')
# plt.scatter(df_results["Length r3"], df_results["Community CUE 3"], label="CUE 3", color='green')

# z1 = np.polyfit(df_results["Length r1"], df_results["Community CUE 1"], 1)
# p1 = np.poly1d(z1)
# plt.plot(df_results["Length r1"], p1(df_results["Length r1"]), "b--")

# z2 = np.polyfit(df_results["Length r2"], df_results["Community CUE 2"], 1)
# p2 = np.poly1d(z2)
# plt.plot(df_results["Length r2"], p2(df_results["Length r2"]), "r--")

# z3 = np.polyfit(df_results["Length r3"], df_results["Community CUE 3"], 1)
# p3 = np.poly1d(z3)
# plt.plot(df_results["Length r3"], p3(df_results["Length r3"]), "g--")

# plt.xlabel("Length_r")
# plt.ylabel("Community CUE")
# plt.legend()
# plt.title("Length of growth rate vs. CUE")
# plt.show()