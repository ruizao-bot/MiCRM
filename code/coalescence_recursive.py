import numpy as np
import os
import sys
sys.path.append(os.path.expanduser("~/Documents/MiCRM/code"))
import param
import CUE
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd

# Storage for results
results = []
# Load the seed list
with open('seeds.txt', 'r') as f:
    seeds = [int(line.strip()) for line in f]
# Loop over the 50 seeds
for seed in seeds:
    np.random.seed(seed)
    N_pool = 1000  # Species pool size
    M_pool = 20     # Resource pool size
    λ = 0.2        # Total leakage rate
    N_modules = 1  # Number of modules
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


    # Function for ODE system
    def dCdt_Rdt(t, y, u, l, N, M, m, rho, omega):
        C, R = y[:N], y[N:]
        dCdt, dRdt = np.zeros(N), np.zeros(M)

        for i in range(N):
            dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - λ) for alpha in range(M)) - C[i] * m[i]

        for alpha in range(M):
            dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]
            dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))
            dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l[i, beta, alpha] for beta in range(M)) for i in range(N))

        return np.concatenate([dCdt, dRdt])

    # Solve ODE for each community
    t_span, t_eval = (0, 500), np.linspace(0, 500, 300)
    C0, R0 = np.full(N1, 0.01), np.full(M1, 1)

    sol1 = solve_ivp(dCdt_Rdt, t_span, np.concatenate([C0, R0]), t_eval=t_eval, args=(u1, l1, N1, M1, m1, rho1, omega1))
    sol2 = solve_ivp(dCdt_Rdt, t_span, np.concatenate([C0, R0]), t_eval=t_eval, args=(u2, l2, N2, M2, m2, rho2, omega2))

    
    # Merge into Community 3
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    uu = u_pool[np.ix_(species_indices3, resource_indices3)]
    ll = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    m3 = np.concatenate([m1, m2])
    lambda_alpha3 = np.full(len(resource_indices3), λ)
    
    omega3 = omega_pool[resource_indices3]
    N3 = N1 + N2
    M3 = len(resource_indices3)
    rho3 = rho_pool[resource_indices3]
    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])
    R0_3 = sol1.y[N1:, -1]  + sol2.y[N2:, -1]
    Y0_3 = np.concatenate([C0_3, R0_3])

    sol3 = solve_ivp(dCdt_Rdt, t_span, np.concatenate([C0_3, R0_3]), t_eval=t_eval, args=(uu, ll, N3, M3, m3, rho3, omega3))
    sol_list = [sol1, sol2, sol3]
    N_list = [N1, N2, N3]
    # Compute Community CUE for each community
    community_CUE1, species_CUE1 = CUE.compute_community_CUE2(sol1, N1, u1, R0, l1, m1)
    community_CUE2, species_CUE2 = CUE.compute_community_CUE2(sol2, N2, u2, R0, l2, m2)
    community_CUE3, species_CUE3 = CUE.compute_community_CUE2(sol3, N3, uu, R0_3, ll, m3)
    # 
    # Calculate relative abundance
    C_final1 = sol1.y[:N1, -1] 
    C_final2 = sol2.y[:N2, -1] 
    C_final3 = sol3.y[:N3, -1] 
    C_final_list = [sol1.y[:N1, -1], sol2.y[:N2, -1], sol3.y[:N3, -1]]
    rel_abundances = [C / np.sum(C) for C in C_final_list]
    # Compute dominance in Community 3
    total_community1, total_community2 = np.sum(C_final3[:N_list[0]]), np.sum(C_final3[N_list[0]:])
    dominant = "Community 1" if total_community1 > total_community2 else "Community 2"
    
    rel_abundance1 = C_final1 / np.sum(C_final1)
    rel_abundance2 = C_final2 / np.sum(C_final2)
    rel_abundance3 = C_final3 / np.sum(C_final3)
    result_entry = {
            "Seed": seed,
            "Community CUE 1": community_CUE1,
            "Community CUE 2": community_CUE2,
            "Community CUE 3": community_CUE3,
            "Total Abundance 1": total_community1,
            "Total Abundance 2": total_community2,
            "Dominant Community": dominant,
        }
    # Community 1
    for i, (val, cue_val, c_val) in enumerate(zip(rel_abundance1, species_CUE1, C_final1)):
        result_entry[f"RelAbun_Comm1_Sp{i+1}"] = val
        result_entry[f"CUE_Comm1_Sp{i+1}"] = cue_val
        result_entry[f"Cfinal_Comm1_Sp{i+1}"] = c_val

    # Community 2
    for i, (val, cue_val, c_val) in enumerate(zip(rel_abundance2, species_CUE2, C_final2)):
        result_entry[f"RelAbun_Comm2_Sp{i+1}"] = val
        result_entry[f"CUE_Comm2_Sp{i+1}"] = cue_val
        result_entry[f"Cfinal_Comm2_Sp{i+1}"] = c_val

    # Community 3
    for i, (val, cue_val, c_val) in enumerate(zip(rel_abundance3, species_CUE3, C_final3)):
        result_entry[f"RelAbun_Comm3_Sp{i+1}"] = val
        result_entry[f"CUE_Comm3_Sp{i+1}"] = cue_val
        result_entry[f"Cfinal_Comm3_Sp{i+1}"] = c_val


    # Store results
    results.append(result_entry)
# Convert results to DataFrame
df_results = pd.DataFrame(results)
df_results.to_csv("data/coal_recursive.csv", index=False)

# Assign Dominance values
df_results["Dominance Community 1"] = df_results["Dominant Community"].apply(lambda x: 1 if x == "Community 1" else 0)
df_results["Dominance Community 2"] = df_results["Dominant Community"].apply(lambda x: 1 if x == "Community 2" else 0)

# Logistics regression and scatter plot for Community CUE vs. Dominance
from sklearn.linear_model import LinearRegression
# Merge df_results into a unified format
df_c1 = df_results[["Community CUE 1", "Dominance Community 1"]].rename(
    columns={"Community CUE 1": "CUE", "Dominance Community 1": "Dominance"}
)
df_c1["Group"] = "Community 1"

# Merge "Community CUE 2" and "Dominance Community 2" into a unified format
df_c2 = df_results[["Community CUE 2", "Dominance Community 2"]].rename(
    columns={"Community CUE 2": "CUE", "Dominance Community 2": "Dominance"}
)
df_c2["Group"] = "Community 2"
# Concatenate both DataFrames
df_combined = pd.concat([df_c1, df_c2], ignore_index=True)
df_combined.to_csv("data/df_combined.csv", index=False)

# # fit model
# import statsmodels.api as sm
# X = sm.add_constant(df_combined["CUE"])
# y = df_combined["Dominance"]


# model = sm.Logit(y, X).fit()
# cue_seq = np.linspace(df_combined["CUE"].min(), df_combined["CUE"].max(), 100)
# X_pred = sm.add_constant(pd.DataFrame({"CUE": cue_seq}))
# predicted = model.predict(X_pred)
# colors = ["blue" if g =="Community 1" else "darkred" for g in df_combined["Group"]]
# plt.scatter(df_combined["CUE"], df_combined["Dominance"], c = colors, alpha = 0.6)
# plt.scatter([], [], color = "blue", alpha = 0.6, label = "Community 1")
# plt.scatter([], [], color = "darkred", alpha = 0.6, label = "Community 2")
# plt.plot(cue_seq, predicted, color = "black", linewidth = 2, label = "Logistic Regression")
# plt.xlabel("CUE vALUE")
# plt.ylabel("Probability of Dominance(1 = Dominant)")
# plt.title("Logistic Regression")
# plt.legend()
# plt.show()