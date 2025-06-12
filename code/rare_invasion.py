import numpy as np
import os
import sys
sys.path.append(os.path.expanduser("~/Documents/MiCRM/code"))
import param
import CUE
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Parameter settings
np.random.seed(37)
N_pool = 1000  # Species pool size
M_pool = 200    # Resource pool size
λ = 0.2        # Total leakage rate
N_modules = 1  # Number of modules
s_ratio = 1 # Modularity ratio
N1 = 100
M1 = 50
m1 = np.full(N1, 0.2)
N2 = 100
M2 = 50
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

# Time span for simulation
t_span = (0, 600)
t_eval = np.linspace(*t_span, 300)

# Simulate Community 1
C0_1 = np.full(N1, 0.01)  # Initial consumer abundance
C0_2 = np.full(N1, 0.01) 
#R0 = np.full(M1, 1)        # Initial resource abundance
R0_1 = np.full(M1, 1)
R0_2 = np.full(M2, 1)
sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1)
ce1 = sol1.y[:N1, -1]  # Consumer abundance at equilibrium
re1 = sol1.y[N1:, -1]  # Resource abundance at equilibrium

# Simulate Community 2
sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2,  R0_2)
ce2 = sol2.y[:N2, -1]
re2 = sol2.y[N2:, -1]

# Merge into Community 3
species_indices3 = np.concatenate([species_indices1, species_indices2])
resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
u3 = u_pool[np.ix_(species_indices3, resource_indices3)]

l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
m3 = np.concatenate([m1, m2])
lambda_alpha3 = np.full(len(resource_indices3), λ)

omega3 = omega_pool[resource_indices3]
N3 = N1 + N2
M3 = len(resource_indices3)
rho3 = rho_pool[resource_indices3]
C0_3 = np.concatenate([ce1, ce2*0.001])
R0_3 = re1 + re2

sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3)
#############################################
# Plot biomass change over time
fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
# Plot for Community 1
cmap1 = plt.get_cmap("Blues")
for i, idx in enumerate(species_indices1):
    axes[0].plot(sol1.t, sol1.y[i], color=cmap1((i + 1) / (N1 + 1)), label=f"c{idx}")
axes[0].set_title('Community 1 Dynamics')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Consumer Abundance')
axes[0].grid(True)
axes[0].legend(loc='upper right', fontsize='small')
# Plot for Community 2
cmap2 = plt.get_cmap("Reds")
for i, idx in enumerate(species_indices2):
    axes[1].plot(sol2.t, sol2.y[i], color=cmap2((i + 1) / (N2 + 1)), label=f"c{idx}")
axes[1].set_title('Community 2 Dynamics ')
axes[1].set_xlabel('Time')
axes[1].grid(True)
axes[1].legend(loc='upper right', fontsize='small')
# Plot for Community 3 (merged)
for i, idx in enumerate(species_indices1):
    axes[2].plot(sol3.t, sol3.y[i], color=cmap1((i + 1) / (N1 + 1)), label=f"c{idx}")
for i, idx in enumerate(species_indices2):
    axes[2].plot(sol3.t, sol3.y[N1 + i], color=cmap2((i + 1) / (N2 + 1)), label=f"c{idx}")
axes[2].set_title('Coalescence Dynamics')
axes[2].set_xlabel('Time')
axes[2].grid(True)
axes[2].legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # leaves space on the right for legends
plt.show()
###### compare the community CUE between survival and extinction######
sol_list = [sol1, sol2, sol3]
N_list = [N1, N2, N3]
u_list = [u1, u2, u3]
R0_list = [R0_1, R0_2, R0_3]

l_list = [l1, l2, l3]
m_list = [m1, m2, m3]
M_list = [M1, M2, M3]
num_communities = len(sol_list)

data_to_save = []
for i in range(num_communities):
    C_final = np.array(sol_list[i].y[:, -1])
    _, species_CUE = CUE.compute_CUE(
        sol_list[i], N_list[i], u_list[i], R0_list[i], l_list[i], m_list[i]
    )
    for j in range(N_list[i]):
        origin = (
            "Resident" if (i == 2 and j < N1) else
            "Invader" if (i == 2 and j >= N1) else
            f"Community {i+1}"
        )
        status = "Survival" if C_final[j] >= 1e-5 else "Extinction"
        data_to_save.append([
            i + 1, f"Sp{j+1}", origin, status, float(species_CUE[j]), float(C_final[j])
        ])

df_out = pd.DataFrame(data_to_save, columns=["Community", "Species", "Origin", "Status", "CUE", "C_final"])

# Analysis
merged = df_out[df_out["Community"] == 3]
invaders = merged[merged["Origin"] == "Invader"]
invader_survival_rate = (invaders["Status"] == "Survival").mean()
print(f"Invader survival rate: {invader_survival_rate:.2%} ({invaders['Status'].value_counts().to_dict()})")

survivors = invaders[invaders["Status"] == "Survival"]
extinct = invaders[invaders["Status"] == "Extinction"]

summary = merged.groupby(["Origin", "Status"]).size().unstack(fill_value=0)
print(summary)

from scipy.stats import mannwhitneyu
if len(survivors) > 0 and len(extinct) > 0:
    stat, p = mannwhitneyu(survivors["CUE"], extinct["CUE"], alternative="greater")
    print(f"Mann-Whitney U test p-value: {p:.3g}")
    plt.boxplot([survivors["CUE"], extinct["CUE"]], labels=["Survivors", "Extinct"])
    plt.title("CUE of Invader Species (Rare Invasion)")
    plt.ylabel("CUE")
    plt.show()
else:
    print("Not enough survivors or extinct invaders for statistical test or plot.")