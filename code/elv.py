import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
import param
import CUE
import pandas as pd
from scipy.stats import truncnorm

# Parameter settings
np.random.seed(37)
N_pool = 1000  # Species pool size
M_pool = 200    # Resource pool size
λ = 0.2        # Total leakage rate
N_modules = 1  # Number of modules
s_ratio = 1.0 # Modularity ratio
N1 = 100
M1 = 50
m1 = truncnorm.rvs((0 - 1) / 0.01, np.inf, loc=1, scale=0.01, size=N1)  # maintaining cost rate
N2 = 100
M2 = 50
m2 = truncnorm.rvs((0 - 1) / 0.01, np.inf, loc=1, scale=0.01, size=N2)
# Generate uptake matrix and leakage tensor for the species pool
u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ)
# Set rho and omega for the resource pool
rho_pool = np.full(M_pool, 0.5)
omega_pool = np.full(M_pool, 0.5)
# Community 1
species_indices1 = np.random.choice(N_pool, N1, replace=False)
resource_indices1 = np.random.choice(M_pool, M1, replace=False)
u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
u1 = 2.5 * u1 / u1.sum(axis=1, keepdims=True)
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
u2 = 2.5 * u2 / u2.sum(axis=1, keepdims=True)
l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]
lambda_alpha2 = np.full(M2, λ)
rho2 = rho_pool[resource_indices2]
omega2 = omega_pool[resource_indices2]
C0_1 = np.full(N1, 0.01)
C0_2 = np.full(N2, 0.01)
sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1)
sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2)
# 对群落 1：
C_hat1 = sol1.y[:N1, -1]
R_hat1 = sol1.y[N1:, -1]
alpha1, r1 = param.compute_alpha_r(C_hat1, R_hat1, N1, M1,u1, l1, m1, lambda_alpha1, omega1)
sol_elv1 = param.solve_elv(alpha1, r1,C0_1 )

# 对群落 2：
C_hat2 = sol2.y[:N2, -1]
R_hat2 = sol2.y[N2:, -1]
alpha2, r2 = param.compute_alpha_r(C_hat2, R_hat2,N2, M2, u2, l2, m2, lambda_alpha2, omega2)
sol_elv2 = param.solve_elv(alpha2, r2, C0_2)



# Merge into Community 3
species_indices3 = np.concatenate([species_indices1, species_indices2])
resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
N3 = N1 + N2
u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
u3 = 2.5 * u3 / u3.sum(axis=1, keepdims=True)
l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
m3 = np.concatenate([m1, m2])
lambda_alpha3 = np.full(len(resource_indices3), λ)
rho3 = rho_pool[resource_indices3]
omega3 = omega_pool[resource_indices3]
C0_3 = np.zeros(N1 + N2)
C0_3[:N1]   = C_hat1
C0_3[N1:]   = C_hat2


M3 = len(resource_indices3)

sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3)
C_hat3 = sol3.y[:N3, -1]
R_hat3 = sol3.y[N3:, -1]

alpha3, r3 = param.compute_alpha_r(C_hat3, R_hat3, N3, M3, u3, l3, m3, lambda_alpha3, omega3)
sol_elv3 = param.solve_elv(alpha3, r3, C0_3 )
# --------- Plotting ---------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
# Plot for Community 1
cmap1 = plt.get_cmap("Blues")
for i, idx in enumerate(species_indices1):
    axes[0].plot(sol_elv1.t, sol_elv1.y[i], color=cmap1((i + 1) / (N1 + 1)), label=f"c{idx}")
axes[0].set_title('Community 1 Dynamics (eLV)')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Consumer Abundance')
axes[0].grid(True)
axes[0].legend(loc='upper right', fontsize='small')
# Plot for Community 2
cmap2 = plt.get_cmap("Reds")
for i, idx in enumerate(species_indices2):
    axes[1].plot(sol_elv2.t, sol_elv2.y[i], color=cmap2((i + 1) / (N2 + 1)), label=f"c{idx}")
axes[1].set_title('Community 2 Dynamics (eLV)')
axes[1].set_xlabel('Time')
axes[1].grid(True)
axes[1].legend(loc='upper right', fontsize='small')
# Plot for Community 3 (merged)
for i, idx in enumerate(species_indices1):
    axes[2].plot(sol_elv3.t, sol_elv3.y[i], color=cmap1((i + 1) / (N1 + 1)), label=f"c{idx}")
for i, idx in enumerate(species_indices2):
    axes[2].plot(sol_elv3.t, sol_elv3.y[N1 + i], color=cmap2((i + 1) / (N2 + 1)), label=f"c{idx}")
axes[2].set_title('Coalescence Dynamics (eLV)')
axes[2].set_xlabel('Time')
axes[2].grid(True)
axes[2].legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()

# storage data
import pandas as pd

def store_elv_data(species_indices, community_id, r, CUE, C_hat):
    """
    Create a DataFrame storing species information for one community:
    - species ID
    - community ID
    - intrinsic growth rate (r)
    - CUE
    - interaction matrix row (α_ij)
    """
    N = len(species_indices)
    records = []

    for i in range(N):
        row = {
            "Species": f"c{species_indices[i]}",
            "Community": community_id,
            "r": r[i],
            "CUE": CUE[i],
            "Abundance": C_hat[i]
        }
        records.append(row)

    return pd.DataFrame(records)

# --- Compute species-level CUEs for each community ---
species_CUE1 = CUE.compute_community_CUE2(sol1, N1, u1, np.full(M1, 1.0), l1, m1)[1]
species_CUE2 = CUE.compute_community_CUE2(sol2, N2, u2, np.full(M2, 1.0), l2, m2)[1]
species_CUE3 = CUE.compute_community_CUE2(sol3, N3, u3, np.full(M3, 1.0), l3, m3)[1]

# --- Build DataFrames for each community ---
df1 = store_elv_data(species_indices1, 1, r1, species_CUE1, C_hat1)
df2 = store_elv_data(species_indices2, 2, r2, species_CUE2, C_hat2)
df3 = store_elv_data(np.concatenate([species_indices1, species_indices2]), 3, r3, species_CUE3, C_hat3)

# --- Combine and export to CSV ---
elv5 = pd.concat([df1, df2, df3], ignore_index=True)
elv5.to_csv("data/elv50.csv", index=False)



"""
########## New elv with indice selection #########
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import truncnorm
import param
import CUE

np.random.seed(37)
N_pool = 1000  # Species pool size
M_pool = 50     # Resource pool size
λ = 0.3        # Total leakage rate
N_modules = 1  # Number of modules
s_ratio = 1 # Modularity ratio
N = 100
M = 50
m = truncnorm.rvs((0 - 1) / 0.01, np.inf, loc=1, scale=0.01, size=N)

# Generate uptake matrix and leakage tensor for the species pool
u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ)
# Set rho and omega for the resource pool
rho_pool = np.full(M_pool, 0.5)
omega_pool = np.full(M_pool, 0.5)
species_indices = np.random.choice(N_pool, N, replace=False)
resource_indices = np.random.choice(M_pool, M, replace=False)
u = u_pool[np.ix_(species_indices, resource_indices)]

u = 2.5 * u / u.sum(axis=1, keepdims=True)

l = l_pool[np.ix_(species_indices, resource_indices, resource_indices)]
lambda_alpha = np.full(M, λ)
rho = rho_pool[resource_indices]
omega = omega_pool[resource_indices]
def dCdt_Rdt(t, y):
    C = y[:N]
    R = y[N:]
    dCdt = np.zeros(N)
    dRdt = np.zeros(M)

    for i in range(N):
        dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - lambda_alpha[alpha]) for alpha in range(M)) - C[i] * m[i]

    for alpha in range(M):
        dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]
        dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))
        dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l[i, beta, alpha] for beta in range(M)) for i in range(N))

    return np.concatenate([dCdt, dRdt])

# 
C0 = np.full(N, 0.01)
R0 = np.full(M, 1.0)
Y0 = np.concatenate([C0, R0])


t_span = (0, 600)
t_eval = np.linspace(*t_span, 300)
sol_mcm = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)

C_hat = sol_mcm.y[:N, -1]
R_hat = sol_mcm.y[N:, -1]

# alpha_ij
D = np.zeros((M, M))
for a in range(M):
    for gamma in range(M):
        if a == gamma:
            D[a, a] = omega[a] + sum(C_hat[i] * u[i, a] for i in range(N))
        else:
            D[a, gamma] = -sum(C_hat[i] * u[i, gamma] * l[i, gamma, a] for i in range(N))

partial_R_C = np.zeros((M, N))
for j in range(N):
    v_j = np.zeros(M)
    for alpha in range(M):
        v_j[alpha] = -R_hat[alpha] * u[j, alpha] + sum(R_hat[beta] * u[j, beta] * l[j, beta, alpha] for beta in range(M))
    partial_R_C[:, j] = np.linalg.solve(D, v_j)

alpha = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        alpha[i, j] = sum(u[i, a] * (1 - lambda_alpha[a]) * partial_R_C[a, j] for a in range(M))

# r_i
r = np.zeros(N)
for i in range(N):
    growth_term = sum(u[i, a] * (1 - lambda_alpha[a]) * R_hat[a] for a in range(M))
    interaction_term = sum(alpha[i, j] * C_hat[j] for j in range(N))
    r[i] = growth_term - m[i] - interaction_term

# eLV
def dCdt_elv(t, C):
    dCdt = np.zeros(N)
    for i in range(N):
        dCdt[i] = C[i] * (r[i] + sum(alpha[i, j] * C[j] for j in range(N)))
    return dCdt


sol_elv = solve_ivp(dCdt_elv, t_span, C0, t_eval=t_eval)
# CUE
community_CUE, species_CUE = CUE.compute_community_CUE2(
        sol_mcm, N, u, R0, l, m
    )

survivors = C_hat > 1e-5

plt.figure()
plt.scatter(species_CUE[survivors], r[survivors], label='Survivors', alpha=0.7)
plt.scatter(species_CUE[~survivors], r[~survivors], label='Extinct', alpha=0.7)
plt.xlabel('Species CUE')
plt.ylabel('Intrinsic growth rate $r_i$')
plt.legend()
plt.grid(True)
plt.show()

# u heatmap
plt.imshow(u, aspect='auto')
plt.title('Uptake matrix heatmap')
plt.colorbar()
plt.show()

# interaction coefficient heatmap
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
plt.imshow(alpha, aspect='auto', cmap='viridis')
plt.colorbar(label='α_ij')
plt.xlabel('j (source species)')
plt.ylabel('i (focal species)')
plt.title('Interaction matrix α_ij')
plt.tight_layout()
plt.show()

# Plot elv biomass change over time
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

for i in range(N):
    plt.plot(sol_elv.t, sol_elv.y[i, :], alpha=0.5, linewidth=1)

plt.xlabel('Time')
plt.ylabel('Species Abundance')
plt.title('Biomass Dynamics of All Species')
plt.grid(True)
plt.tight_layout()
plt.show()

# plot MiCRM
plt.figure(figsize=(10, 6))

for i in range(N):
    plt.plot(sol_mcm.t, sol_mcm.y[i, :], alpha=0.5, linewidth=1)

plt.xlabel('Time')
plt.ylabel('Species Abundance')
plt.title('Biomass Dynamics of All Species')
plt.grid(True)
plt.tight_layout()
plt.show()
"""
"""
########### New elv no indices selection ################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import param
import CUE
from scipy.stats import truncnorm
# Parameter setting
N = 100
M = 50
λ = 0.3
m = truncnorm.rvs((0 - 1) / 0.01, np.inf, loc=1, scale=0.01, size=N)

rho = np.full(M, 0.6)
omega = np.full(M, 0.1)
N_modules = 1
s_ratio = 1

u = param.modular_uptake(N, M, N_modules, s_ratio)
l_tensor = param.generate_l_tensor(N, M, N_modules, s_ratio, λ)
lambda_alpha = np.full(M, λ)


def dCdt_Rdt(t, y):
    C = y[:N]
    R = y[N:]
    dCdt = np.zeros(N)
    dRdt = np.zeros(M)

    for i in range(N):
        dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - lambda_alpha[alpha]) for alpha in range(M)) - C[i] * m[i]

    for alpha in range(M):
        dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]
        dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))
        dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l_tensor[i, beta, alpha] for beta in range(M)) for i in range(N))

    return np.concatenate([dCdt, dRdt])

# 
C0 = np.full(N, 0.01)
R0 = np.full(M, 1.0)
Y0 = np.concatenate([C0, R0])


t_span = (0, 300)
t_eval = np.linspace(*t_span, 100)
sol_mcm = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)

C_hat = sol_mcm.y[:N, -1]
R_hat = sol_mcm.y[N:, -1]

# alpha_ij
D = np.zeros((M, M))
for a in range(M):
    for gamma in range(M):
        if a == gamma:
            D[a, a] = omega[a] + sum(C_hat[i] * u[i, a] for i in range(N))
        else:
            D[a, gamma] = -sum(C_hat[i] * u[i, gamma] * l_tensor[i, gamma, a] for i in range(N))

partial_R_C = np.zeros((M, N))
for j in range(N):
    v_j = np.zeros(M)
    for alpha in range(M):
        v_j[alpha] = -R_hat[alpha] * u[j, alpha] + sum(R_hat[beta] * u[j, beta] * l_tensor[j, beta, alpha] for beta in range(M))
    partial_R_C[:, j] = np.linalg.solve(D, v_j)

alpha = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        alpha[i, j] = sum(u[i, a] * (1 - lambda_alpha[a]) * partial_R_C[a, j] for a in range(M))

# r_i
r = np.zeros(N)
for i in range(N):
    growth_term = sum(u[i, a] * (1 - lambda_alpha[a]) * R_hat[a] for a in range(M))
    interaction_term = sum(alpha[i, j] * C_hat[j] for j in range(N))
    r[i] = growth_term - m[i] - interaction_term

# eLV
def dCdt_elv(t, C):
    dCdt = np.zeros(N)
    for i in range(N):
        dCdt[i] = C[i] * (r[i] + sum(alpha[i, j] * C[j] for j in range(N)))
    return dCdt


sol_elv = solve_ivp(dCdt_elv, t_span, C0, t_eval=t_eval)
community_CUE, species_CUE = CUE.compute_community_CUE2(
        sol_mcm, N, u, R0, l_tensor, m
    )


survivors = C_hat > 1.0e-7

plt.figure()
plt.scatter(species_CUE[survivors], r[survivors], label='Survivors', alpha=0.7)
plt.scatter(species_CUE[~survivors], r[~survivors], label='Extinct', alpha=0.7)
plt.xlabel('Species CUE')
plt.ylabel('Intrinsic growth rate $r_i$')
plt.legend()
plt.grid(True)
plt.show()

# u heatmap
plt.imshow(u, aspect='auto')
plt.title('Uptake matrix heatmap')
plt.colorbar()
plt.show()
"""