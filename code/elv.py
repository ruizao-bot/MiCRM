'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
import param
import CUE
import pandas as pd

# Parameter settings
np.random.seed(37)
N_pool = 1000  # Species pool size
M_pool = 20     # Resource pool size
λ = 0.2        # Total leakage rate
N_modules = 5  # Number of modules
s_ratio = 10.0 # Modularity ratio
N1 = 100
M1 = 5
m1 = np.full(N1, 0.2)  # maintaining cost rate
N2 = 100
M2 = 5
m2 = np.full(N2, 0.2)
# Generate uptake matrix and leakage tensor for the species pool
u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ)
# Set rho and omega for the resource pool
rho_pool = np.full(M_pool, 0.6)
omega_pool = np.full(M_pool, 0)
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
C_guess1 = np.full(N1, 0.1)
R_guess1 = np.full(M1, 1.0)
y_guess1 = np.concatenate([C_guess1, R_guess1])
sol_steady1 = root(lambda y: steady_state_eq(y, N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1), y_guess1)
C_hat1 = sol_steady1.x[:N1]
R_hat1 = sol_steady1.x[N1:]
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
# Solve for the steady state of Community 2
C_guess2 = np.full(N2, 0.1)
R_guess2 = np.full(M2, 1.0)
y_guess2 = np.concatenate([C_guess2, R_guess2])
sol_steady2 = root(lambda y: steady_state_eq(y, N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2), y_guess2)
C_hat2 = sol_steady2.x[:N2]
R_hat2 = sol_steady2.x[N2:]
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
# Solve for the steady state
C_guess3 = np.full(N3, 0.1)
R_guess3 = np.full(M3, 1.0)
y_guess3 = np.concatenate([C_guess3, R_guess3])
sol_steady3 = root(lambda y: steady_state_eq(y, N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3), y_guess3)
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
# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
# Plot for Community 1
cmap1 = plt.get_cmap("Blues")
for i, idx in enumerate(species_indices1):
    axes[0].plot(sol1.t, sol1.y[i], color=cmap1((i + 1) / (N1 + 1)), label=f"c{idx}")
axes[0].set_title('Community 1 Dynamics (eLV)')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Consumer Abundance')
axes[0].grid(True)
axes[0].legend(loc='upper right', fontsize='small')
# Plot for Community 2
cmap2 = plt.get_cmap("Reds")
for i, idx in enumerate(species_indices2):
    axes[1].plot(sol2.t, sol2.y[i], color=cmap2((i + 1) / (N2 + 1)), label=f"c{idx}")
axes[1].set_title('Community 2 Dynamics (eLV)')
axes[1].set_xlabel('Time')
axes[1].grid(True)
axes[1].legend(loc='upper right', fontsize='small')
# Plot for Community 3 (merged)
for i, idx in enumerate(species_indices1):
    axes[2].plot(sol3.t, sol3.y[i], color=cmap1((i + 1) / (N1 + 1)), label=f"c{idx}")
for i, idx in enumerate(species_indices2):
    axes[2].plot(sol3.t, sol3.y[N1 + i], color=cmap2((i + 1) / (N2 + 1)), label=f"c{idx}")
axes[2].set_title('Coalescence Dynamics (eLV)')
axes[2].set_xlabel('Time')
axes[2].grid(True)
axes[2].legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()


print("alpha1:", alpha1)
print("alpha2:", alpha2)
print("alpha3:", alpha3)
print(f"r1: {r1} | r2: {r2} | r3: {r3}")

r1_add = np.pad(r1, (0, N2), mode='constant', constant_values=0)
r2_add = np.pad(r2, (N1, 0), mode='constant', constant_values=0)
length_r1 = np.linalg.norm(r1_add)
length_r2 = np.linalg.norm(r2_add)
length_r3 = np.linalg.norm(r3)

print("Length of r1:",length_r1)
print("Length of r2:",length_r2)
print("Length of r3",length_r3)

dot_product = np.dot(r1_add, r2_add) 
cos_theta = dot_product / (length_r1 * length_r2) 

result = np.sqrt(length_r1**2 + length_r2**2)
print(result)

# Community CUE
sol_list = [sol1, sol2, sol3] 
N_list = [N1, N2, N3] 
u_list = [u1, u2, u3]
R0_list = [R_guess1, R_guess2, R_guess3]
l_list = [l1, l2, l3]
m_list = [m1, m2, m3]
M_list = [M1, M2, M3]
r_list = [r1, r2, r3]

C_final_list = []
cue_results = []
num_communities = len(sol_list) 

for i in range(num_communities):
    C_final = np.array(sol_list[i].y[:, -1]) 
    C_final_list.append(C_final)
    
    community_CUE, species_CUE = CUE.compute_community_CUE2(
        sol_list[i], N_list[i], u_list[i], R0_list[i], l_list[i], m_list[i]
    )
    
    print(f"Community {i+1}: CUE = {community_CUE}")
    
    for j, cue in enumerate(species_CUE):
        cue_results.append({
            "Community": f"Community {i+1}",
            "Species": f"Species {j+1}",
            "r": r_list[i][j],
            "Species CUE": cue
        })
community_CUE1, species_CUE1 = CUE.compute_community_CUE2(
        sol1, N1, u1, R_guess1, l1, m1
    )

# Convert to DataFrame and save
# df_species_CUE = pd.DataFrame(cue_results)
# df_species_CUE.to_csv("../data/df_elv.csv", index=False)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(species_CUE1, r1)
plt.xlabel('Species CUE')
plt.ylabel('Intrinsic growth rate $r_i$')
plt.title('Relationship between $r_i$ and Species CUE')
plt.grid(True)
plt.show()
'''
########## New elv with indice selection #########
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import param
import CUE

np.random.seed(37)
N_pool = 1000  # Species pool size
M_pool = 50     # Resource pool size
λ = 0.2        # Total leakage rate
N_modules = 1  # Number of modules
s_ratio = 1 # Modularity ratio
N = 100
M = 5
m = np.full(N, 0.2) # maintaining cost rate
# Generate uptake matrix and leakage tensor for the species pool
u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ)
# Set rho and omega for the resource pool
rho_pool = np.full(M_pool, 0.6)
omega_pool = np.full(M_pool, 0.1)
species_indices = np.random.choice(N_pool, N, replace=False)
resource_indices = np.random.choice(M_pool, M, replace=False)
u = u_pool[np.ix_(species_indices, resource_indices)]

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


t_span = (0, 300)
t_eval = np.linspace(*t_span, 600)
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
    growth_term = sum(u[i, a] * (1 - lambda_alpha[a]) * R0[a] for a in range(M))
    interaction_term = sum(alpha[i, j] * C0[j] for j in range(N))
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

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(species_CUE, r)
plt.xlabel('Species CUE')
plt.ylabel('Intrinsic growth rate $r_i$')
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

########### New elv no indices selection ################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import param
import CUE

# Parameter setting
N = 10
M = 10
λ = 0.4
m = np.full(N, 0.2)
rho = np.full(M, 1)
omega = np.full(M, 0)
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
C0 = np.full(N, 0.1)
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


import matplotlib.pyplot as plt

plt.figure()
plt.scatter(species_CUE, r)
plt.xlabel('Species CUE')
plt.ylabel('Intrinsic growth rate $r_i$')
plt.title('Relationship between $r_i$ and Species CUE')
plt.grid(True)
plt.show()