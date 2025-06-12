import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
import os

# Manually add the directory where param_3D.py is located
sys.path.append(os.path.expanduser("~/Documents/MiCRM/code"))
import param_3D

np.random.seed(30) 
# Parameters
N = 20 # Number of consumers
M = 12 # Number of resources
λ = 0.3  # Total leakage rate

# Each consumer's uptake scaling factor
λ_u = np.random.uniform(2.4, 2.6, N)  
σ = np.random.uniform(0.05 * λ_u, 0.2 * λ_u)
N_modules = 5  # Number of modules connecting consumers to resources
s_ratio = 4.0  # Strength of modularity

# Generate uptake matrix, defining consumer-resource interaction strengths
u = param_3D.modular_uptake(N, M, N_modules, s_ratio, λ_u, σ)  
print(u)
# Total leakage rate for each resource
lambda_alpha = np.full(M, λ)  

# Mortality rate for each consumer
m = np.full(N, 0.2)  
# Input rate of each resource
rho = np.full(M, 1)  
# Decay rate for each resource
omega = np.full(M, 0.01)  

# Generate leakage tensor representing leakage flow between consumers and resources
l = param_3D.generate_l_tensor(N, M, N_modules, s_ratio, λ)  

# ODE system describing the dynamics of consumers (C) and resources (R)
def dCdt_Rdt(t, y):
    C = y[:N]  # Consumer populations
    R = y[N:]  # Resource concentrations
    dCdt = np.zeros(N)
    dRdt = np.zeros(M)
    
    # Consumer growth equation
    for i in range(N):
        dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - lambda_alpha[alpha]) for alpha in range(M)) - C[i] * m[i]
    
    # Resource depletion and leakage equation
    for alpha in range(M):
        dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]  # Input and decay
        dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))  # Uptake by consumers
        dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l[i, beta, alpha] for beta in range(M)) for i in range(N))  # Leakage contributions
    
    return np.concatenate([dCdt, dRdt])
    
# Initial conditions: assume all consumers and resources start at concentration 1
C0 = np.full(N, 0.1)
R0 = np.full(M, 1)
Y0 = np.concatenate([C0, R0])
    
t_span = (0, 700)
t_eval = np.linspace(*t_span, 300)
sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)
    
# CUE
# Compute CUE at each time step
CUE = np.zeros((N, len(sol.t)))
for i, t in enumerate(sol.t):
    C = sol.y[:N, i]  # Consumer abundances at time t
    R = sol.y[N:, i]  # Resource concentrations at time t
    total_uptake = u @ R# (N × M) @ (M,) -> (N,)
    net_uptake = total_uptake * (1 - λ) - m  # Adjusted for leakage and metabolism
    CUE[:, i] = net_uptake / total_uptake  # Compute CUE per consumer
# Final CUE value 
final_CUE = CUE[:, -1]
print(final_CUE)

# Calculate community CUE
# Compute total resource input (denominator of efficiency)
total_input = np.sum(rho)

# Compute steady-state CUE using final time point
C_final = sol.y[:N, -1]  # Consumer biomass at the final time point
energy_used_final = np.sum(C_final * m)  # Total energy dissipated via mortality at steady state
efficiency_final = energy_used_final / total_input  # Steady-state efficiency

# Compute efficiency trajectory and time-averaged efficiency
C_traj = sol.y[:N, :]  # Consumer biomass over time
energy_used_traj = np.sum(C_traj * m[:, np.newaxis], axis=0)  # Energy dissipated via mortality at each time point
efficiency_traj = energy_used_traj / total_input  # Efficiency trajectory over time
efficiency_avg = np.mean(efficiency_traj)  # Time-averaged efficiency

# Output efficiency results
print(f"Steady-state CUE: {efficiency_final:.4f}")
print(f"Time-averaged CUE: {efficiency_avg:.4f}")

# Plot efficiency over time
plt.figure(figsize=(10, 5))
plt.plot(sol.t, efficiency_traj, label='Efficiency', linewidth=2)
plt.xlabel('Time', fontsize=20)
plt.ylabel('CUE', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.title('CUE Over Time', fontsize=20)
plt.show()
