import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
import os

# Manually add the directory where param_3D.py is located
sys.path.append(os.path.expanduser("~/Documents/MiCRM/code"))
import param_resource
np.random.seed(30) 
# Parameters
N = 10# Number of consumers
M = 10# Number of resources
λ = 0.3  # Total leakage rate
kaf0 = 0.3
epsilon = np.random.normal(0, 0.1, N)
λ_u = np.ones(N)
σ = np.random.uniform(0.1, 1, N)

N_modules = 5  # Number of modules connecting consumers to resources
s_ratio = 10.0  # Strength of modularity

# Generate uptake matrix, defining consumer-resource interaction strengths
u = param_resource.modular_uptake(N, M, N_modules, s_ratio, λ_u, σ)
u = np.where(u < 0.05, 0, 0.5)
print(u)
# Total leakage rate for each resource
lambda_alpha = np.full(M, λ)  

# Input rate of each resource
rho = np.full(M, 1)  

# Decay rate for each resource
omega = np.full(M, 0.05)  

# Generate leakage tensor representing leakage flow between consumers and resources
l = param_resource.generate_l_tensor(N, M, N_modules, s_ratio, λ)  
# Maintenaince cost
# χ0 is the average cost of consuming a given resource
m = param_resource.compute_m(kaf0, epsilon, λ, u)
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
C0 = np.full(N, 1)
R0 = np.full(M, 1)
Y0 = np.concatenate([C0, R0])
    
t_span = (0, 500)
t_eval = np.linspace(*t_span, 300)
sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)
    
# CUE
CUE = np.zeros(N)
total_uptake = np.sum(u * R0, axis=1)  # (N × M) @ (M,) -> (N,)
net_uptake = np.sum(u * R0 *(1-lambda_alpha), axis=1)-m # Adjusted for leakage and metabolism
CUE = net_uptake/total_uptake
print(f"Carbon Use Efficiency (CUE): {CUE.tolist()}")

u_sum = np.sum(u, axis=1)  # Mean uptake per consumer


# -------------------------- Logarithmic Model Fitting --------------------------

# plot
plt.figure(figsize=(10, 5))
for i in range(N):
    plt.plot(sol.t, sol.y[i], label=f'Consumer {i+1}')
for alpha in range(M):
    plt.plot(sol.t, sol.y[N + alpha], label=f'Resource {alpha+1}', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Comsumer / Resource')
plt.legend()
plt.title('Dynamics of Consumers and Resources')
plt.show()

# -------------------------- Logarithmic Model Fitting --------------------------

from scipy.stats import linregress
def plot_regression(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, label='Data points', color='b')
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept
    
    plt.plot(x_fit, y_fit, color='r', linestyle='dashed', label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_value**2:.2f}')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
plot_regression(
    u_sum, CUE,
    xlabel="Sum of Resource uptake type",
    ylabel="Final CUE",
    title="Linear Regression of Final CUE vs. Resource uptake type")