import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import os
import sys
sys.path.append(os.path.expanduser("~/Documents/MiCRM/code"))
import param_3D
import CUE
from scipy.optimize import curve_fit

# Parameters
np.random.seed(30)  # Number of simulations (changing random seed each time)
N = 20  # Number of consumers
M = 10  # Number of resources
λ = 0.3  # Total leakage rate
λ_u = np.random.uniform(0.8, 1, N)
# Storage for results
results = []

# Loop through multiple simulations with different random seeds
for σ in np.arange(0.05 * np.min(λ_u), 0.8 * np.max(λ_u), 0.01):  
    
    N_modules = 4
    s_ratio = 10.0
    # Generate uptake matrix and leakage tensor
    u = param_3D.modular_uptake(N, M, N_modules, s_ratio, λ_u, σ)  
    lambda_alpha = np.full(M, λ)
    m = np.full(N, 0.2)
    rho = np.full(M, 1)
    omega = np.full(M, 0.01)
    l = param_3D.generate_l_tensor(N, M, N_modules, s_ratio, λ)

    # ODE system for consumer and resource dynamics
    def dCdt_Rdt(t, y):
        C = y[:N]
        R = y[N:]
        dCdt = np.zeros(N)
        dRdt = np.zeros(M)

        # Consumer growth equation
        for i in range(N):
            dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - lambda_alpha[alpha]) for alpha in range(M)) - C[i] * m[i]

        # Resource dynamics
        for alpha in range(M):
            dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]
            dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))
            dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l[i, beta, alpha] for beta in range(M)) for i in range(N))

        return np.concatenate([dCdt, dRdt])

    # Initial conditions
    C0 = np.full(N, 0.1)
    R0 = np.full(M, 1)
    Y0 = np.concatenate([C0, R0])

    # Solve ODE system
    t_span = (0, 700)
    t_eval = np.linspace(*t_span, 300)
    sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)

    # Compute CUE 
    community_CUE, species_CUE = CUE.compute_community_CUE2(sol, N, u, R0, l, m)
    species_CUE = np.array(species_CUE, dtype=float)
    C_final = sol.y[:N, -1]

    # Richness count
    species_richness = np.sum(C_final >= 0.1)

    # Compute statistics
    species_mean = np.mean(species_CUE)
    species_var = np.var(species_CUE, ddof=1)
    species_max = np.max(species_CUE)
    deviation_sq_sum = np.sum((species_CUE - species_max) ** 2)

    # Store results
    results.append({
        "Community CUE": community_CUE,
        "Species Mean CUE": species_mean,
        "Species Variance": species_var,
        "Deviation Sum": deviation_sq_sum,
        "Species Richness": species_richness
    })

# Convert results to a DataFrame and display
df_results = pd.DataFrame(results)
from IPython.display import display
display(df_results)

# Plot community CUE vs species mean CUE
plt.figure(figsize=(6, 5))
plt.scatter(df_results["Species Mean CUE"], df_results["Community CUE"], color="red")
plt.xlabel("Species Mean CUE")
plt.ylabel("Community CUE")
plt.title("Community CUE vs Species Mean CUE")
plt.show()

# Plot surviving species count vs species variance
data = pd.DataFrame({
    "Species Variance": df_results["Species Variance"],
    "Species Richness": df_results["Species Richness"]
})

x = data["Species Variance"].values
y = data["Species Richness"].values
def exp_func(x, a, b):
    return a * np.exp(b * x)

popt, _ = curve_fit(exp_func, x, y, p0=(1, 0.1))
y_pred_exp = exp_func(x, *popt)
plt.figure(figsize=(6, 5))
plt.scatter(x, y, color="green", label="Data", alpha=0.6)
plt.plot(x, y_pred_exp, color="purple", linestyle="dotted", label="Exponential Fit")
plt.xlabel("Species Variance")
plt.ylabel("Species Richness")
plt.title("Species Richness vs Species Variance with Fits")
plt.legend()
plt.show()
# Plot surviving species count vs deviation sum
plt.figure(figsize=(6, 5))
plt.scatter(df_results["Deviation Sum"], df_results["Species Richness"], color="purple")
plt.xlabel("Deviation Sum")
plt.ylabel("Species Richness")
plt.title("Species Richness vs Deviation Sum")
plt.show()