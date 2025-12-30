import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import param


# parameters
N = 50  # consumer number
M = 25  # resource number
λ = 0.3  # total leakage rate
λ_u = np.ones(N)

N_modules = 2 #  module number of consumer to resource
s_ratio = 10.0 
# When s_ratio = 1: Resources have a uniform leakage probability；
# When s_ratio > 1: Increases leakage probability within the same module, Increases leakage probability between adjacent modules.


u = param.modular_uptake(N, M, N_modules, s_ratio)  # uptake matrix
row_sums = np.sum(u, axis=1)

lambda_alpha = np.full(M, λ)  # total leakage rate for each resource


m = np.full(N, 0.2)  # mortality rate of N consumers
rho = np.full(M, 0.5)  # input of M resources
omega = np.full(M, 0.5)  # decay rate of M resources

l = param.generate_l_tensor(N, M, N_modules, s_ratio, λ, u) # a tensor for all consumers' leakage matrics

# intial value
C0 = np.full(N, 0.01)  # consumer
R0 = np.full(M, 1)   # resource

# time sacle
t_span = (0, 200000)

# solve ode
sol = param.solve_micrm(N, M, u, l, m, lambda_alpha, rho, omega, C0, R0, t_span)

community_CUE, species_CUE = param.compute_CUE(sol, N, u, R0, lambda_alpha, m)
print(species_CUE)
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

# system analysis
def richness(abundances, thresh=1e-5):
    """Return number of species with abundance > thresh."""
    return int(np.sum(np.asarray(abundances) > thresh))


# Compute richness from final abundances
final_C = sol.y[:N, -1]
SURV_THRESH = 1e-5
rich = richness(final_C, SURV_THRESH)
print(f"Final species richness (abundance > {SURV_THRESH}): {rich} / {N}")


