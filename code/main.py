import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import param


# parameters
N = 100  # consumer number
M = 50  # resource number
λ = 0.3  # total leakage rate
λ_u = np.ones(N)

N_modules = 1 #  module number of consumer to resource
s_ratio = 1.0 
# When s_ratio = 1: Resources have a uniform leakage probability；
# When s_ratio > 1: Increases leakage probability within the same module, Increases leakage probability between adjacent modules.


u = param.modular_uptake(N, M, N_modules, s_ratio)  # uptake matrix
row_sums = np.sum(u, axis=1)

lambda_alpha = np.full(M, λ)  # total leakage rate for each resource


m = np.full(N, 0.1)  # mortality rate of N consumers
rho = np.full(M, 1)  # input of M resources
omega = np.full(M, 1)  # decay rate of M resources

l = param.generate_l_tensor(N, M, N_modules, s_ratio, λ) # a tensor for all consumers' leakage matrics


# ode
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

# intial value
C0 = np.full(N,1)  # consumer
R0 = np.full(M,1)   # resource
Y0 = np.concatenate([C0, R0])


# time sacle
t_span = (0, 200)
t_eval = np.linspace(*t_span, 300)

# solve ode
sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)
# CUE
# Compute CUE at each time step
CUE = np.zeros((N, len(sol.t)))
for i, t in enumerate(sol.t):
    C = sol.y[:N, i]  # Consumer abundances at time t
    R = sol.y[N:, i]  # Resource concentrations at time t
    total_uptake = u @ R  # (N × M) @ (M,) -> (N,)
    net_uptake = total_uptake * (1 - λ) - m  # Adjusted for leakage and metabolism
    CUE[:, i] = net_uptake / total_uptake  # Compute CUE per consumer

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


# plot CUE change
plt.figure(figsize=(10, 5))

for i in range(N):
    plt.plot(sol.t, CUE[i], label=f'Consumer {i+1}')

plt.xlabel('Time')
plt.ylabel('Carbon Use Efficiency (CUE)')
plt.title('CUE Dynamics Over Time')
plt.legend()
plt.show()

# system analysis

