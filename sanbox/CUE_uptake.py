import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import solve_ivp
import seaborn as sns
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
λ_u = np.random.uniform(0.8, 1, N)  
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
CUE = np.zeros(N)
total_uptake = np.sum(u * R0, axis=1)  # (N × M) @ (M,) -> (N,)
net_uptake = np.sum(u * R0 *(1-lambda_alpha), axis=1)-m # Adjusted for leakage and metabolism
CUE = net_uptake/total_uptake
print(f"Carbon Use Efficiency (CUE): {CUE.tolist()}")

# find equilibrium state
# Set the equilibrium threshold
tolerance = 0.01  

# Compute the numerical derivative of dC/dt
dC_dt_numeric = np.gradient(sol.y[:N, :], sol.t, axis=1)  # (N, T)

# Step-by-step check to find the equilibrium state of R and CUE
t_steady = None
for idx in range(len(sol.t)):
    if np.all(np.abs(dC_dt_numeric[:, idx]) < tolerance):  # Check if all dC/dt values are below the threshold
        t_steady = sol.t[idx]  # Record the equilibrium time
        C_steady = sol.y[:N, idx]  # Record the steady-state biomass (C)
        R_steady = sol.y[N:, idx]  # Record the steady-state resource abundance (R)
        break  # Stop searching once equilibrium is found

# Print the results
if t_steady is not None:
    print(f"System reached steady state at t = {t_steady:.2f}")
    print(f"Steady-state Biomass (C): {C_steady}")
    print(f"Steady-state Resources (R): {R_steady}")
else:
    print("System did not reach steady state within the simulation time.")

# Plot the change of dC/dt over time
plt.figure(figsize=(8, 5))
for i in range(N):
    plt.plot(sol.t, dC_dt_numeric[i], label=f'Consumer {i+1}')
if t_steady is not None:
    plt.axvline(x=t_steady, color='r', linestyle='--', label=f'Steady State at t={t_steady:.2f}')
plt.xlabel("Time")
plt.ylabel("dC/dt (Biomass Growth Rate)")
plt.title("Change of dC/dt Over Time")
plt.legend()
plt.show()

# community CUE
# Compute total resource consumed by consumers
total_resource_consumed = np.sum(rho) *t_steady  - np.sum(R_steady)
# Compute biomass growth
biomass_growth = np.sum(C_steady) - np.sum(C0)
CUE_community = biomass_growth/total_resource_consumed
print(f"Community-level Carbon Use Efficiency (CUE): {CUE_community:.4f}")

# Visualize system dynamics
for i in range(N):
    plt.plot(sol.t, sol.y[i], label=f'Consumer {i+1}')
for alpha in range(M):
    plt.plot(sol.t, sol.y[N + alpha], label=f'Resource {alpha+1}', linestyle='dashed')

plt.xlabel('Time')
plt.ylabel('Consumer / Resource')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=2, fontsize=10, columnspacing=1.5)
plt.title('Dynamics of Consumers and Resources')
plt.tight_layout()
plt.savefig("results/dynamics_of_consumers_resources.png", dpi=300, bbox_inches='tight')
plt.show()

# Create an interactive 3D scatter plot
#import plotly.graph_objects as go
#fig = go.Figure(data=[go.Scatter3d(
#    x=u_mean,
#    y=u_variance,
#    z=final_CUE,
#    mode='markers',
#    marker=dict(
#        size=5,
#        color=final_CUE,  # Color mapped to CUE values
#        colorscale='viridis',
#        opacity=0.8
#    )
#)])

# Configure axis labels and title
#fig.update_layout(
#    scene=dict(
#        xaxis_title="Uptake Mean",
#        yaxis_title="Uptake Variance",
#        zaxis_title="CUE"
#    ),
#    title="3D Interactive Scatter Plot of CUE vs. Uptake Mean & Variance"
#)

# Save the plot as an HTML file
#fig.write_html("results/CUE_uptake.html")

# Show the interactive plot
#fig.show()
# linear regression
#from scipy.stats import linregress
#def plot_regression(x, y, xlabel, ylabel, title):
#    plt.figure(figsize=(7, 5))
#    plt.scatter(x, y, label='Data points', color='b')
#    slope, intercept, r_value, p_value, std_err = linregress(x, y)
#    x_fit = np.linspace(min(x), max(x), 100)
#    y_fit = slope * x_fit + intercept
#    
#    plt.plot(x_fit, y_fit, color='r', linestyle='dashed', label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_value**2:.2f}')
#    
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    plt.title(title)
#    plt.legend()
#    plt.show()
#plot_regression(
#    u_mean, final_CUE,
#    xlabel="Uptake Mean",
#    ylabel="Final CUE",
#    title="Linear Regression of Final CUE vs. Uptake Mean"
#)
# Use R equilibrium as system equilibrium
# find equilibrium state
# Set the equilibrium threshold
tolerance = 0.001  

# Compute the numerical derivative of dC/dt
dR_dt_numeric = np.gradient(sol.y[N:, :], sol.t, axis=1)  # (N, T)

# Step-by-step check to find the equilibrium state of R and CUE
t_steady = None
for idx in range(len(sol.t)):
    if np.all(np.abs(dR_dt_numeric[:, idx]) < tolerance):  # Check if all dC/dt values are below the threshold
        t_steady = sol.t[idx]  # Record the equilibrium time
        C_steady = sol.y[:N, idx]  # Record the steady-state biomass (C)
        R_steady = sol.y[N:, idx]  # Record the steady-state resource abundance (R)
        break  # Stop searching once equilibrium is found

# Print the results
if t_steady is not None:
    print(f"System reached steady state at t = {t_steady:.2f}")
    print(f"Steady-state Biomass (C): {C_steady}")
    print(f"Steady-state Resources (R): {R_steady}")
else:
    print("System did not reach steady state within the simulation time.")

# Plot the change of dC/dt over time
plt.figure(figsize=(8, 5))
for i in range(N):
    plt.plot(sol.t, dC_dt_numeric[i], label=f'Consumer {i+1}')
if t_steady is not None:
    plt.axvline(x=t_steady, color='r', linestyle='--', label=f'Steady State at t={t_steady:.2f}')
plt.xlabel("Time")
plt.ylabel("dC/dt (Biomass Growth Rate)")
plt.title("Change of dC/dt Over Time")
plt.legend()
plt.show()

# community CUE
# Compute total resource consumed by consumers
total_resource_consumed = np.sum(rho) *t_steady  - np.sum(R_steady)
# Compute biomass growth
biomass_growth = np.sum(C_steady) - np.sum(C0)
CUE_community = biomass_growth/total_resource_consumed
print(f"Community-level Carbon Use Efficiency (CUE): {CUE_community:.4f}")

# integral method of community CUE
C_values = sol.y[:N, :]
R_values = sol.y[N:, :]

# select fifty time points
time_indices = np.linspace(0, len(sol.t) - 1, 50, dtype=int)
C_selected = C_values[:, time_indices]
R_selected = R_values[:, time_indices]
# cumulative community CUE
#dR = R_selected.sum(axis=0) - R0.sum() + sol.t[time_indices] * 12
#dR = dR[1:] 
#dC = C_selected.sum(axis=0) - C0.sum()
#dC = dC[1:] 
# calculate community level CUE
#community_cue = dC/dR
# print output
#for i, t_idx in enumerate(time_indices[1:]):
#    print(f"Time: {sol.t[t_idx]:.2f}, community CUE: {community_cue[i]:.4f}")

# visualization
#plt.figure(figsize=(8, 5))
#plt.plot(sol.t[time_indices[1:]], community_cue, marker='o', linestyle='-', label='community CUE')
#plt.xlabel('Time')
#plt.ylabel('community CUE')
#plt.title('Cumulative Community CUE Over Time')
#plt.legend()
#plt.grid()
#plt.show()


# instantaneous 
# calculate the instataneous community CUE between adjacent time points
dR2 = (R_selected[:-1].sum(axis=0) - R_selected[0:].sum(axis=0)) 
dR2 = dR2[1:] + 12 * (sol.t[time_indices[1:]] - sol.t[time_indices[:-1]])
dC2 = C_selected[0:].sum(axis=0) - C_selected[:-1].sum(axis=0)
dC2 = dC2 [1:] 
community_cue2 = dC2 / dR2
# print output
for i, t_idx in enumerate(time_indices[1:]):
    print(f"Time: {sol.t[t_idx]:.2f}, community CUE: {community_cue2[i]:.4f}")

# visualization
plt.figure(figsize=(8, 5))
plt.plot(sol.t[time_indices[1:]], community_cue2, marker='o', linestyle='-', label='community CUE')
plt.xlabel('Time')
plt.ylabel('community CUE')
plt.title('Instataneous Community CUE Over Time')
plt.legend()
plt.grid()
plt.show()

# Integral
total_integral = np.trapz(community_cue2, sol.t[time_indices[1:]])
print(f"Total numerical integral of community CUE: {total_integral:.4f}")
# Distribution of species CUE
def identify_best_distribution(data, distributions=None):
    if distributions is None:
        distributions = ['norm', 'lognorm', 'gamma', 'expon', 'beta']
    
    best_fit = {}
    for dist in distributions:
        try:
            params = getattr(stats, dist).fit(data)
            ks_stat, p_value = stats.kstest(data, dist, args=params)
            best_fit[dist] = (ks_stat, p_value)
        except Exception as e:
            print(f"Error fitting {dist}: {e}")
            continue
    
    best_fit_sorted = sorted(best_fit.items(), key=lambda x: x[1][1], reverse=True)
    best_distribution = best_fit_sorted[0][0]
    best_p_value = best_fit_sorted[0][1][1]
    
    return best_distribution, best_p_value



plt.figure(figsize=(8, 5))
sns.histplot(CUE, bins=20, kde=True, edgecolor='black', alpha=0.7)
plt.xlabel("Species CUE")
plt.ylabel("Frequency")
plt.title("Distribution of Species Carbon Use Efficiency (CUE)")
plt.grid(True)
plt.show()


best_dist, best_p = identify_best_distribution(CUE)
print(f"Best fitting distribution: {best_dist} (p-value = {best_p:.4f})")
