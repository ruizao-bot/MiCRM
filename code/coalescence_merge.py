import numpy as np
import os
import sys

# Ensure the repository's `code` directory (where this file lives) is on sys.path
# This makes imports like `import param` and `import CUE` robust regardless of CWD.
script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_dir)

# Project root and results directory (absolute paths)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
results_dir = os.path.join(project_root, "results")
import param
import CUE
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl

# Parameter settings
np.random.seed(37)
λ = 0.6        # Total leakage rate
N_modules = 1  # Number of modules
s_ratio = 1 # Modularity ratio
N1 = 100
M = 50
m1 = np.full(N1, 0.5)
N2 = 100
m2 = np.full(N2, 0.5)

# Community 1
#u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
u1 = param.modular_uptake(N1, M, N_modules, s_ratio)
#l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
l1 = param.generate_l_tensor(N1, M, N_modules, s_ratio, λ)

lambda_alpha1 = np.full(M, λ)
rho1 = np.full(M, 1)
omega1 = np.full(M, 1)
# Community 2

#u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
u2 = param.modular_uptake(N2, M, N_modules, s_ratio)
#l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]
l2 = param.generate_l_tensor(N2, M, N_modules, s_ratio, λ)

lambda_alpha2 = np.full(M, λ)
rho2 = np.full(M, 1)
omega2 = np.full(M, 1)

# Time span for simulation
t_span = (0, 100000)

# Simulate Community 1
C0_1 = np.full(N1, 0.01)  # Initial consumer abundance
C0_2 = np.full(N1, 0.01) 
#R0 = np.full(M1, 1)        # Initial resource abundance
R0_1 = np.full(M, 1)
R0_2 = np.full(M, 1)
sol1 = param.solve_micrm(N1, M, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)
ce1 = sol1.y[:N1, -1]  # Consumer abundance at equilibrium
re1 = sol1.y[N1:, -1]  # Resource abundance at equilibrium

# Simulate Community 2
sol2 = param.solve_micrm(N2, M, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2,  R0_2, t_span)
ce2 = sol2.y[:N2, -1]
re2 = sol2.y[N2:, -1]

# Merge into Community 3 (shared resource columns assumption)
# Combine for community 3 by stacking species (they share the same M3 resource columns)
u3 = np.vstack([u1, u2])            # shape (N3, M3)
l3 = np.concatenate([l1, l2], axis=0)

m3 = np.concatenate([m1, m2])
lambda_alpha3 = np.full(M, λ)

omega3 = np.full(M, 1)
rho3 = np.full(M, 1)
N3 = N1 + N2

# Initial conditions: consumers from previous equilibria, resources uniform
C0_3 = np.full(N3, 0.01)#np.concatenate([ce1, ce2])
R0_3 = np.full(M, 1)  # or choose another initialization such as re1+re2

sol3 = param.solve_micrm(N3, M, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)
# ----------------- Survival rate calculation -----------------
# Species with final abundance > SURV_THRESH are considered survivors
SURV_THRESH = 1e-5
ce3 = sol3.y[:N3, -1]
# counts
n_surv1 = int(np.sum(ce1 > SURV_THRESH))
n_surv2 = int(np.sum(ce2 > SURV_THRESH))
n_surv3 = int(np.sum(ce3 > SURV_THRESH))
# rates
surv_rate1 = n_surv1 / float(N1) if N1 > 0 else 0.0
surv_rate2 = n_surv2 / float(N2) if N2 > 0 else 0.0
surv_rate3 = n_surv3 / float(N3) if N3 > 0 else 0.0

print(f"Survival threshold = {SURV_THRESH}")
print(f"Community1 survivors: {n_surv1}/{N1} => {surv_rate1:.3f}")
print(f"Community2 survivors: {n_surv2}/{N2} => {surv_rate2:.3f}")
print(f"Community3 survivors: {n_surv3}/{N3} => {surv_rate3:.3f}")

# --------------------------------------------------------------
#############################################
# import matplotlib.cm as cm

# # Plot biomass change over time with color gradient for species
# fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)

# # Community 1: blue gradient
# colors1 = cm.Reds(np.linspace(0.3, 1, N1))  # skip very light colors
# for i in range(N1):
#     axes[0].plot(sol1.t, sol1.y[i], color=colors1[i], alpha=0.8, linewidth=1)
# axes[0].set_title('Community 1 Dynamics')
# axes[0].set_xlabel('Time')
# axes[0].set_ylabel('Consumer Abundance')
# axes[0].grid(True)

# # Community 2: red gradient
# colors2 = cm.Greens(np.linspace(0.3, 1, N2))
# for i in range(N2):
#     axes[1].plot(sol2.t, sol2.y[i], color=colors2[i], alpha=0.8, linewidth=1)
# axes[1].set_title('Community 2 Dynamics')
# axes[1].set_xlabel('Time')
# axes[1].grid(True)

# # Community 3 (merged): blue for residents, red for invaders
# colors3_res = cm.Reds(np.linspace(0.3, 1, N1))
# colors3_inv = cm.Greens(np.linspace(0.3, 1, N2))
# for i in range(N1):
#     axes[2].plot(sol3.t, sol3.y[i], color=colors3_res[i], alpha=0.8, linewidth=1)
# for i in range(N2):
#     axes[2].plot(sol3.t, sol3.y[N1 + i], color=colors3_inv[i], alpha=0.8, linewidth=1)
# axes[2].set_title('Coalescence Dynamics')
# axes[2].set_xlabel('Time')
# axes[2].grid(True)

# plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.show()

# # Resource dynamics for the merged community
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

# fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

# # Community 1: Only resource dynamics
# colors_res = cm.Reds(np.linspace(0.4, 1, M))
# for j in range(M):
#     axes[0].plot(sol1.t, sol1.y[N1 + j], color=colors_res[j], alpha=0.8, linewidth=1)
# axes[0].set_title("Community 1 Resources")
# axes[0].set_xlabel("Time")
# axes[0].set_ylabel("Resource abundance")
# axes[0].grid(True)

# # Community 2: Only resource dynamics
# colors_res = cm.Greens(np.linspace(0.4, 1, M))
# for j in range(M):
#     axes[1].plot(sol2.t, sol2.y[N2 + j], color=colors_res[j], alpha=0.8, linewidth=1)
# axes[1].set_title("Community 2 Resources")
# axes[1].set_xlabel("Time")
# axes[1].grid(True)

# # Community 3: Only resource dynamics
# colors_res = cm.Blues(np.linspace(0.4, 1, M))
# for j in range(M):
#     axes[2].plot(sol3.t, sol3.y[N3 + j], color=colors_res[j], alpha=0.8, linewidth=1)
# axes[2].set_title("Community 3 Resources")
# axes[2].set_xlabel("Time")
# axes[2].grid(True)

# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np

# # 全局字体和坐标设置
# plt.rc("font", family="DejaVu Serif", size=24)
# plt.rc("xtick", direction="in")
# plt.rc("ytick", direction="in")
# plt.rc("axes", linewidth=1)

# # 自定义颜色（与 R 配色一致）
# color_map = {
#     "1": "#E74C3C",  # red
#     "2": "#2ECC71",  # green
#     "3": "#3498DB"   # blue
# }
