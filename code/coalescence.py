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
N_pool = 1000  # Species pool size
M_pool = 200    # Resource pool size
λ = 0.2        # Total leakage rate
N_modules = 5  # Number of modules
s_ratio = 10 # Modularity ratio
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
t_span = (0, 5000)


# Simulate Community 1
C0_1 = np.full(N1, 0.01)  # Initial consumer abundance
C0_2 = np.full(N1, 0.01) 
#R0 = np.full(M1, 1)        # Initial resource abundance
R0_1 = np.full(M1, 1)
R0_2 = np.full(M2, 1)
sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)
ce1 = sol1.y[:N1, -1]  # Consumer abundance at equilibrium
re1 = sol1.y[N1:, -1]  # Resource abundance at equilibrium

# Simulate Community 2
sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2,  R0_2, t_span)
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
C0_3 = np.concatenate([ce1, ce2])
R0_3 = np.full(M1, 1)#re1 + re2

sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)
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
# #############################################

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
# colors_res = cm.Reds(np.linspace(0.4, 1, M1))
# for j in range(M1):
#     axes[0].plot(sol1.t, sol1.y[N1 + j], color=colors_res[j], alpha=0.8, linewidth=1)
# axes[0].set_title("Community 1 Resources")
# axes[0].set_xlabel("Time")
# axes[0].set_ylabel("Resource abundance")
# axes[0].grid(True)

# # Community 2: Only resource dynamics
# colors_res = cm.Greens(np.linspace(0.4, 1, M2))
# for j in range(M2):
#     axes[1].plot(sol2.t, sol2.y[N2 + j], color=colors_res[j], alpha=0.8, linewidth=1)
# axes[1].set_title("Community 2 Resources")
# axes[1].set_xlabel("Time")
# axes[1].grid(True)

# # Community 3: Only resource dynamics
# colors_res = cm.Blues(np.linspace(0.4, 1, M3))
# for j in range(M3):
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

# # ---------- 图一：biomass dynamics ----------
# fig, axes = plt.subplots(1, 3, figsize=(21, 8), sharey=True)


# # Community 1
# colors1 = np.linspace(0.3, 1, N1)
# for i in range(N1):
#     axes[0].plot(sol1.t, sol1.y[i], color=cm.Reds(colors1[i]), alpha=0.8, linewidth=1)
# axes[0].set_title('Community 1 Dynamics')
# axes[0].set_xlabel('Time')
# axes[0].set_ylabel('Consumer Abundance')
# axes[0].tick_params(which='both', direction='in')
# axes[0].grid(False)

# # Community 2
# colors2 = np.linspace(0.3, 1, N2)
# for i in range(N2):
#     axes[1].plot(sol2.t, sol2.y[i], color=cm.Greens(colors2[i]), alpha=0.8, linewidth=1)
# axes[1].set_title('Community 2 Dynamics')
# axes[1].set_xlabel('Time')
# axes[1].tick_params(which='both', direction='in')
# axes[1].grid(False)

# # Community 3
# colors3_res = np.linspace(0.3, 1, N1)
# colors3_inv = np.linspace(0.3, 1, N2)
# for i in range(N1):
#     axes[2].plot(sol3.t, sol3.y[i], color=cm.Reds(colors3_res[i]), alpha=0.8, linewidth=1)
# for i in range(N2):
#     axes[2].plot(sol3.t, sol3.y[N1 + i], color=cm.Greens(colors3_inv[i]), alpha=0.8, linewidth=1)
# axes[2].set_title('Coalescence Dynamics')
# axes[2].set_xlabel('Time')
# axes[2].tick_params(which='both', direction='in')
# axes[2].grid(False)

# plt.tight_layout(rect=[0, 0, 0.85, 1])
# os.makedirs(results_dir, exist_ok=True)
# plt.savefig(os.path.join(results_dir, "biomass_dynamics.pdf"), format="pdf", bbox_inches="tight")
# plt.show()


# # ---------- 图二：resource dynamics ----------
# fig, axes = plt.subplots(1, 3, figsize=(21, 8), sharey=True)


# # Community 1 Resources
# colors_res1 = np.linspace(0.4, 1, M1)
# for j in range(M1):
#     axes[0].plot(sol1.t, sol1.y[N1 + j], color=cm.Reds(colors_res1[j]), alpha=0.8, linewidth=1)
# axes[0].set_title("Community 1 Resources")
# axes[0].set_xlabel("Time")
# axes[0].set_ylabel("Resource abundance")
# axes[0].tick_params(which='both', direction='in')
# axes[0].grid(False)

# # Community 2 Resources
# colors_res2 = np.linspace(0.4, 1, M2)
# for j in range(M2):
#     axes[1].plot(sol2.t, sol2.y[N2 + j], color=cm.Greens(colors_res2[j]), alpha=0.8, linewidth=1)
# axes[1].set_title("Community 2 Resources")
# axes[1].set_xlabel("Time")
# axes[1].tick_params(which='both', direction='in')
# axes[1].grid(False)

# # Community 3 Resources
# colors_res3 = np.linspace(0.4, 1, M3)
# for j in range(M3):
#     axes[2].plot(sol3.t, sol3.y[N3 + j], color=cm.Blues(colors_res3[j]), alpha=0.8, linewidth=1)
# axes[2].set_title("Community 3 Resources")
# axes[2].set_xlabel("Time")
# axes[2].tick_params(which='both', direction='in')
# axes[2].grid(False)

# plt.tight_layout()
# os.makedirs(results_dir, exist_ok=True)
# plt.savefig(os.path.join(results_dir, "resource_dynamics.pdf"), format="pdf", bbox_inches="tight")
# plt.show()


# ###### compare the community CUE between survival and extinction######
# sol_list = [sol1, sol2, sol3]
# N_list = [N1, N2, N3]
# u_list = [u1, u2, u3]
# R0_list = [R0_1, R0_2, R0_3]

# l_list = [l1, l2, l3]
# m_list = [m1, m2, m3]
# M_list = [M1, M2, M3]
# num_communities = len(sol_list)

# data_to_save = []

# for i in range(num_communities):
#     C_final = np.array(sol_list[i].y[:, -1])
#     t = sol_list[i].t
#     C_all = sol_list[i].y[:N_list[i], :]

#     _, species_CUE = CUE.compute_CUE(
#         sol_list[i], N_list[i], u_list[i], R0_list[i], l_list[i], m_list[i]
#     )
#     species_CUE = np.array(species_CUE, dtype=float)

#     for j in range(N_list[i]):
#         status = "Survival" if C_final[j] >= 1e-5 else "Extinction"
#         data_to_save.append({
#             "Community": i + 1,
#             "Species": f"Sp{j+1}",
#             "Status": status,
#             "CUE": species_CUE[j],
#             "C_final": C_final[j],
#         })

# df_out = pd.DataFrame(data_to_save)
# # df_out.to_csv("../data/coal_single.csv", index=False)


# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

# fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

# # Community 1: Only resource dynamics
# colors_res = cm.Blues(np.linspace(0.4, 1, M1))
# for j in range(M1):
#     axes[0].plot(sol1.t, sol1.y[N1 + j], color=colors_res[j], alpha=0.8, linewidth=1)
# axes[0].set_title("Community 1 Resources")
# axes[0].set_xlabel("Time")
# axes[0].set_ylabel("Resource abundance")
# axes[0].grid(True)

# # Community 2: Only resource dynamics
# colors_res = cm.Reds(np.linspace(0.4, 1, M2))
# for j in range(M2):
#     axes[1].plot(sol2.t, sol2.y[N2 + j], color=colors_res[j], alpha=0.8, linewidth=1)
# axes[1].set_title("Community 2 Resources")
# axes[1].set_xlabel("Time")
# axes[1].grid(True)

# # Community 3: Only resource dynamics
# colors_res = cm.Greens(np.linspace(0.4, 1, M3))
# for j in range(M3):
#     axes[2].plot(sol3.t, sol3.y[N3 + j], color=colors_res[j], alpha=0.8, linewidth=1)
# axes[2].set_title("Community 3 Resources")
# axes[2].set_xlabel("Time")
# axes[2].grid(True)

# plt.tight_layout()
# plt.show()
