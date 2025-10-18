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
# default values (will be overwritten by sweep)
N_modules = 2  # Number of modules (default)
s_ratio = 10 # Modularity ratio
N1 = 100
M1 = 50
m1 = np.full(N1, 0.2)
N2 = 100
M2 = 50
m2 = np.full(N2, 0.2)
DO_PLOT = False

# set rho and omega for the resource pool (constant across runs)
rho_pool = np.full(M_pool, 0.5)
omega_pool = np.full(M_pool, 0.5)


def run_coalescence(N_modules_run, seed=None, do_plot=False):
    """Run one coalescence simulation with the given N_modules and return richness summary.

    Returns a dict with keys: N_modules, n_surv1, n_surv2, n_surv3
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate uptake/leakage pool for this N_modules
    u_pool = param.modular_uptake(N_pool, M_pool, N_modules_run, s_ratio)
    l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules_run, s_ratio, λ, u_pool)

    # Community 1 sampling
    species_indices1 = np.random.choice(N_pool, N1, replace=False)
    resource_indices1 = np.random.choice(M_pool, M1, replace=False)
    # Community 2 sampling
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

    # Shared resource assumption: choose shared resource indices
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    M3 = len(resource_indices3)

    # Build community matrices using the shared resource columns (extract from pool)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]

    # Combine for community 3 by stacking species (they share the same M3 resource columns)
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]

    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    m3 = np.concatenate([m1, m2])
    lambda_alpha3 = np.full(M3, λ)

    omega3 = omega_pool[resource_indices3]
    rho3 = rho_pool[resource_indices3]
    N3 = N1 + N2

    # Initial conditions and run single-community dynamics
    C0_1 = np.full(N1, 0.01)
    C0_2 = np.full(N2, 0.01)
    R0_1 = np.full(M1, 1)
    R0_2 = np.full(M2, 1)
    t_span = (0, 100000)
    sol1 = param.solve_micrm(N1, M1, u1, l1, m1, np.full(M1, λ), rho1:=rho_pool[resource_indices3][:M1], omega1:=omega_pool[resource_indices3][:M1], C0_1, R0_1, t_span)
    ce1 = sol1.y[:N1, -1]
    sol2 = param.solve_micrm(N2, M2, u2, l2, m2, np.full(M2, λ), rho2:=rho_pool[resource_indices3][:M2], omega2:=omega_pool[resource_indices3][:M2], C0_2, R0_2, t_span)
    ce2 = sol2.y[:N2, -1]

    # Combined initial conditions for coalescence
    C0_3 = np.concatenate([ce1, ce2])
    R0_3 = np.full(M3, 1)

    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)

    # Survival counts
    SURV_THRESH = 1e-5
    ce3 = sol3.y[:N3, -1]
    n_surv1 = int(np.sum(ce1 > SURV_THRESH))
    n_surv2 = int(np.sum(ce2 > SURV_THRESH))
    n_surv3 = int(np.sum(ce3 > SURV_THRESH))

    # Optionally plot (disabled during sweep)
    if do_plot:
        pass  # plotting code can be added here if needed

    return {"N_modules": N_modules_run, "n_surv1": n_surv1, "n_surv2": n_surv2, "n_surv3": n_surv3}


# Run sweep over N_modules
results = []
for nm in range(1, 51):
    print(f"Running coalescence with N_modules={nm}")
    res = run_coalescence(nm, seed=None, do_plot=False)
    # print richness (number of surviving species) for this run
    print(f"  Richness -> Community1: {res['n_surv1']}, Community2: {res['n_surv2']}, Community3: {res['n_surv3']}")
    results.append(res)

# Save sweep results
os.makedirs(results_dir, exist_ok=True)
df_sweep = pd.DataFrame(results)
df_sweep.to_csv(os.path.join(results_dir, "coalescence_survival_sweep.csv"), index=False)

# End of sweep: if you want to run the full plotting for a particular N_modules, call run_coalescence with do_plot=True

# EXIT: stop executing the original plotting blocks below unless DO_PLOT is True
if not DO_PLOT:
    # exit early to avoid plotting 50 figure sets
    sys.exit(0)

