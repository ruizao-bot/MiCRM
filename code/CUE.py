import numpy as np
# # Species CUE
# def compute_CUE_0(u, R0, lambda_alpha, m):
#     """
#     Compute Carbon Use Efficiency (CUE)
    
#     Parameters:
#     u : ndarray (N, M)  # Resource uptake matrix
#     R0 : ndarray (M,)   # Initial resource supply
#     lambda_alpha : ndarray (N, M)  # Leakage coefficient
#     m : ndarray (N,)  # Metabolic loss
    
#     Returns:
#     CUE : ndarray (N,)  # Carbon Use Efficiency
#     """
#     total_uptake = np.sum(u * R0, axis=1)  # Compute total uptake
#     net_uptake = np.sum(u * R0 * (1 - lambda_alpha), axis=1) - m  # Compute net uptake
#     CUE = net_uptake / total_uptake  # Compute CUE
#     return CUE

# # cumulative CUE
# def compute_community_CUE1(sol, N, rho, num_points=50, return_all=False):
#     """
#     Compute the instantaneous Community Carbon Use Efficiency.

#     Parameters:
#     sol : scipy.integrate.solve_ivp solution object
#         Solution object containing time points and state variables.
#     N : int
#         Number of consumers in the community.
#     rho : numpy array
#         Resource input rates for the community.
#     num_points : int, optional
#         Number of time points to sample for computation (default is 50).

#     Returns:
#     time_indices : numpy array
#         Indices of the selected time points.
#     community_cue : numpy array
#         Instantaneous community CUE values.
#     total_integral : float
#         Numerical integral of community CUE over time.
#     """
#     # Extract consumer (C) and resource (R) values from the solution
#     C_values = sol.y[:N, :]
#     R_values = sol.y[N:, :]

#     # Select time points evenly spaced across the time span
#     time_indices = np.linspace(0, len(sol.t) - 1, num_points, dtype=int)
#     C_selected = C_values[:, time_indices]
#     R_selected = R_values[:, time_indices]

#     # Compute resource change (dR) between adjacent time points
#     dR = R_selected.sum(axis=0)[:-1] - R_selected.sum(axis=0)[1:]
#     dR = dR + np.sum(rho) * (sol.t[time_indices[1:]] - sol.t[time_indices[:-1]])

#     # Compute biomass change (dC) between adjacent time points
#     dC = C_selected.sum(axis=0)[1:] - C_selected.sum(axis=0)[:-1]

#     # Compute instantaneous community CUE
#     community_cue = dC / dR

#     # Compute numerical integration of community CUE over time
#     total_integral = np.trapz(community_cue, sol.t[time_indices[1:]])

#     if return_all:
#         return time_indices, community_cue, total_integral
#     else:
#         return total_integral



# Weighted average
def compute_CUE(sol, N, u, R0, λ, m):
    """
    Compute the community Carbon Use Efficiency (CUE) based on the weighted average of species CUE.
    
    Parameters:
    sol: ODE solution object (output of solve_ivp)
    N: Number of species (consumers)
    u: Resource uptake matrix (N × M)
    R0: Initial resource concentration (M,)
    leakge rate: Leakage fraction for each species N (M,M)
    m: Maintenance cost for each species (N,)

    Returns:
    community_CUE: The weighted average of species CUE
    species_CUE: Individual CUE for each species (N,)
    """

    # Extract the steady-state biomass (last time point)
    C_values = sol.y[:N, -1]  # Shape (N,)

    # Compute total resource uptake per species
    total_uptake = np.sum(u * R0, axis=1)  # Shape (N,)
    
    # Compute net resource uptake (adjusted for leakage and metabolism)
    net_uptake = np.sum(u * R0 * (1 - λ), axis=1) - m  # Shape (N,)
    
    # Compute species-level CUE
    species_CUE = net_uptake / total_uptake  # Shape (N,)

    # Compute community CUE as the weighted average of species CUE
    community_CUE = np.sum(C_values * species_CUE) / np.sum(C_values)

    return community_CUE, species_CUE
