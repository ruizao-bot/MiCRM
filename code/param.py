import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def modular_uptake(N, M, N_modules, s_ratio, rng):
    """generate a modular uptake matrix u"""
    assert N_modules <= M and N_modules <= N, "N_modules must be less than or equal to both M and N"

    sR = M // N_modules
    dR = M - (N_modules * sR)

    sC = N // N_modules
    dC = N - (N_modules * sC)

    diffR = np.full(N_modules, sR, dtype=int)
    if dR > 0:
        diffR[rng.choice(N_modules, dR, replace=False)] += 1
    mR = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffR) - diffR + 1), np.cumsum(diffR))]

    diffC = np.full(N_modules, sC, dtype=int)
    if dC > 0:
        diffC[rng.choice(N_modules, dC, replace=False)] += 1
    mC = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffC) - diffC + 1), np.cumsum(diffC))]

    u = rng.random((N, M))

    for x, y in zip(mC, mR):
        u[np.ix_(x, y)] *= s_ratio

    row_sums = np.sum(u, axis=1, keepdims=True)
    u = u / row_sums
    return u


def modular_leakage(M, N_modules, s_ratio, lam, rng):
    """generate a modular leakage matrix l"""
    assert N_modules <= M, "N_modules must be less than or equal to M"

    sR = M // N_modules
    dR = M - (N_modules * sR)

    diffR = np.full(N_modules, sR, dtype=int)
    if dR > 0:
        diffR[rng.choice(N_modules, dR, replace=False)] += 1
    mR = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffR) - diffR + 1), np.cumsum(diffR))]

    l = rng.random((M, M))

    for i, x in enumerate(mR):
        for j, y in enumerate(mR):
            if i == j or i + 1 == j:
                l[np.ix_(x, y)] *= s_ratio

    row_sums = np.sum(l, axis=1, keepdims=True)
    l = lam * l / row_sums
    return l


def generate_l_tensor(N, M, N_modules, s_ratio, lam, u, rng):
    """generate a 3D leakage tensor l"""
    l_tensor = np.zeros((N, M, M))
    for i in range(N):
        l_tensor[i] = modular_leakage(M, N_modules, s_ratio, lam, rng)
    return l_tensor


def safe_weighted_average(values, weights):
    """Compute a weighted average"""
    total_weight = np.sum(weights)
    if total_weight <= 0:
        return np.nan
    return np.sum(values * weights) / total_weight


def compute_species_CUE(u, R_ref, lam, m):
    """Compute species-level CUE"""
    total_uptake = np.sum(u * R_ref, axis=1)
    net_uptake = np.sum(u * R_ref * (1 - lam), axis=1) - m
    species_CUE = net_uptake / (total_uptake + 1e-12)
    return species_CUE


def compute_community_CUE(species_CUE, C_ref):
    """Compute community-level CUE as the weighted average of species CUE, weighted by biomass"""
    return safe_weighted_average(species_CUE, C_ref)


def compute_CUE(sol, N, u, R0, λ, m):
    """
    Compute the community Carbon Use Efficiency (CUE) based on the weighted average of species CUE.
    
    Parameters:
    sol: ODE solution object (output of solve_ivp)
    N: Number of species (consumers)
    u: Resource uptake matrix (N × M)
    R0: Initial resource concentration (M,)
    λ: Leakage rate (scalar)
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

def solve_micrm(
    N, M, u, l, m, lambda_alpha, rho, omega, C0, R0,
    t_span, t_eval=None, tol=1e-5, method='BDF'
):
    """Solve the MICRM model using solve_ivp with an event to detect equilibrium"""
    def dCdt_Rdt(t, y):
        C = y[:N]
        R = y[N:]
        uptake = u * (R * (1 - lambda_alpha))  # (N, M)
        dCdt = C * (np.sum(uptake, axis=1) - m)
        dRdt = rho - omega * R
        consumption = np.sum(C[:, None] * u * R, axis=0)  # (M,)
        dRdt -= consumption
        leakage = np.einsum('i,j,ij,ijk->k', C, R, u, l)
        dRdt += leakage
        return np.concatenate([dCdt, dRdt])

    def equilibrium_event(t, y):
        deriv = dCdt_Rdt(t, y)
        return np.max(np.abs(deriv)) - tol

    equilibrium_event.terminal = True
    equilibrium_event.direction = -1

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 100)

    Y0 = np.concatenate([C0, R0])

    sol = solve_ivp(
        dCdt_Rdt,
        t_span,
        Y0,
        t_eval=t_eval,
        method=method,
        events=equilibrium_event
    )
    return sol


def average_cosine_similarity(u):
    """
    Compute average cosine similarity (niche overlap) for a community.
    u: (N_species, N_resources) uptake matrix
    Returns: scalar average cosine similarity
    """
    N = u.shape[0]
    if N < 2:
        return np.nan
    # Normalize each row (species vector)
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_norm = u / (norms)
    # Cosine similarity matrix
    sim_matrix = np.dot(u_norm, u_norm.T)
    # Exclude diagonal, average over all pairs
    mask = ~np.eye(N, dtype=bool)
    avg_sim = np.sum(sim_matrix[mask]) / (N * (N - 1))
    return avg_sim


def calculate_effective_leakage(u, l):
    """Calculate effective leakage for each species as the sum of leakage contributions from all resources."""
    return np.einsum('ia,iab->ib', u, l)

def calculate_community_feedback(L_eff, u):
    """
    Calculate community-level facilitation as the mean over species of total leakage per species.
    
    Parameters:
    L_eff: effective leakage matrix (N x M)
    u: uptake matrix (N x M) (not used)
    
    Returns:
    F: community-level facilitation (scalar)
    """
    return np.mean(np.sum(L_eff, axis=1))

def community_level_competition(u):
    """calculate community-level competition based on average cosine similarity between species' uptake vectors"""
    N, _ = u.shape
    if N < 2:
        return np.nan

    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_normalized = u / (norms + 1e-10)
    similarity = u_normalized @ u_normalized.T

    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            total += similarity[i, j]

    return 2 * total / (N * (N - 1))


def species_level_competition(u):
    """calculate species-level competition based on average cosine similarity to other species"""
    N, _ = u.shape
    if N < 2:
        return np.full(N, np.nan)

    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_normalized = u / (norms + 1e-10)
    similarity = u_normalized @ u_normalized.T
    np.fill_diagonal(similarity, 0.0)

    comp = np.sum(similarity, axis=1) / (N - 1)
    return comp


def species_level_competition_dot(u):
    """calculate species-level competition based on dot product (CCT)"""
    N, _ = u.shape
    if N < 2:
        return np.full(N, np.nan)

    comp_matrix = u @ u.T
    np.fill_diagonal(comp_matrix, 0.0)
    comp = np.sum(comp_matrix, axis=1) / (N - 1)
    return comp


def compute_uptake_variance(u):
    """Calculate the variance of uptake for each species across resources."""
    return np.var(u, axis=1)

def compute_uptake_variance(u):
    """Calculate the variance of uptake for each species across resources."""
    return np.var(u, axis=1)


def extract_state_at_target_time(sol, N, omega):
    """Get consumer abundances and resource concentrations at t = 1 / omega"""
    t_target = t_eff / np.mean(omega)
    idx = np.argmin(np.abs(sol.t - t_target))
    C_at_t = sol.y[:N, idx]
    R_at_t = sol.y[N:, idx]
    return C_at_t, R_at_t


def choose_resources_for_second_community(M_pool, M1, M2, resource_indices1, rng):
    """Select resource indices for the second community based on the first community's resources and the desired number of resources M2."""
    if M1 > M2:
        return rng.choice(resource_indices1, M2, replace=False)
    if M1 < M2:
        remaining_resources = np.setdiff1d(np.arange(M_pool), resource_indices1)
        additional_resources = rng.choice(remaining_resources, M2 - M1, replace=False)
        return np.concatenate([resource_indices1, additional_resources])
    return resource_indices1.copy()
