import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def modular_uptake(N, M, N_modules, s_ratio):
    assert N_modules <= M and N_modules <= N, "N_modules must be less than or equal to both M and N"

    # Baseline calculations
    sR = M // N_modules
    dR = M - (N_modules * sR)

    sC = N // N_modules
    dC = N - (N_modules * sC)

    # Get module sizes for M
    diffR = np.full(N_modules, sR, dtype=int)
    diffR[np.random.choice(N_modules, dR, replace=False)] += 1
    mR = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffR) - diffR + 1), np.cumsum(diffR))]

    # Get module sizes for N
    diffC = np.full(N_modules, sC, dtype=int)
    diffC[np.random.choice(N_modules, dC, replace=False)] += 1
    mC = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffC) - diffC + 1), np.cumsum(diffC))]

    # Preallocate u matrix
    u = np.random.rand(N, M)

    # Apply scaling
    for x, y in zip(mC, mR):
        u[np.ix_(x, y)] *= s_ratio
        
    # Normalize each row
    for i in range(N):
        u[i, :] /= np.sum(u[i, :])
    return u


def modular_leakage(M, N_modules, s_ratio, λ):
    assert N_modules <= M, "N_modules must be less than or equal to M"

    # Baseline
    sR = M // N_modules
    dR = M - (N_modules * sR)

    # Get module sizes and add to make to M
    diffR = np.full(N_modules, sR, dtype=int)
    diffR[np.random.choice(N_modules, dR, replace=False)] += 1
    mR = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffR) - diffR + 1), np.cumsum(diffR))]

    l = np.random.rand(M, M)

    for i, x in enumerate(mR):
        for j, y in enumerate(mR):
            if i == j or i + 1 == j:
                l[np.ix_(x, y)] *= s_ratio

    for i in range(M):
        l[i, :] = λ * l[i, :] / np.sum(l[i, :])

    return l


def generate_l_tensor(N, M, N_modules, s_ratio, λ):
    l_tensor = np.array([modular_leakage(M, N_modules, s_ratio, λ) for _ in range(N)])
    return l_tensor

def compute_CUE(sol, N, u, R0, l, m):
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
    net_uptake = np.sum(u * R0 * (1 - np.sum(l, axis=1)), axis=1) - m  # Shape (N,)
    
    # Compute species-level CUE
    species_CUE = net_uptake / total_uptake  # Shape (N,)

    # Compute community CUE as the weighted average of species CUE
    community_CUE = np.sum(C_values * species_CUE) / np.sum(C_values)

    return community_CUE, species_CUE

def solve_micrm(
    N, M, u, l, m, lambda_alpha, rho, omega, C0, R0,
    t_span, t_eval=None, tol=1e-5, method='BDF'
):
    """
    Integrate the MiCRM ODEs until equilibrium or t_span is reached.

    Parameters:
        N, M: int
            Number of consumers and resources.
        u, l, m, lambda_alpha, rho, omega: model parameters.
        C0, R0: initial conditions for consumers and resources.
        t_span: tuple
            Time span for integration (default: (0, 1000)).
        t_eval: array or None
            Time points to evaluate solution (default: 300 points in t_span).
        tol: float
            Tolerance for equilibrium detection.
        method: str
            Integration method for solve_ivp.

    Returns:
        sol: OdeResult
            Solution object from scipy.integrate.solve_ivp.
    """
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
        dCdt_Rdt, t_span, Y0, t_eval=t_eval, method=method,
        events=equilibrium_event
    )
    return sol

def compute_alpha_r(C_hat, R_hat, N, M, u, l, m, lambda_alpha, omega):

    D = np.diag(omega + np.sum(C_hat[:, np.newaxis] * u, axis=0))
    D -= np.einsum('i,ig,iag->ag', C_hat, u * R_hat, l)
    partial_R_C = np.zeros((M, N))
    for j in range(N):
        v_j = -R_hat * u[j] + np.einsum('b,b,ba->a', R_hat, u[j], l[j])
        partial_R_C[:, j] = np.linalg.solve(D, v_j)
    alpha = np.einsum('ia,a,aj->ij', u, 1 - lambda_alpha, partial_R_C)
    r = np.sum(u * (1 - lambda_alpha) * R_hat, axis=1) - m - np.sum(alpha * C_hat, axis=1)
    return alpha, r


def solve_elv(alpha, r, C0, t_span=(0, 50000), t_eval=None):
    N = len(C0)
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 300)

    def dCdt_elv(t, C):
        dCdt = np.zeros(N)
        for i in range(N):
            dCdt[i] = C[i] * (r[i] + sum(alpha[i, j] * C[j] for j in range(N)))
        return dCdt

    sol = solve_ivp(dCdt_elv, t_span, C0, t_eval=t_eval, method="BDF")
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
    u_norm = u / (norms + 1e-12)
    # Cosine similarity matrix
    sim_matrix = np.dot(u_norm, u_norm.T)
    # Exclude diagonal, average over all pairs
    mask = ~np.eye(N, dtype=bool)
    avg_sim = np.sum(sim_matrix[mask]) / (N * (N - 1))
    return avg_sim


def depletion_steady(rho, omega, R_final):

    inflow_rate   = rho.sum()          
    outflow_rate  = np.sum(omega * R_final) 

    # ---- 消耗比例 ----
    depletion_frac = 1.0 - outflow_rate / (inflow_rate)
    return depletion_frac

def calculate_effective_leakage(u, l):
    """
    Calculate effective leakage vector for each consumer.
    
    L_i^eff = Σ_{α=1}^{M} u_i_α * l_i_α
    
    Parameters:
    u: uptake matrix (N x M)
    l: leakage tensor (N x M x M)
    
    Returns:
    L_eff: effective leakage matrix (N x M)
    """
    N, M = u.shape
    L_eff = np.zeros((N, M))
    
    for i in range(N):
        for alpha in range(M):
            # L_i^eff = Σ_{α=1}^{M} u_i_α * l_i_α
            L_eff[i] += u[i, alpha] * l[i, alpha, :]
    
    return L_eff

def calculate_community_feedback(L_eff, u):
    N, M = L_eff.shape
    total_similarity = 0.0
    
    for i in range(N):
        for j in range(N):
            if i != j:
                dot_product = np.dot(L_eff[i], u[j])
                norm_L_eff = np.linalg.norm(L_eff[i])
                norm_u_j = np.linalg.norm(u[j])
                
                if norm_L_eff > 0 and norm_u_j > 0:
                    cosine_sim = dot_product / (norm_L_eff * norm_u_j)
                    total_similarity += cosine_sim
    

    if N > 1:
        C_feed = (1 * total_similarity) / (N * (N - 1))
    else:
        C_feed = 0.0
    
    return C_feed