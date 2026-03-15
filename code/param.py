def competition2(C):
    """
    Compute species-level competition based on the dot product (CCT).
    For each species i, compute the average dot product with all other species:
        comp_i = (1/(N-1)) * sum_{j!=i} dot(C_i, C_j)
    where dot(C_i, C_j) = sum_k C[i, k] * C[j, k]

    Args:
        C: (N, M) array, metabolic preference matrix for N species and M resources.

    Returns:
        comp: (N,) array, average competition pressure for each species.
    """
    N, M = C.shape
    if N < 2:
        # If only one species, cannot compute competition with others
        return np.full(N, np.nan)

    # 1. Compute competition matrix (dot products)
    comp_matrix = C @ C.T

    # 2. Exclude diagonal (self-competition)
    np.fill_diagonal(comp_matrix, 0.0)

    # 3. Average over other species
    comp = np.sum(comp_matrix, axis=1) / (N - 1)
    return comp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def modular_uptake(N, M, N_modules, s_ratio, rng):
    """生成模块化资源摄取矩阵 u。"""
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
    """生成模块化泄漏矩阵 l。"""
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
    """为每个物种独立生成泄漏矩阵，构成三维泄漏张量。"""
    l_tensor = np.zeros((N, M, M))
    for i in range(N):
        l_tensor[i] = modular_leakage(M, N_modules, s_ratio, lam, rng)
    return l_tensor


def compute_maintenance(chi0, epsilon, lambda_scalar, u_matrix):
    """
    Compute species maintenance cost vector using the formula:
        z_alpha = chi0 * (1 + epsilon_alpha) * (1 - lambda) * sum_j u_alpha_j

    Parameters:
    - chi0: scalar baseline maintenance coefficient
    - epsilon: array-like of length N (or scalar) with species-specific efficiency term
    - lambda_scalar: scalar leakage rate (λ)
    - u_matrix: (N, M) uptake matrix for the species set

    Returns:
    - m: numpy array of length N with maintenance costs
    """
    u = np.asarray(u_matrix, dtype=float)
    N = u.shape[0]
    eps = np.asarray(epsilon)
    if eps.ndim == 0:
        eps = np.full(N, float(eps))
    if eps.shape[0] != N:
        raise ValueError("epsilon must be scalar or have length matching number of species in u_matrix")

    m = chi0 * (1.0 + eps) * (1.0 - float(lambda_scalar)) * np.sum(u, axis=1)
    return m

def safe_weighted_average(values, weights):
    total_weight = np.sum(weights)
    if total_weight <= 0:
        return np.nan
    return np.sum(values * weights) / total_weight


def compute_species_CUE(u, R_ref, lam, m):
    total_uptake = np.sum(u * R_ref, axis=1)
    net_uptake = np.sum(u * R_ref * (1 - lam), axis=1) - m
    species_CUE = net_uptake / (total_uptake + 1e-12)
    return species_CUE


def compute_community_CUE(species_CUE, C_ref):
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
    u_norm = u / (norms)
    # Cosine similarity matrix
    sim_matrix = np.dot(u_norm, u_norm.T)
    # Exclude diagonal, average over all pairs
    mask = ~np.eye(N, dtype=bool)
    avg_sim = np.sum(sim_matrix[mask]) / (N * (N - 1))
    return avg_sim


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
    """
    Compute community-level average competition based on cosine similarity.
    
    Average cosine similarity across all pairs of species:
        C_comp = 2/(N(N-1)) * Σ_{1≤i<j≤N} cos_sim(u_i, u_j)
    where cos_sim(u_i, u_j) = (u_i · u_j) / (||u_i|| ||u_j||)
    
    Returns:
        scalar: average cosine similarity across all species pairs
    """
    N, M = u.shape
    if N < 2:
        return np.nan
    
    # Normalize each species vector
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_normalized = u / (norms + 1e-10)
    
    # Compute cosine similarity matrix
    similarity = u_normalized @ u_normalized.T
    
    # Sum over upper triangle (i < j) and multiply by 2
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            total += similarity[i, j]
    
    return 2 * total / (N * (N - 1))


def species_level_competition(u):
    """
    Compute species-level competition based on cosine similarity of uptake vectors.

    For each species i, compute average cosine similarity with all other species:
        comp_i = (1/(N-1)) * Σ_{j≠i} cos_sim(u_i, u_j)
    where cos_sim(u_i, u_j) = (u_i · u_j) / (||u_i|| ||u_j||)

    Returns:
        comp: (N,) array with species-level competition values.
    """
    N, M = u.shape
    if N < 2:
        return np.full(N, np.nan)

    # Normalize each species vector
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_normalized = u / (norms + 1e-10)  # Add small value to avoid division by zero
    
    # Compute cosine similarity matrix
    similarity = u_normalized @ u_normalized.T
    np.fill_diagonal(similarity, 0.0)
    
    # Average similarity with all other species
    comp = np.sum(similarity, axis=1) / (N - 1)
    return comp