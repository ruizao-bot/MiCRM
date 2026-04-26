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
    t_span, t_eval=None, tol=1e-5, method='BDF', n_save_points=100
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

    if t_eval is None and n_save_points is not None and n_save_points > 0:
        t_eval = np.linspace(t_span[0], t_span[1], int(n_save_points))

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
def calculate_elv_params(C_hat, R_hat, N, M, u, l_tensor, m, rho, omega, lambda_vec):
    C_u = C_hat[:, None] * u  # (N, M)

    L = np.einsum('ib,iab->ab', C_u, l_tensor, optimize='optimal')  # (M, M)

    diag = omega + np.sum(C_u, axis=0)  # (M,)
    D = np.diag(diag) - L               # (M, M)

    term1 = -u * R_hat  # (N, M)
    term2 = np.einsum('ib,iab->ia', u * R_hat, l_tensor, optimize='optimal')  # (N, M)
    V = term1 + term2  # (N, M)

    partial_R_C = np.linalg.solve(D, V.T)  # (M, N)

    row_scale = (1 - lambda_vec)[:, None]        # (N,1)
    weighted_u = row_scale * u                   # (N,M)

    alpha = weighted_u @ partial_R_C             # (N,N)

    growth_terms = (1 - lambda_vec) * np.sum(u * R_hat, axis=1)
    interaction_terms = alpha @ C_hat
    r = growth_terms - m - interaction_terms

    return alpha, r

def dCdt_elv(t, C, r, alpha):
    return C * (r + alpha @ C)

def make_micrm_equilibrium_event(N, M, u, l, m, rho, omega, lambda_vec, tol=1e-5):
    def equilibrium_event(t, y):
        deriv = dCdt_Rdt(t, y, N, M, u, l, m, rho, omega, lambda_vec)
        return np.max(np.abs(deriv)) - tol
    equilibrium_event.terminal = True
    equilibrium_event.direction = -1
    return equilibrium_event

def make_elv_equilibrium_event(r, alpha, tol=1e-5):
    def event(t, C):
        deriv = dCdt_elv(t, C, r, alpha)
        return np.max(np.abs(deriv)) - tol
    event.terminal = True
    event.direction = -1
    return event

def LV_jac(alpha, C_eq, threshold=1e-5):
    mask = C_eq > threshold
    if not np.any(mask):
        raise ValueError("No feasible species above threshold.")
    C = C_eq[mask]
    A = alpha[np.ix_(mask, mask)]
    return np.diag(C) @ A

def MiCRM_jac(N, M, u, l, m, rho, omega, lambda_vec, sol):
    state = sol.y[:, -1]
    C = state[:N]
    R = state[N:]

    # CC block
    cc_diag = -m + ((1 - lambda_vec)[:, None] * u * R[None, :]).sum(axis=1)
    CC = np.diag(cc_diag)

    # CR block
    CR = C[:, None] * (1 - lambda_vec)[:, None] * u

    # RR block
    P = C[:, None, None] * u[:, :, None] * l
    RR = P.sum(axis=0).T
    diag_val = np.diag(RR)
    sub_diag = (C[:, None] * u).sum(axis=0)
    diag_rr = diag_val - sub_diag - omega
    np.fill_diagonal(RR, diag_rr)

    # RC block
    Q = u * R[None, :]
    Ql = Q[:, :, None] * l
    term2 = Ql.sum(axis=1)
    RC = (term2 - Q).T

    top    = np.hstack([CC, CR])
    bottom = np.hstack([RC, RR])
    return np.vstack([top, bottom])

def leading_eigenvalue(J):
    eigvals = np.linalg.eigvals(J)
    re = np.real(eigvals)
    return float(re[np.argmax(re)])

def leading_hermitian_eigenvalue(J):
    H = (J + J.T) / 2
    eigvals = np.linalg.eigvalsh(H)
    return np.max(eigvals)


def feasibility_prob(alpha, eps=0.0, maxpts=None, abseps=1e-6, releps=0):
    """
    Compute P[X >= 0] using R mvtnorm::pmvnorm (Genz–Bretz algorithm),
    where X ~ N(0, Sigma) and Sigma = (A^-1)(A^-1)^T.
    """
    # Lazy import rpy2 to avoid ModuleNotFoundError when rpy2 is not installed
    import rpy2.robjects as ro
    from rpy2.robjects import FloatVector
    from rpy2.robjects.packages import importr
    
    S = alpha.shape[0]
    if S == 0:
        return 0.0
    A = alpha.copy()
    if eps > 0.0:
        A = A + eps * np.eye(S)
    try:
        X = np.linalg.solve(A, np.eye(S))
    except np.linalg.LinAlgError:
        return 0.0
    Sigma = X @ X.T
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = Sigma + 1e-10 * np.eye(S)

    mvtnorm = importr('mvtnorm')
    r = ro.r
    lower = FloatVector([0.0] * S)
    upper = FloatVector([r('Inf')[0]] * S)
    mean = FloatVector([0.0] * S)
    r_sigma = r['matrix'](ro.FloatVector(Sigma.flatten('F')), nrow=S, ncol=S)
    if maxpts is None:
        maxpts = int(max(250000, 10000 * S))
    algo = mvtnorm.GenzBretz(maxpts=maxpts, abseps=abseps, releps=releps)
    res = mvtnorm.pmvnorm(lower=lower, upper=upper, mean=mean,
                          sigma=r_sigma, algorithm=algo)
    return float(res[0])


# =============================================================================
# Feasibility and Stability Functions
# =============================================================================

def build_effective_competition_matrix(u, R_eq):
    """
    Build the effective Lotka-Volterra competition matrix from MiCRM.

    In MacArthur-style consumer-resource models, the effective competition
    between species i and j mediated by resource k is:
        alpha_ij = sum_k (u_ik * u_jk) / K_k
    where K_k is the equilibrium resource concentration (proxy for supply).

    Parameters
    ----------
    u     : (N, M) uptake matrix for surviving species
    R_eq  : (M,)   equilibrium resource concentrations from simulation

    Returns
    -------
    alpha : (N, N) effective competition matrix (off-diagonal negative,
                   diagonal set to -1 for self-regulation)
    """
    # Avoid division by zero: replace zero resource levels with a small value
    R_safe = np.where(R_eq > 1e-12, R_eq, 1e-12)
    # alpha_ij = sum_k u_ik * u_jk / R_k  →  matrix form: U @ diag(1/R) @ U.T
    alpha = u @ np.diag(1.0 / R_safe) @ u.T
    # Negate because competition reduces growth, then set diagonal to -1
    alpha = -alpha
    np.fill_diagonal(alpha, -1.0)
    return alpha


def compute_stability(alpha, C_eq):
    """
    Compute local asymptotic stability via the community (Jacobian) matrix.

    J_ij = C_i* * alpha_ij   (standard LV Jacobian at equilibrium)

    Returns
    -------
    lambda_max : float  largest real part of eigenvalues.
                        Negative → locally stable.
    """
    from scipy.linalg import eigvals
    J = np.diag(C_eq) @ alpha
    eigenvalues = eigvals(J)
    return float(np.max(eigenvalues.real))


def compute_feasibility(alpha, n_samples=10000):
    """
    Compute feasibility probability using multivariate normal distribution.
    
    Uses R's mvtnorm package (Genz-Bretz algorithm) to compute P[X >= 0]
    where X ~ N(0, Sigma) and Sigma = (A^{-1})(A^{-1})^T.
    
    This is more accurate than Monte Carlo sampling and follows the approach
    in Song et al. (2018).

    Parameters
    ----------
    alpha     : (S, S) interaction matrix (must be invertible)
    n_samples : int    ignored (kept for compatibility)

    Returns
    -------
    feas_prob : float  feasibility probability (0 – 1 scale)
    """
    S = alpha.shape[0]
    if S < 2:
        return np.nan
    
    # Use the existing feasibility_prob function with mvtnorm
    try:
        prob = feasibility_prob(alpha, eps=0.0, maxpts=None, abseps=1e-6, releps=0)
        return float(prob)
    except Exception:
        return np.nan


def compute_feasibility_domain_shape(alpha):
    """
    Compute the heterogeneity of the feasibility domain shape following
    Grilli et al. (2017): measure the variance of pairwise side lengths
    of the convex polyhedral cone (feasibility domain).

    Side length between species i and j is defined via the cosine of the
    angle between the corresponding rows of -A^{-1}:
        cos(eta_ij) = (-A^{-1})_i · (-A^{-1})_j
                      / (|(-A^{-1})_i| * |(-A^{-1})_j|)

    A larger variance indicates a more anisotropic domain, meaning the
    community is more sensitive to perturbations in certain directions.

    Returns
    -------
    mean_cos  : float  mean of cosines across all species pairs
    std_cos   : float  std of cosines (heterogeneity measure)
    """
    S = alpha.shape[0]
    if S < 2:
        return np.nan, np.nan

    try:
        A_inv = -np.linalg.inv(alpha)   # rows correspond to species
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    # Normalise rows to unit vectors
    norms = np.linalg.norm(A_inv, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1e-12)
    A_inv_norm = A_inv / norms

    # Cosine matrix
    cos_matrix = A_inv_norm @ A_inv_norm.T

    # Extract upper-triangle (unique pairs, exclude diagonal)
    idx = np.triu_indices(S, k=1)
    cosines = cos_matrix[idx]

    return float(np.mean(cosines)), float(np.std(cosines))


def compute_community_metrics(u, C_eq, R_eq, survivors_idx):
    """
    Wrapper: given the full uptake matrix and equilibrium state,
    extract the survivor sub-community and compute all metrics.

    Parameters
    ----------
    u            : (N, M) full uptake matrix for this community
    C_eq         : (N,)   equilibrium species abundances
    R_eq         : (M,)   equilibrium resource concentrations
    survivors_idx: 1-D array of surviving species indices

    Returns
    -------
    dict with keys:
        lambda_max     : float  stability (leading eigenvalue of Jacobian)
        xi             : float  feasibility probability (0-1), computed via mvtnorm
        mean_cos_side  : float  mean cosine of feasibility domain angles
        std_cos_side   : float  heterogeneity of feasibility domain shape
    """
    if len(survivors_idx) < 2:
        return dict(lambda_max=np.nan, xi=np.nan,
                    mean_cos_side=np.nan, std_cos_side=np.nan)

    u_s   = u[survivors_idx, :]          # (S, M)
    C_s   = C_eq[survivors_idx]          # (S,)
    alpha = build_effective_competition_matrix(u_s, R_eq)

    lambda_max              = compute_stability(alpha, C_s)
    xi                      = compute_feasibility(alpha)
    mean_cos, std_cos       = compute_feasibility_domain_shape(alpha)

    return dict(
        lambda_max     = lambda_max,
        xi             = xi,
        mean_cos_side  = mean_cos,
        std_cos_side   = std_cos,
    )

# ===================== Adaptive ridge parameters =====================
COND_MAX = 1e10
EPS_REL_INIT = 1e-10
EPS_REL_CAP = 1e-2
EPS_GROWTH = 10.0
EPS_REFINE_STEPS = 20

def estimate_scale_for_eps(A):
    diag = np.diag(A)
    diag_abs = np.abs(diag[np.isfinite(diag)])
    diag_abs = diag_abs[diag_abs > 0]
    if diag_abs.size > 0:
        scale = float(np.median(diag_abs))
    else:
        a_abs = np.abs(A[np.isfinite(A)])
        a_abs = a_abs[a_abs > 0]
        scale = float(np.median(a_abs)) if a_abs.size > 0 else 1.0
    return max(scale, 1e-12)

def adaptive_ridge(A,
                  cond_max=COND_MAX,
                  eps_rel_init=EPS_REL_INIT,
                  eps_rel_cap=EPS_REL_CAP,
                  growth=EPS_GROWTH,
                  refine_steps=EPS_REFINE_STEPS):
    """
    Find the smallest eps_abs = eps_rel * scale such that:
      cond(A + eps_abs I) <= cond_max
    """
    S = A.shape[0]
    if S == 0:
        return None
    I = np.eye(S)
    scale = estimate_scale_for_eps(A)

    def ok_at(eps_rel):
        eps_abs = eps_rel * scale
        A_reg = A + eps_abs * I
        try:
            cond = float(np.linalg.cond(A_reg))
        except np.linalg.LinAlgError:
            return False, None
        if (not np.isfinite(cond)) or (cond > cond_max):
            return False, None
        return True, {"A_reg": A_reg, "cond": cond, "eps_abs": eps_abs, "eps_rel": eps_rel, "scale": scale}

    eps_rel = eps_rel_init
    prev = None
    ok, met = ok_at(eps_rel)
    while (not ok) and (eps_rel < eps_rel_cap):
        prev = eps_rel
        eps_rel *= growth
        ok, met = ok_at(eps_rel)

    if not ok:
        return None

    lo = prev if prev is not None else eps_rel_init / growth
    lo = max(lo, eps_rel_init / growth)
    hi = eps_rel
    best = met

    for _ in range(refine_steps):
        mid = np.sqrt(lo * hi)
        okm, metm = ok_at(mid)
        if okm:
            hi = mid
            best = metm
        else:
            lo = mid

    best["status"] = "ok"
    return best

# ===================== C2: logdet / spectral proxies + feasibility-domain mapping =====================
def compute_proxies_from_A(A_reg):
    """
    Returns:
      - logdet(A A^T) = 2 * sum(log sigma_i)
      - log_volume = -0.5 * logdet(A A^T)
      - log_sigma_min, log_inv_spec, log_inv_frob
      - sigma_min, sigma_max
    """
    s = np.linalg.svd(A_reg, compute_uv=False)
    smin = float(np.min(s))
    smax = float(np.max(s))

    tiny = 1e-300
    logdet_AAT = float(2.0 * np.sum(np.log(s + tiny)))
    log_volume = float(-0.5 * logdet_AAT)
    log_smin = float(np.log(smin + tiny))
    log_inv_spec = float(-log_smin)

    inv_frob = float(np.sqrt(np.sum(1.0 / (s * s + tiny))))
    log_inv_frob = float(np.log(inv_frob + tiny))

    return {
        "sigma_min": smin,
        "sigma_max": smax,
        "logdet_AAT": logdet_AAT,
        "log_volume": log_volume,
        "log_sigma_min": log_smin,
        "log_inv_spec": log_inv_spec,
        "log_inv_frob": log_inv_frob,
    }


def map_log_volume_to_feasible_proxies(log_volume: float, S: int):
    S = max(int(S), 1)
    lv = float(log_volume)
    feasible_volume_proxy = float(np.exp(np.clip(lv, -700, 700)))
    feasible_scale_per_dim = float(np.exp(np.clip(lv / S, -700, 700)))

    log10_feasible_volume_proxy = float(lv / np.log(10.0))
    log10_feasible_scale_per_dim = float((lv / S) / np.log(10.0))

    return {
        "feasible_volume_proxy": feasible_volume_proxy,
        "feasible_scale_per_dim": feasible_scale_per_dim,
        "log10_feasible_volume_proxy": log10_feasible_volume_proxy,
        "log10_feasible_scale_per_dim": log10_feasible_scale_per_dim,
    }