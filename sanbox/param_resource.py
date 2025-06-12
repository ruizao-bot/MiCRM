import numpy as np

def modular_uptake(N, M, N_modules, s_ratio, λ_u, σ=1):
    assert N_modules <= M and N_modules <= N, "N_modules must be less than or equal to both M and N"
    
    # Ensure σ is an array of shape (N,)
    if np.isscalar(σ):
        σ = np.full(N, σ)
    
    # Baseline calculations
    sR = M // N_modules
    dR = M - (N_modules * sR)
    
    sC = N // N_modules
    dC = N - (N_modules * sC)
    
    # Get module sizes for M
    diffR = np.full(N_modules, sR, dtype=int)
    diffR[np.random.choice(N_modules, dR, replace=False)] += 1
    mR = [list(range(x, y)) for x, y in zip(np.cumsum(diffR) - diffR, np.cumsum(diffR))]
    
    # Get module sizes for N
    diffC = np.full(N_modules, sC, dtype=int)
    diffC[np.random.choice(N_modules, dC, replace=False)] += 1
    mC = [list(range(x, y)) for x, y in zip(np.cumsum(diffC) - diffC, np.cumsum(diffC))]
    
    # Initialize u_raw with uniform random values in [0,1]
    u_raw = np.random.rand(N, M)
    
    # Apply scaling to modular structure
    for x, y in zip(mC, mR):
        u_raw[np.ix_(x, y)] *= s_ratio
    
    # Standardize u_raw (zero mean, unit variance per row)
    row_mean = np.mean(u_raw, axis=1, keepdims=True) 
    row_std = np.std(u_raw, axis=1, keepdims=True) 
    u_raw = (u_raw - row_mean) / row_std  
    
    # Apply scaling with σ
    u = u_raw * σ[:, np.newaxis]  # Ensure broadcasting across columns
    
    # Normalize u to [0,1]
    u_min = np.min(u)
    u_max = np.max(u)
    u = (u - u_min) / (u_max - u_min)
    
    # Adjust row sums using λ_u
    row_sums = np.sum(u, axis=1, keepdims=True)
    u = u * (λ_u[:, np.newaxis] / row_sums)  # Scale each row to match λ_u
    
    return u



def modular_leakage(M, N_modules, s_ratio, λ_l):
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
        l[i, :] = λ_l * l[i, :] / np.sum(l[i, :])

    return l


def generate_l_tensor(N, M, N_modules, s_ratio, λ):
    l_tensor = np.array([modular_leakage(M, N_modules, s_ratio, λ) for _ in range(N)])
    return l_tensor


def compute_m(kaf0, epsilon, λ, u):
    """
    Compute the m vector for all species using a loop, ensuring that each species 
    has its own l_sum and u_sum indexed by i.

    Parameters:
    - kaf0: Scalar parameter (chi_0)
    - epsilon: (N, ) array representing species-specific modifications (epsilon_alpha)
    - u: (N, M) matrix, where u[i, :] represents uptake for species i over resources

    Returns:
    - arraym: (N, ) NumPy array containing computed m values for each species
    """
  # Sum uptake across all resources for each species
    m = kaf0 * (1 + epsilon) * (1 - λ)*np.sum(u, axis=1) # Vectorized computation
    return m
