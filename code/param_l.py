import numpy as np


def modular_uptake(N, M, N_modules, s_ratio, λ_u):
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
    u_raw = np.random.rand(N, M)
    
    # Apply scaling
    for x, y in zip(mC, mR):
        u_raw[np.ix_(x, y)] *= s_ratio

    # Normalize each row and apply λ scaling
    for i in range(N):
        u_raw[i, :] = u_raw[i, :] / np.sum(u_raw[i, :])
        u = u_raw * 2.5 * λ_u[:, np.newaxis]
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

