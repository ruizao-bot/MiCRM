import numpy as np
from scipy.stats import dirichlet

def modular_uptake(N, M, **kw):
    if 'T' in kw:
        u_sum = kw['tt'][:, 0] 
    else:
        u_sum = np.full(M, 2.5)
    
    diri = dirichlet.rvs([1]*M, size=N).T 
    u = diri * u_sum[:, np.newaxis]
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

