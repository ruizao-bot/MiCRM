from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import param

# Random seed and simulation parameters
BASE_SEED = 50
N_SIMULATIONS = 20
N_INVASIONS = 10

# Exported file names
OUTPUT_FILE = "successive_coalescence.csv"
PLOT_FILE = "successive_coalescence_trends.png"

# Species pool and resource pool parameters
N_POOL = 1000
M_POOL = 100
N_MODULES = 1
S_RATIO = 1
LEAKAGE_RATE = 0.2

# Initial community parameters
N_INITIAL = 100
M_INITIAL = 50

# Invader community parameters
N_INVADER = 50
M_INVADER = 50

# Physiological parameters
MAINTENANCE_COST = 0.2
RHO_VALUE = 0.6
OMEGA_VALUE = 0.1
T_SPAN = (0, 100000)

# Initial conditions
C0_VALUE = 0.01
R0_VALUE = 1

# Survival threshold
SURVIVAL_THRESHOLD = 1e-5


def simulate_single_invasion(resident_indices, resident_C_final, resource_indices,
                              u_pool, l_pool, invasion_num, rng,
                              used_species_set):
    """
    Simulate a single invasion event.

    FIX 2: used_species_set tracks ALL species ever present (including
            extinct ones) so they cannot re-invade.
    """

    # FIX 2: exclude all historically present species, not just current residents
    remaining_species = np.setdiff1d(np.arange(N_POOL), list(used_species_set))
    if len(remaining_species) == 0:
        return None  # Pool exhausted

    invader_indices = rng.choice(
        remaining_species,
        min(N_INVADER, len(remaining_species)),
        replace=False
    )

    # Combine resident and invader species
    combined_indices = np.concatenate([resident_indices, invader_indices])
    N_combined = len(combined_indices)
    M_combined = len(resource_indices)

    # Extract uptake and leakage matrices
    u_combined = u_pool[np.ix_(combined_indices, resource_indices)]
    l_combined = l_pool[np.ix_(combined_indices, resource_indices, resource_indices)]

    # Set parameters
    lambda_alpha = np.full(M_combined, LEAKAGE_RATE)
    rho = np.full(M_combined, RHO_VALUE)
    omega = np.full(M_combined, OMEGA_VALUE)

    # Initial conditions: residents keep their abundance, invaders start low
    C0_combined = np.concatenate([
        resident_C_final,
        np.full(len(invader_indices), C0_VALUE)
    ])
    R0_combined = np.full(M_combined, R0_VALUE)

    # Solve dynamics
    sol = param.solve_micrm(N_combined, M_combined, u_combined, l_combined,
                            MAINTENANCE_COST, lambda_alpha, rho, omega,
                            C0_combined, R0_combined, T_SPAN)

    C_final = sol.y[:N_combined, -1]
    R_final = sol.y[N_combined:, -1]

    return combined_indices, C_final, R_final, u_combined, l_combined, sol


def simulate(seed):
    """Simulate successive coalescence events for a single seed."""
    rng = np.random.default_rng(seed)

    # Generate pool-level uptake and leakage matrices
    u_pool = param.modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = param.generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO,
                                     LEAKAGE_RATE, u_pool, rng)

    # Initialize the first community
    species_indices = rng.choice(N_POOL, N_INITIAL, replace=False)
    resource_indices = rng.choice(M_POOL, M_INITIAL, replace=False)

    u = u_pool[np.ix_(species_indices, resource_indices)]
    l = l_pool[np.ix_(species_indices, resource_indices, resource_indices)]

    lambda_alpha = np.full(M_INITIAL, LEAKAGE_RATE)
    rho = np.full(M_INITIAL, RHO_VALUE)
    omega = np.full(M_INITIAL, OMEGA_VALUE)
    C0 = np.full(N_INITIAL, C0_VALUE)
    R0 = np.full(M_INITIAL, R0_VALUE)

    sol = param.solve_micrm(N_INITIAL, M_INITIAL, u, l, MAINTENANCE_COST,
                            lambda_alpha, rho, omega, C0, R0, T_SPAN)

    current_indices = species_indices
    current_C_final = sol.y[:N_INITIAL, -1]
    current_R_final = sol.y[N_INITIAL:, -1]
    current_u = u
    current_l = l
    current_sol = sol

    # FIX 2: track ALL species ever used (including extinct ones)
    used_species_set = set(species_indices.tolist())

    all_data = []

    # Record initial community (invasion 0)
    all_data.extend(
        record_community_data(
            seed, 0, current_indices, current_C_final, current_R_final,
            current_u, current_l, current_sol,
            lambda_alpha, rho, omega,
            N_INITIAL, M_INITIAL, 0, 0
        )
    )

    # Successive invasions
    for invasion_num in range(1, N_INVASIONS + 1):
        # Only keep surviving species before next invasion
        survivors_mask = current_C_final > SURVIVAL_THRESHOLD
        current_indices = current_indices[survivors_mask]
        current_C_final = current_C_final[survivors_mask]

        if len(current_indices) == 0:
            break

        result = simulate_single_invasion(
            current_indices, current_C_final, resource_indices,
            u_pool, l_pool, invasion_num, rng,
            used_species_set          # FIX 2: pass full history
        )

        if result is None:
            print(f"Seed {seed}: species pool exhausted at invasion {invasion_num}")
            break

        new_indices, new_C_final, new_R_final, new_u, new_l, new_sol = result

        # FIX 2: add newly introduced species to the historical set
        used_species_set.update(new_indices.tolist())

        N_new = len(new_indices)
        M_new = len(resource_indices)
        lambda_alpha_new = np.full(M_new, LEAKAGE_RATE)
        rho_new = np.full(M_new, RHO_VALUE)
        omega_new = np.full(M_new, OMEGA_VALUE)

        n_resident = len(current_indices)
        n_invader = N_new - n_resident

        all_data.extend(
            record_community_data(
                seed, invasion_num, new_indices, new_C_final,
                new_R_final, new_u, new_l, new_sol,
                lambda_alpha_new, rho_new, omega_new,
                N_new, M_new, n_resident, n_invader
            )
        )

        # Update current state
        current_indices = new_indices
        current_C_final = new_C_final
        current_R_final = new_R_final
        current_u = new_u
        current_l = new_l
        current_sol = new_sol

    return all_data


def record_community_data(seed, invasion_num, species_indices, C_final, R_final,
                          u, l, sol, lambda_alpha, rho, omega, N, M,
                          n_resident, n_invader):
    """
    Record community-level and species-level data.

    FIX 4: Jacobian is computed only over surviving species, avoiding
            spurious eigenvalue inflation from near-zero abundances.
    """

    survivors_mask = C_final > SURVIVAL_THRESHOLD
    n_survivors = int(np.sum(survivors_mask))

    # Species CUE
    R0 = np.full(M, R0_VALUE)
    species_CUE = param.compute_species_CUE(u, R0, lambda_alpha, MAINTENANCE_COST)

    # Community CUE (abundance-weighted over survivors)
    community_CUE = param.safe_weighted_average(
        species_CUE[survivors_mask], C_final[survivors_mask]
    )

    # Interaction quantities
    L_eff = param.calculate_effective_leakage(u, l)
    facilitation = np.mean(L_eff, axis=1)
    competition_comm = param.community_level_competition(u)
    competition_species = param.species_level_competition(u)
    competition_dot = param.species_level_competition_dot(u)

    uptake_var = param.compute_uptake_variance(u)
    depletion = np.sum(R_final)
    total_abundance = np.sum(C_final)

    if invasion_num > 0:
        resident_abundance = np.sum(C_final[:n_resident])
        invader_abundance = np.sum(C_final[n_resident:])
    else:
        resident_abundance = total_abundance
        invader_abundance = 0.0

    # Feasibility — restricted to survivor sub-matrix
    lambda_vec = np.full(N, LEAKAGE_RATE)
    alpha, r = param.calculate_elv_params(
        C_final, R_final, N, M, u, l,
        MAINTENANCE_COST, rho, omega, lambda_vec
    )
    feas = param.feasibility_prob(
        alpha[np.ix_(survivors_mask, survivors_mask)], eps=0.0
    )

    # FIX 4: Jacobian restricted to surviving species only
    if n_survivors > 0:
        u_surv = u[survivors_mask]
        l_surv = l[survivors_mask]
        lambda_vec_surv = np.full(n_survivors, LEAKAGE_RATE)
        
        # Create a mock solution object with only survivors
        C_surv = C_final[survivors_mask]
        state_surv = np.concatenate([C_surv, R_final])
        
        # Create a simple object to mimic sol structure
        class MockSol:
            def __init__(self, y):
                self.y = y
        
        sol_surv = MockSol(state_surv[:, None])

        J = param.MiCRM_jac(
            n_survivors, M, u_surv, l_surv,
            MAINTENANCE_COST, rho, omega,
            lambda_vec_surv, sol_surv
        )
        ev = param.leading_eigenvalue(J)
    else:
        ev = np.nan

    # Build species-level records
    species_data = []
    for i in range(N):
        if invasion_num == 0:
            origin = "Initial"
        elif i < n_resident:
            origin = "Resident"
        else:
            origin = "Invader"

        species_data.append({
            "Seed": seed,
            "Invasion_Num": invasion_num,
            "Species_ID": int(species_indices[i]),
            "Local_ID": i,
            "Origin": origin,
            "Species_CUE": species_CUE[i],
            "Community_CUE": community_CUE,
            "Abundance": C_final[i],
            "Survived": bool(C_final[i] > SURVIVAL_THRESHOLD),
            "Total_Abundance": total_abundance,
            "Resident_Abundance": resident_abundance,
            "Invader_Abundance": invader_abundance,
            "Competition_Community": competition_comm,
            "Competition_Species": competition_species[i],
            "Competition_Dot": competition_dot[i],
            "Facilitation": facilitation[i],
            "Uptake_Variance": uptake_var[i],
            "Depletion": depletion,
            "N_Survivors": n_survivors,
            "N_Total": N,
            "N_Resident": n_resident,
            "N_Invader": n_invader,
            "Feasibility": float(feas),
            "Leading_Eigenvalue": float(ev) if not np.isnan(ev) else None
        })

    return species_data


def main():
    seed_generator = np.random.default_rng(BASE_SEED)
    seeds = seed_generator.integers(0, 2**32 - 1, size=N_SIMULATIONS,
                                    dtype=np.uint32).tolist()

    print(f"Starting {N_SIMULATIONS} simulations with {N_INVASIONS} "
          f"successive invasions each...")

    with Pool(cpu_count()) as pool:
        all_data_nested = pool.map(simulate, seeds)

    all_data = [
        row
        for one_seed_result in all_data_nested
        if one_seed_result
        for row in one_seed_result
    ]

    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Output Leading Eigenvalue summary by invasion number
    print("\n" + "="*60)
    print("LEADING EIGENVALUE BY INVASION NUMBER")
    print("="*60)
    invasion_summary = df.groupby(['Seed', 'Invasion_Num']).first().reset_index()
    ev_summary = invasion_summary.groupby('Invasion_Num')['Leading_Eigenvalue'].agg(['mean', 'std', 'min', 'max'])
    print(ev_summary.to_string())
    print("="*60 + "\n")

    plot_invasion_trends(df)


def plot_invasion_trends(df):
    """Plot feasibility and stability trends across successive invasions."""

    invasion_summary = df.groupby(['Seed', 'Invasion_Num']).agg({
        'Feasibility': 'first',
        'Leading_Eigenvalue': 'first',
        'N_Survivors': 'first',
        'Community_CUE': 'first',
        'Total_Abundance': 'first',
        'Resident_Abundance': 'first',
        'Invader_Abundance': 'first'
    }).reset_index()

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def plot_panel(ax, col, color, ylabel, title, hline=None):
        grp = invasion_summary.groupby('Invasion_Num')[col]
        mean_v = grp.mean()
        std_v = grp.std()
        ax.plot(mean_v.index, mean_v.values, 'o-', linewidth=2,
                markersize=8, color=color)
        ax.fill_between(mean_v.index,
                        np.maximum(mean_v.values - std_v.values, 0),
                        mean_v.values + std_v.values,
                        alpha=0.3, color=color)
        if hline is not None:
            ax.axhline(y=hline, color='red', linestyle='--', alpha=0.7,
                       label='Stability threshold')
            ax.legend()
        ax.set_xlabel('Invasion Number', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plot_panel(axes[0, 0], 'Feasibility',        '#2E86AB',
               'Feasibility Probability',  'Feasibility vs Invasion Number')
    plot_panel(axes[0, 1], 'Leading_Eigenvalue', '#A23B72',
               'Leading Eigenvalue',       'Stability vs Invasion Number', hline=0)
    plot_panel(axes[1, 0], 'N_Survivors',        '#F18F01',
               'Number of Survivors',      'Species Richness vs Invasion Number')
    plot_panel(axes[1, 1], 'Community_CUE',      '#06A77D',
               'Community CUE',            'Community CUE vs Invasion Number')

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {PLOT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
