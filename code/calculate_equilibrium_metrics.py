import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import braycurtis
import param

# Define the project root and data directory
code_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(code_path, os.pardir))
data_dir = os.path.join(project_root, "data")


def calculate_metrics(seed):
    """
    Simulate coalescence of two parent communities and calculate equilibrium metrics.
    Similar to thesis.py structure.
    """
    np.random.seed(seed)

    # Define parameters (similar to thesis.py)
    N_pool, M_pool, λ, N_modules, s_ratio = 1000, 100, 0.2, 1, 1.0
    N1, M1, N2, M2 = 100, 50, 100, 50
    t_span = (0, 100000)

    # Generate pool of uptake and leakage matrices
    u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
    l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ, u_pool)

    # ==================== Parent Community 1 ====================
    species_indices1 = np.random.choice(N_pool, N1, replace=False)
    resource_indices1 = np.random.choice(M_pool, M1, replace=False)
    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
    
    m1 = 0.2
    lambda_alpha1 = np.full(M1, λ)
    rho1, omega1 = np.full(M1, 0.6), np.full(M1, 0.1)
    C0_1 = np.full(N1, 0.01)
    R0_1 = np.full(M1, 1.0)
    
    sol1 = param.solve_micrm(N1, M1, u1, l1, m1, lambda_alpha1, rho1, omega1, C0_1, R0_1, t_span)
    
    # Extract equilibrium values for community 1
    C_final1 = sol1.y[:N1, -1]
    R_final1 = sol1.y[N1:, -1]
    
    # Calculate CUE and similarity for community 1
    community_CUE1, species_CUE1 = param.compute_CUE(sol1, N1, u1, R_final1, lambda_alpha1, m1)
    # ==================== Parent Community 2 ====================
    # Ensure resource overlap (similar to thesis.py)
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
    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]
    
    m2 = 0.2
    lambda_alpha2 = np.full(M2, λ)
    rho2, omega2 = np.full(M2, 0.6), np.full(M2, 0.1)
    C0_2 = np.full(N2, 0.01)
    R0_2 = np.full(M2, 1.0)
    
    sol2 = param.solve_micrm(N2, M2, u2, l2, m2, lambda_alpha2, rho2, omega2, C0_2, R0_2, t_span)
    
    # Extract equilibrium values for community 2
    C_final2 = sol2.y[:N2, -1]
    R_final2 = sol2.y[N2:, -1]
    
    # Calculate CUE and similarity for community 2
    community_CUE2, species_CUE2 = param.compute_CUE(sol2, N2, u2, R_final2, lambda_alpha2, m2)

    # ==================== Daughter Community (Coalesced) ====================
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    
    N3, M3 = N1 + N2, len(resource_indices3)
    lambda_alpha3 = np.full(M3, λ)
    omega3 = np.full(M3, 0.1)
    rho3 = np.full(M3, 0.6)
    m3 = 0.2
    
    # Initial conditions for coalescence: use final abundances from parents
    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])
    R0_3 = np.full(M3, 1.0)
    
    sol3 = param.solve_micrm(N3, M3, u3, l3, m3, lambda_alpha3, rho3, omega3, C0_3, R0_3, t_span)
    
    # Extract equilibrium values for community 3
    C_final3 = sol3.y[:N3, -1]
    R_final3 = sol3.y[N3:, -1]
    
    # Calculate CUE and similarity for community 3
    community_CUE3, species_CUE3 = param.compute_CUE(sol3, N3, u3, R_final3, lambda_alpha3, m3)

    # Determine dominance
    total_1 = np.sum(C_final3[:N1])
    total_2 = np.sum(C_final3[N1:])
    dominant = "Community 1" if total_1 > total_2 else "Community 2"

    # ==================== Calculate Bray-Curtis Similarity ====================
    # Create abundance vectors in a shared species space (similar to Global_Species_ID in thesis.R)
    # Community 1 has the first N1 species, Community 2 has the next N2 species
    N_total = N1 + N2
    
    # Abundance vector for parent community 1 (in global space)
    abund_comm1 = np.zeros(N_total)
    abund_comm1[:N1] = C_final1
    
    # Abundance vector for parent community 2 (in global space)
    abund_comm2 = np.zeros(N_total)
    abund_comm2[N1:] = C_final2
    
    # Abundance vector for coalesced community 3 (in global space)
    abund_comm3 = np.zeros(N_total)
    abund_comm3[:N1] = C_final3[:N1]  # Species from community 1
    abund_comm3[N1:] = C_final3[N1:]  # Species from community 2
    
    # Calculate Bray-Curtis dissimilarity
    bray_3vs1 = braycurtis(abund_comm3, abund_comm1)
    bray_3vs2 = braycurtis(abund_comm3, abund_comm2)
    
    # Convert to similarity (1 - dissimilarity)
    sim_3vs1 = 1 - bray_3vs1
    sim_3vs2 = 1 - bray_3vs2

    # Return results as a dictionary
    return {
        "Seed": seed,
        "CUE_1": community_CUE1,
        "CUE_2": community_CUE2,
        "CUE_3": community_CUE3,
        "Sim_3vs1": sim_3vs1,
        "Sim_3vs2": sim_3vs2,
        "Dominant_Community": dominant,
        "Total_Abundance_1": total_1,
        "Total_Abundance_2": total_2
    }


if __name__ == "__main__":
    # Read seeds from file
    seeds_file = os.path.join(code_path, 'seeds.txt')
    with open(seeds_file, 'r') as f:
        seeds = [int(line.strip()) for line in f]

    # Run simulations in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(calculate_metrics, seeds)

    # Save results to CSV
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, "coalescence_metrics.csv")

    # Convert results to DataFrame (one row per seed)
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Coalescence metrics saved to {output_file}")
    print(f"Columns: {list(df.columns)}")
    print(f"Number of simulations: {len(df)}")