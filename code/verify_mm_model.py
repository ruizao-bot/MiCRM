import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import sys

# Setup paths
code_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(code_path, os.pardir))
data_dir = os.path.join(project_root, "data")

# Define the Michaelis-Menten model function
def mm_model(epsilon, C_max, epsilon_c, h):
    """
    Michaelis-Menten type model for abundance as a function of CUE.
    
    Parameters:
    -----------
    epsilon : array
        CUE values (epsilon)
    C_max : float
        Maximum abundance
    epsilon_c : float
        Critical CUE threshold (fixed at 0.42)
    h : float
        Half-saturation constant
    
    Returns:
    --------
    array : Predicted abundance values
    """
    epsilon_shifted = np.maximum(epsilon - epsilon_c, 0)
    return C_max * (epsilon_shifted / (epsilon_shifted + h))


def mm_model_fixed_epsilon_c(epsilon, C_max, h):
    """
    Michaelis-Menten model with epsilon_c fixed at 0.42.
    
    Parameters:
    -----------
    epsilon : array
        CUE values
    C_max : float
        Maximum abundance
    h : float
        Half-saturation constant
    
    Returns:
    --------
    array : Predicted abundance values
    """
    epsilon_c = 0.42
    return mm_model(epsilon, C_max, epsilon_c, h)


def fit_mm_model_to_community(df, community_id, epsilon_c=0.42, 
                               abundance_col='Abundance', cue_col='Species_CUE',
                               use_survivors_only=False, abundance_threshold=1e-5):
    """
    Fit the Michaelis-Menten model to a specific community.
    
    Parameters:
    -----------
    df : DataFrame
        Complete dataset
    community_id : int
        Community ID (1, 2, or 3)
    epsilon_c : float
        Critical CUE threshold (default: 0.42)
    abundance_col : str
        Column name for abundance
    cue_col : str
        Column name for CUE
    use_survivors_only : bool
        If True, only fit to species with abundance > threshold
    abundance_threshold : float
        Threshold for survivor classification
    
    Returns:
    --------
    dict : Fitted parameters and statistics
    """
    # Filter for specific community
    comm_df = df[df['Community'] == community_id].copy()
    
    if use_survivors_only:
        comm_df = comm_df[comm_df[abundance_col] > abundance_threshold]
    
    if len(comm_df) < 3:
        print(f"Warning: Community {community_id} has insufficient data points ({len(comm_df)})")
        return None
    
    cue_data = comm_df[cue_col].values
    abundance_data = comm_df[abundance_col].values
    
    # Filter out non-positive abundances for fitting
    valid_mask = (abundance_data > 0) & np.isfinite(cue_data) & np.isfinite(abundance_data)
    cue_data = cue_data[valid_mask]
    abundance_data = abundance_data[valid_mask]
    
    if len(cue_data) < 3:
        print(f"Warning: Community {community_id} has insufficient valid data points")
        return None
    
    # Initial parameter guesses
    C_max_guess = np.max(abundance_data)
    h_guess = 0.01
    
    try:
        # Fit the model (epsilon_c is fixed at 0.42)
        popt, pcov = curve_fit(
            mm_model_fixed_epsilon_c, 
            cue_data, 
            abundance_data, 
            p0=[C_max_guess, h_guess],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=10000
        )
        
        C_max_fit, h_fit = popt
        
        # Calculate R-squared
        y_pred = mm_model(cue_data, C_max_fit, epsilon_c, h_fit)
        ss_res = np.sum((abundance_data - y_pred) ** 2)
        ss_tot = np.sum((abundance_data - np.mean(abundance_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate parameter standard errors
        perr = np.sqrt(np.diag(pcov))
        C_max_err, h_err = perr
        
        return {
            'community_id': community_id,
            'C_max': C_max_fit,
            'C_max_err': C_max_err,
            'epsilon_c': epsilon_c,
            'h': h_fit,
            'h_err': h_err,
            'r_squared': r_squared,
            'n_points': len(cue_data),
            'cue_data': cue_data,
            'abundance_data': abundance_data,
            'use_survivors_only': use_survivors_only
        }
        
    except Exception as e:
        print(f"Error fitting Community {community_id}: {str(e)}")
        return None


def plot_mm_fit(results_list, save_path=None):
    """
    Plot the fitted Michaelis-Menten models for all communities.
    
    Parameters:
    -----------
    results_list : list of dict
        List of fitting results from fit_mm_model_to_community
    save_path : str, optional
        Path to save the figure
    """
    n_communities = len([r for r in results_list if r is not None])
    
    if n_communities == 0:
        print("No valid results to plot")
        return
    
    fig, axes = plt.subplots(1, n_communities, figsize=(6*n_communities, 5))
    
    if n_communities == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    plot_idx = 0
    for results in results_list:
        if results is None:
            continue
            
        ax = axes[plot_idx]
        
        comm_id = results['community_id']
        cue_data = results['cue_data']
        abundance_data = results['abundance_data']
        C_max = results['C_max']
        epsilon_c = results['epsilon_c']
        h = results['h']
        r_squared = results['r_squared']
        
        # Plot data points
        ax.scatter(cue_data, abundance_data, alpha=0.5, s=20, 
                  color=colors[comm_id-1], label='Data')
        
        # Plot fitted curve
        epsilon_range = np.linspace(max(epsilon_c, np.min(cue_data)), 
                                   np.max(cue_data) * 1.1, 200)
        abundance_pred = mm_model(epsilon_range, C_max, epsilon_c, h)
        ax.plot(epsilon_range, abundance_pred, 'r-', linewidth=2, 
               label=f'MM Model (R²={r_squared:.3f})')
        
        # Add vertical line at epsilon_c
        ax.axvline(x=epsilon_c, color='gray', linestyle='--', alpha=0.5, 
                  label=f'ε_c = {epsilon_c}')
        
        # Labels and title
        community_name = "Daughter" if comm_id == 3 else f"Parent {comm_id}"
        ax.set_xlabel('Species CUE (ε)', fontsize=12)
        ax.set_ylabel('Abundance', fontsize=12)
        ax.set_title(f'Community {comm_id} ({community_name})\n' + 
                    f'C_max={C_max:.3f}, h={h:.4f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def compare_parental_vs_daughter(df, epsilon_c=0.42, use_survivors_only=False):
    """
    Compare parental communities (1 and 2) with daughter community (3).
    
    Parameters:
    -----------
    df : DataFrame
        Complete dataset
    epsilon_c : float
        Critical CUE threshold
    use_survivors_only : bool
        If True, only analyze survivors
    
    Returns:
    --------
    dict : Summary statistics comparing communities
    """
    results = {}
    
    # Fit each community
    for comm_id in [1, 2, 3]:
        fit_result = fit_mm_model_to_community(
            df, comm_id, epsilon_c=epsilon_c, 
            use_survivors_only=use_survivors_only
        )
        results[f'community_{comm_id}'] = fit_result
    
    # Create summary comparison
    summary = []
    for comm_id in [1, 2, 3]:
        result = results[f'community_{comm_id}']
        if result:
            community_type = "Daughter" if comm_id == 3 else f"Parent {comm_id}"
            summary.append({
                'Community': f'{comm_id} ({community_type})',
                'C_max': f"{result['C_max']:.4f} ± {result['C_max_err']:.4f}",
                'h': f"{result['h']:.5f} ± {result['h_err']:.5f}",
                'R²': f"{result['r_squared']:.4f}",
                'N': result['n_points']
            })
    
    summary_df = pd.DataFrame(summary)
    
    return {
        'results': results,
        'summary': summary_df
    }


def main():
    """Main function to run the analysis."""
    
    # Load data
    data_path = os.path.join(data_dir, "coal.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run thesis.py first to generate the data.")
        return
    
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Total records: {len(df)}")
    print(f"Communities: {df['Community'].unique()}")
    print(f"Seeds: {len(df['Seed'].unique())}")
    print("-" * 70)
    
    # Analysis 1: All species
    print("\n" + "="*70)
    print("ANALYSIS 1: Fitting MM Model to ALL Species")
    print("="*70)
    
    comparison_all = compare_parental_vs_daughter(df, epsilon_c=0.42, 
                                                   use_survivors_only=False)
    
    print("\nFitted Parameters (All Species):")
    print(comparison_all['summary'].to_string(index=False))
    
    # Plot results
    results_all = [comparison_all['results'][f'community_{i}'] for i in [1, 2, 3]]
    plot_mm_fit(results_all, 
                save_path=os.path.join(data_dir, "mm_model_fit_all.png"))
    
    # Analysis 2: Survivors only
    print("\n" + "="*70)
    print("ANALYSIS 2: Fitting MM Model to SURVIVORS Only (Abundance > 1e-5)")
    print("="*70)
    
    comparison_surv = compare_parental_vs_daughter(df, epsilon_c=0.42, 
                                                    use_survivors_only=True)
    
    print("\nFitted Parameters (Survivors Only):")
    print(comparison_surv['summary'].to_string(index=False))
    
    # Plot results
    results_surv = [comparison_surv['results'][f'community_{i}'] for i in [1, 2, 3]]
    plot_mm_fit(results_surv, 
                save_path=os.path.join(data_dir, "mm_model_fit_survivors.png"))
    
    # Statistical comparison
    print("\n" + "="*70)
    print("COMPARISON: Parental vs Daughter Communities")
    print("="*70)
    
    for analysis_name, comparison in [("All Species", comparison_all), 
                                     ("Survivors Only", comparison_surv)]:
        print(f"\n{analysis_name}:")
        
        # Average parental parameters
        parent_results = [comparison['results'][f'community_{i}'] 
                         for i in [1, 2]]
        parent_results = [r for r in parent_results if r is not None]
        
        daughter_result = comparison['results']['community_3']
        
        if len(parent_results) == 2 and daughter_result:
            avg_parent_C_max = np.mean([r['C_max'] for r in parent_results])
            avg_parent_h = np.mean([r['h'] for r in parent_results])
            avg_parent_r2 = np.mean([r['r_squared'] for r in parent_results])
            
            print(f"  Average Parental C_max: {avg_parent_C_max:.4f}")
            print(f"  Daughter C_max: {daughter_result['C_max']:.4f}")
            print(f"  Ratio (Daughter/Parent): {daughter_result['C_max']/avg_parent_C_max:.3f}")
            print(f"  Average Parental h: {avg_parent_h:.5f}")
            print(f"  Daughter h: {daughter_result['h']:.5f}")
            print(f"  Ratio (Daughter/Parent): {daughter_result['h']/avg_parent_h:.3f}")
            print(f"  Average Parental R²: {avg_parent_r2:.4f}")
            print(f"  Daughter R²: {daughter_result['r_squared']:.4f}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
