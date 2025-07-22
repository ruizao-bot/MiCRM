import pytest
import pandas as pd
from code.coal_sameR0_hpc import simulate

# test_coal_sameR0_hpc.py


def test_uptakevar_cue_correlation():
    # Use a fixed seed for reproducibility
    seed = 42
    species_data = simulate(seed)
    df = pd.DataFrame(species_data)
    assert "UptakeVar" in df.columns
    assert "Species_CUE" in df.columns
    # Check for non-constant columns
    assert df["UptakeVar"].nunique() > 1
    assert df["Species_CUE"].nunique() > 1
    # Compute correlation
    corr = df["UptakeVar"].corr(df["Species_CUE"])
    assert isinstance(corr, float)
    print(f"Correlation between UptakeVar and Species_CUE: {corr:.3f}")