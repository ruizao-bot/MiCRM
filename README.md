# Code Directory

This directory contains the core simulation and visualization code for coalescence using the MiCRM (Microbial Consumer Resource Model)
## Files Overview

### Core Library

#### `param.py`
Core library module containing all MiCRM simulation functions and utilities.

**Key Functions:**
- `modular_uptake()` - Generates modular resource uptake matrix
- `modular_leakage()` - Generates modular leakage matrix  
- `generate_l_tensor()` - Creates 3D leakage tensor
- `solve_micrm()` - Solves MiCRM ODE system to equilibrium
- `compute_species_CUE()` - Calculates species-level carbon use efficiency
- `compute_community_CUE()` - Calculates community-level CUE
- `calculate_effective_leakage()` - Computes effective leakage (optimized with einsum)
- `community_level_competition()` - Cosine similarity-based competition metric
- `species_level_competition()` - Per-species competition calculation
- `species_level_competition_dot()` - Dot product-based competition
- `compute_uptake_variance()` - Calculates uptake vector variance
- `extract_state_at_target_time()` - Extracts system state at specific time
- `choose_resources_for_second_community()` - Resource allocation logic

**Dependencies:** numpy, scipy, matplotlib, micrm

---

### Simulation Scripts

#### `main.py`
Main community coalescence simulation script.

**Purpose:** Simulates coalescence of two microbial communities and tracks comprehensive ecological metrics.

**Output:** `coal.csv` - Detailed species-level data with metrics:
- Species abundance, CUE, competition, facilitation
- Community-level CUE, competition, facilitation
- Resource depletion, uptake variance
- Dominant community identification

**Usage:**
```bash
conda activate micrm
python code/main.py
```

---

#### `coal_resource_overlap.py`
Resource overlap coalescence experiment.

**Purpose:** Tests the effect of resource overlap (25%, 50%, 75%) between communities on coalescence outcomes.


**Output:** `data/coal_resource.csv` - Contains:
- All standard coalescence metrics (CUE, competition, facilitation) under different overlap ratio

**Usage:**
```bash
conda activate micrm
python code/coal_resource_overlap.py
```

---

#### `rare_invasion.py`
Rare species invasion experiment.

**Purpose:** Simulates invasion of rare species from one community into another at different dilution rates.

**Output:** 
- `data/rare.csv` - Detailed species-level invasion data

**Usage:**
```bash
conda activate micrm
python code/rare_invasion.py
```

---

### Visualization

#### `plot.py`
Generates all publication-quality figures from simulation data.

**Purpose:** Creates comprehensive visualization suite for thesis/publications.

**Figures Generated:**

1. **Figure 1: Bray-Curtis Similarity with Dominance**
   - `results/dom_sim.pdf` - CUE difference vs similarity difference colored by dominant community

2. **Figure 1B: Resource Overlap Analysis**
   - `results/cue_dominance_overlap.pdf` - Boxplots showing overlap effects on CUE-similarity relationship

3. **Figure 2: Species CUE vs Abundance**
   - `results/cue_abund.pdf` - Scatter with side histograms for each community

4. **Figure 3: Rare Species Invasion**
   - `results/rare_survival.pdf` - Stacked bar charts by CUE bins and rarity level

5. **Figure 4: Facilitation vs Community CUE**
   - `results/Facilitation_vs_communityCUE.pdf` - Three-panel scatter plots

6. **Figure 5: CUE vs Depletion**
   - `results/Residual_vs_CUE.pdf` - Combined scatter and boxplot

7. **Figure SI: Competition vs Community CUE**
   - `results/Competition_vs_communityCUE.pdf` - Community level resource usage similarity vs CUE
   - `results/species_competition.pdf` - Species-level analysis

**Requirements:**
- Input data: `coal.csv`, `data/coal_resource.csv`, `data/rare.csv`
- Output directory: `results/` (auto-created)

**Usage:**
```bash
conda activate micrm
python code/plot.py
```

---

## Dependencies

All scripts require the `micrm` conda environment:

```bash
conda activate micrm
```

**Required packages:**
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- micrm

## Author

Created for MiCRM project analyzing microbial community coalescence dynamics.

**Date:** March 2026
