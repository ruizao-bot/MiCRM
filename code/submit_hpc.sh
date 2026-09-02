#!/bin/bash
#SBATCH --partition=large_336
#SBATCH --job-name=micrm
#SBATCH --output=logs/micrm_%j.out
#SBATCH --error=logs/micrm_%j.err

# Ensure logs directory exists
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate micrm

# Ensure shared libraries from this conda env are discoverable at runtime.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Using CONDA_PREFIX: $CONDA_PREFIX"

python code/main.py

echo "Job completed at $(date)"
