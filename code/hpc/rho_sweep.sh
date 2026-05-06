#!/bin/bash
#PBS -N rho_sweep
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=32:mem=64gb
#PBS -o logs/rho_sweep.out
#PBS -e logs/rho_sweep.err

set -euo pipefail

REPO_DIR="$HOME/micrm"
CODE_DIR="$REPO_DIR/code"
CONDA_ENV="micrm"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$REPO_DIR/logs"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"
python "$CODE_DIR/main_rho_sweep.py"
