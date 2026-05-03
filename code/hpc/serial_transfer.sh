#!/bin/bash
#PBS -N serial_transfer
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=32:mem=64gb
#PBS -o logs/serial.out
#PBS -e logs/serial.err

set -euo pipefail

REPO_DIR="$HOME/micrm"
CODE_DIR="$REPO_DIR/code"
CONDA_ENV="micrm"
N_CORES=32

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$REPO_DIR/logs"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"
python "$CODE_DIR/hpc/main_serial_transfer_hpc.py" \
    --task-id 0 \
    --n-tasks 1 \
    --n-cores "$N_CORES"
