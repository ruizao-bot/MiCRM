#!/bin/bash
#PBS -N micrm_coal
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/jc224/home/micrm/logs/coal.out
#PBS -e /rds/general/user/jc224/home/micrm/logs/coal.err

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/rds/general/user/jc224/home/micrm"
CODE_DIR="$PROJECT_DIR/code"
DATA_DIR="$PROJECT_DIR/data"
LOGS_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOGS_DIR" "$DATA_DIR/main_hpc"
cd "$PROJECT_DIR"
export PROJECT_DIR
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate micrm

# ── Run task ──────────────────────────────────────────────────────────────────
python "$CODE_DIR/main.py" \
    --n-cores  8

echo "Done"
