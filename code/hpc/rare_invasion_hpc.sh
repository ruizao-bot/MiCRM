#!/bin/bash
#PBS -N micrm_rare
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/jc224/home/micrm/logs/rare.out
#PBS -e /rds/general/user/jc224/home/micrm/logs/rare.err

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/rds/general/user/jc224/home/micrm"
CODE_DIR="$PROJECT_DIR/code"
DATA_DIR="$PROJECT_DIR/data"
LOGS_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOGS_DIR" "$DATA_DIR"
cd "$PROJECT_DIR"
export PROJECT_DIR
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate micrm

# ── Run ───────────────────────────────────────────────────────────────────────
python "$CODE_DIR/rare_invasion.py"

echo "Done"
