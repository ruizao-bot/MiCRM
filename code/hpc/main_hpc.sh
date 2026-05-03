#!/bin/bash
#PBS -N micrm_coal
#PBS -J 0-9
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/jc224/home/micrm/logs/coal_^array_index^.out
#PBS -e /rds/general/user/jc224/home/micrm/logs/coal_^array_index^.err

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
N_TASKS=10   # must match -J upper bound + 1

python "$CODE_DIR/hpc/main_hpc.py" \
    --task-id  "$PBS_ARRAY_INDEX" \
    --n-tasks  "$N_TASKS" \
    --n-cores  8

echo "Task $PBS_ARRAY_INDEX done"
