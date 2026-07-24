#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

DATA_PATH="${1:-${DATA_PATH:-./MSA_Datasets}}"
COST_ROOT="${2:-${COST_ROOT:-runs/computational_cost}}"
PROFILE_ROOT="${3:-${PROFILE_ROOT:-runs/complexity_profile}}"
GPU_ID="${4:-${GPU_ID:-0}}"
DATASETS_CSV="${5:-${DATASETS:-SIMS,MOSI}}"
SEEDS_CSV="${SEEDS:-3328683074,4136559363,1686802513}"
PYTHON="${PYTHON:-python}"

PROJECT_DIR="$(resolve_project_dir)"
require_project_file "$PROJECT_DIR" "pdcc_multiseed_tools/run_diagnostics_retrain.py"
require_project_file "$PROJECT_DIR" "pdcc_multiseed_tools/summarize_retrain_diagnostics.py"
require_project_file "$PROJECT_DIR" "pdcc_multiseed_tools/run_complexity_table.py"

csv_to_array "$DATASETS_CSV" DATASETS_ARR
cd "$PROJECT_DIR"

for dataset in "${DATASETS_ARR[@]}"; do
  echo "[PDCC-MER] Full two-stage cost dataset=$dataset"
  "$PYTHON" pdcc_multiseed_tools/run_diagnostics_retrain.py \
    --project-dir "$PROJECT_DIR" \
    --entry pdcc_main.py \
    --python "$PYTHON" \
    --dataset "$dataset" \
    --data-path "$DATA_PATH" \
    --run-root "$COST_ROOT" \
    --seeds "$SEEDS_CSV" \
    --gpu "$GPU_ID" \
    --suites cost
done

"$PYTHON" pdcc_multiseed_tools/summarize_retrain_diagnostics.py \
  --root "$COST_ROOT"

echo "[PDCC-MER] Reuse COST checkpoints for latency, memory, and FLOP profiling"
"$PYTHON" pdcc_multiseed_tools/run_complexity_table.py \
  --project-dir "$PROJECT_DIR" \
  --entry pdcc_main.py \
  --python "$PYTHON" \
  --data-path "$DATA_PATH" \
  --run-root "$PROFILE_ROOT" \
  --trained-cost-root "$COST_ROOT" \
  --datasets "$DATASETS_CSV" \
  --variants Base,Base+TPLR,Base+PCRP,Base+RCCR,PDCC-MER \
  --seeds "$SEEDS_CSV" \
  --gpu "$GPU_ID"

echo "[PDCC-MER] Cost table: $COST_ROOT/summary/cost_table.md"
echo "[PDCC-MER] Profiling table: $PROFILE_ROOT/complexity_table.md"
