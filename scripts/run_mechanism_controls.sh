#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

DATA_PATH="${1:-${DATA_PATH:-./MSA_Datasets}}"
RUN_ROOT="${2:-${RUN_ROOT:-runs/mechanism_controls}}"
GPU_ID="${3:-${GPU_ID:-0}}"
DATASETS_CSV="${4:-${DATASETS:-SIMS,MOSI}}"
SEEDS_CSV="${SEEDS:-3328683074,4136559363,1686802513}"
PYTHON="${PYTHON:-python}"

PROJECT_DIR="$(resolve_project_dir)"
require_project_file "$PROJECT_DIR" "pdcc_multiseed_tools/run_mechanism_controls.py"

csv_to_array "$DATASETS_CSV" DATASETS_ARR

cd "$PROJECT_DIR"

for dataset in "${DATASETS_ARR[@]}"; do
  echo "[PDCC-MER] Mechanism controls dataset=$dataset"
  "$PYTHON" pdcc_multiseed_tools/run_mechanism_controls.py \
    --project-dir "$PROJECT_DIR" \
    --entry pdcc_main.py \
    --python "$PYTHON" \
    --data-path "$DATA_PATH" \
    --run-root "$RUN_ROOT" \
    --dataset "$dataset" \
    --gpu "$GPU_ID" \
    --seeds "$SEEDS_CSV"
done

if [[ -f "$PROJECT_DIR/pdcc_multiseed_tools/summarize_mechanism_controls.py" ]]; then
  "$PYTHON" pdcc_multiseed_tools/summarize_mechanism_controls.py --root "$RUN_ROOT"
fi

echo "[PDCC-MER] Mechanism-control runs completed."

