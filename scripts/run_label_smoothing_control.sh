#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

DATA_PATH="${1:-${DATA_PATH:-./MSA_Datasets}}"
RUN_ROOT="${2:-${RUN_ROOT:-runs/label_smoothing_control_v1}}"
GPU_ID="${3:-${GPU_ID:-0}}"
DATASET="${4:-${DATASET:-SIMS}}"
SEEDS_CSV="${SEEDS:-3328683074,4136559363,1686802513}"
EPSILONS_CSV="${EPSILONS:-0.05,0.1,0.2}"
PYTHON="${PYTHON:-python}"

PROJECT_DIR="$(resolve_project_dir)"
require_project_file "$PROJECT_DIR" "pdcc_multiseed_tools/run_label_smoothing_control.py"

cd "$PROJECT_DIR"

"$PYTHON" pdcc_multiseed_tools/run_label_smoothing_control.py \
  --project-dir "$PROJECT_DIR" \
  --entry pdcc_main.py \
  --python "$PYTHON" \
  --dataset "$DATASET" \
  --data-path "$DATA_PATH" \
  --run-root "$RUN_ROOT" \
  --epsilons "$EPSILONS_CSV" \
  --seeds "$SEEDS_CSV" \
  --gpu "$GPU_ID"

echo "[PDCC-MER] Label-smoothing control completed for dataset=$DATASET."
echo "[PDCC-MER] Summarize after both datasets finish:"
echo "  $PYTHON pdcc_multiseed_tools/summarize_label_smoothing_control.py --root $RUN_ROOT"
