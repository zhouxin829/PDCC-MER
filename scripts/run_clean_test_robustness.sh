#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

DATA_PATH="${1:-${DATA_PATH:-./MSA_Datasets}}"
RUN_ROOT="${2:-${RUN_ROOT:-runs/clean_test_robustness}}"
GPU_ID="${3:-${GPU_ID:-0}}"
DATASETS_CSV="${4:-${DATASETS:-SIMS,MOSI}}"
SEEDS_CSV="${SEEDS:-3328683074,4136559363,1686802513}"
MODELS_CSV="${MODELS:-BASE,ROUTER_ONLY,PDCC_MER}"
PYTHON="${PYTHON:-python}"

PROJECT_DIR="$(resolve_project_dir)"
require_project_file "$PROJECT_DIR" "pdcc_multiseed_tools/run_clean_test_robustness.py"
require_project_file "$PROJECT_DIR" "pdcc_multiseed_tools/summarize_clean_test_robustness.py"

csv_to_array "$DATASETS_CSV" DATASETS_ARR
cd "$PROJECT_DIR"

EXTRA_ARGS=()
if [[ -n "${CONDITIONS:-}" ]]; then
  EXTRA_ARGS+=(--conditions "$CONDITIONS")
fi
if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--eval-only)
fi
if [[ "${NO_PER_SAMPLE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-per-sample)
fi
if [[ "${FORCE_EVAL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-eval)
fi

for dataset in "${DATASETS_ARR[@]}"; do
  echo "[PDCC-MER] Clean-train/corrupted-test dataset=$dataset"
  "$PYTHON" pdcc_multiseed_tools/run_clean_test_robustness.py \
    --project-dir "$PROJECT_DIR" \
    --entry pdcc_main.py \
    --python "$PYTHON" \
    --data-path "$DATA_PATH" \
    --run-root "$RUN_ROOT" \
    --dataset "$dataset" \
    --gpu "$GPU_ID" \
    --models "$MODELS_CSV" \
    --seeds "$SEEDS_CSV" \
    "${EXTRA_ARGS[@]}"
done

"$PYTHON" pdcc_multiseed_tools/summarize_clean_test_robustness.py \
  --root "$RUN_ROOT"

echo "[PDCC-MER] Clean-train/corrupted-test runs completed."
