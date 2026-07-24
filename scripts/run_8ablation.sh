#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

DATA_PATH="${1:-${DATA_PATH:-./MSA_Datasets}}"
RUN_ROOT="${2:-${RUN_ROOT:-runs/pdcc_8ablation}}"
GPU_ID="${3:-${GPU_ID:-0}}"
DATASETS_CSV="${4:-${DATASETS:-SIMS,MOSI}}"
SEEDS_CSV="${SEEDS:-3328683074,4136559363,1686802513}"
PYTHON="${PYTHON:-python}"

PROJECT_DIR="$(resolve_project_dir)"
require_project_file "$PROJECT_DIR" "pdcc_main.py"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

csv_to_array "$DATASETS_CSV" DATASETS_ARR
csv_to_array "$SEEDS_CSV" SEEDS_ARR

COMBOS=(
  "BASE:"
  "TPLR:--use_tplr"
  "PCRP:--use_pcrp"
  "RCCR:--use_rccr"
  "TPLR_PCRP:--use_tplr --use_pcrp"
  "TPLR_RCCR:--use_tplr --use_rccr"
  "PCRP_RCCR:--use_pcrp --use_rccr"
  "PDCC_FULL:--use_tplr --use_pcrp --use_rccr"
)

cd "$PROJECT_DIR"

for dataset in "${DATASETS_ARR[@]}"; do
  for combo in "${COMBOS[@]}"; do
    combo_name="${combo%%:*}"
    combo_flags="${combo#*:}"
    run_idx=1

    for seed in "${SEEDS_ARR[@]}"; do
      run_name="$(printf "run_%02d" "$run_idx")"
      run_dir="$RUN_ROOT/$dataset/$combo_name/$run_name"
      mkdir -p "$run_dir/models" "$run_dir/logs"

      echo "[PDCC-MER] Ablation dataset=$dataset combo=$combo_name run=$run_name seed=$seed"

      "$PYTHON" -u pdcc_main.py \
        --dataset "$dataset" \
        --data_path "$DATA_PATH" \
        --model_path "$run_dir/models" \
        --run_dir "$run_dir" \
        --use_best \
        --is_pseudo \
        $combo_flags \
        --seed "$seed" \
        2>&1 | tee "$run_dir/logs/stage1.log"

      "$PYTHON" -u pdcc_main.py \
        --dataset "$dataset" \
        --data_path "$DATA_PATH" \
        --model_path "$run_dir/models" \
        --run_dir "$run_dir" \
        --use_best \
        --is_pseudo \
        --finetune \
        --pretrained_model \
        $combo_flags \
        --seed "$seed" \
        2>&1 | tee "$run_dir/logs/stage2.log"

      run_idx=$((run_idx + 1))
    done
  done
done

if [[ -f "$PROJECT_DIR/collect_pdcc_8ablation_results.py" ]]; then
  "$PYTHON" collect_pdcc_8ablation_results.py "$RUN_ROOT"
fi

echo "[PDCC-MER] Ablation runs completed."
