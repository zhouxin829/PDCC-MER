#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

DATASET="${1:-${DATASET:-SIMS}}"
DATA_PATH="${2:-${DATA_PATH:-./MSA_Datasets}}"
RUN_DIR="${3:-${RUN_DIR:-runs/quickstart/${DATASET}/seed_3328683074}}"
GPU_ID="${4:-${GPU_ID:-0}}"
SEED="${5:-${SEED:-3328683074}}"
PYTHON="${PYTHON:-python}"

PROJECT_DIR="$(resolve_project_dir)"
require_project_file "$PROJECT_DIR" "pdcc_main.py"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

cd "$PROJECT_DIR"
mkdir -p "$RUN_DIR/models" "$RUN_DIR/logs"

echo "[PDCC-MER] Project: $PROJECT_DIR"
echo "[PDCC-MER] Dataset: $DATASET"
echo "[PDCC-MER] Data path: $DATA_PATH"
echo "[PDCC-MER] Run dir: $RUN_DIR"
echo "[PDCC-MER] GPU: $CUDA_VISIBLE_DEVICES"
echo "[PDCC-MER] Seed: $SEED"

echo "[PDCC-MER] Stage 1: consensus pretraining with TPLR"
"$PYTHON" -u pdcc_main.py \
  --dataset "$DATASET" \
  --data_path "$DATA_PATH" \
  --model_path "$RUN_DIR/models" \
  --run_dir "$RUN_DIR" \
  --use_best \
  --is_pseudo \
  --use_tplr \
  --seed "$SEED" \
  2>&1 | tee "$RUN_DIR/logs/stage1.log"

echo "[PDCC-MER] Stage 2: finetuning with TPLR + PCRP + RCCR"
"$PYTHON" -u pdcc_main.py \
  --dataset "$DATASET" \
  --data_path "$DATA_PATH" \
  --model_path "$RUN_DIR/models" \
  --run_dir "$RUN_DIR" \
  --use_best \
  --is_pseudo \
  --finetune \
  --pretrained_model \
  --use_tplr \
  --use_pcrp \
  --use_rccr \
  --seed "$SEED" \
  2>&1 | tee "$RUN_DIR/logs/stage2.log"

echo "[PDCC-MER] Done. Metrics: $RUN_DIR/metrics.json"
