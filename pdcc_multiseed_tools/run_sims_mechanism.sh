#!/usr/bin/env bash
set -euo pipefail
cd /data/Lab105/zhouxin/pdcc-revise/pdcc
python pdcc_multiseed_tools/run_mechanism_controls.py \
  --project-dir /data/Lab105/zhouxin/pdcc-revise/pdcc \
  --entry pdcc_main.py \
  --python /home/shenxiang/miniconda3/envs/cmc/bin/python \
  --dataset SIMS \
  --data-path /data/Lab105/zhouxin/MSA_Datasets \
  --run-root /data/Lab105/zhouxin/pdcc-revise/pdcc/runs/mechanism_controls_v1 \
  --seeds 3328683074,4136559363,1686802513 \
  --gpu 5
