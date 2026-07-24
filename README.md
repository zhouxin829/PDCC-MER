# PDCC-MER

Official implementation scaffold for **PDCC-MER: Progressive Denoising and Calibrated Consensus Learning for Multimodal Emotion Recognition**.

PDCC-MER is a multimodal emotion recognition framework for learning reliable cross-modal consensus. The implementation contains the three main modules used in the paper:

- **TPLR**
- **PCRP**
- **RCCR**

This `github/` folder is prepared as a release-ready GitHub package. When publishing the code, place this README, `environment.yml`, `requirements.txt`, and the `scripts/` directory at the same repository level as `pdcc_main.py`, `src/`, `pdcc_best_config.py`, and `pdcc_multiseed_tools/`.

## Repository Layout

Expected layout after copying these files into the code repository:

```text
pdcc/
  README.md
  environment.yml
  requirements.txt
  pdcc_main.py
  pdcc_best_config.py
  summarize_pdcc_all_5runs.py
  collect_pdcc_8ablation_results.py
  src/
  pdcc_multiseed_tools/
    run_mechanism_controls.py
    summarize_mechanism_controls.py
    run_robustness_retrain.py
    summarize_robustness_retrain.py
    robustness_protocol.py
    run_clean_test_robustness.py
    eval_clean_test_robustness.py
    summarize_clean_test_robustness.py
    run_diagnostics_retrain.py
    analyze_retrained_rccr.py
    analyze_retrained_tplr.py
    summarize_retrain_diagnostics.py
    run_complexity_table.py
  scripts/
    train_pdcc_mer.sh
    evaluate_saved_metrics.py
    run_all_5seeds.sh
    run_8ablation.sh
    run_robustness.sh
    run_clean_test_robustness.sh
    run_mechanism_controls.sh
    run_diagnostics.sh
    run_computational_cost.sh
    summarize_results.sh
```

## Environment

The experiments in the revised manuscript were run with Python 3.10 and PyTorch 2.7.1.

Create a conda environment:

```bash
conda env create -f environment.yml
conda activate pdcc
```

Or install with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA, install the PyTorch build that matches your local driver and CUDA runtime. The provided environment file uses the CUDA-enabled PyTorch channel. For CPU-only debugging, replace the PyTorch installation with the official CPU build.

## Datasets

PDCC-MER expects preprocessed multimodal features. The code supports:

- `SIMS`
- `SIMS-v2`
- `MOSI`
- `MOSEI`

## Quick Start

Run the two-stage PDCC-MER training and evaluation pipeline on one dataset:

```bash
bash scripts/train_pdcc_mer.sh SIMS "$DATA_PATH" runs/quickstart/SIMS/seed_3328683074 0 3328683074
```

Arguments:

```text
bash scripts/train_pdcc_mer.sh <dataset> <data_path> <run_dir> <gpu_id> <seed>
```

Example for MOSI:

```bash
bash scripts/train_pdcc_mer.sh MOSI "$DATA_PATH" runs/quickstart/MOSI/seed_3328683074 0 3328683074
```

The script runs:

1. Stage 1 consensus pretraining with TPLR.
2. Stage 2 finetuning with TPLR, PCRP, and RCCR.

Main outputs are saved under `run_dir`:

```text
run_dir/
  logs/
    stage1.log
    stage2.log
  models/
  pseudo_labels/
  stage1_summary.json
  metrics.json
```

## Direct Commands

The single-run script is equivalent to the following commands.

Stage 1:

```bash
python -u pdcc_main.py \
  --dataset SIMS \
  --data_path "$DATA_PATH" \
  --model_path runs/quickstart/SIMS/seed_3328683074/models \
  --run_dir runs/quickstart/SIMS/seed_3328683074 \
  --use_best \
  --is_pseudo \
  --use_tplr \
  --seed 3328683074
```

Stage 2:

```bash
python -u pdcc_main.py \
  --dataset SIMS \
  --data_path "$DATA_PATH" \
  --model_path runs/quickstart/SIMS/seed_3328683074/models \
  --run_dir runs/quickstart/SIMS/seed_3328683074 \
  --use_best \
  --is_pseudo \
  --finetune \
  --pretrained_model \
  --use_tplr \
  --use_pcrp \
  --use_rccr \
  --seed 3328683074
```

Use `--use_best` to load the dataset-specific hyperparameters from `pdcc_best_config.py`.

## Reproduce Main Multi-Seed Results

Run five seeds on one or more datasets:

```bash
bash scripts/run_all_5seeds.sh "$DATA_PATH" runs/pdcc_all_5runs 0 SIMS,MOSI,MOSEI
```

Default seeds:

```text
3328683074, 4136559363, 1686802513, 2124692648, 964165003
```

Override the seed list:

```bash
SEEDS=3328683074,1974074723,1686464603 \
bash scripts/run_all_5seeds.sh "$DATA_PATH" runs/pdcc_all_3runs 0 SIMS,MOSI
```

After training, summarize the results:

```bash
bash scripts/summarize_results.sh 5run runs/pdcc_all_5runs
```

## Ablation Study

Run the eight module-combination ablations:

```bash
bash scripts/run_8ablation.sh "$DATA_PATH" runs/pdcc_8ablation 0 SIMS,MOSI
```

The combinations include:

```text
BASE
TPLR
PCRP
RCCR
TPLR_PCRP
TPLR_RCCR
PCRP_RCCR
PDCC_FULL
```

Summarize ablations:

```bash
bash scripts/summarize_results.sh ablation runs/pdcc_8ablation
```

## Mechanism Controls

Run the mechanism-control experiments used to analyze module interactions beyond the eight ablation combinations:

```bash
bash scripts/run_mechanism_controls.sh "$DATA_PATH" runs/mechanism_controls 0 SIMS,MOSI
```

Summarize:

```bash
bash scripts/summarize_results.sh mechanism runs/mechanism_controls
```

## Robustness Evaluation

Two complementary protocols are provided. For generalization to unseen
corruption, use the clean-train/corrupted-test protocol:

```bash
bash scripts/run_clean_test_robustness.sh \
  "$DATA_PATH" runs/clean_test_robustness 0 SIMS,MOSI
```

For each training seed, this protocol:

- trains `BASE`, `ROUTER_ONLY`, and full `PDCC_MER` on clean data;
- shares the exact Stage-1 TPLR artifacts between `ROUTER_ONLY` and `PDCC_MER`;
- freezes each final checkpoint;
- evaluates fixed missing modalities, three noise levels (`0.1`, `0.2`, `0.4`),
  sample-random missing modalities, and cross-sample modality misalignment;
- reports binary performance, paired degradation from Clean, ECE/NLL/Brier,
  per-modality entropy, entropy-derived reliability priors, and router weights.

The summary files are written to:

```text
runs/clean_test_robustness/summary/per_seed_metrics.csv
runs/clean_test_robustness/summary/mean_std_metrics.csv
runs/clean_test_robustness/summary/binary_performance.md
runs/clean_test_robustness/summary/binary_calibration.md
runs/clean_test_robustness/summary/rccr_reliability_response.md
runs/clean_test_robustness/summary/binary_performance_table.tex
```

To regenerate only the summary after all evaluations finish:

```bash
bash scripts/summarize_results.sh clean-robustness runs/clean_test_robustness
```

The older condition-specific adaptation protocol remains available:

```bash
bash scripts/run_robustness.sh "$DATA_PATH" runs/robustness_retrain 0 SIMS,MOSI
```

It retrains the complete two-stage model independently under every degraded
training/validation/test condition. Its results measure condition-specific
adaptability and must not be described as clean-train/corrupted-test robustness.

Summarize:

```bash
bash scripts/summarize_results.sh robustness runs/robustness_retrain
```

## RCCR, TPLR, and Cost Diagnostics

Run the retrained diagnostic experiments:

```bash
bash scripts/run_diagnostics.sh "$DATA_PATH" runs/retrain_diagnostics 0 SIMS,MOSI
```

This wrapper runs:

- `run_diagnostics_retrain.py`
- `analyze_retrained_rccr.py`
- `analyze_retrained_tplr.py`
- `summarize_retrain_diagnostics.py`

Summarize:

```bash
bash scripts/summarize_results.sh diagnostics runs/retrain_diagnostics
```

## Computational Cost and FLOPs

Run the complete two-stage cost experiment and then reuse the same Stage 2
checkpoints for deployable-parameter, train-step, inference, CUDA-memory, and
FLOP profiling:

```bash
bash scripts/run_computational_cost.sh \
  "$DATA_PATH" \
  runs/computational_cost \
  runs/complexity_profile \
  0 \
  SIMS,MOSI
```

The default protocol uses three seeds from the `SEEDS` environment variable.
It produces:

```text
runs/computational_cost/summary/cost_table.csv
runs/computational_cost/summary/cost_table.md
runs/computational_cost/summary/cost_table_latex.txt
runs/computational_cost/summary/cost_measurement_notes.md
runs/complexity_profile/complexity_per_seed.csv
runs/complexity_profile/complexity_table.csv
runs/complexity_profile/complexity_table.md
runs/complexity_profile/complexity_table_latex.txt
runs/complexity_profile/complexity_measurement_notes.md
```

If SIMS and MOSI were profiled in separate processes but written below the
same profile root, regenerate the combined table without repeating GPU
profiling:

```bash
python pdcc_multiseed_tools/run_complexity_table.py \
  --run-root runs/complexity_profile \
  --datasets SIMS,MOSI \
  --variants Base,Base+TPLR,Base+PCRP,Base+RCCR,PDCC-MER \
  --seeds 3328683074,4136559363,1686802513 \
  --summary-only
```

`Total time` is computed per seed as Stage 1 plus Stage 2 before aggregation.
`Pseudo-label storage` is the combined size of the train/validation/test
pseudo-label files. Stage-level peak GPU memory is process memory sampled with
`nvidia-smi`. The profiling-table memory column is inference-only allocated
CUDA memory from `torch.cuda.max_memory_allocated()` and is intentionally
reported under a different name.

FLOPs are operator-level estimates from
`torch.profiler.profile(with_flops=True)`. They cover supported PyTorch
operators in the profiled train step or deployed forward pass; they do not
include data I/O, checkpoint serialization, EMA parameter updates, or
pseudo-label memory operations. Use `--skip-flops` only when the installed
PyTorch build cannot provide FLOP profiling.

## Inspect Saved Metrics

Print saved `metrics.json` files:

```bash
python scripts/evaluate_saved_metrics.py runs/quickstart/SIMS/seed_3328683074
```

Scan a whole run root and save a CSV:

```bash
python scripts/evaluate_saved_metrics.py runs/pdcc_all_5runs --csv runs/pdcc_all_5runs/metrics_index.csv
```

## Common Options

All shell wrappers support these environment variables:

```bash
PYTHON=python
PROJECT_DIR=/path/to/pdcc
SEEDS=3328683074,1974074723,1686464603
CUDA_VISIBLE_DEVICES=0
```

The scripts also accept a GPU id argument and set `CUDA_VISIBLE_DEVICES` internally.

## Naming and Legacy Artifacts

- The paper and method name is `PDCC-MER`.
- Python modules, scripts, environment names, and repository paths use lowercase `pdcc`.
- New method artifacts use the `PDCC` tag, for example
  `SIMS_PDCC_PDCCModel_Has0_acc_2.pt` and
  `SIMS_PDCC_train_pseudo_labels.pkl`.
- The lightweight legacy entry points and artifact resolvers can still read
  checkpoints and pseudo-label banks created before the rename. New runs do
  not write legacy names.

## Notes on Reproducibility

- Use the same processed feature files and the same random seeds when comparing with reported results.
- Use `--use_best` for the tuned dataset-specific hyperparameters.
- Each run should have an isolated `run_dir` so that checkpoints and pseudo labels do not overlap.
- Report mean and standard deviation over multiple seeds. The summary scripts output table-ready statistics from saved run artifacts.
