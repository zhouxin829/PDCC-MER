#!/usr/bin/env python3
"""Run only the fixed label-smoothing control on one dataset.

For each candidate epsilon and seed, this runner executes the complete two-stage
pipeline in an isolated directory.  TPLR is replaced by fixed label smoothing
in Stage 1, while Stage 2 retains PCRP and RCCR.  Epsilon selection is performed
later by summarize_label_smoothing_control.py using validation metrics only.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from artifact_naming import all_artifacts_exist, artifact_exists


TAG = "PDCC"


def parse_csv(text, cast):
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("comma-separated argument cannot be empty")
    if len(values) != len(set(values)):
        raise ValueError("comma-separated argument must contain unique values")
    return values


def epsilon_token(epsilon):
    return f"EPS_{epsilon:g}".replace(".", "p")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def expected_stage1(run_dir, dataset):
    return [
        run_dir / "models" / f"{dataset}_{TAG}_Pseudo_TextModel.pt",
        run_dir / "models" / f"{dataset}_{TAG}_Pseudo_AudioModel.pt",
        run_dir / "models" / f"{dataset}_{TAG}_Pseudo_VisionModel.pt",
        run_dir / "pseudo_labels" / f"{dataset}_{TAG}_train_pseudo_labels.pkl",
        run_dir / "pseudo_labels" / f"{dataset}_{TAG}_valid_pseudo_labels.pkl",
        run_dir / "pseudo_labels" / f"{dataset}_{TAG}_test_pseudo_labels.pkl",
        run_dir / "stage1_summary.json",
    ]


def expected_stage2(run_dir, dataset):
    return [
        run_dir / "models" / f"{dataset}_{TAG}_PDCCModel_Has0_acc_2.pt",
        run_dir / "metrics.json",
    ]


def complete(paths):
    return all_artifacts_exist(paths)


def run_command(command, cwd, env, log_path, dry_run):
    print("[CMD]", shlex.join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Exit={process.returncode}; inspect {log_path}")


def common_command(args, run_dir, seed, epsilon):
    command = [
        args.python,
        args.entry,
        "--dataset",
        args.dataset,
        "--data_path",
        args.data_path,
        "--model_path",
        str(run_dir / "models"),
        "--run_dir",
        str(run_dir),
        "--use_best",
        "--is_pseudo",
        "--seed",
        str(seed),
        "--use_label_smoothing_control",
        "--label_smoothing_epsilon",
        f"{epsilon:g}",
    ]
    if args.num_workers is not None:
        command += ["--num_workers", str(args.num_workers)]
    return command


def run_one(args, epsilon, seed):
    project_dir = Path(args.project_dir).resolve()
    dataset_dir = Path(args.run_root).resolve() / args.dataset
    run_dir = dataset_dir / epsilon_token(epsilon) / f"seed_{seed}"
    for path in (run_dir, run_dir / "models", run_dir / "pseudo_labels", run_dir / "logs"):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": "fixed_label_smoothing_control_v1",
        "dataset": args.dataset,
        "seed": seed,
        "epsilon": epsilon,
        "selection_metric": "valid/Has0_acc_2",
        "target_definition": "(1-epsilon)*one_hot+epsilon/3",
        "ema_teacher": False,
        "progressive_perturbation": False,
        "historical_pseudo_label_memory": False,
        "stage2_modules": ["PCRP", "RCCR"],
        "stage1_flags": [
            "--use_label_smoothing_control",
            "--label_smoothing_epsilon",
            f"{epsilon:g}",
        ],
        "stage2_flags": [
            "--use_label_smoothing_control",
            "--label_smoothing_epsilon",
            f"{epsilon:g}",
            "--use_pcrp",
            "--use_rccr",
        ],
        "created_at": time.time(),
    }
    write_json(run_dir / "label_smoothing_manifest.json", manifest)

    stage1_files = expected_stage1(run_dir, args.dataset)
    stage2_files = expected_stage2(run_dir, args.dataset)
    if complete(stage2_files) and not args.force:
        print(
            f"[SKIP] {args.dataset} epsilon={epsilon:g} seed={seed} is complete",
            flush=True,
        )
        return

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    base = common_command(args, run_dir, seed, epsilon)
    if not complete(stage1_files) or args.force:
        print(
            f"[RUN] {args.dataset} LABEL_SMOOTHING epsilon={epsilon:g} "
            f"seed={seed} Stage-1",
            flush=True,
        )
        run_command(
            base,
            project_dir,
            env,
            run_dir / "logs" / "stage1.log",
            args.dry_run,
        )
        if not args.dry_run:
            missing = [str(path) for path in stage1_files if not artifact_exists(path)]
            if missing:
                raise FileNotFoundError(
                    "Stage-1 outputs are incomplete:\n" + "\n".join(missing)
                )
    else:
        print(f"[RESUME] Stage-1 ready: {run_dir}", flush=True)

    print(
        f"[RUN] {args.dataset} LABEL_SMOOTHING epsilon={epsilon:g} "
        f"seed={seed} Stage-2",
        flush=True,
    )
    run_command(
        base
        + [
            "--finetune",
            "--pretrained_model",
            "--use_pcrp",
            "--use_rccr",
        ],
        project_dir,
        env,
        run_dir / "logs" / "stage2.log",
        args.dry_run,
    )
    if not args.dry_run:
        missing = [str(path) for path in stage2_files if not artifact_exists(path)]
        if missing:
            raise FileNotFoundError(
                "Stage-2 outputs are incomplete:\n" + "\n".join(missing)
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--entry", default="pdcc_main.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset", required=True, choices=["SIMS", "MOSI"])
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--epsilons",
        default="0.05,0.1,0.2",
        help="positive candidates selected using validation Has0 Acc-2",
    )
    parser.add_argument(
        "--seeds", default="3328683074,4136559363,1686802513"
    )
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    if not (project_dir / args.entry).is_file():
        raise FileNotFoundError(project_dir / args.entry)
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    epsilons = parse_csv(args.epsilons, float)
    if any(not 0.0 < epsilon < 1.0 for epsilon in epsilons):
        raise ValueError("--epsilons must contain values strictly between 0 and 1")
    seeds = parse_csv(args.seeds, int)

    dataset_dir = Path(args.run_root).resolve() / args.dataset
    write_json(
        dataset_dir / "label_smoothing_experiment_config.json",
        {
            "protocol": "fixed_label_smoothing_control_v1",
            "dataset": args.dataset,
            "epsilons": epsilons,
            "seeds": seeds,
            "selection_split": "valid",
            "selection_metric": "Has0_acc_2",
            "selection_rule": (
                "maximum mean validation Has0_acc_2 across configured seeds; "
                "lower epsilon breaks an exact tie"
            ),
            "test_policy": "test metrics are reported only after epsilon selection",
            "target_definition": "(1-epsilon)*one_hot+epsilon/3",
            "stage2_modules": ["PCRP", "RCCR"],
            "gpu": args.gpu,
            "num_workers": args.num_workers,
        },
    )

    for epsilon in epsilons:
        for seed in seeds:
            run_one(args, epsilon, seed)

    print(f"[DONE] {dataset_dir}")
    print("After both datasets finish, run:")
    print(
        f"  {shlex.quote(args.python)} "
        "pdcc_multiseed_tools/summarize_label_smoothing_control.py "
        f"--root {shlex.quote(str(Path(args.run_root).resolve()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
