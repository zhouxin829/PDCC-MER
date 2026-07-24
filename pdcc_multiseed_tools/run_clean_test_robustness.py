#!/usr/bin/env python3
"""Train on clean data once, then evaluate frozen models under corruption.

ROUTER_ONLY and PDCC_MER share the exact same Stage-1 TPLR artifacts for each
model seed. Their Stage-2 comparison therefore isolates reliability calibration
without introducing a different unimodal initialization or pseudo-label bank.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from artifact_naming import (
    all_artifacts_exist,
    artifact_exists,
    resolve_existing_artifact,
)
from robustness_protocol import DEFAULT_CONDITIONS, parse_conditions, parse_seeds


MODEL_VARIANTS = {
    "BASE": {
        "stage1_group": "BASE",
        "stage1_flags": [],
        "stage2_flags": [],
        "description": "base fusion model without TPLR, PCRP, or RCCR",
    },
    "ROUTER_ONLY": {
        "stage1_group": "TPLR",
        "stage1_flags": ["--use_tplr"],
        "stage2_flags": ["--use_tplr", "--use_pcrp"],
        "description": "standard learned router without RCCR reliability calibration",
    },
    "PDCC_MER": {
        "stage1_group": "TPLR",
        "stage1_flags": ["--use_tplr"],
        "stage2_flags": ["--use_tplr", "--use_pcrp", "--use_rccr"],
        "description": "complete PDCC-MER with reliability-calibrated routing",
    },
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_models(text: str) -> list[str]:
    aliases = {
        "FULL": "PDCC_MER",
        "PDCC-MER": "PDCC_MER",
        "ROUTER": "ROUTER_ONLY",
    }
    models = []
    for item in [part.strip().upper() for part in text.split(",") if part.strip()]:
        model = aliases.get(item, item)
        if model not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model {item!r}; choose from {','.join(MODEL_VARIANTS)}"
            )
        if model not in models:
            models.append(model)
    if not models:
        raise ValueError("--models is empty")
    return models


def tag_from_flags(flags: list[str]) -> str:
    return "PDCC" if any(
        flag in flags for flag in ("--use_tplr", "--use_pcrp", "--use_rccr")
    ) else "BASE"


def expected_stage1(run_dir: Path, dataset: str, tag: str) -> list[Path]:
    return [
        run_dir / "models" / f"{dataset}_{tag}_Pseudo_TextModel.pt",
        run_dir / "models" / f"{dataset}_{tag}_Pseudo_AudioModel.pt",
        run_dir / "models" / f"{dataset}_{tag}_Pseudo_VisionModel.pt",
        run_dir / "pseudo_labels" / f"{dataset}_{tag}_train_pseudo_labels.pkl",
        run_dir / "pseudo_labels" / f"{dataset}_{tag}_valid_pseudo_labels.pkl",
        run_dir / "pseudo_labels" / f"{dataset}_{tag}_test_pseudo_labels.pkl",
        run_dir / "stage1_summary.json",
    ]


def expected_stage2(run_dir: Path, dataset: str, tag: str) -> list[Path]:
    return [
        run_dir / "models" / f"{dataset}_{tag}_PDCCModel_Has0_acc_2.pt",
        run_dir / "metrics.json",
    ]


def all_exist(paths: list[Path]) -> bool:
    return all_artifacts_exist(paths)


def base_command(
    args: argparse.Namespace,
    run_dir: Path,
    seed: int,
) -> list[str]:
    return [
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
        "--robust_mode",
        "clean",
        "--robust_modality",
        "none",
        "--robust_level",
        "0",
        "--robust_scope",
        "all",
        "--robust_seed",
        str(args.robust_seed),
        "--num_workers",
        str(args.num_workers),
    ]


def run_command(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        print("[DRY]", shlex.join(command), flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(command) + "\n\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Exit={process.returncode}; inspect {log_path}")


def materialize_file(source: Path, target: Path, overwrite: bool) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not overwrite:
            return "existing"
        target.unlink()
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def materialize_stage1(
    source_dir: Path,
    target_dir: Path,
    dataset: str,
    tag: str,
    overwrite: bool,
) -> dict[str, str]:
    actions: dict[str, str] = {}
    for canonical_source in expected_stage1(source_dir, dataset, tag):
        source = resolve_existing_artifact(canonical_source)
        if not source.is_file():
            raise FileNotFoundError(canonical_source)
        if source.name == "stage1_summary.json":
            target = target_dir / source.name
        elif source.parent.name == "models":
            target = target_dir / "models" / source.name
        else:
            target = target_dir / "pseudo_labels" / source.name
        actions[str(target)] = materialize_file(source, target, overwrite)
    return actions


def ensure_stage1(
    args: argparse.Namespace,
    root: Path,
    group: str,
    flags: list[str],
    seed: int,
    environment: dict[str, str],
) -> Path:
    run_dir = root / args.dataset / "_SHARED_STAGE1" / group / f"seed_{seed}"
    for subdir in ("models", "pseudo_labels", "logs"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    tag = tag_from_flags(flags)
    command = base_command(args, run_dir, seed) + flags
    outputs = expected_stage1(run_dir, args.dataset, tag)
    write_json(
        run_dir / "stage1_manifest.json",
        {
            "dataset": args.dataset,
            "stage1_group": group,
            "seed": seed,
            "tag": tag,
            "training_condition": "clean",
            "command": command,
        },
    )
    if not all_exist(outputs) or args.force_stage1:
        print(
            f"[RUN] {args.dataset} shared Stage-1 {group} seed={seed}",
            flush=True,
        )
        run_command(
            command,
            Path(args.project_dir),
            environment,
            run_dir / "logs" / "stage1.log",
            args.dry_run,
        )
    else:
        print(f"[RESUME] Shared Stage-1 exists: {run_dir}", flush=True)
    if not args.dry_run:
        missing = [str(path) for path in outputs if not artifact_exists(path)]
        if missing:
            raise FileNotFoundError(
                "Stage-1 outputs missing:\n" + "\n".join(missing)
            )
    return run_dir


def train_stage2(
    args: argparse.Namespace,
    root: Path,
    model_name: str,
    seed: int,
    stage1_dir: Path,
    environment: dict[str, str],
) -> tuple[Path, Path]:
    config = MODEL_VARIANTS[model_name]
    run_dir = root / args.dataset / model_name / f"seed_{seed}"
    for subdir in ("models", "pseudo_labels", "logs", "evaluations"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    stage1_tag = tag_from_flags(config["stage1_flags"])
    stage2_tag = tag_from_flags(config["stage2_flags"])

    materialization = {}
    if not args.dry_run:
        materialization = materialize_stage1(
            stage1_dir,
            run_dir,
            args.dataset,
            stage1_tag,
            overwrite=args.force_stage1,
        )
    command = (
        base_command(args, run_dir, seed)
        + ["--finetune", "--pretrained_model"]
        + config["stage2_flags"]
    )
    checkpoint = (
        run_dir
        / "models"
        / f"{args.dataset}_{stage2_tag}_PDCCModel_Has0_acc_2.pt"
    )
    outputs = expected_stage2(run_dir, args.dataset, stage2_tag)
    write_json(
        run_dir / "clean_training_manifest.json",
        {
            "protocol": "clean training followed by frozen corrupted-test evaluation",
            "dataset": args.dataset,
            "model": model_name,
            "description": config["description"],
            "seed": seed,
            "training_condition": "clean",
            "stage1_group": config["stage1_group"],
            "shared_stage1_dir": str(stage1_dir),
            "stage1_materialization": materialization,
            "stage2_command": command,
        },
    )

    if not all_exist(outputs) or args.force_stage2:
        print(
            f"[RUN] {args.dataset} {model_name} seed={seed} Stage-2",
            flush=True,
        )
        run_command(
            command,
            Path(args.project_dir),
            environment,
            run_dir / "logs" / "stage2.log",
            args.dry_run,
        )
    else:
        print(f"[RESUME] Stage-2 exists: {run_dir}", flush=True)
    if not args.dry_run:
        missing = [str(path) for path in outputs if not artifact_exists(path)]
        if missing:
            raise FileNotFoundError(
                "Stage-2 outputs missing:\n" + "\n".join(missing)
            )
    return run_dir, resolve_existing_artifact(checkpoint)


def evaluate_checkpoint(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: Path,
    model_name: str,
    seed: int,
    conditions_text: str,
    environment: dict[str, str],
) -> None:
    command = [
        args.python,
        args.evaluator,
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        args.dataset,
        "--data-path",
        args.data_path,
        "--output-dir",
        str(run_dir / "evaluations"),
        "--model-name",
        model_name,
        "--model-seed",
        str(seed),
        "--gpu",
        str(args.gpu),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.eval_workers),
        "--robust-seed",
        str(args.robust_seed),
        "--ece-bins",
        str(args.ece_bins),
        "--splits",
        args.splits,
        "--conditions",
        conditions_text,
    ]
    if args.no_per_sample:
        command.append("--no-per-sample")
    if args.force_eval:
        command.append("--force")
    print(
        f"[EVAL] {args.dataset} {model_name} seed={seed} frozen checkpoint",
        flush=True,
    )
    run_command(
        command,
        Path(args.project_dir),
        environment,
        run_dir / "logs" / "corrupted_test.log",
        args.dry_run,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--entry", default="pdcc_main.py")
    parser.add_argument(
        "--evaluator",
        default="pdcc_multiseed_tools/eval_clean_test_robustness.py",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset", required=True, choices=["SIMS", "MOSI"])
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--models", default="BASE,ROUTER_ONLY,PDCC_MER")
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    parser.add_argument("--splits", default="auto")
    parser.add_argument("--num-workers", type=int, default=14)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--robust-seed", type=int, default=20260707)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-per-sample", action="store_true")
    parser.add_argument("--force-stage1", action="store_true")
    parser.add_argument("--force-stage2", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    for relative in (args.entry, args.evaluator):
        if not (project / relative).is_file():
            raise FileNotFoundError(project / relative)
    if not Path(args.data_path).is_dir() and not args.dry_run:
        raise FileNotFoundError(f"Dataset directory not found: {args.data_path}")
    if args.num_workers < 0 or args.eval_workers < 0:
        raise ValueError("worker counts must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    root = Path(args.run_root).resolve()
    models = parse_models(args.models)
    seeds = parse_seeds(args.seeds)
    conditions = parse_conditions(args.conditions)
    conditions_text = ",".join(
        (
            "clean"
            if condition.mode == "clean"
            else (
                f"{condition.mode}:{condition.modality}"
                if condition.mode == "missing"
                else f"{condition.mode}:{condition.modality}:{condition.level}"
            )
        )
        for condition in conditions
    )
    write_json(
        root / args.dataset / "clean_test_robustness_manifest.json",
        {
            "protocol": "clean-train/corrupted-test",
            "dataset": args.dataset,
            "models": models,
            "seeds": seeds,
            "robust_seed": args.robust_seed,
            "splits": args.splits,
            "conditions": [condition.as_dict() for condition in conditions],
            "stage1_reuse": (
                "ROUTER_ONLY and PDCC_MER share identical TPLR Stage-1 artifacts "
                "within each model seed"
            ),
        },
    )

    for seed in seeds:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        environment["PYTHONHASHSEED"] = str(seed)
        stage1_dirs: dict[str, Path] = {}
        if not args.eval_only:
            for model_name in models:
                config = MODEL_VARIANTS[model_name]
                group = config["stage1_group"]
                if group not in stage1_dirs:
                    stage1_dirs[group] = ensure_stage1(
                        args,
                        root,
                        group,
                        config["stage1_flags"],
                        seed,
                        environment,
                    )

        for model_name in models:
            if args.eval_only:
                run_dir = root / args.dataset / model_name / f"seed_{seed}"
                tag = tag_from_flags(MODEL_VARIANTS[model_name]["stage2_flags"])
                checkpoint = resolve_existing_artifact(
                    run_dir
                    / "models"
                    / f"{args.dataset}_{tag}_PDCCModel_Has0_acc_2.pt"
                )
                if not checkpoint.is_file() and not args.dry_run:
                    raise FileNotFoundError(checkpoint)
            else:
                group = MODEL_VARIANTS[model_name]["stage1_group"]
                run_dir, checkpoint = train_stage2(
                    args,
                    root,
                    model_name,
                    seed,
                    stage1_dirs[group],
                    environment,
                )
            evaluate_checkpoint(
                args,
                run_dir,
                checkpoint,
                model_name,
                seed,
                conditions_text,
                environment,
            )

    print("[DONE] Clean-train/corrupted-test protocol completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
