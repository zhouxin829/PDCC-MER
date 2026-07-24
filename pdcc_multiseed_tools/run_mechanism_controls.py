#!/usr/bin/env python3
"""Mechanism-control experiments for TPLR, PCRP and RCCR.

The runner executes each dataset sequentially on one GPU.  Run one copy for
SIMS and a second copy for MOSI on another GPU.  Every run gets an isolated
model/pseudo-label/log directory.

Protocol:
  TPLR: No TPLR, EMA-only, student-only, one-step TPLR, full TPLR.
  PCRP: 0, 1, 2, 3, 4 propagation steps.  The dataset's tuned step count is
        represented by the common FULL reference run.
  RCCR: router-only, reliability-prior-only, full calibrated routing.

All commands include --use_best.  The patched pdcc_main.py preserves an explicit
mechanism control (e.g. --pcrp_steps 1) while applying every other dataset-
specific value from pdcc_best_config.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from artifact_naming import (
    all_artifacts_exist,
    artifact_exists,
    resolve_existing_artifact,
)

TAG = "PDCC"
TRACKS = ("tplr", "pcrp", "rccr")


def parse_seeds(text: str) -> list[int]:
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--seeds must be a non-empty comma-separated unique list")
    return values


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_best(project_dir: Path, dataset: str) -> dict[str, Any]:
    cfg_path = project_dir / "pdcc_best_config.py"
    spec = importlib.util.spec_from_file_location("pdcc_best_config", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return dict(module.BEST_CONFIGS[dataset])
    except KeyError as exc:
        raise ValueError(f"No use_best configuration for dataset={dataset}") from exc


def stage1_required(run_dir: Path, dataset: str) -> list[Path]:
    model_dir = run_dir / "models"
    pseudo_dir = run_dir / "pseudo_labels"
    return [
        model_dir / f"{dataset}_{TAG}_Pseudo_TextModel.pt",
        model_dir / f"{dataset}_{TAG}_Pseudo_AudioModel.pt",
        model_dir / f"{dataset}_{TAG}_Pseudo_VisionModel.pt",
        pseudo_dir / f"{dataset}_{TAG}_train_pseudo_labels.pkl",
        pseudo_dir / f"{dataset}_{TAG}_valid_pseudo_labels.pkl",
        pseudo_dir / f"{dataset}_{TAG}_test_pseudo_labels.pkl",
    ]


def completed_stage2(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").is_file()


def exists_all(paths: list[Path]) -> bool:
    return all_artifacts_exist(paths)


def ensure_dirs(run_dir: Path) -> tuple[Path, Path, Path]:
    models = run_dir / "models"
    pseudos = run_dir / "pseudo_labels"
    logs = run_dir / "logs"
    for item in (run_dir, models, pseudos, logs):
        item.mkdir(parents=True, exist_ok=True)
    return models, pseudos, logs


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path, dry_run: bool) -> None:
    print("[CMD]", shlex.join(cmd), flush=True)
    if dry_run:
        return
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(cmd) + "\n\n")
        log.flush()
        process = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Exit={process.returncode}; inspect {log_path}")


def base_command(args: argparse.Namespace, run_dir: Path, seed: int) -> list[str]:
    return [
        args.python, args.entry,
        "--dataset", args.dataset,
        "--data_path", args.data_path,
        "--model_path", str(run_dir / "models"),
        "--run_dir", str(run_dir),
        "--use_best", "--is_pseudo", "--seed", str(seed),
    ]


def run_two_stages(
    args: argparse.Namespace,
    run_dir: Path,
    seed: int,
    stage1_flags: list[str],
    stage2_flags: list[str],
    name: str,
) -> None:
    models, pseudos, logs = ensure_dirs(run_dir)
    stage1 = stage1_required(run_dir, args.dataset)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    write_json(run_dir / "mechanism_manifest.json", {
        "dataset": args.dataset,
        "seed": seed,
        "name": name,
        "stage1_flags": stage1_flags,
        "stage2_flags": stage2_flags,
        "use_best": True,
        "entry": args.entry,
        "created_at": time.time(),
    })

    if not exists_all(stage1) or args.force:
        print(f"[RUN] {args.dataset} {name} seed={seed} Stage-1", flush=True)
        run_command(base_command(args, run_dir, seed) + stage1_flags,
                    Path(args.project_dir), env, logs / "stage1.log", args.dry_run)
        if not args.dry_run:
            missing = [str(x) for x in stage1 if not artifact_exists(x)]
            if missing:
                raise FileNotFoundError("Stage-1 outputs missing:\n" + "\n".join(missing))
    else:
        print(f"[RESUME] Stage-1 ready: {run_dir}", flush=True)

    if completed_stage2(run_dir) and not args.force:
        print(f"[SKIP] Stage-2 ready: {run_dir}", flush=True)
        return

    print(f"[RUN] {args.dataset} {name} seed={seed} Stage-2", flush=True)
    cmd = base_command(args, run_dir, seed) + ["--finetune", "--pretrained_model"] + stage2_flags
    run_command(cmd, Path(args.project_dir), env, logs / "stage2.log", args.dry_run)
    if not args.dry_run and not completed_stage2(run_dir):
        raise FileNotFoundError(f"Stage-2 did not create {run_dir / 'metrics.json'}")


def symlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def materialize_full_stage1(full_dir: Path, target_dir: Path, dataset: str) -> None:
    """Reuse the same full-TPLR Stage-1 artifacts for Stage-2-only controls."""
    source_files = stage1_required(full_dir, dataset)
    if not exists_all(source_files):
        raise FileNotFoundError(f"Full reference Stage-1 is incomplete: {full_dir}")
    ensure_dirs(target_dir)
    for canonical_source in source_files:
        source = resolve_existing_artifact(canonical_source)
        if source.parent.name == "models":
            target = target_dir / "models" / source.name
        else:
            target = target_dir / "pseudo_labels" / source.name
        symlink_or_copy(source, target)
    summary = full_dir / "stage1_summary.json"
    if summary.is_file():
        symlink_or_copy(summary, target_dir / "stage1_summary.json")


def run_stage2_from_full_stage1(
    args: argparse.Namespace,
    full_dir: Path,
    target_dir: Path,
    seed: int,
    stage2_flags: list[str],
    name: str,
) -> None:
    # In dry-run mode the FULL Stage-1 files do not exist by design; only print
    # the commands and planned directories.  A real run materializes them first.
    if not args.dry_run:
        materialize_full_stage1(full_dir, target_dir, args.dataset)
    _, _, logs = ensure_dirs(target_dir)
    if completed_stage2(target_dir) and not args.force:
        print(f"[SKIP] Stage-2 ready: {target_dir}", flush=True)
        return

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    write_json(target_dir / "mechanism_manifest.json", {
        "dataset": args.dataset,
        "seed": seed,
        "name": name,
        "stage1_source": str(full_dir),
        "stage2_flags": stage2_flags,
        "use_best": True,
        "entry": args.entry,
        "created_at": time.time(),
    })
    print(f"[RUN] {args.dataset} {name} seed={seed} Stage-2 (reuse FULL Stage-1)", flush=True)
    cmd = base_command(args, target_dir, seed) + ["--finetune", "--pretrained_model"] + stage2_flags
    run_command(cmd, Path(args.project_dir), env, logs / "stage2.log", args.dry_run)
    if not args.dry_run and not completed_stage2(target_dir):
        raise FileNotFoundError(f"Stage-2 did not create {target_dir / 'metrics.json'}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True)
    p.add_argument("--entry", default="pdcc_main.py")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dataset", required=True, choices=["SIMS", "MOSI"])
    p.add_argument("--data-path", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--seeds", default="3328683074,4136559363,1686802513")
    p.add_argument("--gpu", default=None)
    p.add_argument("--tracks", nargs="+", choices=TRACKS, default=list(TRACKS))
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    project = Path(args.project_dir).resolve()
    args.project_dir = str(project)
    if not (project / args.entry).is_file():
        raise FileNotFoundError(project / args.entry)
    best = load_best(project, args.dataset)
    default_pcrp_steps = int(best["pcrp_steps"])
    root = Path(args.run_root).resolve() / args.dataset
    seeds = parse_seeds(args.seeds)

    run_config = {
        "dataset": args.dataset,
        "seeds": seeds,
        "tracks": args.tracks,
        "best_config": best,
        "entry": args.entry,
        "gpu": args.gpu,
        "protocol_version": "mechanism_controls_v1",
    }
    write_json(root / "experiment_config.json", run_config)

    # Full reference: Stage-1 TPLR plus full Stage-2 PCRP/RCCR.
    for seed in seeds:
        full_dir = root / "FULL" / f"seed_{seed}"
        run_two_stages(args, full_dir, seed, ["--use_tplr"],
                       ["--use_tplr", "--use_pcrp", "--use_rccr"], "FULL")

    if "tplr" in args.tracks:
        # The absence of TPLR must still produce an PDCC-tagged Stage-1 checkpoint
        # because Stage-2 keeps PCRP/RCCR.  These flags are ignored by Stage-1 but
        # ensure its checkpoint/pseudo-label filenames match Stage-2.
        tplr_variants = {
            "NO_TPLR": (
                ["--use_pcrp", "--use_rccr"],
                ["--use_pcrp", "--use_rccr"],
            ),
            "EMA_ONLY": (
                ["--use_tplr", "--tplr_mode", "teacher_only"],
                ["--use_tplr", "--tplr_mode", "teacher_only", "--use_pcrp", "--use_rccr"],
            ),
            "STUDENT_ONLY": (
                ["--use_tplr", "--tplr_mode", "student_only"],
                ["--use_tplr", "--tplr_mode", "student_only", "--use_pcrp", "--use_rccr"],
            ),
            "ONE_STEP": (
                ["--use_tplr", "--tplr_steps", "1"],
                ["--use_tplr", "--tplr_steps", "1", "--use_pcrp", "--use_rccr"],
            ),
        }
        for seed in seeds:
            for name, (s1, s2) in tplr_variants.items():
                run_two_stages(args, root / "TPLR" / name / f"seed_{seed}", seed, s1, s2, f"TPLR/{name}")

    if "pcrp" in args.tracks:
        # Full reference represents the tuned default step count; all other step
        # controls reuse exactly its Stage-1 checkpoint and pseudo labels.
        for seed in seeds:
            full_dir = root / "FULL" / f"seed_{seed}"
            for steps in range(0, 5):
                if steps == default_pcrp_steps:
                    continue
                name = f"STEPS_{steps}"
                flags = ["--use_tplr", "--use_rccr"]
                if steps > 0:
                    flags += ["--use_pcrp", "--pcrp_steps", str(steps)]
                run_stage2_from_full_stage1(args, full_dir,
                    root / "PCRP" / name / f"seed_{seed}", seed, flags, f"PCRP/{name}")

    if "rccr" in args.tracks:
        for seed in seeds:
            full_dir = root / "FULL" / f"seed_{seed}"
            rccr_variants = {
                "ROUTER_ONLY": ["--use_tplr", "--use_pcrp"],
                "PRIOR_ONLY": ["--use_tplr", "--use_pcrp", "--use_rccr", "--rccr_lambda", "1.0"],
            }
            for name, flags in rccr_variants.items():
                run_stage2_from_full_stage1(args, full_dir,
                    root / "RCCR" / name / f"seed_{seed}", seed, flags, f"RCCR/{name}")

    print("[DONE]", root)
    print("Run the summary command after completion:")
    print(f"  {shlex.quote(args.python)} pdcc_multiseed_tools/summarize_mechanism_controls.py --root {shlex.quote(str(Path(args.run_root).resolve()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
