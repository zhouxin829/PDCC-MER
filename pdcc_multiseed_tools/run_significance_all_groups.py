#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from artifact_naming import all_artifacts_exist


METHODS = {
    "BASE": {
        "stage1_flags": [],
        "stage2_flags": [],
        "tag": "BASE",
        "description": "Baseline with --use_best backbone/config and no module flags.",
    },
    "PDCC": {
        "stage1_flags": ["--use_tplr"],
        "stage2_flags": ["--use_tplr", "--use_pcrp", "--use_rccr"],
        "tag": "PDCC",
        "description": "Full PDCC-MER with TPLR + PCRP + RCCR and --use_best config.",
    },
}


def parse_csv(text: str, cast=str) -> list:
    values = [cast(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("empty comma-separated argument")
    return values


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[CMD]", shlex.join(cmd), flush=True)
    if dry_run:
        return
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Exit={proc.returncode}; inspect {log_path}")


def expected_stage1(run_dir: Path, dataset: str, tag: str) -> list[Path]:
    return [
        run_dir / "models" / f"{dataset}_{tag}_Pseudo_TextModel.pt",
        run_dir / "models" / f"{dataset}_{tag}_Pseudo_AudioModel.pt",
        run_dir / "models" / f"{dataset}_{tag}_Pseudo_VisionModel.pt",
        run_dir / "pseudo_labels" / f"{dataset}_{tag}_train_pseudo_labels.pkl",
        run_dir / "pseudo_labels" / f"{dataset}_{tag}_valid_pseudo_labels.pkl",
        run_dir / "pseudo_labels" / f"{dataset}_{tag}_test_pseudo_labels.pkl",
    ]


def expected_stage2(run_dir: Path, dataset: str, tag: str) -> list[Path]:
    return [
        run_dir / "models" / f"{dataset}_{tag}_PDCCModel_Has0_acc_2.pt",
        run_dir / "metrics.json",
    ]


def complete(paths: list[Path]) -> bool:
    return all_artifacts_exist(paths)


def common_cmd(args: argparse.Namespace, dataset: str, run_dir: Path, seed: int) -> list[str]:
    return [
        args.python,
        args.entry,
        "--dataset", dataset,
        "--data_path", args.data_path,
        "--model_path", str(run_dir / "models"),
        "--run_dir", str(run_dir),
        "--use_best",
        "--is_pseudo",
        "--seed", str(seed),
    ]


def run_one(args: argparse.Namespace, dataset: str, method: str, seed: int) -> dict[str, Any]:
    cfg = METHODS[method]
    tag = cfg["tag"]
    project_dir = Path(args.project_dir).resolve()
    run_dir = Path(args.run_root).resolve() / dataset / method / f"seed_{seed}"

    if args.force and run_dir.exists():
        print(f"[FORCE] removing {run_dir}", flush=True)
        shutil.rmtree(run_dir)

    for d in (run_dir, run_dir / "models", run_dir / "pseudo_labels", run_dir / "logs"):
        d.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "significance_manifest.json", {
        "dataset": dataset,
        "method": method,
        "tag": tag,
        "seed": seed,
        "description": cfg["description"],
        "run_dir": str(run_dir),
        "model_path": str(run_dir / "models"),
        "pseudo_dir": str(run_dir / "pseudo_labels"),
        "stage1_flags": cfg["stage1_flags"],
        "stage2_flags": ["--finetune", "--pretrained_model"] + cfg["stage2_flags"],
    })

    s1_files = expected_stage1(run_dir, dataset, tag)
    s2_files = expected_stage2(run_dir, dataset, tag)

    if complete(s2_files) and not args.force:
        print(f"[SKIP] completed: {dataset}/{method}/seed_{seed}", flush=True)
        return {"dataset": dataset, "method": method, "seed": seed, "status": "skipped_completed", "run_dir": str(run_dir)}

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    base = common_cmd(args, dataset, run_dir, seed)
    t0 = time.time()

    if complete(s1_files) and not args.force:
        print(f"[RESUME] Stage 1 exists: {dataset}/{method}/seed_{seed}", flush=True)
        s1_status = "skipped_existing"
    else:
        print(f"[RUN] {dataset} {method} seed={seed} Stage 1", flush=True)
        run_command(base + cfg["stage1_flags"], project_dir, env, run_dir / "logs" / "stage1.log", args.dry_run)
        s1_status = "ran"
        if not args.dry_run:
            missing = [str(p) for p in s1_files if not p.is_file()]
            if missing:
                raise FileNotFoundError("Stage 1 outputs missing:\n" + "\n".join(missing))

    print(f"[RUN] {dataset} {method} seed={seed} Stage 2", flush=True)
    run_command(
        base + ["--finetune", "--pretrained_model"] + cfg["stage2_flags"],
        project_dir,
        env,
        run_dir / "logs" / "stage2.log",
        args.dry_run,
    )
    if not args.dry_run:
        missing = [str(p) for p in s2_files if not p.is_file()]
        if missing:
            raise FileNotFoundError("Stage 2 outputs missing:\n" + "\n".join(missing))

    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "status": "ok",
        "stage1_status": s1_status,
        "stage2_status": "ran",
        "elapsed_seconds": time.time() - t0,
        "run_dir": str(run_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--entry", default="pdcc_main.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-root", default="/data1/Lab105/zhouxin/runs/significance_v2")
    parser.add_argument("--datasets", default="SIMS,MOSI")
    parser.add_argument("--methods", default="BASE,PDCC")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    datasets = parse_csv(args.datasets, str)
    methods = [m.upper() for m in parse_csv(args.methods, str)]
    seeds = parse_csv(args.seeds, int)

    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method={m}; allowed={list(METHODS)}")

    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in datasets:
        if dataset not in {"SIMS", "MOSI"}:
            raise ValueError(f"unsupported dataset={dataset}")
        for method in methods:
            for seed in seeds:
                rows.append(run_one(args, dataset, method, seed))
                write_csv(root / "significance_run_status.csv", rows)

    print(f"[DONE] wrote {root / 'significance_run_status.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
