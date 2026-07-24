#!/usr/bin/env python3
"""Run and profile the PDCC-MER complexity table.

Table columns:
  Dataset | Model | Deployable params (M) | Profiled train step (ms/batch)
  | Train-step GFLOPs/batch | Inference (ms/sample)
  | Inference GFLOPs/sample | Inference peak allocated CUDA (MiB)

For each dataset and variant, this script:
  1) runs Stage 1 and Stage 2 with --use_best in an isolated directory;
  2) loads the final checkpoint;
  3) counts parameters;
  4) profiles forward+backward training step speed on train split;
  5) estimates train-step and inference FLOPs with torch.profiler;
  6) profiles inference speed on test split;
  7) writes CSV / Markdown / LaTeX table.

Default output root:
  /data1/Lab105/zhouxin/runs/complexity_table_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import MMDataset
from artifact_naming import (
    all_artifacts_exist,
    artifact_exists,
    resolve_existing_artifact,
)


VARIANTS = {
    "Base": {
        "stage1_flags": [],
        "stage2_flags": [],
        "tag": "BASE",
    },
    "Base+TPLR": {
        "stage1_flags": ["--use_tplr"],
        "stage2_flags": ["--use_tplr"],
        "tag": "PDCC",
    },
    "Base+PCRP": {
        # PCRP is a Stage-2 module. Passing --use_pcrp in Stage 1 only makes
        # the isolated pseudo-label/checkpoint tag become PDCC for this variant.
        "stage1_flags": ["--use_pcrp"],
        "stage2_flags": ["--use_pcrp"],
        "tag": "PDCC",
    },
    "Base+RCCR": {
        # RCCR is a Stage-2 module. Passing --use_rccr in Stage 1 only makes
        # the isolated pseudo-label/checkpoint tag become PDCC for this variant.
        "stage1_flags": ["--use_rccr"],
        "stage2_flags": ["--use_rccr"],
        "tag": "PDCC",
    },
    "PDCC-MER": {
        "stage1_flags": ["--use_tplr"],
        "stage2_flags": ["--use_tplr", "--use_pcrp", "--use_rccr"],
        "tag": "PDCC",
    },
}

ORDER = ["Base", "Base+TPLR", "Base+PCRP", "Base+RCCR", "PDCC-MER"]
COST_VARIANT_DIR = {
    "Base": "BASE",
    "Base+TPLR": "TPLR",
    "Base+PCRP": "PCRP",
    "Base+RCCR": "RCCR",
    "PDCC-MER": "FULL",
}


def parse_csv_arg(text: str, cast=str) -> list:
    values = [cast(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("empty comma-separated argument")
    return values


def safe_name(name: str) -> str:
    return name.replace("+", "_plus_").replace("-", "_")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path, dry_run: bool) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[CMD]", shlex.join(cmd), flush=True)
    if dry_run:
        return float("nan")
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"Exit={proc.returncode}; inspect {log_path}")
    return elapsed


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


def all_exist(paths: list[Path]) -> bool:
    return all_artifacts_exist(paths)


def build_common(args: argparse.Namespace, dataset: str, run_dir: Path, seed: int) -> list[str]:
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
        "--num_workers", str(args.num_workers),
    ]


def make_loader(dataset: str, data_path: str, split: str, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    ds = MMDataset(SimpleNamespace(dataset=dataset, data_path=data_path), split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def sum_profiled_flops(profiler: torch.profiler.profile) -> float:
    total = 0.0
    for event in profiler.key_averages():
        value = getattr(event, "flops", 0)
        if value:
            total += float(value)
    return total


def profile_train_step_flops(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    text: torch.Tensor,
    audio: torch.Tensor,
    vision: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
    ) as profiler:
        output = model(text, audio, vision)
        loss = criterion(output["pred"], labels)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    return sum_profiled_flops(profiler)


def profile_inference_flops(
    model: nn.Module,
    text: torch.Tensor,
    audio: torch.Tensor,
    vision: torch.Tensor,
) -> float:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    model.eval()
    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
        ) as profiler:
            _ = model(text, audio, vision)
            torch.cuda.synchronize()
    return sum_profiled_flops(profiler)


def profile_checkpoint(
    checkpoint: Path,
    dataset: str,
    data_path: str,
    gpu: str,
    batch_size: int,
    workers: int,
    train_warmup_batches: int,
    train_profile_batches: int,
    infer_warmup_batches: int,
    infer_repeats: int,
    skip_flops: bool,
) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for stable timing.")

    model = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError(f"checkpoint is not a torch.nn.Module: {type(model)}")
    model.to(device)

    total_params = count_params(model)
    train_loader = make_loader(dataset, data_path, "train", batch_size, workers, shuffle=True)
    test_loader = make_loader(dataset, data_path, "test", batch_size, workers, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=1e-5)

    # Training speed: forward + CE loss + backward + optimizer step.
    model.train()
    timed = []
    seen = 0
    iterator = iter(train_loader)
    total_needed = train_warmup_batches + train_profile_batches
    for step in range(total_needed):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        text = batch["text"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        vision = batch["vision"].to(device, non_blocking=True)
        y = batch["labels"]["M"].view(-1).long().to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        out = model(text, audio, vision)
        loss = criterion(out["pred"], y)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if step >= train_warmup_batches:
            timed.append(elapsed_ms)
            seen += 1

    train_ms = float(np.mean(timed)) if timed else float("nan")
    train_std = float(np.std(timed, ddof=1)) if len(timed) > 1 else 0.0

    flop_train_batch = next(iter(train_loader))
    flop_train_text = flop_train_batch["text"].to(device)
    flop_train_audio = flop_train_batch["audio"].to(device)
    flop_train_vision = flop_train_batch["vision"].to(device)
    flop_train_labels = flop_train_batch["labels"]["M"].view(-1).long().to(device)
    train_flops = float("nan")
    if not skip_flops:
        train_flops = profile_train_step_flops(
            model,
            optimizer,
            criterion,
            flop_train_text,
            flop_train_audio,
            flop_train_vision,
            flop_train_labels,
        )
        if train_flops <= 0:
            raise RuntimeError(
                "torch.profiler reported zero train-step FLOPs; "
                "use --skip-flops or inspect unsupported/custom operators"
            )

    # Inference speed over the whole test split.
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            _ = model(batch["text"].to(device), batch["audio"].to(device), batch["vision"].to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i + 1 >= infer_warmup_batches:
                break

    flop_test_batch = next(iter(test_loader))
    flop_test_text = flop_test_batch["text"].to(device)
    flop_test_audio = flop_test_batch["audio"].to(device)
    flop_test_vision = flop_test_batch["vision"].to(device)
    flop_test_samples = int(len(flop_test_batch["labels"]["M"]))
    inference_flops = float("nan")
    if not skip_flops:
        inference_flops = profile_inference_flops(
            model,
            flop_test_text,
            flop_test_audio,
            flop_test_vision,
        )
        if inference_flops <= 0:
            raise RuntimeError(
                "torch.profiler reported zero inference FLOPs; "
                "use --skip-flops or inspect unsupported/custom operators"
            )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    infer_ms_values = []
    num_samples = 0
    with torch.no_grad():
        for _ in range(infer_repeats):
            n = 0
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for batch in test_loader:
                _ = model(batch["text"].to(device), batch["audio"].to(device), batch["vision"].to(device))
                n += len(batch["labels"]["M"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            infer_ms_values.append(elapsed * 1000.0 / max(1, n))
            num_samples = n

    peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0

    del model
    torch.cuda.empty_cache()

    return {
        "params": total_params,
        "params_M": total_params / 1e6,
        "train_ms_per_batch": train_ms,
        "train_ms_per_batch_std": train_std,
        "train_profile_batches": int(seen),
        "train_step_flops_per_batch": train_flops,
        "train_step_gflops_per_batch": train_flops / 1e9,
        "train_flop_batch_size": int(len(flop_train_labels)),
        "infer_ms_per_sample": float(np.mean(infer_ms_values)),
        "infer_ms_per_sample_std": float(np.std(infer_ms_values, ddof=1)) if len(infer_ms_values) > 1 else 0.0,
        "infer_repeats": int(infer_repeats),
        "infer_samples": int(num_samples),
        "inference_flops_per_sample": inference_flops / max(1, flop_test_samples),
        "inference_gflops_per_sample": inference_flops / max(1, flop_test_samples) / 1e9,
        "inference_flop_batch_size": flop_test_samples,
        "flop_backend": "torch.profiler(with_flops=True); supported operators only",
        "flop_input_shapes": {
            "text": list(flop_test_text.shape),
            "audio": list(flop_test_audio.shape),
            "vision": list(flop_test_vision.shape),
        },
        "inference_peak_allocated_cuda_MiB": float(peak_mib),
        # Backward-compatible alias for existing result readers.
        "peak_cuda_memory_MiB": float(peak_mib),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "profile_batch_size": batch_size,
        "checkpoint": str(checkpoint),
    }


def run_variant(args: argparse.Namespace, dataset: str, variant: str, seed: int) -> dict[str, Any]:
    cfg = VARIANTS[variant]
    tag = cfg["tag"]
    output_root = Path(args.run_root).resolve()
    profile_dir = output_root / "profiles" / dataset / safe_name(variant) / f"seed_{seed}"
    profile_dir.mkdir(parents=True, exist_ok=True)

    if args.trained_cost_root:
        run_dir = (
            Path(args.trained_cost_root).resolve()
            / dataset
            / "COST"
            / COST_VARIANT_DIR[variant]
            / f"seed_{seed}"
        )
    else:
        run_dir = output_root / dataset / safe_name(variant) / f"seed_{seed}"
        for directory in [run_dir, run_dir / "models", run_dir / "pseudo_labels", run_dir / "logs"]:
            directory.mkdir(parents=True, exist_ok=True)
        if args.force and run_dir.exists():
            shutil.rmtree(run_dir)
            for directory in [run_dir, run_dir / "models", run_dir / "pseudo_labels", run_dir / "logs"]:
                directory.mkdir(parents=True, exist_ok=True)

    write_json(profile_dir / "complexity_manifest.json", {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "tag": tag,
        "stage1_flags": cfg["stage1_flags"],
        "stage2_flags": cfg["stage2_flags"],
        "source_run_dir": str(run_dir),
        "trained_cost_root": args.trained_cost_root,
    })

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    s2_files = expected_stage2(run_dir, dataset, tag)

    stage1_wall = float("nan")
    stage2_wall = float("nan")

    if args.trained_cost_root:
        if not all_exist(s2_files):
            missing = [str(path) for path in s2_files if not artifact_exists(path)]
            raise FileNotFoundError(
                "Existing COST checkpoint/metrics are incomplete:\n" + "\n".join(missing)
            )
        print(f"[REUSE] trained COST run: {run_dir}", flush=True)
    elif all_exist(s2_files) and not args.force:
        print(f"[SKIP] trained: {dataset}/{variant}/seed_{seed}", flush=True)
    else:
        common = build_common(args, dataset, run_dir, seed)
        s1_files = expected_stage1(run_dir, dataset, tag)
        if all_exist(s1_files) and not args.force:
            print(f"[RESUME] Stage 1 exists: {dataset}/{variant}/seed_{seed}", flush=True)
        else:
            print(f"[RUN] {dataset} {variant} seed={seed} Stage 1", flush=True)
            stage1_wall = run_cmd(
                common + cfg["stage1_flags"],
                Path(args.project_dir).resolve(),
                env,
                run_dir / "logs" / "stage1.log",
                args.dry_run,
            )
            if not args.dry_run:
                missing = [str(p) for p in s1_files if not artifact_exists(p)]
                if missing:
                    raise FileNotFoundError("Stage 1 outputs missing:\n" + "\n".join(missing))

        print(f"[RUN] {dataset} {variant} seed={seed} Stage 2", flush=True)
        stage2_wall = run_cmd(
            common + ["--finetune", "--pretrained_model"] + cfg["stage2_flags"],
            Path(args.project_dir).resolve(),
            env,
            run_dir / "logs" / "stage2.log",
            args.dry_run,
        )
        if not args.dry_run:
            missing = [str(p) for p in s2_files if not artifact_exists(p)]
            if missing:
                raise FileNotFoundError("Stage 2 outputs missing:\n" + "\n".join(missing))

    checkpoint = resolve_existing_artifact(s2_files[0])
    if args.dry_run:
        prof = {
            "params": float("nan"),
            "params_M": float("nan"),
            "train_ms_per_batch": float("nan"),
            "train_ms_per_batch_std": float("nan"),
            "infer_ms_per_sample": float("nan"),
            "infer_ms_per_sample_std": float("nan"),
            "train_step_gflops_per_batch": float("nan"),
            "inference_gflops_per_sample": float("nan"),
            "inference_peak_allocated_cuda_MiB": float("nan"),
            "peak_cuda_memory_MiB": float("nan"),
            "checkpoint": str(checkpoint),
        }
    else:
        print(f"[PROFILE] {dataset} {variant} seed={seed}", flush=True)
        prof = profile_checkpoint(
            checkpoint=checkpoint,
            dataset=dataset,
            data_path=args.data_path,
            gpu=str(args.gpu),
            batch_size=args.profile_batch_size,
            workers=args.profile_workers,
            train_warmup_batches=args.train_warmup_batches,
            train_profile_batches=args.train_profile_batches,
            infer_warmup_batches=args.infer_warmup_batches,
            infer_repeats=args.infer_repeats,
            skip_flops=args.skip_flops,
        )

    row = {
        "dataset": dataset,
        "model": variant,
        "seed": seed,
        "stage1_wall_s": stage1_wall,
        "stage2_wall_s": stage2_wall,
        **prof,
        "source_run_dir": str(run_dir),
        "profile_dir": str(profile_dir),
    }
    write_json(profile_dir / "complexity_profile.json", row)
    return row


def summarize(rows: list[dict[str, Any]], out_root: Path) -> None:
    write_csv(out_root / "complexity_per_seed.csv", rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((r["dataset"], r["model"]), []).append(r)

    summary_rows = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for model in ORDER:
            key = (dataset, model)
            if key not in grouped:
                continue
            vals = grouped[key]
            def mean(k):
                x = np.array([float(v.get(k, float("nan"))) for v in vals], dtype=float)
                return float(np.nanmean(x))
            def std(k):
                x = np.array([float(v.get(k, float("nan"))) for v in vals], dtype=float)
                x = x[np.isfinite(x)]
                return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
            summary_rows.append({
                "Dataset": "CH-SIMS" if dataset == "SIMS" else "CMU-MOSI",
                "Dataset_raw": dataset,
                "Model": model,
                "Params_M": mean("params_M"),
                "Train_ms_batch": mean("train_ms_per_batch"),
                "Train_ms_batch_std": std("train_ms_per_batch"),
                "Train_step_GFLOPs_batch": mean("train_step_gflops_per_batch"),
                "Train_step_GFLOPs_batch_std": std("train_step_gflops_per_batch"),
                "Infer_ms_sample": mean("infer_ms_per_sample"),
                "Infer_ms_sample_std": std("infer_ms_per_sample"),
                "Inference_GFLOPs_sample": mean("inference_gflops_per_sample"),
                "Inference_GFLOPs_sample_std": std("inference_gflops_per_sample"),
                "Inference_peak_allocated_cuda_MiB": mean("inference_peak_allocated_cuda_MiB"),
                "Inference_peak_allocated_cuda_MiB_std": std("inference_peak_allocated_cuda_MiB"),
                "n": len(vals),
            })

    write_csv(out_root / "complexity_table.csv", summary_rows)

    md = [
        "# Per-step training and inference profiling",
        "",
        "> FLOPs are operator-level estimates from torch.profiler(with_flops=True). "
        "They exclude data I/O, checkpoint serialization, EMA parameter updates, "
        "and pseudo-label memory operations.",
        "",
        "| Dataset | Model | Deployable params (M) | Profiled train step (ms/batch) | "
        "Train-step GFLOPs/batch | Test inference (ms/sample) | "
        "Inference GFLOPs/sample | Inference peak allocated CUDA (MiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        md.append(
            f"| {r['Dataset']} | {r['Model']} | {r['Params_M']:.2f} | "
            f"{r['Train_ms_batch']:.2f} +/- {r['Train_ms_batch_std']:.2f} | "
            f"{r['Train_step_GFLOPs_batch']:.3f} +/- {r['Train_step_GFLOPs_batch_std']:.3f} | "
            f"{r['Infer_ms_sample']:.3f} +/- {r['Infer_ms_sample_std']:.3f} | "
            f"{r['Inference_GFLOPs_sample']:.3f} +/- {r['Inference_GFLOPs_sample_std']:.3f} | "
            f"{r['Inference_peak_allocated_cuda_MiB']:.2f} +/- "
            f"{r['Inference_peak_allocated_cuda_MiB_std']:.2f} |"
        )
    (out_root / "complexity_table.md").write_text("\n".join(md), encoding="utf-8")

    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Model & Params (M) & Train (ms/batch) & Train GFLOPs/batch & "
        r"Infer. (ms/sample) & Infer. GFLOPs/sample & Infer. peak CUDA (MiB) \\",
        r"\midrule",
    ]
    last_ds = None
    for r in summary_rows:
        ds = r["Dataset"] if r["Dataset"] != last_ds else ""
        last_ds = r["Dataset"]
        lines.append(
            f"{ds} & {r['Model']} & {r['Params_M']:.2f} & "
            f"{r['Train_ms_batch']:.2f} $\\pm$ {r['Train_ms_batch_std']:.2f} & "
            f"{r['Train_step_GFLOPs_batch']:.3f} $\\pm$ {r['Train_step_GFLOPs_batch_std']:.3f} & "
            f"{r['Infer_ms_sample']:.3f} $\\pm$ {r['Infer_ms_sample_std']:.3f} & "
            f"{r['Inference_GFLOPs_sample']:.3f} $\\pm$ {r['Inference_GFLOPs_sample_std']:.3f} & "
            f"{r['Inference_peak_allocated_cuda_MiB']:.2f} $\\pm$ "
            f"{r['Inference_peak_allocated_cuda_MiB_std']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_root / "complexity_table_latex.txt").write_text("\n".join(lines), encoding="utf-8")

    notes = [
        "# Profiling protocol",
        "",
        "- Deployable parameters are counted from the final Stage 2 checkpoint and exclude "
        "the training-only EMA teacher and pseudo-label memory.",
        "- Profiled train-step latency covers model forward, cross-entropy loss, backward, "
        "and one AdamW update after warm-up. Data loading and host-to-device transfer are excluded.",
        "- Test inference latency is averaged over the full test split after warm-up and "
        "includes DataLoader iteration and host-to-device transfer.",
        "- Inference peak allocated CUDA is measured with torch.cuda.max_memory_allocated() "
        "after resetting peak statistics immediately before timed inference.",
        "- FLOPs are torch.profiler operator-level estimates. Unsupported/custom operations, "
        "data I/O, EMA updates, and pseudo-label memory operations are not included.",
    ]
    (out_root / "complexity_measurement_notes.md").write_text("\n".join(notes), encoding="utf-8")


def load_existing_profiles(
    out_root: Path,
    datasets: list[str],
    variants: list[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Load completed per-seed profiles without running GPU profiling again."""
    rows: list[dict[str, Any]] = []
    missing: list[Path] = []
    for dataset in datasets:
        for variant in variants:
            for seed in seeds:
                path = (
                    out_root
                    / "profiles"
                    / dataset
                    / safe_name(variant)
                    / f"seed_{seed}"
                    / "complexity_profile.json"
                )
                if not path.is_file():
                    missing.append(path)
                    continue
                try:
                    row = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError(f"Invalid profile JSON {path}: {error}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Profile JSON must contain an object: {path}")
                expected = (dataset, variant, seed)
                actual = (
                    str(row.get("dataset", "")),
                    str(row.get("model", "")),
                    int(row.get("seed", -1)),
                )
                if actual != expected:
                    raise ValueError(
                        f"Profile metadata mismatch in {path}: "
                        f"expected={expected}, actual={actual}"
                    )
                rows.append(row)
    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        suffix = (
            f"\n... and {len(missing) - 20} more"
            if len(missing) > 20
            else ""
        )
        raise FileNotFoundError(
            "Cannot summarize an incomplete profile set. Missing:\n"
            + preview
            + suffix
        )
    return rows


def cleanup_heavy(run_root: Path) -> None:
    targets = []
    for name in ["models", "pseudo_labels", "logs"]:
        targets.extend([p for p in run_root.rglob(name) if p.is_dir()])
    for p in sorted(targets):
        print(f"[CLEANUP] remove {p}", flush=True)
        shutil.rmtree(p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--entry", default="pdcc_main.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--run-root", default="/data1/Lab105/zhouxin/runs/complexity_table_v1")
    parser.add_argument(
        "--trained-cost-root",
        default=None,
        help=(
            "reuse checkpoints from run_diagnostics_retrain.py --suites cost; "
            "expected layout: ROOT/DATASET/COST/VARIANT/seed_SEED"
        ),
    )
    parser.add_argument("--datasets", default="SIMS,MOSI")
    parser.add_argument("--variants", default="Base,Base+TPLR,Base+PCRP,Base+RCCR,PDCC-MER")
    parser.add_argument("--seeds", default="3328683074")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--profile-workers", type=int, default=0)
    parser.add_argument("--profile-batch-size", type=int, default=64)
    parser.add_argument("--train-warmup-batches", type=int, default=5)
    parser.add_argument("--train-profile-batches", type=int, default=30)
    parser.add_argument("--infer-warmup-batches", type=int, default=5)
    parser.add_argument("--infer-repeats", type=int, default=5)
    parser.add_argument(
        "--skip-flops",
        action="store_true",
        help="skip torch.profiler FLOP estimation when profiling support is unavailable",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "read existing profiles/<dataset>/<variant>/seed_<seed>/"
            "complexity_profile.json files and regenerate combined tables "
            "without loading checkpoints or running GPU profiling"
        ),
    )
    parser.add_argument("--cleanup-after-profile", action="store_true")
    args = parser.parse_args()

    datasets = parse_csv_arg(args.datasets, str)
    variants = parse_csv_arg(args.variants, str)
    seeds = parse_csv_arg(args.seeds, int)
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"Unknown variant={v}; allowed={list(VARIANTS)}")
    for d in datasets:
        if d not in {"SIMS", "MOSI"}:
            raise ValueError(f"Unknown dataset={d}; allowed=SIMS,MOSI")

    out_root = Path(args.run_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        rows = load_existing_profiles(out_root, datasets, variants, seeds)
        summarize(rows, out_root)
        print(
            f"[DONE] summarized {len(rows)} existing profiles without GPU profiling",
            flush=True,
        )
        print(f"[DONE] wrote {out_root / 'complexity_table.csv'}", flush=True)
        print(f"[DONE] wrote {out_root / 'complexity_table.md'}", flush=True)
        print(
            f"[DONE] wrote {out_root / 'complexity_table_latex.txt'}",
            flush=True,
        )
        return 0

    if not args.project_dir:
        raise ValueError("--project-dir is required unless --summary-only is used")
    if not args.data_path:
        raise ValueError("--data-path is required unless --summary-only is used")

    rows = []
    for dataset in datasets:
        for variant in variants:
            for seed in seeds:
                row = run_variant(args, dataset, variant, seed)
                rows.append(row)
                summarize(rows, out_root)

    summarize(rows, out_root)
    print(f"[DONE] wrote {out_root / 'complexity_table.csv'}", flush=True)
    print(f"[DONE] wrote {out_root / 'complexity_table.md'}", flush=True)
    print(f"[DONE] wrote {out_root / 'complexity_table_latex.txt'}", flush=True)

    if args.cleanup_after_profile and not args.dry_run:
        cleanup_heavy(out_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
