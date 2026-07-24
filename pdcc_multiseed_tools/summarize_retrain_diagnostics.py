#!/usr/bin/env python3
"""Aggregate retrained diagnostics into mean/std and paper-ready cost tables."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


COST_ORDER = ["BASE", "TPLR", "PCRP", "RCCR", "FULL"]
COST_METRICS = [
    "stage1_wall_s",
    "stage1_peak_gpu_mib",
    "stage2_wall_s",
    "stage2_peak_gpu_mib",
    "total_wall_s",
    "pseudo_label_storage_mib",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(root: Path) -> list[dict]:
    rows = []
    allowed = {
        "diagnostic_retrain_cost.csv",
        "tplr_final_quality.csv",
        "rccr_metrics_long.csv",
    }
    for csv_path in root.glob("**/*.csv"):
        if csv_path.name not in allowed:
            continue
        with csv_path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(csv_path)
                rows.append(row)
    return rows


def add_group_value(groups: dict, key: tuple, value: str) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(number):
        groups[key].append(number)


def collect_groups(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row.get("value", "") != "":
            key = (
                row.get("dataset"),
                row.get("suite"),
                row.get("variant"),
                row.get("split", ""),
                row.get("modality", ""),
                row.get("metric", "value"),
            )
            add_group_value(groups, key, row["value"])
            continue

        if row.get("agreement", "") != "":
            for metric in ["agreement", "mean_confidence", "mean_entropy", "ece", "brier"]:
                key = (
                    row.get("dataset"),
                    row.get("suite"),
                    row.get("variant"),
                    row.get("split", ""),
                    row.get("modality", ""),
                    metric,
                )
                add_group_value(groups, key, row.get(metric, ""))
            continue

        for metric in COST_METRICS:
            if row.get(metric, "") in ("", "nan", "NaN"):
                continue
            key = (
                row.get("dataset"),
                row.get("suite"),
                row.get("variant"),
                "",
                "",
                metric,
            )
            add_group_value(groups, key, row[metric])
    return groups


def aggregate(groups: dict) -> list[dict]:
    out_rows = []
    for (dataset, suite, variant, split, modality, metric), values in sorted(groups.items()):
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if len(array) == 0:
            continue
        mean = float(array.mean())
        std = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
        mean_std = f"{mean:.4f} +/- {std:.4f}" if math.isfinite(std) else f"{mean:.4f}"
        out_rows.append(
            {
                "dataset": dataset,
                "suite": suite,
                "variant": variant,
                "split": split,
                "modality": modality,
                "metric": metric,
                "n": len(array),
                "mean": mean,
                "std": std,
                "mean_std": mean_std,
            }
        )
    return out_rows


def format_mean_std(row: dict | None, digits: int = 2) -> str:
    if row is None:
        return "NA"
    mean = float(row["mean"])
    std = float(row["std"])
    if math.isfinite(std):
        return f"{mean:.{digits}f} +/- {std:.{digits}f}"
    return f"{mean:.{digits}f}"


def write_cost_tables(out: Path, out_rows: list[dict]) -> None:
    lookup = {
        (row["dataset"], row["variant"], row["metric"]): row
        for row in out_rows
        if row["suite"] == "COST"
    }
    datasets = sorted({key[0] for key in lookup})
    table_rows = []
    md_rows = []
    latex_rows = []

    for dataset in datasets:
        for variant in COST_ORDER:
            metrics = {
                metric: lookup.get((dataset, variant, metric))
                for metric in COST_METRICS
            }
            if not any(metrics.values()):
                continue
            display_dataset = "CH-SIMS" if dataset == "SIMS" else "CMU-MOSI"
            display_variant = "PDCC-MER" if variant == "FULL" else variant
            row = {
                "Dataset": display_dataset,
                "Dataset_raw": dataset,
                "Configuration": display_variant,
            }
            for metric, metric_row in metrics.items():
                row[f"{metric}_mean"] = metric_row["mean"] if metric_row else ""
                row[f"{metric}_std"] = metric_row["std"] if metric_row else ""
                row[f"{metric}_n"] = metric_row["n"] if metric_row else ""
            table_rows.append(row)

            formatted = [format_mean_std(metrics[metric]) for metric in COST_METRICS]
            md_rows.append(
                f"| {display_dataset} | {display_variant} | "
                + " | ".join(formatted)
                + " |"
            )
            latex_values = [value.replace("+/-", r"$\pm$") for value in formatted]
            latex_rows.append(
                f"{display_dataset} & {display_variant} & "
                + " & ".join(latex_values)
                + r" \\"
            )

    write_csv(out / "cost_table.csv", table_rows)
    md = [
        "# Full two-stage training-pipeline cost",
        "",
        "> Total time is computed per run as Stage 1 + Stage 2 before aggregation. "
        "Pseudo-label storage is the combined size of train/valid/test pseudo-label files.",
        "",
        "| Dataset | Configuration | Stage 1 time (s) | Stage 1 peak GPU (MiB) | "
        "Stage 2 time (s) | Stage 2 peak GPU (MiB) | Total time (s) | "
        "Pseudo-label storage (MiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *md_rows,
    ]
    (out / "cost_table.md").write_text("\n".join(md), encoding="utf-8")

    latex = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Configuration & Stage 1 (s) & Stage 1 GPU (MiB) & "
        r"Stage 2 (s) & Stage 2 GPU (MiB) & Total (s) & Pseudo labels (MiB) \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (out / "cost_table_latex.txt").write_text("\n".join(latex), encoding="utf-8")

    notes = [
        "# Cost measurement protocol",
        "",
        "- Stage 1 is the full unimodal pseudo-label pretraining process. For TPLR/FULL, "
        "it includes EMA-teacher updates, progressive refinement, validation, "
        "checkpoint serialization, and pseudo-label I/O.",
        "- Stage 2 is the full multimodal fine-tuning process using the Stage 1 "
        "checkpoint and generated pseudo labels.",
        "- Total time is computed for each seed before mean and sample-standard-deviation aggregation.",
        "- Stage-level peak GPU memory is process used_memory sampled with nvidia-smi; "
        "it is not an exact per-module memory decomposition.",
        "- Pseudo-label storage is disk storage and is separate from peak CUDA memory.",
    ]
    (out / "cost_measurement_notes.md").write_text("\n".join(notes), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root / "summary"
    rows = read_rows(root)
    write_csv(out / "all_raw_rows.csv", rows)
    out_rows = aggregate(collect_groups(rows))
    write_csv(out / "mean_std.csv", out_rows)
    write_cost_tables(out, out_rows)

    md = ["# Retrained diagnostic summary", ""]
    for row in out_rows:
        md.append(
            f"- {row['dataset']} | {row['suite']} | {row['variant']} | "
            f"{row['split']} | {row['modality']} | {row['metric']}: "
            f"{row['mean_std']} (n={row['n']})"
        )
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("[DONE]", out)


if __name__ == "__main__":
    main()
