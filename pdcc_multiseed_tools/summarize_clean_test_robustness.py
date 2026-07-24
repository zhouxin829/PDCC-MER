#!/usr/bin/env python3
"""Summarize clean-train/corrupted-test robustness and reliability results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


MODEL_ORDER = {"BASE": 0, "ROUTER_ONLY": 1, "PDCC_MER": 2}
MODE_ORDER = {
    "clean": 0,
    "missing": 1,
    "noise": 2,
    "random_missing": 3,
    "misalign": 4,
}
MODALITY_ORDER = {"none": 0, "text": 1, "audio": 2, "vision": 3}
PERFORMANCE_METRICS = ("Has0_acc_2", "Has0_F1_score")
CALIBRATION_METRICS = ("ece", "nll", "brier")


def numeric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def add_row(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    family: str,
    metric: str,
    value: Any,
    modality: str = "",
) -> None:
    if numeric(value):
        rows.append(
            {
                **base,
                "family": family,
                "metric": metric,
                "metric_modality": modality,
                "value": float(value),
            }
        )


def discover_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/**/seed_*/evaluations/*.json")):
        if path.name == "evaluation_manifest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"[WARN] Skipping {path}: {error}")
            continue
        meta = payload.get("meta", {})
        condition = meta.get("condition", {})
        model = str(meta.get("model", ""))
        dataset = str(meta.get("dataset", ""))
        seed = meta.get("model_seed")
        if not model or not dataset or not isinstance(seed, int):
            print(f"[WARN] Missing metadata in {path}")
            continue
        base_condition = {
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "condition": str(condition.get("name", path.stem)),
            "mode": str(condition.get("mode", "")),
            "condition_modality": str(condition.get("modality", "")),
            "level": float(condition.get("level", 0.0)),
            "source": str(path),
        }
        for split, split_payload in payload.get("splits", {}).items():
            base = {**base_condition, "split": split}
            for metric, value in split_payload.get("performance", {}).items():
                add_row(rows, base, "performance", metric, value)
            for family in ("calibration_2", "calibration_3", "corruption"):
                for metric, value in split_payload.get(family, {}).items():
                    add_row(rows, base, family, metric, value)
            for modality, values in split_payload.get("modalities", {}).items():
                for metric, value in values.items():
                    add_row(rows, base, "modality", metric, value, modality)
    return rows


def reference_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["dataset"],
        row["model"],
        row["seed"],
        row["split"],
        row["family"],
        row["metric"],
        row["metric_modality"],
    )


def add_clean_comparisons(rows: list[dict[str, Any]]) -> None:
    clean = {
        reference_key(row): row["value"]
        for row in rows
        if row["mode"] == "clean"
    }
    for row in rows:
        reference = clean.get(reference_key(row))
        if reference is None:
            row["clean_value"] = ""
            row["delta_from_clean"] = ""
            row["retention"] = ""
            continue
        row["clean_value"] = reference
        row["delta_from_clean"] = row["value"] - reference
        row["retention"] = (
            row["value"] / reference if abs(reference) > 1e-12 else ""
        )


def group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["dataset"],
        row["model"],
        row["condition"],
        row["mode"],
        row["condition_modality"],
        row["level"],
        row["split"],
        row["family"],
        row["metric"],
        row["metric_modality"],
    )


def mean_std(values: list[float]) -> tuple[float, float]:
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    result = []
    fields = (
        "dataset",
        "model",
        "condition",
        "mode",
        "condition_modality",
        "level",
        "split",
        "family",
        "metric",
        "metric_modality",
    )
    for key, group in groups.items():
        mean, std = mean_std([row["value"] for row in group])
        deltas = [
            float(row["delta_from_clean"])
            for row in group
            if row["delta_from_clean"] != ""
        ]
        retentions = [
            float(row["retention"])
            for row in group
            if row["retention"] != ""
        ]
        delta_mean, delta_std = (
            mean_std(deltas) if deltas else (float("nan"), float("nan"))
        )
        retention_mean, retention_std = (
            mean_std(retentions)
            if retentions
            else (float("nan"), float("nan"))
        )
        result.append(
            {
                **dict(zip(fields, key)),
                "n_seeds": len(group),
                "mean": mean,
                "std": std,
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "retention_mean": retention_mean,
                "retention_std": retention_std,
            }
        )
    return sorted(result, key=sort_key)


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["dataset"],
        row["split"],
        MODEL_ORDER.get(row["model"], 99),
        MODE_ORDER.get(row["mode"], 99),
        MODALITY_ORDER.get(row["condition_modality"], 99),
        row["level"],
        row["family"],
        row["metric_modality"],
        row["metric"],
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def index_summary(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (
            row["dataset"],
            row["model"],
            row["condition"],
            row["split"],
            row["family"],
            row["metric_modality"],
            row["metric"],
        ): row
        for row in rows
    }


def pct(row: dict[str, Any] | None) -> str:
    if row is None:
        return "-"
    return f"{100.0 * row['mean']:.2f} ± {100.0 * row['std']:.2f}"


def delta_pp(row: dict[str, Any] | None) -> str:
    if row is None or not math.isfinite(row["delta_mean"]):
        return "-"
    return f"{100.0 * row['delta_mean']:+.2f} ± {100.0 * row['delta_std']:.2f}"


def scalar(row: dict[str, Any] | None, digits: int = 3) -> str:
    if row is None:
        return "-"
    return f"{row['mean']:.{digits}f} ± {row['std']:.{digits}f}"


def binary_performance_markdown(
    rows: list[dict[str, Any]],
    index: dict[tuple[Any, ...], dict[str, Any]],
) -> str:
    condition_rows = {
        (
            row["dataset"],
            row["model"],
            row["condition"],
            row["split"],
            row["mode"],
            row["condition_modality"],
            row["level"],
        )
        for row in rows
        if row["family"] == "performance"
        and row["metric"] in PERFORMANCE_METRICS
    }
    lines = [
        "# Clean-train / corrupted-test binary performance",
        "",
        "Values are mean ± sample standard deviation over model-training seeds. "
        "Delta values are paired within seed against the same model's Clean result.",
        "",
        "| Dataset | Split | Model | Condition | Acc-2 (%) | Delta Acc (pp) | F1 (%) | Delta F1 (pp) |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset, model, condition, split, mode, modality, level in sorted(
        condition_rows,
        key=lambda item: (
            item[0],
            item[3],
            MODEL_ORDER.get(item[1], 99),
            MODE_ORDER.get(item[4], 99),
            MODALITY_ORDER.get(item[5], 99),
            item[6],
        ),
    ):
        acc = index.get(
            (dataset, model, condition, split, "performance", "", "Has0_acc_2")
        )
        f1 = index.get(
            (
                dataset,
                model,
                condition,
                split,
                "performance",
                "",
                "Has0_F1_score",
            )
        )
        lines.append(
            f"| {dataset} | {split} | {model} | {condition} | {pct(acc)} | "
            f"{delta_pp(acc)} | {pct(f1)} | {delta_pp(f1)} |"
        )
    return "\n".join(lines) + "\n"


def calibration_markdown(
    rows: list[dict[str, Any]],
    index: dict[tuple[Any, ...], dict[str, Any]],
) -> str:
    condition_rows = {
        (row["dataset"], row["model"], row["condition"], row["split"])
        for row in rows
        if row["family"] == "calibration_2"
    }
    lines = [
        "# Binary calibration under test corruption",
        "",
        "Lower ECE, NLL, and Brier are better. Binary probabilities are the "
        "renormalized negative/positive logits used by the project's Acc-2 protocol.",
        "",
        "| Dataset | Split | Model | Condition | ECE | NLL | Brier |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for dataset, model, condition, split in sorted(
        condition_rows,
        key=lambda item: (
            item[0],
            item[3],
            MODEL_ORDER.get(item[1], 99),
            item[2],
        ),
    ):
        values = [
            index.get(
                (dataset, model, condition, split, "calibration_2", "", metric)
            )
            for metric in CALIBRATION_METRICS
        ]
        lines.append(
            f"| {dataset} | {split} | {model} | {condition} | "
            + " | ".join(scalar(value) for value in values)
            + " |"
        )
    return "\n".join(lines) + "\n"


def reliability_markdown(
    rows: list[dict[str, Any]],
    index: dict[tuple[Any, ...], dict[str, Any]],
) -> str:
    conditions = {
        (
            row["dataset"],
            row["model"],
            row["condition"],
            row["split"],
            row["condition_modality"],
        )
        for row in rows
        if row["family"] == "modality"
        and row["model"] == "PDCC_MER"
        and row["mode"] != "clean"
        and row["metric_modality"] == row["condition_modality"]
    }
    lines = [
        "# RCCR reliability and routing response",
        "",
        "Rows report the deliberately corrupted modality in the full PDCC-MER model. "
        "A positive entropy delta and negative prior/gate deltas indicate that RCCR "
        "detects and downweights the corrupted modality. Prior AUC tests whether the "
        "entropy-derived reliability prior ranks correct modality-expert predictions "
        "above incorrect ones. Quality AUC is additionally available for sample-random "
        "missing and misaligned conditions, where both affected and unaffected samples "
        "are present.",
        "",
        "| Dataset | Split | Condition | Expert Acc-3 | Entropy | Delta entropy | Prior | Delta prior | Gate | Delta gate | Correctness AUC | Quality AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, model, condition, split, modality in sorted(conditions):
        def get(metric: str) -> dict[str, Any] | None:
            return index.get(
                (dataset, model, condition, split, "modality", modality, metric)
            )

        expert_acc = get("expert_acc_3")
        entropy_row = get("expert_mean_entropy_3")
        prior = get("mean_reliability_prior")
        gate = get("mean_gate")
        auc = get("reliability_prior_auc_for_expert_correct")
        quality_auc = get("reliability_prior_auc_for_uncorrupted_quality")
        lines.append(
            f"| {dataset} | {split} | {condition} | {pct(expert_acc)} | "
            f"{scalar(entropy_row)} | {scalar_delta(entropy_row)} | "
            f"{scalar(prior)} | {scalar_delta(prior)} | {scalar(gate)} | "
            f"{scalar_delta(gate)} | {scalar(auc)} | {scalar(quality_auc)} |"
        )
    return "\n".join(lines) + "\n"


def scalar_delta(row: dict[str, Any] | None, digits: int = 3) -> str:
    if row is None or not math.isfinite(row["delta_mean"]):
        return "-"
    return f"{row['delta_mean']:+.{digits}f} ± {row['delta_std']:.{digits}f}"


def latex_binary_table(
    rows: list[dict[str, Any]],
    index: dict[tuple[Any, ...], dict[str, Any]],
) -> str:
    condition_rows = {
        (
            row["dataset"],
            row["model"],
            row["condition"],
            row["split"],
            row["mode"],
            row["condition_modality"],
            row["level"],
        )
        for row in rows
        if row["family"] == "performance"
        and row["metric"] == "Has0_acc_2"
    }
    lines = [
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Dataset & Split & Model & Condition & Acc-2 & $\Delta$Acc & F1 & $\Delta$F1 \\",
        r"\midrule",
    ]
    for dataset, model, condition, split, mode, modality, level in sorted(
        condition_rows,
        key=lambda item: (
            item[0],
            item[3],
            MODEL_ORDER.get(item[1], 99),
            MODE_ORDER.get(item[4], 99),
            MODALITY_ORDER.get(item[5], 99),
            item[6],
        ),
    ):
        acc = index.get(
            (dataset, model, condition, split, "performance", "", "Has0_acc_2")
        )
        f1 = index.get(
            (
                dataset,
                model,
                condition,
                split,
                "performance",
                "",
                "Has0_F1_score",
            )
        )
        values = [
            pct(acc).replace("±", r"$\pm$"),
            delta_pp(acc).replace("±", r"$\pm$"),
            pct(f1).replace("±", r"$\pm$"),
            delta_pp(f1).replace("±", r"$\pm$"),
        ]
        lines.append(
            " & ".join(
                [
                    dataset.replace("_", r"\_"),
                    split.replace("_", r"\_"),
                    model.replace("_", r"\_"),
                    condition.replace("_", r"\_"),
                    *values,
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.out).resolve() if args.out else root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    rows = discover_rows(root)
    if not rows:
        raise FileNotFoundError(
            f"No evaluation JSON files found below {root}; expected "
            "<dataset>/<model>/seed_<seed>/evaluations/<condition>.json"
        )
    add_clean_comparisons(rows)
    summary = aggregate(rows)
    index = index_summary(summary)

    write_csv(output / "per_seed_metrics.csv", rows)
    write_csv(output / "mean_std_metrics.csv", summary)
    (output / "binary_performance.md").write_text(
        binary_performance_markdown(summary, index), encoding="utf-8"
    )
    (output / "binary_calibration.md").write_text(
        calibration_markdown(summary, index), encoding="utf-8"
    )
    (output / "rccr_reliability_response.md").write_text(
        reliability_markdown(summary, index), encoding="utf-8"
    )
    (output / "binary_performance_table.tex").write_text(
        latex_binary_table(summary, index), encoding="utf-8"
    )
    print(f"[DONE] {output}")
    print(f"[INFO] {len(rows)} per-seed metric rows; {len(summary)} aggregates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
