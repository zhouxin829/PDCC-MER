#!/usr/bin/env python3
"""Create paper-ready mean±sample-std tables for mechanism controls."""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np

MAIN_METRICS = ("Has0_acc_2", "Has0_F1_score", "Acc_3", "F1_score_3")
TRACK_ORDER = {
    "TPLR": ["NO_TPLR", "EMA_ONLY", "STUDENT_ONLY", "ONE_STEP", "FULL"],
    "PCRP": ["STEPS_0", "STEPS_1", "STEPS_2", "STEPS_3", "STEPS_4"],
    "RCCR": ["ROUTER_ONLY", "PRIOR_ONLY", "FULL"],
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def read_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    dtest = data.get("metrics", {}).get("D_test", {})
    return {key: float(dtest[key]) for key in MAIN_METRICS if key in dtest}


def mean_std(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) >= 2 else float("nan")


def source_for(root: Path, dataset: str, track: str, variant: str, seed: str) -> Path:
    if variant == "FULL":
        return root/dataset/"FULL"/seed/"metrics.json"
    return root/dataset/track/variant/seed/"metrics.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="runs/mechanism_controls_v1")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root/"summary"

    rows: list[dict[str, Any]] = []
    md = ["# Mechanism-control summary", ""]
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir() and (p/"experiment_config.json").is_file()):
        dataset = dataset_dir.name
        config = json.loads((dataset_dir/"experiment_config.json").read_text(encoding="utf-8"))
        best_steps = int(config["best_config"]["pcrp_steps"])
        seeds = [f"seed_{int(s)}" for s in config["seeds"]]
        md += [f"## {dataset}", ""]
        for track, variants in TRACK_ORDER.items():
            if track.lower() not in config["tracks"]:
                continue
            md += [f"### {track}", "", "| Variant | Has0 Acc-2 | Has0 F1 | Acc-3 | F1-3 | n |", "|---|---:|---:|---:|---:|---:|"]
            for variant in variants:
                # PCRP full is represented by the FULL reference at the dataset's tuned depth.
                if track == "PCRP" and variant == f"STEPS_{best_steps}":
                    variant_for_file = "FULL"
                    shown = f"{variant} (FULL)"
                elif variant == "FULL":
                    variant_for_file = "FULL"; shown = "FULL"
                else:
                    variant_for_file = variant; shown = variant
                collected: dict[str, list[float]] = defaultdict(list)
                for seed in seeds:
                    path = source_for(root, dataset, track, variant_for_file, seed)
                    if not path.is_file():
                        continue
                    metrics = read_metrics(path)
                    for key, value in metrics.items():
                        collected[key].append(value)
                        rows.append({"dataset": dataset, "track": track, "variant": shown, "seed": seed.removeprefix("seed_"), "metric": key, "value": value, "metrics_path": str(path)})
                display = []
                for key in MAIN_METRICS:
                    values = collected.get(key, [])
                    if values:
                        mu, sd = mean_std(values)
                        display.append(f"{mu*100:.2f} ± {sd*100:.2f}" if math.isfinite(sd) else f"{mu*100:.2f}")
                    else:
                        display.append("–")
                md.append(f"| {shown} | " + " | ".join(display) + f" | {len(collected.get('Has0_acc_2', []))} |")
            md.append("")
    if not rows:
        raise SystemExit(f"No metrics.json found under {root}")
    write_csv(out/"per_seed_metrics.csv", rows)
    # Aggregate CSV
    agg: list[dict[str, Any]] = []
    bucket: dict[tuple[str,str,str,str], list[float]] = defaultdict(list)
    for row in rows:
        bucket[(row["dataset"],row["track"],row["variant"],row["metric"])].append(row["value"])
    for (dataset,track,variant,metric), values in sorted(bucket.items()):
        mu, sd = mean_std(values)
        agg.append({"dataset":dataset,"track":track,"variant":variant,"metric":metric,"n":len(values),"mean":mu,"std_sample":sd,"mean_percent":mu*100,"std_percent":sd*100})
    write_csv(out/"mean_std.csv", agg)
    (out/"summary.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print(f"[SAVED] {out/'summary.md'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
