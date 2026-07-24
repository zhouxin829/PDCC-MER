#!/usr/bin/env python
"""Inspect saved PDCC-MER metrics.json files.

This script does not retrain or re-evaluate a checkpoint. It reads metrics
already produced by pdcc_main.py after Stage 2 and prints a compact index.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PREFERRED_KEYS = [
    "MAE",
    "Corr",
    "Acc_2",
    "Acc_2_non_neg",
    "Acc_3",
    "Acc_5",
    "Acc_7",
    "F1_score",
    "F1_score_non_neg",
    "Has0_acc_2",
    "Has0_F1_score",
    "Non0_acc_2",
    "Non0_F1_score",
]


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict):
      items: dict[str, Any] = {}
      for key, value in obj.items():
          child = f"{prefix}.{key}" if prefix else str(key)
          items.update(flatten(value, child))
      return items
    return {prefix: obj}


def find_metric_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    direct = path / "metrics.json"
    if direct.exists():
        return [direct]
    return sorted(path.rglob("metrics.json"))


def compact_row(metric_file: Path) -> dict[str, Any]:
    with metric_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    flat = flatten(data)
    row: dict[str, Any] = {"metrics_file": str(metric_file)}

    for key in PREFERRED_KEYS:
        matches = [name for name in flat if name.endswith(key)]
        if matches:
            row[key] = flat[matches[0]]

    if len(row) == 1:
        for key, value in flat.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                row[key] = value

    return row


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No metrics.json files found.")
        return

    columns = ["metrics_file"] + [
        key for key in PREFERRED_KEYS if any(key in row for row in rows)
    ]

    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }

    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect saved PDCC-MER metrics.")
    parser.add_argument("path", type=Path, help="metrics.json, run directory, or run root")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    metric_files = find_metric_files(args.path)
    rows = [compact_row(path) for path in metric_files]
    print_table(rows)

    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()

