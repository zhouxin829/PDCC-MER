#!/usr/bin/env python3
"""Select label smoothing on validation data and report selected test results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


TEST_METRICS = (
    ("Has0_acc_2", "Acc-2"),
    ("Has0_F1_score", "F1"),
)


def epsilon_token(epsilon):
    return f"EPS_{epsilon:g}".replace(".", "p")


def mean_std(values):
    values = [float(value) for value in values]
    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if len(values) >= 2 else 0.0
    return mean, std


def format_percent(values):
    mean, std = mean_std(values)
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_validation(path):
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    metrics = payload.get("metrics", {})
    containers = (
        metrics.get("selection", {}),
        payload.get("selection", {}),
        metrics,
    )
    for container in containers:
        if isinstance(container, dict) and "best_valid_Has0_acc_2" in container:
            return float(container["best_valid_Has0_acc_2"]), "metrics.json"

    # Compatibility with runs produced before pdcc_train_run.py started writing
    # the selection block. The log records the running best validation score
    # after every epoch, so its maximum is the same model-selection statistic.
    log_path = path.parent / "logs" / "stage2.log"
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        patterns = (
            r"Current Best VALID Has0_acc_2:\s*([0-9]*\.?[0-9]+)",
            r"Saved best (?:PDCC|DCC)Model_Has0_acc_2 by VALID:\s*([0-9]*\.?[0-9]+)",
        )
        values = [
            float(match)
            for pattern in patterns
            for match in re.findall(pattern, text)
        ]
        values = [value for value in values if 0.0 <= value <= 1.0]
        if values:
            value = max(values)
            print(
                "[VALIDATION FALLBACK] "
                f"{path.parent}: best_valid_Has0_acc_2={value:.4f} "
                f"from {log_path}",
                flush=True,
            )
            return value, "logs/stage2.log"

    raise KeyError(
        "Cannot recover best validation Has0_acc_2. Missing "
        f"metrics.selection.best_valid_Has0_acc_2 in {path} and no matching "
        f"validation record in {log_path}. Test metrics are intentionally not "
        "used as a fallback."
    )


def read_test(path):
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    d_test = payload.get("metrics", {}).get("D_test", {})
    return {key: float(d_test[key]) for key, _ in TEST_METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True, help="root used by run_label_smoothing_control.py"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root / "summary"
    configs = sorted(root.glob("*/label_smoothing_experiment_config.json"))
    if not configs:
        raise SystemExit(f"No label-smoothing experiment configs found under {root}")

    validation_rows = []
    selected_rows = []
    aggregate_rows = []
    selection_payload = {
        "protocol": "fixed_label_smoothing_control_v1",
        "selection_uses_test_metrics": False,
        "datasets": {},
    }
    markdown = [
        "# Fixed label-smoothing control",
        "",
        "Epsilon is selected independently for each dataset using mean validation "
        "Has0 Acc-2 across all configured seeds. Test metrics from non-selected "
        "candidates are not used in selection or reported below.",
        "",
        "| Dataset | Selected epsilon | Valid Acc-2 | Test Acc-2 | Test F1 | Seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for config_path in configs:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        dataset = str(config["dataset"])
        epsilons = [float(value) for value in config["epsilons"]]
        seeds = [int(value) for value in config["seeds"]]
        dataset_dir = config_path.parent
        candidate_records = {}

        for epsilon in epsilons:
            records = []
            for seed in seeds:
                metrics_path = (
                    dataset_dir
                    / epsilon_token(epsilon)
                    / f"seed_{seed}"
                    / "metrics.json"
                )
                if not metrics_path.is_file():
                    raise FileNotFoundError(
                        f"Missing configured run: {metrics_path}. "
                        "Selection requires every epsilon/seed pair."
                    )
                validation, validation_source = read_validation(metrics_path)
                records.append(
                    {
                        "seed": seed,
                        "validation": validation,
                        "validation_source": validation_source,
                        "metrics_path": str(metrics_path),
                    }
                )
                validation_rows.append(
                    {
                        "dataset": dataset,
                        "epsilon": epsilon,
                        "seed": seed,
                        "valid_Has0_acc_2": validation,
                        "validation_source": validation_source,
                        "metrics_path": str(metrics_path),
                    }
                )

            valid_values = [record["validation"] for record in records]
            valid_mean, valid_std = mean_std(valid_values)
            candidate_records[epsilon] = {
                "records": records,
                "valid_mean": valid_mean,
                "valid_std_sample": valid_std,
            }
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "epsilon": epsilon,
                    "split": "valid",
                    "metric": "Has0_acc_2",
                    "n": len(valid_values),
                    "mean": valid_mean,
                    "std_sample": valid_std,
                    "selected": False,
                }
            )

        selected_epsilon = max(
            epsilons,
            key=lambda value: (
                candidate_records[value]["valid_mean"],
                -value,
            ),
        )
        selected = candidate_records[selected_epsilon]
        for row in aggregate_rows:
            if row["dataset"] == dataset and math.isclose(
                float(row["epsilon"]), selected_epsilon
            ):
                row["selected"] = True

        test_values = {key: [] for key, _ in TEST_METRICS}
        for record in selected["records"]:
            test = read_test(Path(record["metrics_path"]))
            selected_row = {
                "dataset": dataset,
                "epsilon": selected_epsilon,
                "seed": record["seed"],
                "valid_Has0_acc_2": record["validation"],
                "validation_source": record["validation_source"],
                "metrics_path": record["metrics_path"],
            }
            for key, _ in TEST_METRICS:
                value = test[key]
                test_values[key].append(value)
                selected_row[f"test_{key}"] = value
            selected_rows.append(selected_row)

        for key, label in TEST_METRICS:
            metric_mean, metric_std = mean_std(test_values[key])
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "epsilon": selected_epsilon,
                    "split": "test",
                    "metric": key,
                    "display_metric": label,
                    "n": len(test_values[key]),
                    "mean": metric_mean,
                    "std_sample": metric_std,
                    "selected": True,
                }
            )

        selection_payload["datasets"][dataset] = {
            "selected_epsilon": selected_epsilon,
            "selection_metric": "valid/Has0_acc_2",
            "selection_mean": selected["valid_mean"],
            "selection_std_sample": selected["valid_std_sample"],
            "candidate_validation": {
                f"{epsilon:g}": {
                    "mean": candidate_records[epsilon]["valid_mean"],
                    "std_sample": candidate_records[epsilon]["valid_std_sample"],
                    "n": len(candidate_records[epsilon]["records"]),
                }
                for epsilon in epsilons
            },
        }
        markdown.append(
            f"| {dataset} | {selected_epsilon:g} | "
            f"{format_percent([r['validation'] for r in selected['records']])} | "
            f"{format_percent(test_values['Has0_acc_2'])} | "
            f"{format_percent(test_values['Has0_F1_score'])} | "
            f"{len(seeds)} |"
        )

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "validation_grid_per_seed.csv", validation_rows)
    write_csv(out / "selected_test_per_seed.csv", selected_rows)
    write_csv(out / "mean_std.csv", aggregate_rows)
    (out / "selection.json").write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "label_smoothing_table.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(f"[SAVED] {out / 'label_smoothing_table.md'}")
    print(f"[SAVED] {out / 'selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
