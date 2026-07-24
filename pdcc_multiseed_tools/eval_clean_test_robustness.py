#!/usr/bin/env python3
"""Evaluate one clean-trained checkpoint under deterministic test corruption.

The checkpoint is never updated. Each requested condition reconstructs only the
evaluation dataset, so all reported degradation is clean-train/corrupted-test
generalization rather than condition-specific adaptation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

TOOL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TOOL_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataloader import MMDataset  # noqa: E402
from robustness_protocol import (  # noqa: E402
    DEFAULT_CONDITIONS,
    MODALITIES,
    Condition,
    parse_conditions,
)


EXPERT_KEYS = {
    "text": "pred_t",
    "audio": "pred_a",
    "vision": "pred_v",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int,
) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            result += (
                abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
                * float(mask.mean())
            )
    return float(result)


def calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int,
) -> dict[str, float]:
    n = len(labels)
    selected = np.clip(probabilities[np.arange(n), labels], 1e-12, 1.0)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return {
        "ece": expected_calibration_error(probabilities, labels, bins),
        "nll": float(-np.log(selected).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "mean_entropy": float(entropy(probabilities).mean()),
    }


def binary_probabilities(probabilities: np.ndarray) -> np.ndarray:
    binary = probabilities[:, [0, 2]].astype(np.float64, copy=True)
    denominator = binary.sum(axis=1, keepdims=True)
    return binary / np.clip(denominator, 1e-12, None)


def classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    dataset: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    predicted_3 = probabilities.argmax(axis=1)
    binary_prob = binary_probabilities(probabilities)
    predicted_2 = binary_prob.argmax(axis=1)
    if dataset == "SIMS":
        labels_2 = (labels == 2).astype(np.int64)
    else:
        labels_2 = (labels >= 1).astype(np.int64)

    non_neutral = labels != 1
    nonzero_labels = (labels[non_neutral] == 2).astype(np.int64)
    nonzero_predicted = predicted_2[non_neutral]
    metrics = {
        "Has0_acc_2": float(accuracy_score(labels_2, predicted_2)),
        "Has0_F1_score": float(
            f1_score(labels_2, predicted_2, average="weighted", zero_division=0)
        ),
        "Non0_acc_2": float(
            accuracy_score(nonzero_labels, nonzero_predicted)
        ) if non_neutral.any() else float("nan"),
        "Non0_F1_score": float(
            f1_score(
                nonzero_labels,
                nonzero_predicted,
                average="weighted",
                zero_division=0,
            )
        ) if non_neutral.any() else float("nan"),
        "Acc_3": float(accuracy_score(labels, predicted_3)),
        "F1_score_3": float(
            f1_score(labels, predicted_3, average="weighted", zero_division=0)
        ),
    }
    return metrics, binary_prob, labels_2


def safe_mean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else float("nan")


def safe_auc(correct: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(correct) & np.isfinite(score)
    if not mask.any() or len(np.unique(correct[mask])) != 2:
        return float("nan")
    try:
        return float(roc_auc_score(correct[mask], score[mask]))
    except ValueError:
        return float("nan")


def correctness_gap(correct: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(correct) & np.isfinite(score)
    correct_mask = mask & (correct == 1)
    wrong_mask = mask & (correct == 0)
    if not correct_mask.any() or not wrong_mask.any():
        return float("nan")
    return float(score[correct_mask].mean() - score[wrong_mask].mean())


def normalize_splits(dataset: str, text: str) -> list[tuple[str, str]]:
    if text.strip().lower() == "auto":
        return [("D_test", ""), ("D_msi", "D_msi")] if dataset == "SIMS" else [
            ("D_test", "")
        ]
    result: list[tuple[str, str]] = []
    for raw in [item.strip() for item in text.split(",") if item.strip()]:
        normalized = raw.lower()
        if normalized in {"test", "d_test"}:
            item = ("D_test", "")
        elif normalized == "d_msi" and dataset == "SIMS":
            item = ("D_msi", "D_msi")
        elif normalized == "d_msc" and dataset == "SIMS":
            item = ("D_msc", "D_msc")
        else:
            raise ValueError(f"Unsupported split for {dataset}: {raw}")
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError("--splits is empty")
    return result


def dataset_args(args: argparse.Namespace, condition: Condition) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=args.dataset,
        data_path=args.data_path,
        robust_mode=condition.mode,
        robust_modality=condition.modality,
        robust_level=condition.level,
        robust_scope="test_only",
        robust_seed=args.robust_seed,
    )


def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: str,
    condition: Condition,
    device: torch.device,
    ece_bins: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    applied_parts: list[np.ndarray] = []
    donor_parts: list[np.ndarray] = []
    ids: list[str] = []
    expert_probability_parts = {modality: [] for modality in MODALITIES}
    prior_parts: list[np.ndarray] = []
    gate_raw_parts: list[np.ndarray] = []
    gate_parts: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                batch["text"].to(device, non_blocking=True),
                batch["audio"].to(device, non_blocking=True),
                batch["vision"].to(device, non_blocking=True),
            )
            if not isinstance(outputs, dict) or "pred" not in outputs:
                raise TypeError("Checkpoint model must return a dict containing 'pred'")
            labels = batch["labels"]["M"].view(-1).long().cpu().numpy()
            probabilities = torch.softmax(outputs["pred"], dim=1).cpu().numpy()
            labels_parts.append(labels)
            probability_parts.append(probabilities)
            applied_parts.append(
                batch["robust_applied"].view(-1).cpu().numpy().astype(bool)
            )
            donor_parts.append(
                batch["robust_donor_index"].view(-1).cpu().numpy().astype(np.int64)
            )
            ids.extend(str(value) for value in batch["id"])

            for modality in MODALITIES:
                logits = outputs.get(EXPERT_KEYS[modality])
                if logits is None:
                    expert_probability_parts[modality].append(
                        np.full_like(probabilities, np.nan)
                    )
                else:
                    expert_probability_parts[modality].append(
                        torch.softmax(logits, dim=1).cpu().numpy()
                    )

            for key, target in (
                ("reliability_prior", prior_parts),
                ("gate_raw", gate_raw_parts),
                ("gate", gate_parts),
            ):
                value = outputs.get(key)
                if value is None:
                    target.append(np.full((len(labels), 3), np.nan, dtype=np.float32))
                else:
                    target.append(value.detach().cpu().numpy())

    labels = np.concatenate(labels_parts)
    probabilities = np.concatenate(probability_parts)
    applied = np.concatenate(applied_parts)
    donors = np.concatenate(donor_parts)
    priors = np.concatenate(prior_parts)
    gate_raw = np.concatenate(gate_raw_parts)
    gates = np.concatenate(gate_parts)
    expert_probabilities = {
        modality: np.concatenate(parts)
        for modality, parts in expert_probability_parts.items()
    }

    performance, binary_prob, binary_labels = classification_metrics(
        probabilities, labels, dataset
    )
    calibration_3 = calibration_metrics(probabilities, labels, ece_bins)
    calibration_2 = calibration_metrics(binary_prob, binary_labels, ece_bins)
    final_predicted = probabilities.argmax(axis=1)
    final_entropy = entropy(probabilities)

    modality_summary: dict[str, dict[str, float]] = {}
    expert_state: dict[str, dict[str, np.ndarray]] = {}
    for index, modality in enumerate(MODALITIES):
        expert_prob = expert_probabilities[modality]
        expert_predicted = expert_prob.argmax(axis=1)
        expert_correct = (expert_predicted == labels).astype(np.float64)
        expert_entropy = entropy(expert_prob)
        prior = priors[:, index]
        raw = gate_raw[:, index]
        gate = gates[:, index]
        if (
            modality == condition.modality
            and condition.mode in {"random_missing", "misalign"}
        ):
            quality = (~applied).astype(np.float64)
        else:
            quality = np.ones(len(applied), dtype=np.float64)
        modality_summary[modality] = {
            "expert_acc_3": float(expert_correct.mean()),
            "expert_mean_entropy_3": float(expert_entropy.mean()),
            "mean_reliability_prior": safe_mean(prior),
            "mean_gate_raw": safe_mean(raw),
            "mean_gate": safe_mean(gate),
            "reliability_prior_auc_for_expert_correct": safe_auc(
                expert_correct, prior
            ),
            "reliability_prior_correctness_gap": correctness_gap(
                expert_correct, prior
            ),
            "gate_raw_auc_for_expert_correct": safe_auc(expert_correct, raw),
            "gate_auc_for_expert_correct": safe_auc(expert_correct, gate),
            "gate_correctness_gap": correctness_gap(expert_correct, gate),
            "reliability_prior_auc_for_uncorrupted_quality": safe_auc(
                quality, prior
            ),
            "reliability_prior_quality_gap": correctness_gap(quality, prior),
            "gate_auc_for_uncorrupted_quality": safe_auc(quality, gate),
            "gate_quality_gap": correctness_gap(quality, gate),
        }
        expert_state[modality] = {
            "predicted": expert_predicted,
            "correct": expert_correct,
            "entropy": expert_entropy,
            "prior": prior,
            "gate_raw": raw,
            "gate": gate,
        }

    summary = {
        "n": int(len(labels)),
        "performance": performance,
        "calibration_3": calibration_3,
        "calibration_2": calibration_2,
        "corruption": {
            "applied_count": int(applied.sum()),
            "applied_fraction": float(applied.mean()),
            "donor_count": int((donors >= 0).sum()),
        },
        "modalities": modality_summary,
    }

    per_sample: list[dict[str, Any]] = []
    for row_index in range(len(labels)):
        row: dict[str, Any] = {
            "id": ids[row_index],
            "label_3": int(labels[row_index]),
            "final_pred_3": int(final_predicted[row_index]),
            "final_correct_3": int(final_predicted[row_index] == labels[row_index]),
            "final_confidence_3": float(probabilities[row_index].max()),
            "final_entropy_3": float(final_entropy[row_index]),
            "robust_applied": int(applied[row_index]),
            "robust_donor_index": int(donors[row_index]),
        }
        for modality in MODALITIES:
            state = expert_state[modality]
            row.update(
                {
                    f"{modality}_expert_pred_3": int(
                        state["predicted"][row_index]
                    ),
                    f"{modality}_expert_correct_3": int(
                        state["correct"][row_index]
                    ),
                    f"{modality}_expert_entropy_3": float(
                        state["entropy"][row_index]
                    ),
                    f"{modality}_reliability_prior": float(
                        state["prior"][row_index]
                    ),
                    f"{modality}_gate_raw": float(state["gate_raw"][row_index]),
                    f"{modality}_gate": float(state["gate"][row_index]),
                }
            )
        per_sample.append(json_safe(row))
    return summary, per_sample


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["SIMS", "MOSI"])
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--robust-seed", type=int, default=20260707)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--splits", default="auto")
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    parser.add_argument("--no-per-sample", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.ece_bins < 2:
        raise ValueError("--ece-bins must be >= 2")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not Path(args.data_path).is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_path}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditions = parse_conditions(args.conditions)
    splits = normalize_splits(args.dataset, args.splits)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {checkpoint}", flush=True)
    model = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected nn.Module checkpoint, got {type(model)}")
    model = model.to(device).eval()

    manifest = {
        "protocol": "clean-train/corrupted-test; checkpoint frozen for every condition",
        "dataset": args.dataset,
        "model": args.model_name,
        "model_seed": args.model_seed,
        "checkpoint": str(checkpoint),
        "robust_seed": args.robust_seed,
        "ece_bins": args.ece_bins,
        "splits": [name for name, _ in splits],
        "conditions": [condition.as_dict() for condition in conditions],
        "corruption_definitions": {
            "missing": "the complete selected modality is zeroed for every test sample",
            "noise_text": "valid lexical BERT tokens are replaced by [UNK] with probability level",
            "noise_audio_vision": (
                "Gaussian noise is added on the valid region with standard deviation "
                "level times the sample-level feature RMS"
            ),
            "random_missing": (
                "the selected modality is zeroed independently per sample with "
                "probability level"
            ),
            "misalign": (
                "the selected modality is replaced by that modality from a "
                "deterministically selected different test sample with probability level"
            ),
        },
    }
    write_json(output_dir / "evaluation_manifest.json", manifest)

    for condition in conditions:
        condition_path = output_dir / f"{condition.name}.json"
        if condition_path.is_file() and not args.force:
            print(f"[RESUME] {condition.name}", flush=True)
            continue
        print(f"[EVAL] {condition.name}", flush=True)
        payload: dict[str, Any] = {
            "meta": {
                **manifest,
                "condition": condition.as_dict(),
            },
            "splits": {},
        }
        for split_name, split_mode in splits:
            dataset = MMDataset(
                dataset_args(args, condition),
                mode="test",
                split_mode=split_mode,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            summary, rows = evaluate_split(
                model, loader, args.dataset, condition, device, args.ece_bins
            )
            payload["splits"][split_name] = summary
            if not args.no_per_sample:
                write_csv(
                    output_dir / f"{condition.name}__{split_name}_per_sample.csv",
                    rows,
                )
        write_json(condition_path, payload)

    print(f"[DONE] {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
