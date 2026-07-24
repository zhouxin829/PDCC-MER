#!/usr/bin/env python3
"""Shared condition definitions for clean-train/corrupted-test evaluation."""
from __future__ import annotations

from dataclasses import dataclass


MODALITIES = ("text", "audio", "vision")
STOCHASTIC_MODES = ("noise", "random_missing", "misalign")


@dataclass(frozen=True)
class Condition:
    mode: str
    modality: str
    level: float

    @property
    def name(self) -> str:
        if self.mode == "clean":
            return "CLEAN"
        if self.mode == "missing":
            return f"MISSING_{self.modality.upper()}"
        prefix = {
            "noise": "NOISE",
            "random_missing": "RANDOM_MISSING",
            "misalign": "MISALIGN",
        }[self.mode]
        level_tag = "L" if self.mode == "noise" else "P"
        return (
            f"{prefix}_{self.modality.upper()}_"
            f"{level_tag}{int(round(self.level * 100)):02d}"
        )

    def as_dict(self) -> dict[str, str | float]:
        return {
            "name": self.name,
            "mode": self.mode,
            "modality": self.modality,
            "level": self.level,
        }


def parse_condition(item: str) -> Condition:
    pieces = [piece.strip().lower() for piece in item.split(":")]
    if pieces == ["clean"]:
        return Condition("clean", "none", 0.0)
    if (
        len(pieces) == 2
        and pieces[0] == "missing"
        and pieces[1] in MODALITIES
    ):
        return Condition("missing", pieces[1], 1.0)
    if (
        len(pieces) == 3
        and pieces[0] in STOCHASTIC_MODES
        and pieces[1] in MODALITIES
    ):
        level = float(pieces[2])
        if not 0.0 < level <= 1.0:
            raise ValueError(f"Condition level must be in (0,1]: {item}")
        return Condition(pieces[0], pieces[1], level)
    raise ValueError(
        f"Invalid condition {item!r}. Expected clean, missing:text, "
        "noise:audio:0.2, random_missing:vision:0.3, or misalign:text:0.3."
    )


def parse_conditions(text: str) -> list[Condition]:
    conditions = [
        parse_condition(item.strip())
        for item in text.split(",")
        if item.strip()
    ]
    if not conditions:
        raise ValueError("--conditions is empty")
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("--conditions contains duplicate normalized conditions")
    return conditions


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds is empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds contains duplicates")
    for seed in seeds:
        if not 0 < seed < 2**32 - 1:
            raise ValueError(f"Seed must be in (0, 2**32-1): {seed}")
    return seeds


DEFAULT_CONDITIONS = ",".join(
    [
        "clean",
        "missing:text",
        "missing:audio",
        "missing:vision",
        "noise:text:0.1",
        "noise:text:0.2",
        "noise:text:0.4",
        "noise:audio:0.1",
        "noise:audio:0.2",
        "noise:audio:0.4",
        "noise:vision:0.1",
        "noise:vision:0.2",
        "noise:vision:0.4",
        "random_missing:text:0.3",
        "random_missing:audio:0.3",
        "random_missing:vision:0.3",
        "misalign:text:0.3",
        "misalign:audio:0.3",
        "misalign:vision:0.3",
    ]
)
