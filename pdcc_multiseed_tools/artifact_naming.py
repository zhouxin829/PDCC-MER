"""Canonical PDCC-MER artifact names with read-only legacy fallbacks."""

from __future__ import annotations

from pathlib import Path


def artifact_candidates(path: Path) -> list[Path]:
    """Return a canonical artifact path followed by pre-rename aliases."""
    path = Path(path)
    names = [path.name]
    replacements = (
        ("_PDCC_", "_OURS_"),
        ("PDCCModel", "DCCModel"),
    )

    index = 0
    while index < len(names):
        name = names[index]
        index += 1
        for current, legacy in replacements:
            candidate = name.replace(current, legacy)
            if candidate != name and candidate not in names:
                names.append(candidate)
    return [path.with_name(name) for name in names]


def resolve_existing_artifact(path: Path) -> Path:
    for candidate in artifact_candidates(path):
        if candidate.is_file():
            return candidate
    return Path(path)


def artifact_exists(path: Path) -> bool:
    return resolve_existing_artifact(path).is_file()


def all_artifacts_exist(paths: list[Path]) -> bool:
    return all(artifact_exists(path) for path in paths)
