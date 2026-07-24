#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() or p.is_symlink():
                total += p.lstat().st_size
        except FileNotFoundError:
            pass
    return total


def gib(n: int) -> float:
    return n / (1024 ** 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--names", default="models,pseudo_labels,logs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    target_names = {x.strip() for x in args.names.split(",") if x.strip()}
    candidates = sorted(p for p in root.rglob("*") if p.is_dir() and p.name in target_names)

    total = 0
    for p in candidates:
        size = dir_size(p)
        total += size
        prefix = "[DRY-RUN]" if args.dry_run else "[DELETE]"
        print(f"{prefix} {p} size={gib(size):.3f} GiB")
        if not args.dry_run:
            shutil.rmtree(p)

    print(f"[TOTAL] {len(candidates)} directories, {gib(total):.3f} GiB")
    if args.dry_run:
        print("[INFO] dry-run only; run again without --dry-run to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
