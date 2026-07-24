#!/usr/bin/env python3
"""Summarize condition-specific PDCC-MER re-training metrics as mean ± sample std."""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np

REPORTABLE_PREFIXES = ('D_test.', 'D_msc.', 'D_msi.', 'D_test_reg.')


def flatten(obj: Any, prefix: str = '') -> dict[str, float]:
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f'{prefix}.{key}' if prefix else str(key)
            out.update(flatten(value, name))
    elif isinstance(obj, (int, float, np.integer, np.floating)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def reportable(key: str) -> bool:
    return key.startswith(REPORTABLE_PREFIXES)


def scale(key: str) -> float:
    leaf = key.rsplit('.', 1)[-1].lower()
    return 100.0 if any(x in leaf for x in ('acc', 'f1', 'precision', 'recall')) else 1.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root / 'summary'

    per_seed = []
    for metrics_path in sorted(root.glob('*/*/seed_*/metrics.json')):
        # root/DATASET/CONDITION/seed_S/metrics.json
        try:
            dataset, condition, seed_dir = metrics_path.parts[-4:-1]
            seed = int(seed_dir.removeprefix('seed_'))
            payload = json.loads(metrics_path.read_text(encoding='utf-8'))
            values = {key: value for key, value in flatten(payload.get('metrics', {})).items() if reportable(key)}
        except Exception as exc:
            print(f'[WARN] skipped {metrics_path}: {exc}')
            continue
        for metric, value in values.items():
            per_seed.append({'dataset': dataset, 'condition': condition, 'seed': seed,
                             'metric': metric, 'value': value, 'metrics_path': str(metrics_path)})
    if not per_seed:
        raise SystemExit(f'No reportable metrics found in {root}')

    write_csv(out / 'per_seed_metrics.csv', per_seed)
    groups = defaultdict(list)
    for row in per_seed:
        groups[(row['dataset'], row['condition'], row['metric'])].append(row['value'])

    rows = []
    for (dataset, condition, metric), vals in sorted(groups.items()):
        vals = np.asarray(vals, dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else float('nan')
        factor = scale(metric)
        display = f'{mean * factor:.2f} ± {std * factor:.2f}' if math.isfinite(std) else f'{mean * factor:.2f}'
        rows.append({'dataset': dataset, 'condition': condition, 'metric': metric, 'n': int(vals.size),
                     'mean_raw': mean, 'std_raw': std, 'display_scale': factor, 'mean_std': display})
    write_csv(out / 'mean_std.csv', rows)

    md = ['# Condition-specific re-training robustness summary', '',
          '> Each row is a separately trained two-stage PDCC-MER model with separately saved pseudo labels.', '']
    for dataset in sorted({row['dataset'] for row in rows}):
        md += [f'## {dataset}', '']
        metrics = sorted({row['metric'] for row in rows if row['dataset'] == dataset})
        conditions = sorted({row['condition'] for row in rows if row['dataset'] == dataset})
        for metric in metrics:
            md += [f'### {metric}', '', '| Condition | n | Mean ± std |', '|---|---:|---:|']
            for condition in conditions:
                match = next((r for r in rows if r['dataset'] == dataset and r['condition'] == condition and r['metric'] == metric), None)
                if match:
                    md.append(f"| {condition} | {match['n']} | {match['mean_std']} |")
            md.append('')
    (out / 'summary.md').write_text('\n'.join(md), encoding='utf-8')
    print(f'[DONE] wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
