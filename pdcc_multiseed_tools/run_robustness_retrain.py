#!/usr/bin/env python3
"""Re-train full PDCC-MER for each missing/noise condition in isolated directories.

This is a *condition-specific re-training* experiment. Every condition starts a new
Stage-1 TPLR and Stage-2 PDCC-MER run, then saves its own model and pseudo-label bank.
It is not a replacement for clean-train / corrupted-test robustness evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifact_naming import all_artifacts_exist, artifact_exists


@dataclass(frozen=True)
class Condition:
    mode: str
    modality: str
    level: float

    @property
    def name(self) -> str:
        if self.mode == 'clean':
            return 'CLEAN'
        if self.mode == 'missing':
            return f'MISSING_{self.modality.upper()}'
        return f'NOISE_{self.modality.upper()}_L{int(round(self.level * 100)):02d}'


def parse_conditions(text: str) -> list[Condition]:
    """Syntax: clean,missing:text,noise:audio:0.2,noise:vision:0.4"""
    conditions: list[Condition] = []
    seen = set()
    for item in [part.strip() for part in text.split(',') if part.strip()]:
        pieces = [piece.strip().lower() for piece in item.split(':')]
        if pieces == ['clean']:
            cond = Condition('clean', 'none', 0.0)
        elif len(pieces) == 2 and pieces[0] == 'missing' and pieces[1] in {'text', 'audio', 'vision'}:
            cond = Condition('missing', pieces[1], 1.0)
        elif len(pieces) == 3 and pieces[0] == 'noise' and pieces[1] in {'text', 'audio', 'vision'}:
            level = float(pieces[2])
            if not 0.0 < level <= 1.0:
                raise ValueError(f'Noise level must be in (0,1]: {item}')
            cond = Condition('noise', pieces[1], level)
        else:
            raise ValueError(
                f'Invalid condition {item!r}. Use clean, missing:text, or noise:audio:0.2.'
            )
        if cond.name in seen:
            raise ValueError(f'Duplicate condition: {cond.name}')
        seen.add(cond.name)
        conditions.append(cond)
    if not conditions:
        raise ValueError('--conditions is empty')
    return conditions


def parse_seeds(text: str) -> list[int]:
    seeds = [int(part.strip()) for part in text.split(',') if part.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('--seeds must contain at least one unique integer seed')
    for seed in seeds:
        if not 0 < seed < 2**32 - 1:
            raise ValueError(f'Invalid seed: {seed}')
    return seeds


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')


def run_command(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as f:
        f.write('$ ' + shlex.join(command) + '\n\n')
        f.flush()
        proc = subprocess.run(command, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'Exit={proc.returncode}; inspect {log_path}')


def expected_stage1(run_dir: Path, dataset: str) -> list[Path]:
    return [
        run_dir / 'models' / f'{dataset}_PDCC_Pseudo_TextModel.pt',
        run_dir / 'models' / f'{dataset}_PDCC_Pseudo_AudioModel.pt',
        run_dir / 'models' / f'{dataset}_PDCC_Pseudo_VisionModel.pt',
        run_dir / 'pseudo_labels' / f'{dataset}_PDCC_train_pseudo_labels.pkl',
        run_dir / 'pseudo_labels' / f'{dataset}_PDCC_valid_pseudo_labels.pkl',
        run_dir / 'pseudo_labels' / f'{dataset}_PDCC_test_pseudo_labels.pkl',
    ]


def expected_stage2(run_dir: Path, dataset: str) -> list[Path]:
    return [
        run_dir / 'models' / f'{dataset}_PDCC_PDCCModel_Has0_acc_2.pt',
        run_dir / 'metrics.json',
    ]


def all_exist(paths: list[Path]) -> bool:
    return all_artifacts_exist(paths)


def base_command(args: argparse.Namespace, run_dir: Path, seed: int, condition: Condition) -> list[str]:
    return [
        args.python, args.entry,
        '--dataset', args.dataset,
        '--data_path', args.data_path,
        '--model_path', str(run_dir / 'models'),
        '--run_dir', str(run_dir),
        '--use_best', '--is_pseudo', '--seed', str(seed),
        '--robust_mode', condition.mode,
        '--robust_modality', condition.modality,
        '--robust_level', str(condition.level),
        '--robust_scope', args.robust_scope,
        '--robust_seed', str(args.robust_seed),
        '--num_workers', str(args.num_workers),
    ]


def run_one(args: argparse.Namespace, root: Path, condition: Condition, seed: int) -> None:
    run_dir = root / args.dataset / condition.name / f'seed_{seed}'
    for subdir in ('models', 'pseudo_labels', 'logs'):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    stage1 = base_command(args, run_dir, seed, condition) + ['--use_tplr']
    stage2 = base_command(args, run_dir, seed, condition) + [
        '--finetune', '--pretrained_model', '--use_tplr', '--use_pcrp', '--use_rccr'
    ]
    write_json(run_dir / 'condition_manifest.json', {
        'dataset': args.dataset,
        'condition': {'name': condition.name, 'mode': condition.mode, 'modality': condition.modality,
                      'level': condition.level, 'scope': args.robust_scope, 'corruption_seed': args.robust_seed},
        'seed': seed,
        'protocol': 'independent condition-specific two-stage PDCC-MER retraining with --use_best',
        'stage1_command': stage1,
        'stage2_command': stage2,
    })

    stage1_paths = expected_stage1(run_dir, args.dataset)
    stage2_paths = expected_stage2(run_dir, args.dataset)
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.dry_run:
        print('[DRY STAGE1]', shlex.join(stage1))
        print('[DRY STAGE2]', shlex.join(stage2))
        return

    if not all_exist(stage1_paths) or args.force:
        print(f'[RUN] {args.dataset} {condition.name} seed={seed} Stage-1', flush=True)
        run_command(stage1, Path(args.project_dir), env, run_dir / 'logs' / 'stage1.log')
        missing = [str(path) for path in stage1_paths if not artifact_exists(path)]
        if missing:
            raise FileNotFoundError('Stage-1 did not create isolated models/pseudo labels:\n' + '\n'.join(missing))
    else:
        print(f'[RESUME] Stage-1 exists: {run_dir}', flush=True)

    if not all_exist(stage2_paths) or args.force:
        print(f'[RUN] {args.dataset} {condition.name} seed={seed} Stage-2', flush=True)
        run_command(stage2, Path(args.project_dir), env, run_dir / 'logs' / 'stage2.log')
        missing = [str(path) for path in stage2_paths if not artifact_exists(path)]
        if missing:
            raise FileNotFoundError('Stage-2 did not create isolated model/metrics:\n' + '\n'.join(missing))
    else:
        print(f'[SKIP] Completed: {run_dir}', flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', required=True)
    parser.add_argument('--entry', default='pdcc_main.py')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dataset', choices=['SIMS', 'MOSI'], required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--seeds', required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--num-workers', type=int, default=14)
    parser.add_argument('--robust-seed', type=int, default=20260707)
    parser.add_argument('--robust-scope', choices=['all', 'test_only'], default='all')
    parser.add_argument(
        '--conditions',
        default='clean,missing:text,missing:audio,missing:vision,noise:text:0.2,noise:audio:0.2,noise:vision:0.2',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    if not (project / args.entry).is_file():
        raise FileNotFoundError(f'Entry not found: {project / args.entry}')
    if args.num_workers < 0:
        raise ValueError('--num-workers must be >= 0')

    root = Path(args.run_root).resolve()
    conditions = parse_conditions(args.conditions)
    seeds = parse_seeds(args.seeds)
    write_json(root / args.dataset / 'robustness_retrain_manifest.json', {
        'dataset': args.dataset,
        'entry': args.entry,
        'seeds': seeds,
        'gpu': str(args.gpu),
        'num_workers': args.num_workers,
        'robust_scope': args.robust_scope,
        'robust_seed': args.robust_seed,
        'conditions': [condition.__dict__ | {'name': condition.name} for condition in conditions],
    })

    for condition in conditions:
        for seed in seeds:
            run_one(args, root, condition, seed)
    print('[DONE] All requested re-training runs completed.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
