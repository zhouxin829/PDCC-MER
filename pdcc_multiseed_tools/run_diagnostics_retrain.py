#!/usr/bin/env python3
"""Retrain diagnostic experiments with isolated models and pseudo labels.

Suites:
  rccr: ROUTER_ONLY / PRIOR_ONLY / FULL_RCCR
  tplr: NO_TPLR / EMA_ONLY / STUDENT_ONLY / ONE_STEP / FULL_TPLR
  cost: BASE / TPLR / PCRP / RCCR / FULL

Every run writes to:
  <run_root>/<DATASET>/<SUITE>/<VARIANT>/seed_<SEED>/
      models/ pseudo_labels/ logs/ stage1_summary.json metrics.json
      cost_measurement.json

All runs use --use_best.  Variant-specific overrides such as --tplr_steps 1 or
--rccr_lambda 1.0 are intentionally passed after --use_best; this assumes your
current pdcc_main.py keeps explicit mechanism overrides effective.
"""
from __future__ import annotations
import argparse, csv, json, math, os, subprocess, sys, time, threading
from pathlib import Path
from typing import Any

from artifact_naming import all_artifacts_exist, artifact_exists

RCCR = {
    "ROUTER_ONLY": (["--use_tplr"], ["--use_tplr", "--use_pcrp"]),
    "PRIOR_ONLY":  (["--use_tplr"], ["--use_tplr", "--use_pcrp", "--use_rccr", "--rccr_lambda", "1.0"]),
    "FULL_RCCR":   (["--use_tplr"], ["--use_tplr", "--use_pcrp", "--use_rccr"]),
}
TPLR = {
    # Stage-1 uses --use_pcrp only to force PDCC filenames; PCRP is ignored by pseudo pretraining.
    "NO_TPLR":      (["--use_pcrp"], ["--use_pcrp", "--use_rccr"]),
    "EMA_ONLY":     (["--use_tplr", "--tplr_mode", "teacher_only"], ["--use_tplr", "--use_pcrp", "--use_rccr"]),
    "STUDENT_ONLY": (["--use_tplr", "--tplr_mode", "student_only"], ["--use_tplr", "--use_pcrp", "--use_rccr"]),
    "ONE_STEP":     (["--use_tplr", "--tplr_steps", "1"], ["--use_tplr", "--use_pcrp", "--use_rccr"]),
    "FULL_TPLR":    (["--use_tplr"], ["--use_tplr", "--use_pcrp", "--use_rccr"]),
}
COST = {
    "BASE": ([ ], [ ]),
    "TPLR": (["--use_tplr"], ["--use_tplr"]),
    "PCRP": (["--use_pcrp"], ["--use_pcrp"]),
    "RCCR": (["--use_rccr"], ["--use_rccr"]),
    "FULL": (["--use_tplr"], ["--use_tplr", "--use_pcrp", "--use_rccr"]),
}
SUITES = {"rccr": RCCR, "tplr": TPLR, "cost": COST}


def parse_seeds(text: str) -> list[int]:
    seeds = [int(x.strip()) for x in text.split(',') if x.strip()]
    if not seeds:
        raise ValueError("--seeds is empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds contains duplicates")
    return seeds


def tag_from_flags(flags: list[str]) -> str:
    return "PDCC" if any(f in flags for f in ("--use_tplr", "--use_pcrp", "--use_rccr")) else "BASE"


def expected_stage1(run_dir: Path, dataset: str, tag: str) -> list[Path]:
    m, p = run_dir / "models", run_dir / "pseudo_labels"
    return [
        m / f"{dataset}_{tag}_Pseudo_TextModel.pt",
        m / f"{dataset}_{tag}_Pseudo_AudioModel.pt",
        m / f"{dataset}_{tag}_Pseudo_VisionModel.pt",
        p / f"{dataset}_{tag}_train_pseudo_labels.pkl",
        p / f"{dataset}_{tag}_valid_pseudo_labels.pkl",
        p / f"{dataset}_{tag}_test_pseudo_labels.pkl",
    ]


def pseudo_label_paths(run_dir: Path, dataset: str, tag: str) -> list[Path]:
    p = run_dir / "pseudo_labels"
    return [
        p / f"{dataset}_{tag}_train_pseudo_labels.pkl",
        p / f"{dataset}_{tag}_valid_pseudo_labels.pkl",
        p / f"{dataset}_{tag}_test_pseudo_labels.pkl",
    ]


def expected_stage2(run_dir: Path, dataset: str, tag: str) -> list[Path]:
    return [run_dir / "models" / f"{dataset}_{tag}_PDCCModel_Has0_acc_2.pt", run_dir / "metrics.json"]


def ok(paths: list[Path]) -> bool:
    return all_artifacts_exist(paths)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def gpu_name(gpu: str) -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi", "-i", str(gpu),
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip().splitlines()[0].strip()
    except Exception:
        return "unknown"


def file_size_mib(paths: list[Path]) -> float:
    return sum(path.stat().st_size for path in paths if path.is_file()) / (1024.0 ** 2)


def monitor_gpu(
    pid: int,
    gpu: str,
    peak: list[float],
    stop: threading.Event,
    poll_interval_s: float,
) -> None:
    while not stop.is_set():
        try:
            out = subprocess.check_output([
                "nvidia-smi", "-i", str(gpu),
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits"
            ], text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                parts = [x.strip() for x in line.split(',')]
                if len(parts) >= 2 and parts[0] == str(pid):
                    peak[0] = max(peak[0], float(parts[1]))
        except Exception:
            pass
        time.sleep(poll_interval_s)


def run_cmd(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log: Path,
    gpu: str,
    dry: bool,
    poll_interval_s: float,
) -> tuple[float, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print("[DRY]", " ".join(cmd), flush=True)
        return 0.0, 0.0
    start = time.perf_counter(); peak = [0.0]; stop = threading.Event()
    with log.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n"); f.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        t = threading.Thread(
            target=monitor_gpu,
            args=(proc.pid, gpu, peak, stop, poll_interval_s),
            daemon=True,
        )
        t.start()
        rc = proc.wait(); stop.set(); t.join(timeout=1)
    wall = time.perf_counter() - start
    if rc != 0:
        raise RuntimeError(f"Exit={rc}; inspect {log}")
    return wall, peak[0]


def make_common(args, run_dir: Path, seed: int) -> list[str]:
    return [
        args.python, args.entry,
        "--dataset", args.dataset,
        "--data_path", args.data_path,
        "--model_path", str(run_dir / "models"),
        "--run_dir", str(run_dir),
        "--use_best", "--is_pseudo",
        "--seed", str(seed),
    ]


def run_one(args, suite: str, variant: str, seed: int, s1_flags: list[str], s2_flags: list[str]) -> dict[str, Any]:
    run_dir = Path(args.run_root).resolve() / args.dataset / suite.upper() / variant / f"seed_{seed}"
    for d in (run_dir, run_dir/"models", run_dir/"pseudo_labels", run_dir/"logs"):
        d.mkdir(parents=True, exist_ok=True)
    s1_tag, s2_tag = tag_from_flags(s1_flags), tag_from_flags(s2_flags)
    common = make_common(args, run_dir, seed)
    cmd1 = common + s1_flags
    cmd2 = common + ["--finetune", "--pretrained_model"] + s2_flags
    manifest = {"dataset": args.dataset, "suite": suite, "variant": variant, "seed": seed,
                "stage1_tag": s1_tag, "stage2_tag": s2_tag,
                "stage1_cmd": cmd1, "stage2_cmd": cmd2, "run_dir": str(run_dir)}
    write_json(run_dir / "diagnostic_manifest.json", manifest)
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(args.gpu); env["PYTHONHASHSEED"] = str(seed)

    measurement_path = run_dir / "cost_measurement.json"
    previous = load_json(measurement_path)
    s1_time = float(previous.get("stage1_wall_s", float("nan")))
    s1_mem = float(previous.get("stage1_peak_gpu_mib", float("nan")))
    s2_time = float(previous.get("stage2_wall_s", float("nan")))
    s2_mem = float(previous.get("stage2_peak_gpu_mib", float("nan")))
    if not ok(expected_stage1(run_dir, args.dataset, s1_tag)) or args.force:
        print(f"[RUN] {args.dataset} {suite}/{variant} seed={seed} Stage-1", flush=True)
        s1_time, s1_mem = run_cmd(
            cmd1,
            Path(args.project_dir),
            env,
            run_dir/"logs"/"stage1.log",
            args.gpu,
            args.dry_run,
            args.gpu_poll_interval,
        )
    else:
        print(f"[RESUME] Stage-1 exists: {run_dir}", flush=True)
    if not args.dry_run:
        missing = [
            str(x)
            for x in expected_stage1(run_dir, args.dataset, s1_tag)
            if not artifact_exists(x)
        ]
        if missing: raise FileNotFoundError("Stage-1 outputs missing:\n" + "\n".join(missing))

    if not ok(expected_stage2(run_dir, args.dataset, s2_tag)) or args.force:
        print(f"[RUN] {args.dataset} {suite}/{variant} seed={seed} Stage-2", flush=True)
        s2_time, s2_mem = run_cmd(
            cmd2,
            Path(args.project_dir),
            env,
            run_dir/"logs"/"stage2.log",
            args.gpu,
            args.dry_run,
            args.gpu_poll_interval,
        )
    else:
        print(f"[RESUME] Stage-2 exists: {run_dir}", flush=True)
    if not args.dry_run:
        missing = [
            str(x)
            for x in expected_stage2(run_dir, args.dataset, s2_tag)
            if not artifact_exists(x)
        ]
        if missing: raise FileNotFoundError("Stage-2 outputs missing:\n" + "\n".join(missing))
    total_time = s1_time + s2_time if math.isfinite(s1_time) and math.isfinite(s2_time) else float("nan")
    pseudo_storage = (
        file_size_mib(pseudo_label_paths(run_dir, args.dataset, s1_tag))
        if not args.dry_run else float("nan")
    )
    result = {
        "dataset": args.dataset,
        "suite": suite.upper(),
        "variant": variant,
        "seed": seed,
        "stage1_wall_s": s1_time,
        "stage1_peak_gpu_mib": s1_mem,
        "stage2_wall_s": s2_time,
        "stage2_peak_gpu_mib": s2_mem,
        "total_wall_s": total_time,
        "pseudo_label_storage_mib": pseudo_storage,
        "gpu_name": gpu_name(args.gpu) if not args.dry_run else "dry-run",
        "gpu_poll_interval_s": args.gpu_poll_interval,
        "peak_gpu_measurement": "nvidia-smi process used_memory",
        "stage1_scope": (
            "full process: unimodal pretraining, validation, checkpoints, and pseudo-label I/O; "
            "TPLR variants also include EMA-teacher and progressive refinement"
        ),
        "stage2_scope": "full process: multimodal fine-tuning, validation, and checkpoint I/O",
        "run_dir": str(run_dir),
    }
    write_json(measurement_path, result)
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True)
    p.add_argument("--entry", default="pdcc_main.py")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dataset", required=True, choices=["SIMS", "MOSI"])
    p.add_argument("--data-path", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument(
        "--gpu-poll-interval",
        type=float,
        default=0.25,
        help="nvidia-smi process-memory sampling interval in seconds",
    )
    p.add_argument("--suites", default="rccr,tplr,cost", help="comma list: rccr,tplr,cost")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for option, value in [
        ("--project-dir", args.project_dir),
        ("--entry", args.entry),
        ("--python", args.python),
        ("--data-path", args.data_path),
        ("--run-root", args.run_root),
        ("--gpu", args.gpu),
    ]:
        if not str(value).strip():
            raise ValueError(
                f"{option} is empty. Define the corresponding shell variable "
                "or pass an explicit path/value."
            )
    if args.gpu_poll_interval <= 0:
        raise ValueError("--gpu-poll-interval must be positive")
    if not (Path(args.project_dir) / args.entry).is_file():
        raise FileNotFoundError(Path(args.project_dir) / args.entry)
    if not Path(args.data_path).is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_path}")
    seeds = parse_seeds(args.seeds)
    rows = []
    for suite in [s.strip().lower() for s in args.suites.split(',') if s.strip()]:
        if suite not in SUITES: raise ValueError(f"Unknown suite: {suite}")
        for variant, (s1, s2) in SUITES[suite].items():
            for seed in seeds:
                rows.append(run_one(args, suite, variant, seed, s1, s2))
    if rows:
        out = Path(args.run_root).resolve() / args.dataset / "diagnostic_retrain_cost.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("[DONE]", out, flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
