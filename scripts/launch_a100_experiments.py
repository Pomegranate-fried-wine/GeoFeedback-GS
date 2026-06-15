#!/usr/bin/env python3
"""Launch GeoGuardGS A100 experiments.

This script is intentionally lightweight: it creates experiment manifests,
assigns configs to GPU ids, and either prints commands or starts subprocesses.
It does not modify configs in-place.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--output-root", default="outputs/a100_main_experiments")
    parser.add_argument("--train-entry", default="scripts/train.py")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("No GPU ids provided.")

    runs = []
    procs = []
    for idx, cfg in enumerate(args.configs):
        cfg_path = Path(cfg)
        exp_name = cfg_path.stem
        gpu = gpus[idx % len(gpus)]
        exp_dir = root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, args.train_entry, "--config", str(cfg_path)]
        if args.resume:
            cmd.extend(["resume", "True"])
        if args.extra_args:
            cmd.extend(args.extra_args.split())
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        record = {
            "experiment": exp_name,
            "config": str(cfg_path),
            "gpu": gpu,
            "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "output_dir": str(exp_dir),
            "command": cmd,
            "dry_run": bool(args.dry_run),
        }
        runs.append(record)
        print(f"[GeoGuardGS] GPU {gpu} CUDA_VISIBLE_DEVICES={gpu}: {' '.join(cmd)}")
        if not args.dry_run:
            log_path = exp_dir / "launch.log"
            log = open(log_path, "a", encoding="utf-8")
            procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(root),
        "runs": runs,
    }
    with open(root / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    for proc in procs:
        proc.wait()
    if procs and any(p.returncode != 0 for p in procs):
        raise SystemExit("At least one experiment failed. Check launch.log files.")


if __name__ == "__main__":
    main()
