#!/usr/bin/env python3
"""Launch GeoFeedback-GS A100 experiments.

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


DEFAULT_FOUR_GROUP_CONFIGS = [
    "configs/experiments/a100_baseline_streetgs.yaml",
    "configs/experiments/a100_no_lidar_supervision_control.yaml",
    "configs/experiments/a100_da3_periodic_group_softpatch.yaml",
    "configs/experiments/a100_pv_da3_feedback_obj.yaml",
]

GROUP_LABELS = {
    "a100_baseline_streetgs": "A",
    "a100_no_lidar_supervision_control": "B",
    "a100_da3_periodic_group_softpatch": "C",
    "a100_pv_da3_feedback_obj": "PV-C",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_FOUR_GROUP_CONFIGS,
        help="Experiment configs. Defaults to the official A/B/C/PV-C four-group matrix.",
    )
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--train-entry", default="scripts/train.py")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_config_model_path(cfg_path):
    for line in Path(cfg_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("model_path:"):
            return stripped.split(":", 1)[1].strip().strip("\"'")
    return ""


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
        configured_model_path = read_config_model_path(cfg_path)
        exp_dir = Path(configured_model_path) if configured_model_path else root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, args.train_entry, "--config", str(cfg_path)]
        if args.resume:
            cmd.extend(["resume", "True"])
        if args.extra_args:
            cmd.extend(args.extra_args.split())
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        record = {
            "group": GROUP_LABELS.get(exp_name, ""),
            "experiment": exp_name,
            "config": str(cfg_path),
            "gpu": gpu,
            "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "output_dir": str(exp_dir),
            "command": cmd,
            "dry_run": bool(args.dry_run),
        }
        runs.append(record)
        group = f"{record['group']} " if record["group"] else ""
        print(f"[GeoGuardGS] {group}GPU {gpu} CUDA_VISIBLE_DEVICES={gpu}: {' '.join(cmd)}")
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
