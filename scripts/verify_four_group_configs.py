#!/usr/bin/env python3
"""Verify the official GeoGuardGS A/B/C/PV-C experiment matrix."""

import argparse
import json
from pathlib import Path

import yaml


OFFICIAL_GROUPS = {
    "A": "configs/experiments/a100_baseline_streetgs.yaml",
    "B": "configs/experiments/a100_no_lidar_supervision_control.yaml",
    "C": "configs/experiments/a100_da3_periodic_group_softpatch.yaml",
    "PV-C": "configs/experiments/a100_pv_da3_feedback_obj.yaml",
}

EXPECTED_MODEL_PATHS = {
    "A": "outputs/A_streetgs_lidar_init_lidar_sup",
    "B": "outputs/B_lidar_init_no_lidar_sup",
    "C": "outputs/C_lidar_init_da3_feedback",
    "PV-C": "outputs/PVC_no_lidar_init_da3_feedback",
}

SNAPSHOT_ITERS = [1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000]


def deep_merge(base, child):
    out = dict(base)
    for key, value in child.items():
        if key == "_BASE_":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.get("_BASE_")
    if base:
        return deep_merge(load_config((path.parent / base).resolve()), cfg)
    return cfg


def get(cfg, dotted, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def as_int_list(value):
    return [int(v) for v in (value or [])]


def check_common(cfg):
    checks = []
    checks.append(("iterations_30000", int(get(cfg, "train.iterations", 0)) == 30000))
    for key in ("train.test_iterations", "train.save_iterations", "train.checkpoint_iterations"):
        checks.append((key.replace(".", "_"), as_int_list(get(cfg, key, [])) == SNAPSHOT_ITERS))
    checks.append(("eval_full_interval_5000", int(get(cfg, "train.eval_full_interval", 0)) == 5000))
    checks.append(("full_snapshot_iterations", as_int_list(get(cfg, "train.full_snapshot_iterations", [])) == SNAPSHOT_ITERS))
    checks.append(("full_snapshot_train_test", {str(v) for v in get(cfg, "train.full_snapshot_splits", [])} == {"train", "test"}))
    return checks


def verify_group(group, cfg):
    checks = check_common(cfg)
    checks.append(("model_path_group_output_dir", str(get(cfg, "model_path", "")) == EXPECTED_MODEL_PATHS[group]))
    lidar_init = bool(get(cfg, "data.allow_lidar_initialization", False)) and not bool(get(cfg, "data.require_no_lidar_initialization", False))
    no_lidar_init = bool(get(cfg, "data.require_no_lidar_initialization", False)) and not bool(get(cfg, "data.allow_lidar_initialization", True))
    lidar_sup = bool(get(cfg, "train.guided_feedback.use_lidar_depth", False)) or float(get(cfg, "optim.lambda_depth_lidar", 0.0) or 0.0) > 0.0
    da3_structure = bool(get(cfg, "train.guided_feedback.use_da3_structure", False))
    feedback_enabled = bool(get(cfg, "train.feedback_controller.enabled", False))
    feedback_interval = int(get(cfg, "train.feedback_controller.interval", 0) or 0)
    feedback_start = int(get(cfg, "train.feedback_controller.start_iter", 0) or 0)
    risk_source = str(get(cfg, "train.feedback_controller.risk_source", ""))

    if group == "A":
        checks.extend([
            ("A_lidar_initialization_enabled", lidar_init),
            ("A_lidar_supervision_enabled", lidar_sup),
            ("A_da3_feedback_disabled", not da3_structure and not feedback_enabled),
        ])
    elif group == "B":
        checks.extend([
            ("B_lidar_initialization_enabled", lidar_init),
            ("B_lidar_supervision_disabled", not lidar_sup),
            ("B_da3_only_disabled", not da3_structure),
            ("B_feedback_controller_disabled", not feedback_enabled),
        ])
    elif group == "C":
        checks.extend([
            ("C_lidar_initialization_enabled", lidar_init),
            ("C_lidar_supervision_disabled", not lidar_sup),
            ("C_da3_feedback_enabled", da3_structure and feedback_enabled),
            ("C_feedback_interval_1000", feedback_start == 1000 and feedback_interval == 1000),
            ("C_da3_boundary_risk", risk_source == "da3_boundary"),
        ])
    elif group == "PV-C":
        checks.extend([
            ("PVC_lidar_initialization_disabled", no_lidar_init),
            ("PVC_lidar_supervision_disabled", not lidar_sup),
            ("PVC_da3_feedback_enabled", da3_structure and feedback_enabled),
            ("PVC_feedback_interval_1000", feedback_start == 1000 and feedback_interval == 1000),
            ("PVC_da3_boundary_risk", risk_source == "da3_boundary"),
        ])
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    root = Path(args.repo_root)
    rows = []
    failed = []
    for group, rel_path in OFFICIAL_GROUPS.items():
        cfg = load_config(root / rel_path)
        group_checks = verify_group(group, cfg)
        rows.append({
            "group": group,
            "config": rel_path,
            "checks": {name: bool(ok) for name, ok in group_checks},
        })
        failed.extend(f"{group}:{name}" for name, ok in group_checks if not ok)
    payload = {
        "status": "passed" if not failed else "failed",
        "official_groups": OFFICIAL_GROUPS,
        "snapshot_iterations": SNAPSHOT_ITERS,
        "groups": rows,
        "failed": failed,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
