#!/usr/bin/env python3
"""Safety checker for GeoGuardGS closed-loop configs."""

import argparse
import json
from pathlib import Path

import yaml


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
        base_path = (path.parent / base).resolve()
        return deep_merge(load_config(base_path), cfg)
    return cfg


def get(cfg, dotted, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
    checks = []

    supervision = get(cfg, "train.guided_feedback.supervision_mode", get(cfg, "train.feedback_controller.supervision_mode", ""))
    uses_lidar = bool(get(cfg, "train.guided_feedback.use_lidar_depth", False))
    lambda_lidar = float(get(cfg, "optim.lambda_depth_lidar", 0.0) or 0.0)
    risk_source = get(cfg, "train.feedback_controller.risk_source", "")
    control_mode = get(cfg, "train.gaussian_control.control_mode", "off")
    allow_param = bool(get(cfg, "train.gaussian_control.allow_parameter_modification", False))
    allow_prune = bool(get(cfg, "train.gaussian_control.allow_real_prune", False))
    allow_split = bool(get(cfg, "train.gaussian_control.allow_real_split", False))
    allow_shrink = bool(get(cfg, "train.gaussian_control.allow_real_shrink", False))
    cameras = _as_int_list(get(cfg, "data.cameras", []))
    lambda_sky_scale = get(cfg, "optim.lambda_sky_scale", [])

    if supervision == "da3_unsupervised":
        checks.append(("da3_no_lidar_loss", not uses_lidar and lambda_lidar == 0.0))
        checks.append(("da3_no_lidar_risk_source", risk_source != "lidar_error"))
    checks.append(("real_prune_disabled", not allow_prune))
    checks.append(("real_split_disabled", not allow_split))
    checks.append(("real_shrink_disabled", not allow_shrink))
    if control_mode == "opacity_decay_apply":
        checks.append(("opacity_decay_requires_parameter_permission", allow_param))
    else:
        checks.append(("no_parameter_modification_unless_decay", not allow_param))
    if isinstance(lambda_sky_scale, list) and lambda_sky_scale:
        max_cam = max(cameras) if cameras else -1
        checks.append(("lambda_sky_scale_covers_cameras", max_cam < len(lambda_sky_scale)))

    failed = [name for name, ok in checks if not ok]
    payload = {
        "config": args.config,
        "status": "passed" if not failed else "failed",
        "checks": {name: bool(ok) for name, ok in checks},
        "failed": failed,
        "summary": {
            "supervision_mode": supervision,
            "risk_source": risk_source,
            "control_mode": control_mode,
            "uses_lidar_depth": uses_lidar,
            "lambda_depth_lidar": lambda_lidar,
            "cameras": cameras,
            "lambda_sky_scale_length": len(lambda_sky_scale) if isinstance(lambda_sky_scale, list) else None,
        },
    }
    if "lambda_sky_scale_covers_cameras" in failed:
        payload["error"] = (
            "optim.lambda_sky_scale is too short for data.cameras: "
            f"max(data.cameras)={max(cameras) if cameras else 'none'}, "
            f"len(lambda_sky_scale)={len(lambda_sky_scale)}. "
            "Use a scale list that covers all camera ids, e.g. [1, 1, 0, 0, 0] for cameras [0,1,2,3,4]."
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
