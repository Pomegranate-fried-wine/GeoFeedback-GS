#!/usr/bin/env python3
"""GeoGuardGS training wrapper.

The current release keeps the runnable Street Gaussian training code under
third_party/street_gaussian for compatibility. This wrapper launches that
entrypoint from the correct working directory and passes through all arguments.
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml


def _deep_merge(base, override):
    result = dict(base or {})
    for key, value in (override or {}).items():
        if key == "_BASE_":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_merged_config(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        current = yaml.safe_load(f) or {}
    base_ref = current.get("_BASE_")
    if not base_ref:
        return current
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    return _deep_merge(_load_merged_config(base_path), current)


def _materialize_config(repo_root, config_path):
    payload = _load_merged_config(config_path)
    payload["workspace"] = str(repo_root)
    out_dir = repo_root / ".geoguardgs_merged_configs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / Path(config_path).name
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return out_path, payload


def _config_visible_devices(payload):
    gpus = payload.get("gpus")
    if not isinstance(gpus, list) or -1 in gpus:
        return None
    return ",".join(str(gpu) for gpu in gpus)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    streetgs_root = repo_root / "third_party" / "street_gaussian"
    train_entry = streetgs_root / "train.py"
    if not train_entry.exists():
        raise SystemExit(f"Missing Street Gaussian train entry: {train_entry}")
    env = os.environ.copy()
    pythonpath = [str(streetgs_root), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    args = []
    config_payloads = []
    for arg in sys.argv[1:]:
        if arg.endswith(".yaml") or arg.endswith(".yml"):
            p = Path(arg)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            materialized_path, payload = _materialize_config(repo_root, p)
            config_payloads.append(payload)
            args.append(str(materialized_path))
        else:
            args.append(arg)
    if not env.get("CUDA_VISIBLE_DEVICES") and config_payloads:
        visible_devices = _config_visible_devices(config_payloads[0])
        if visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = visible_devices
    print(
        "[GeoGuardGS][CUDA] Launching StreetGS with "
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    cmd = [sys.executable, str(train_entry)] + args
    raise SystemExit(subprocess.call(cmd, cwd=str(streetgs_root), env=env))


if __name__ == "__main__":
    main()
