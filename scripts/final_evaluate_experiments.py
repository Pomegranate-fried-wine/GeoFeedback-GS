#!/usr/bin/env python3
"""Run paper-grade final evaluation for GeoGuardGS/StreetGS experiments.

This evaluates a trained checkpoint directly, without relying on the sampled
training-time diagnostic CSV. It writes full-image, object-region, and
background-region metrics for train/test splits.
"""

import argparse
import csv
import json
import math
import os
import shutil
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


def _materialize_config(repo_root, config_path, out_dir, loaded_iter):
    payload = _load_merged_config(config_path)
    payload["workspace"] = str(repo_root)
    payload["mode"] = "evaluate"
    payload["loaded_iter"] = int(loaded_iter)
    existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing_visible:
        payload["gpus"] = [-1]
        print(
            "[FinalEval][CUDA] Respect existing "
            f"CUDA_VISIBLE_DEVICES={existing_visible}; disable cfg.gpus override"
        )
    else:
        print(
            "[FinalEval][CUDA] CUDA_VISIBLE_DEVICES is unset; "
            f"StreetGS may use cfg.gpus={payload.get('gpus', '<missing>')}"
        )
    payload.setdefault("eval", {})
    payload["eval"]["skip_train"] = False
    payload["eval"]["skip_test"] = False
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(config_path).stem}_final_eval.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return out_path, payload


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def to_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def summarize(rows, scope, split):
    subset = [r for r in rows if r.get("scope") == scope and r.get("split") == split and r.get("status") == "valid"]
    out = {"scope": scope, "split": split, "view_count": len(subset)}
    for metric in ["l1", "psnr", "ssim", "lpips"]:
        values = [to_float(r.get(metric)) for r in subset]
        values = [v for v in values if v is not None]
        if values:
            out[f"{metric}_mean"] = sum(values) / len(values)
            sorted_values = sorted(values)
            mid = len(sorted_values) // 2
            out[f"{metric}_median"] = sorted_values[mid] if len(sorted_values) % 2 else 0.5 * (sorted_values[mid - 1] + sorted_values[mid])
            mean = out[f"{metric}_mean"]
            out[f"{metric}_std"] = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            out[f"{metric}_min"] = min(values)
            out[f"{metric}_max"] = max(values)
        else:
            out[f"{metric}_mean"] = ""
            out[f"{metric}_median"] = ""
            out[f"{metric}_std"] = ""
            out[f"{metric}_min"] = ""
            out[f"{metric}_max"] = ""
    return out


def _save_panel(path, title_images):
    import cv2
    import numpy as np

    rows = []
    for title, img in title_images:
        arr = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        arr = (arr * 255).astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.putText(arr, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(arr, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(arr)
    panel = np.concatenate(rows, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def _metric_row(torch, loss_utils, lpips_fn, split, idx, camera, image, gt, mask, scope, include_lpips):
    mask = mask.bool()
    valid = int(torch.count_nonzero(mask).item())
    if valid == 0:
        return {
            "split": split,
            "scope": scope,
            "view_index": idx,
            "cam_id": camera.meta.get("cam", ""),
            "frame": camera.meta.get("frame", ""),
            "image_name": getattr(camera, "image_name", ""),
            "valid_pixel_count": 0,
            "status": "not_applicable",
            "warning": "empty_scope_mask",
        }
    row = {
        "split": split,
        "scope": scope,
        "view_index": idx,
        "cam_id": camera.meta.get("cam", ""),
        "frame": camera.meta.get("frame", ""),
        "frame_idx": camera.meta.get("frame_idx", ""),
        "image_name": getattr(camera, "image_name", ""),
        "valid_pixel_count": valid,
        "status": "valid",
        "warning": "",
    }
    row["l1"] = float(loss_utils.l1_loss(image, gt, mask).detach().cpu().item())
    row["psnr"] = float(loss_utils.psnr(image, gt, mask).detach().cpu().item())
    try:
        row["ssim"] = float(loss_utils.ssim(image, gt, mask=mask).detach().cpu().item())
    except Exception as exc:
        row["ssim"] = ""
        row["warning"] = f"ssim_failed:{exc}"
    if include_lpips:
        try:
            masked_image = torch.where(mask, image, torch.zeros_like(image))
            masked_gt = torch.where(mask, gt, torch.zeros_like(gt))
            row["lpips"] = float(lpips_fn(masked_image, masked_gt, net_type="alex").detach().cpu().item())
        except Exception as exc:
            row["lpips"] = ""
            row["warning"] = (row["warning"] + ";" if row["warning"] else "") + f"lpips_failed:{exc}"
    else:
        row["lpips"] = ""
    return row


def evaluate_one(repo_root, config_path, final_root, loaded_iter, max_panels, include_lpips):
    streetgs_root = repo_root / "third_party" / "street_gaussian"
    sys.path.insert(0, str(streetgs_root))
    sys.path.insert(0, str(repo_root))
    os.chdir(streetgs_root)

    exp_out = final_root / Path(config_path).stem
    materialized, payload = _materialize_config(repo_root, config_path, exp_out / "configs", loaded_iter)
    sys.argv = ["final_evaluate_experiments.py", "--config", str(materialized)]

    import torch
    from lib.config import cfg
    from lib.datasets.dataset import Dataset
    from lib.models.scene import Scene
    from lib.models.street_gaussian_model import StreetGaussianModel
    from lib.models.street_gaussian_renderer import StreetGaussianRenderer
    from lib.utils import loss_utils
    from lib.utils.lpipsPyTorch import lpips as lpips_fn

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for final evaluation.")

    rows = []
    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        split_cameras = {
            "train": scene.getTrainCameras(),
            "test": scene.getTestCameras(),
        }
        for split, cameras in split_cameras.items():
            for idx, camera in enumerate(cameras):
                render_pkg = renderer.render(camera, gaussians)
                image = torch.clamp(render_pkg["rgb"], 0.0, 1.0)
                gt = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
                full_mask = torch.ones_like(gt[0:1]).bool()
                if "mask" in camera.guidance:
                    full_mask = camera.guidance["mask"].to("cuda").bool()
                obj_mask = camera.guidance.get("obj_bound")
                if obj_mask is None:
                    obj_mask = torch.zeros_like(full_mask)
                else:
                    obj_mask = obj_mask.to("cuda").bool()
                    if obj_mask.ndim == 2:
                        obj_mask = obj_mask[None]
                bg_mask = full_mask & (~obj_mask)
                rows.append(_metric_row(torch, loss_utils, lpips_fn, split, idx, camera, image, gt, full_mask, "full_image", include_lpips))
                rows.append(_metric_row(torch, loss_utils, lpips_fn, split, idx, camera, image, gt, obj_mask & full_mask, "object_region", include_lpips))
                rows.append(_metric_row(torch, loss_utils, lpips_fn, split, idx, camera, image, gt, bg_mask, "background_region", include_lpips))
                if idx < max_panels:
                    err = torch.clamp(torch.abs(image - gt) * 4.0, 0.0, 1.0)
                    panel_path = exp_out / "figures" / "final_comparison_panels" / split / f"{camera.image_name}_panel.jpg"
                    _save_panel(panel_path, [("GT RGB", gt), ("Rendered RGB", image), ("RGB Error x4", err)])

    fields = [
        "split", "scope", "view_index", "cam_id", "frame", "frame_idx", "image_name",
        "valid_pixel_count", "status", "l1", "psnr", "ssim", "lpips", "warning",
    ]
    write_csv(exp_out / "metrics_full_image.csv", [r for r in rows if r["scope"] == "full_image"], fields)
    write_csv(exp_out / "metrics_object_region.csv", [r for r in rows if r["scope"] == "object_region"], fields)
    write_csv(exp_out / "metrics_background_region.csv", [r for r in rows if r["scope"] == "background_region"], fields)
    summary_rows = [summarize(rows, scope, split) for scope in ["full_image", "object_region", "background_region"] for split in ["train", "test"]]
    summary_fields = ["scope", "split", "view_count"]
    for metric in ["l1", "psnr", "ssim", "lpips"]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_median", f"{metric}_std", f"{metric}_min", f"{metric}_max"])
    write_csv(exp_out / "summary_by_scope.csv", summary_rows, summary_fields)
    manifest = {
        "experiment": Path(config_path).stem,
        "model_path": payload.get("model_path", ""),
        "loaded_iter": int(loaded_iter),
        "eval_protocol": "full_final_evaluation",
        "splits": {split: len(cameras) for split, cameras in split_cameras.items()},
        "include_obj": payload.get("model", {}).get("nsg", {}).get("include_obj", ""),
        "include_lpips": include_lpips,
    }
    (exp_out / "final_evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return exp_out, summary_rows, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-root", default="outputs/final_evaluation_full_scene_v2")
    parser.add_argument("--loaded-iter", type=int, default=30000)
    parser.add_argument("--max-panels-per-split", type=int, default=12)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--single-config-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    final_root = repo_root / args.output_root
    if final_root.exists():
        print(f"[FinalEval] Using existing output root: {final_root}")
    final_root.mkdir(parents=True, exist_ok=True)

    if not args.single_config_worker and len(args.configs) > 1:
        for config in args.configs:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--configs",
                config,
                "--output-root",
                args.output_root,
                "--loaded-iter",
                str(args.loaded_iter),
                "--max-panels-per-split",
                str(args.max_panels_per_split),
                "--single-config-worker",
            ]
            if args.skip_lpips:
                cmd.append("--skip-lpips")
            ret = subprocess.call(cmd, cwd=str(repo_root))
            if ret != 0:
                raise SystemExit(ret)
        aggregate_existing(final_root)
        return

    main_rows = []
    by_scope_rows = []
    for config in args.configs:
        exp_dir, summaries, manifest = evaluate_one(
            repo_root,
            Path(config).resolve(),
            final_root,
            args.loaded_iter,
            args.max_panels_per_split,
            include_lpips=not args.skip_lpips,
        )
        for row in summaries:
            out = {"experiment": exp_dir.name, "eval_protocol": "full_final_evaluation", **row}
            by_scope_rows.append(out)
            if row["scope"] == "full_image" and row["split"] == "test":
                main_rows.append(out)

    summary_fields = ["experiment", "eval_protocol", "scope", "split", "view_count"]
    for metric in ["l1", "psnr", "ssim", "lpips"]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_median", f"{metric}_std", f"{metric}_min", f"{metric}_max"])
    write_csv(final_root / "summary_main.csv", main_rows, summary_fields)
    write_csv(final_root / "summary_by_scope.csv", by_scope_rows, summary_fields)
    print(json.dumps({
        "output_root": str(final_root),
        "experiment_count": len(args.configs),
        "summary_main": str(final_root / "summary_main.csv"),
        "summary_by_scope": str(final_root / "summary_by_scope.csv"),
    }, indent=2, ensure_ascii=False))


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def aggregate_existing(final_root):
    by_scope_rows = []
    for path in sorted(final_root.glob("*/summary_by_scope.csv")):
        exp = path.parent.name
        for row in read_csv(path):
            by_scope_rows.append({"experiment": exp, "eval_protocol": "full_final_evaluation", **row})
    summary_fields = ["experiment", "eval_protocol", "scope", "split", "view_count"]
    for metric in ["l1", "psnr", "ssim", "lpips"]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_median", f"{metric}_std", f"{metric}_min", f"{metric}_max"])
    main_rows = [
        row for row in by_scope_rows
        if row.get("scope") == "full_image" and row.get("split") == "test"
    ]
    write_csv(final_root / "summary_main.csv", main_rows, summary_fields)
    write_csv(final_root / "summary_by_scope.csv", by_scope_rows, summary_fields)
    print(json.dumps({
        "output_root": str(final_root),
        "experiment_count": len({row["experiment"] for row in by_scope_rows}),
        "summary_main": str(final_root / "summary_main.csv"),
        "summary_by_scope": str(final_root / "summary_by_scope.csv"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
