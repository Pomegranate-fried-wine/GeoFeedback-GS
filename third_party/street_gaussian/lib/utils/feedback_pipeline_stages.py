import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from lib.utils.cuda_contribution_utils import capture_contributions_cuda_live, write_live_contribution_outputs

try:
    import cv2
except ImportError:
    cv2 = None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _grad_mag_np(x):
    dx = np.zeros_like(x, dtype=np.float32)
    dy = np.zeros_like(x, dtype=np.float32)
    dx[:, :-1] = x[:, 1:] - x[:, :-1]
    dy[:-1, :] = x[1:, :] - x[:-1, :]
    return np.sqrt(dx * dx + dy * dy + 1e-8)


def _normalize(x, mask):
    valid = mask & np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[valid], [5, 95])
    return np.clip((x - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)


def _safe_view_id(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "unknown"))


def _copy_existing(src, dst):
    if src and os.path.exists(src):
        ensure_dir(os.path.dirname(dst))
        shutil.copyfile(src, dst)
        return dst
    return ""


def _load_da3_risk_cache(cache_dir, view_id, depth_shape):
    if not cache_dir:
        return None
    view_cache_dir = os.path.join(cache_dir, _safe_view_id(view_id))
    manifest_path = os.path.join(view_cache_dir, "risk_cache_manifest.json")
    risk_matrix_path = os.path.join(view_cache_dir, "risk_score_matrix.npy")
    selected_path = os.path.join(view_cache_dir, "selected_pixels.npy")
    selected_risk_path = os.path.join(view_cache_dir, "selected_pixel_risk_scores.npy")
    if not (os.path.exists(manifest_path) and os.path.exists(risk_matrix_path) and os.path.exists(selected_path) and os.path.exists(selected_risk_path)):
        return None
    try:
        manifest = read_json(manifest_path)
        if list(manifest.get("image_shape", [])) != [int(depth_shape[0]), int(depth_shape[1])]:
            return None
        return {
            "view_cache_dir": view_cache_dir,
            "manifest": manifest,
            "risk_matrix_path": risk_matrix_path,
            "risk_band_mask_path": os.path.join(view_cache_dir, "risk_band_mask.npy"),
            "selected_pixels_path": selected_path,
            "selected_pixel_risk_scores_path": selected_risk_path,
            "risk_score_heatmap_path": os.path.join(view_cache_dir, "risk_score_heatmap.png"),
            "risk_band_mask_png_path": os.path.join(view_cache_dir, "risk_band_mask.png"),
            "da3_edge_score_path": os.path.join(view_cache_dir, "da3_edge_score.npy"),
            "da3_edge_score_png_path": os.path.join(view_cache_dir, "da3_edge_score.png"),
            "rendered_edge_score_path": os.path.join(view_cache_dir, "rendered_edge_score.npy"),
            "rendered_edge_score_png_path": os.path.join(view_cache_dir, "rendered_edge_score.png"),
        }
    except Exception:
        return None


def _rgb_quality_risk(rendered_outputs, depth_shape):
    rgb = rendered_outputs.get("rgb")
    camera = rendered_outputs.get("camera")
    gt = getattr(camera, "original_image", None) if camera is not None else None
    if torch.is_tensor(rgb):
        rgb = rgb.detach().float().cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.detach().float().cpu().numpy()
    if rgb is None or gt is None:
        return np.zeros(depth_shape, dtype=np.float32), False, "missing_rendered_or_gt_rgb"
    rgb = np.asarray(rgb, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if rgb.ndim == 3 and rgb.shape[0] in {1, 3, 4}:
        rgb = np.transpose(rgb[:3], (1, 2, 0))
    if gt.ndim == 3 and gt.shape[0] in {1, 3, 4}:
        gt = np.transpose(gt[:3], (1, 2, 0))
    if rgb.shape[:2] != tuple(depth_shape) or gt.shape[:2] != tuple(depth_shape):
        if cv2 is None:
            return np.zeros(depth_shape, dtype=np.float32), False, "rgb_shape_mismatch_without_cv2"
        rgb = cv2.resize(rgb, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_LINEAR)
    rgb = np.clip(rgb, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)
    err = np.mean(np.abs(rgb - gt), axis=-1).astype(np.float32)
    gray_render = np.mean(rgb, axis=-1).astype(np.float32)
    gray_gt = np.mean(gt, axis=-1).astype(np.float32)
    edge_gap = np.abs(_grad_mag_np(gray_render) - _grad_mag_np(gray_gt)).astype(np.float32)
    mask = np.isfinite(err) & np.isfinite(edge_gap)
    rgb_risk = 0.70 * _normalize(err, mask) + 0.30 * _normalize(edge_gap, mask)
    return rgb_risk.astype(np.float32), True, "rendered_rgb_l1_plus_rgb_edge_mismatch"


def build_da3_boundary_risk_stage(rendered_outputs, views, out_dir, max_pixels_per_region=64, cache_dir=""):
    """Build an original-resolution DA3-prior risk matrix and selected risk band."""
    ensure_dir(out_dir)
    depth = rendered_outputs.get("depth")
    acc = rendered_outputs.get("acc")
    da3_depth = rendered_outputs.get("da3_depth")
    view_id = rendered_outputs.get("view_id", views[0] if views else "unknown")
    if torch.is_tensor(depth):
        depth = depth.detach().float().cpu().numpy().squeeze()
    if torch.is_tensor(acc):
        acc = acc.detach().float().cpu().numpy().squeeze()
    if torch.is_tensor(da3_depth):
        da3_depth = da3_depth.detach().float().cpu().numpy().squeeze()
    if depth is None:
        payload = {"status": "failed", "reason": "missing rendered depth", "views": views}
        write_json(os.path.join(out_dir, "risk_summary.json"), payload)
        return payload
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    acc = np.ones_like(depth, dtype=np.float32) if acc is None else np.asarray(acc, dtype=np.float32).squeeze()
    cached = _load_da3_risk_cache(cache_dir, view_id, depth.shape)
    if cached is not None:
        base_risk = np.load(cached["risk_matrix_path"]).astype(np.float32)
        rgb_risk, rgb_available, rgb_note = _rgb_quality_risk(rendered_outputs, depth.shape)
        valid = np.isfinite(depth) & np.isfinite(acc) & (acc > 0.03)
        risk = np.where(valid, np.clip(0.70 * base_risk + 0.30 * rgb_risk, 0.0, 1.0), 0.0).astype(np.float32)
        if np.any(valid):
            thr = np.percentile(risk[valid], 92)
            band = valid & (risk >= thr) & (risk > 0)
        else:
            band = np.zeros_like(valid, dtype=bool)
        if cv2 is not None and np.any(band):
            band = cv2.dilate(band.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool) & valid
        ys, xs = np.where(band)
        truncated = False
        if len(xs) > max_pixels_per_region:
            order = np.argsort(risk[ys, xs])[::-1][:max_pixels_per_region]
            xs, ys = xs[order], ys[order]
            truncated = True
        selected = np.stack([xs, ys], axis=1).astype(np.int64) if len(xs) else np.zeros((0, 2), dtype=np.int64)
        selected_risk = risk[selected[:, 1], selected[:, 0]].astype(np.float32) if len(selected) else np.zeros((0,), dtype=np.float32)
        local_risk_path = os.path.join(out_dir, f"{view_id}_risk_score_matrix.npy")
        local_band_path = os.path.join(out_dir, f"{view_id}_risk_band_mask.npy")
        local_selected_path = os.path.join(out_dir, f"{view_id}_selected_pixels.npy")
        local_selected_risk_path = os.path.join(out_dir, f"{view_id}_selected_pixel_risk_scores.npy")
        local_rgb_risk_path = os.path.join(out_dir, f"{view_id}_rendered_rgb_quality_risk.npy")
        np.save(local_risk_path, risk.astype(np.float32))
        np.save(local_band_path, band.astype(np.uint8))
        np.save(local_selected_path, selected)
        np.save(local_selected_risk_path, selected_risk)
        np.save(local_rgb_risk_path, rgb_risk.astype(np.float32))
        risk_heatmap_path = os.path.join(out_dir, f"{view_id}_risk_score_heatmap.png")
        rgb_risk_heatmap_path = os.path.join(out_dir, f"{view_id}_rendered_rgb_quality_risk.png")
        risk_band_png_path = os.path.join(out_dir, f"{view_id}_risk_band_mask.png")
        if cv2 is not None:
            cv2.imwrite(risk_heatmap_path, _colorize01(risk))
            cv2.imwrite(rgb_risk_heatmap_path, _colorize01(rgb_risk))
            cv2.imwrite(risk_band_png_path, (band.astype(np.uint8) * 255))
        payload = {
            "status": "valid",
            "risk_source": "da3_boundary",
            "selected_pixel_source": "cached_da3_boundary_prior_plus_dynamic_rendered_rgb_quality",
            "uses_lidar_selected_pixels": False,
            "view_id": view_id,
            "views": views,
            "selected_pixels_count": int(len(selected)),
            "risk_band_pixel_count": int(np.count_nonzero(band)),
            "selected_pixels_truncated": bool(truncated),
            "risk_threshold_percentile": 92,
            "risk_map_path": local_risk_path,
            "risk_score_matrix_path": local_risk_path,
            "risk_band_mask_path": local_band_path,
            "risk_score_heatmap_path": risk_heatmap_path if cv2 is not None else "",
            "risk_band_mask_png_path": risk_band_png_path if cv2 is not None else "",
            "selected_pixel_risk_scores_path": local_selected_risk_path,
            "da3_edge_score_path": cached["da3_edge_score_path"] if os.path.exists(cached["da3_edge_score_path"]) else "",
            "da3_edge_score_png_path": cached["da3_edge_score_png_path"] if os.path.exists(cached["da3_edge_score_png_path"]) else "",
            "rendered_edge_score_path": cached["rendered_edge_score_path"] if os.path.exists(cached["rendered_edge_score_path"]) else "",
            "rendered_edge_score_png_path": cached["rendered_edge_score_png_path"] if os.path.exists(cached["rendered_edge_score_png_path"]) else "",
            "rendered_rgb_quality_risk_path": local_rgb_risk_path,
            "rendered_rgb_quality_risk_png_path": rgb_risk_heatmap_path if cv2 is not None else "",
            "rendered_rgb_quality_available": bool(rgb_available),
            "selected_pixels_path": local_selected_path,
            "da3_depth_available": bool(cached["manifest"].get("da3_depth_available", True)),
            "risk_cache_hit": True,
            "risk_cache_dir": cached["view_cache_dir"],
            "note": f"Loaded cached original-view DA3 risk prior, then dynamically boosted risk by current rendered RGB quality ({rgb_note}); current Gaussian contribution is recomputed downstream.",
        }
        write_json(os.path.join(out_dir, "risk_summary.json"), payload)
        return payload
    valid = np.isfinite(depth) & np.isfinite(acc) & (acc > 0.03)
    norm_depth = _normalize(depth / np.maximum(acc, 1e-6), valid)
    rendered_edge = _grad_mag_np(norm_depth)
    da3_depth_available = da3_depth is not None and np.size(da3_depth) > 0
    if da3_depth_available:
        da3_depth = np.asarray(da3_depth, dtype=np.float32).squeeze()
        if da3_depth.shape != depth.shape:
            if cv2 is None:
                da3_depth_available = False
            else:
                da3_depth = cv2.resize(da3_depth, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
    if da3_depth_available:
        da3_norm = _normalize(da3_depth, np.isfinite(da3_depth))
        da3_edge = _grad_mag_np(da3_norm)
        edge_gap = np.clip(da3_edge - rendered_edge, 0.0, None)
        edge_mismatch = np.abs(da3_edge - rendered_edge)
        risk = 0.65 * _normalize(da3_edge, np.isfinite(da3_edge)) + 0.25 * _normalize(edge_gap, valid) + 0.10 * _normalize(edge_mismatch, valid)
        risk_source_note = "DA3 prior edge strength plus rendered-depth edge mismatch."
    else:
        da3_edge = np.zeros_like(rendered_edge, dtype=np.float32)
        risk = _normalize(rendered_edge, valid)
        risk_source_note = "Fallback rendered-depth edge risk because DA3 prior was unavailable."
    risk = np.where(valid & np.isfinite(risk), risk, 0.0).astype(np.float32)
    da3_prior_risk = risk.copy()
    rgb_risk, rgb_available, rgb_note = _rgb_quality_risk(rendered_outputs, depth.shape)
    risk = np.where(valid, np.clip(0.70 * risk + 0.30 * rgb_risk, 0.0, 1.0), 0.0).astype(np.float32)
    if np.any(valid):
        thr = np.percentile(risk[valid], 92)
        band = valid & (risk >= thr) & (risk > 0)
    else:
        band = np.zeros_like(valid, dtype=bool)
    if cv2 is not None and np.any(band):
        band = cv2.dilate(band.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool) & valid
    ys, xs = np.where(band)
    truncated = False
    if len(xs) > max_pixels_per_region:
        order = np.argsort(risk[ys, xs])[::-1][:max_pixels_per_region]
        xs, ys = xs[order], ys[order]
        truncated = True
    selected = np.stack([xs, ys], axis=1).astype(np.int64) if len(xs) else np.zeros((0, 2), dtype=np.int64)
    risk_map_path = os.path.join(out_dir, f"{view_id}_da3_boundary_risk.npy")
    np.save(risk_map_path, risk.astype(np.float32))
    risk_matrix_path = os.path.join(out_dir, f"{view_id}_risk_score_matrix.npy")
    np.save(risk_matrix_path, risk.astype(np.float32))
    risk_band_path = os.path.join(out_dir, f"{view_id}_risk_band_mask.npy")
    np.save(risk_band_path, band.astype(np.uint8))
    da3_edge_path = os.path.join(out_dir, f"{view_id}_da3_edge_score.npy")
    np.save(da3_edge_path, da3_edge.astype(np.float32))
    rendered_edge_path = os.path.join(out_dir, f"{view_id}_rendered_edge_score.npy")
    np.save(rendered_edge_path, rendered_edge.astype(np.float32))
    selected_path = os.path.join(out_dir, f"{view_id}_selected_pixels.npy")
    np.save(selected_path, selected)
    selected_risk_path = os.path.join(out_dir, f"{view_id}_selected_pixel_risk_scores.npy")
    selected_risk = risk[selected[:, 1], selected[:, 0]].astype(np.float32) if len(selected) else np.zeros((0,), dtype=np.float32)
    np.save(selected_risk_path, selected_risk)
    rgb_risk_path = os.path.join(out_dir, f"{view_id}_rendered_rgb_quality_risk.npy")
    np.save(rgb_risk_path, rgb_risk.astype(np.float32))
    risk_heatmap_path = ""
    risk_band_png_path = ""
    da3_edge_png_path = ""
    rendered_edge_png_path = ""
    rgb_risk_heatmap_path = ""
    if cv2 is not None:
        risk_heatmap_path = os.path.join(out_dir, f"{view_id}_risk_score_heatmap.png")
        risk_band_png_path = os.path.join(out_dir, f"{view_id}_risk_band_mask.png")
        da3_edge_png_path = os.path.join(out_dir, f"{view_id}_da3_edge_score.png")
        rendered_edge_png_path = os.path.join(out_dir, f"{view_id}_rendered_edge_score.png")
        rgb_risk_heatmap_path = os.path.join(out_dir, f"{view_id}_rendered_rgb_quality_risk.png")
        cv2.imwrite(risk_heatmap_path, _colorize01(risk))
        cv2.imwrite(risk_band_png_path, (band.astype(np.uint8) * 255))
        cv2.imwrite(da3_edge_png_path, _colorize01(_normalize(da3_edge, np.isfinite(da3_edge))))
        cv2.imwrite(rendered_edge_png_path, _colorize01(_normalize(rendered_edge, np.isfinite(rendered_edge))))
        cv2.imwrite(rgb_risk_heatmap_path, _colorize01(rgb_risk))
    payload = {
        "status": "valid",
        "risk_source": "da3_boundary",
        "selected_pixel_source": "da3_boundary_risk_map",
        "uses_lidar_selected_pixels": False,
        "view_id": view_id,
        "views": views,
        "selected_pixels_count": int(len(selected)),
        "risk_band_pixel_count": int(np.count_nonzero(band)),
        "selected_pixels_truncated": bool(truncated),
        "risk_threshold_percentile": 92,
        "risk_map_path": risk_map_path,
        "risk_score_matrix_path": risk_matrix_path,
        "risk_band_mask_path": risk_band_path,
        "risk_score_heatmap_path": risk_heatmap_path,
        "risk_band_mask_png_path": risk_band_png_path,
        "selected_pixel_risk_scores_path": selected_risk_path,
        "da3_edge_score_path": da3_edge_path,
        "da3_edge_score_png_path": da3_edge_png_path,
        "rendered_edge_score_path": rendered_edge_path,
        "rendered_edge_score_png_path": rendered_edge_png_path,
        "rendered_rgb_quality_risk_path": rgb_risk_path,
        "rendered_rgb_quality_risk_png_path": rgb_risk_heatmap_path,
        "rendered_rgb_quality_available": bool(rgb_available),
        "selected_pixels_path": selected_path,
        "da3_depth_available": bool(da3_depth_available),
        "risk_cache_hit": False,
        "risk_cache_dir": "",
        "note": f"{risk_source_note} Dynamic risk is additionally boosted by current rendered RGB quality ({rgb_note}).",
    }
    if cache_dir:
        view_cache_dir = os.path.join(cache_dir, _safe_view_id(view_id))
        ensure_dir(view_cache_dir)
        cache_paths = {
            "risk_score_matrix_path": os.path.join(view_cache_dir, "risk_score_matrix.npy"),
            "risk_band_mask_path": os.path.join(view_cache_dir, "risk_band_mask.npy"),
            "selected_pixels_path": os.path.join(view_cache_dir, "selected_pixels.npy"),
            "selected_pixel_risk_scores_path": os.path.join(view_cache_dir, "selected_pixel_risk_scores.npy"),
            "da3_edge_score_path": os.path.join(view_cache_dir, "da3_edge_score.npy"),
            "rendered_edge_score_path": os.path.join(view_cache_dir, "rendered_edge_score.npy"),
        }
        np.save(cache_paths["risk_score_matrix_path"], da3_prior_risk.astype(np.float32))
        np.save(cache_paths["risk_band_mask_path"], band.astype(np.uint8))
        np.save(cache_paths["selected_pixels_path"], selected)
        np.save(cache_paths["selected_pixel_risk_scores_path"], selected_risk)
        np.save(cache_paths["da3_edge_score_path"], da3_edge.astype(np.float32))
        np.save(cache_paths["rendered_edge_score_path"], rendered_edge.astype(np.float32))
        if cv2 is not None:
            cache_paths["risk_score_heatmap_path"] = os.path.join(view_cache_dir, "risk_score_heatmap.png")
            cache_paths["risk_band_mask_png_path"] = os.path.join(view_cache_dir, "risk_band_mask.png")
            cache_paths["da3_edge_score_png_path"] = os.path.join(view_cache_dir, "da3_edge_score.png")
            cache_paths["rendered_edge_score_png_path"] = os.path.join(view_cache_dir, "rendered_edge_score.png")
            cv2.imwrite(cache_paths["risk_score_heatmap_path"], _colorize01(risk))
            cv2.imwrite(cache_paths["risk_band_mask_png_path"], (band.astype(np.uint8) * 255))
            cv2.imwrite(cache_paths["da3_edge_score_png_path"], _colorize01(_normalize(da3_edge, np.isfinite(da3_edge))))
            cv2.imwrite(cache_paths["rendered_edge_score_png_path"], _colorize01(_normalize(rendered_edge, np.isfinite(rendered_edge))))
        cache_manifest = {
            "view_id": view_id,
            "image_shape": [int(depth.shape[0]), int(depth.shape[1])],
            "risk_source": "da3_boundary",
            "risk_threshold_percentile": 92,
            "risk_band_pixel_count": int(np.count_nonzero(band)),
            "selected_pixels_count": int(len(selected)),
            "selected_pixels_truncated": bool(truncated),
            "da3_depth_available": bool(da3_depth_available),
            **cache_paths,
        }
        write_json(os.path.join(view_cache_dir, "risk_cache_manifest.json"), cache_manifest)
        payload["risk_cache_dir"] = view_cache_dir
    write_json(os.path.join(out_dir, "risk_summary.json"), payload)
    return payload


def _colorize01(x):
    x = np.asarray(x, dtype=np.float32)
    u8 = (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def build_lidar_error_risk_stage(rendered_outputs, views, out_dir, max_pixels_per_region=64):
    ensure_dir(out_dir)
    depth = rendered_outputs.get("depth")
    acc = rendered_outputs.get("acc")
    camera = rendered_outputs.get("camera")
    view_id = rendered_outputs.get("view_id", views[0] if views else "unknown")
    lidar_depth = None
    if camera is not None and hasattr(camera, "guidance") and "lidar_depth" in camera.guidance:
        lidar_depth = camera.guidance["lidar_depth"]
    if torch.is_tensor(depth):
        depth = depth.detach().float().cpu().numpy().squeeze()
    if torch.is_tensor(acc):
        acc = acc.detach().float().cpu().numpy().squeeze()
    if torch.is_tensor(lidar_depth):
        lidar_depth = lidar_depth.detach().float().cpu().numpy().squeeze()
    if depth is None or lidar_depth is None:
        payload = {
            "status": "failed",
            "risk_source": "lidar_error",
            "views": views,
            "reason": "missing rendered depth or lidar_depth for lidar_error risk stage",
        }
        write_json(os.path.join(out_dir, "risk_summary.json"), payload)
        return payload
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    acc = np.ones_like(depth, dtype=np.float32) if acc is None else np.asarray(acc, dtype=np.float32).squeeze()
    lidar_depth = np.asarray(lidar_depth, dtype=np.float32).squeeze()
    rendered_depth = depth / np.maximum(acc, 1e-6)
    valid = (
        np.isfinite(rendered_depth)
        & np.isfinite(lidar_depth)
        & (lidar_depth > 1.0)
        & (lidar_depth < 80.0)
        & (rendered_depth > 1.0)
        & (rendered_depth < 80.0)
        & (acc > 0.03)
    )
    error = np.zeros_like(rendered_depth, dtype=np.float32)
    error[valid] = np.abs(rendered_depth[valid] - lidar_depth[valid])
    ys, xs = np.where(valid & (error > 0))
    if len(xs) > max_pixels_per_region:
        order = np.argsort(error[ys, xs])[::-1][:max_pixels_per_region]
        xs, ys = xs[order], ys[order]
    selected = np.stack([xs, ys], axis=1).astype(np.int64) if len(xs) else np.zeros((0, 2), dtype=np.int64)
    risk_map_path = os.path.join(out_dir, f"{view_id}_lidar_error_risk.npy")
    selected_path = os.path.join(out_dir, f"{view_id}_selected_pixels.npy")
    np.save(risk_map_path, error.astype(np.float32))
    np.save(selected_path, selected)
    payload = {
        "status": "valid" if len(selected) else "low_evidence",
        "risk_source": "lidar_error",
        "selected_pixel_source": "lidar_error_map",
        "uses_lidar_selected_pixels": True,
        "view_id": view_id,
        "views": views,
        "selected_pixels_count": int(len(selected)),
        "valid_lidar_count": int(np.count_nonzero(valid)),
        "risk_map_path": risk_map_path,
        "selected_pixels_path": selected_path,
        "uses_lidar_for_labeling": True,
        "note": "LiDAR branch is supervised/reference only; invalid sparse LiDAR pixels are not treated as depth 0.",
    }
    write_json(os.path.join(out_dir, "risk_summary.json"), payload)
    return payload


def run_cuda_contribution_stage(
    risk_summary,
    contribution_summary_path,
    out_dir,
    use_cached=True,
    contribution_source="cached_summary",
    model=None,
    camera=None,
    renderer=None,
    top_k=16,
):
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "contribution_summary.json")
    if contribution_source == "live_current_model":
        selected_path = risk_summary.get("selected_pixels_path", "")
        if model is None or camera is None:
            payload = {
                "status": "failed",
                "mode": "live_current_model",
                "path": out_path,
                "reason": "model or camera is missing for live CUDA contribution",
                "live_cuda_contribution": False,
                "uses_cached_contribution": False,
            }
            write_json(out_path, payload)
            write_json(os.path.join(out_dir, "live_contribution_summary.json"), payload)
            return payload
        if not selected_path or not os.path.exists(selected_path):
            payload = {
                "status": "low_evidence",
                "mode": "live_current_model",
                "path": out_path,
                "reason": "selected pixels are missing",
                "live_cuda_contribution": False,
                "uses_cached_contribution": False,
            }
            write_json(out_path, payload)
            write_json(os.path.join(out_dir, "live_contribution_summary.json"), payload)
            return payload
        selected_pixels = np.load(selected_path)
        result = capture_contributions_cuda_live(
            model=model,
            camera=camera,
            renderer=renderer,
            selected_pixels=selected_pixels,
            top_k=top_k,
        )
        risk_score_path = risk_summary.get("selected_pixel_risk_scores_path", "")
        if risk_score_path and os.path.exists(risk_score_path):
            try:
                result["selected_risk_scores"] = np.load(risk_score_path).astype(np.float32)
            except Exception:
                result["selected_risk_scores"] = np.ones((len(selected_pixels),), dtype=np.float32)
        view_id = risk_summary.get("view_id", "live")
        summary_path, _ = write_live_contribution_outputs(result, out_dir, view_id=view_id, region_id="live")
        shutil.copyfile(summary_path, out_path)
        status = result.get("status", "failed")
        return {
            "status": status,
            "mode": "live_current_model",
            "path": out_path,
            "live_summary_path": summary_path,
            "live_cuda_contribution": bool(result.get("live_cuda_contribution", False)),
            "uses_cached_contribution": False,
            "cuda_ok_count": 1 if status == "valid" else 0,
            "low_evidence_count": 0 if status == "valid" else 1,
            "selected_pixels_count": int(len(selected_pixels)),
            "stable_id_map_available": bool(result.get("stable_id_map_available", False)),
            "unmapped_id_count": int(result.get("unmapped_id_count", 0) or 0),
            "cuda_runtime_sec": float(result.get("runtime_sec", 0.0) or 0.0),
            "error": result.get("reason", ""),
        }
    if use_cached and contribution_summary_path and os.path.exists(contribution_summary_path):
        shutil.copyfile(contribution_summary_path, out_path)
        payload = read_json(out_path)
        frames = payload.get("frames", [])
        return {
            "status": "valid",
            "mode": "cached_cuda_dump_summary",
            "path": out_path,
            "cuda_ok_count": int(sum(1 for f in frames if f.get("status") == "ok")),
            "low_evidence_count": int(sum(1 for f in frames if f.get("status") != "ok")),
        }
    payload = {
        "status": "skipped",
        "mode": "dynamic_cuda_dump_not_inlined",
        "path": out_path,
        "reason": "Selected-pixel CUDA dump is currently exposed by debug script; controller keeps this as a stage boundary.",
        "risk_summary": risk_summary,
    }
    write_json(out_path, payload)
    return payload


def select_da3_responsible_group_stage(contribution_summary_path, out_dir, max_regions=30, spatial_cell_size=96):
    ensure_dir(out_dir)
    script = Path("script/select_da3_boundary_responsible_gaussian_groups.py")
    if not script.exists() or not contribution_summary_path or not os.path.exists(contribution_summary_path):
        payload = {"status": "skipped", "reason": "missing group script or contribution summary"}
        write_json(os.path.join(out_dir, "responsible_group_summary.json"), payload)
        return payload
    cmd = [
        sys.executable,
        str(script),
        "--contribution-summary",
        contribution_summary_path,
        "--output-dir",
        out_dir,
        "--max-regions",
        str(max_regions),
        "--spatial-cell-size",
        str(spatial_cell_size),
    ]
    subprocess.run(cmd, cwd=str(script.parent.parent), check=True)
    summary_path = os.path.join(out_dir, "group_counterfactual_summary.json")
    payload = read_json(summary_path) if os.path.exists(summary_path) else {"status": "valid"}
    payload["status"] = "valid"
    write_json(os.path.join(out_dir, "responsible_group_summary.json"), payload)
    return payload


def build_softpatch_feedback_stage(source_signal_path, group_summary, out_dir, mode="group_softpatch"):
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "feedback_signal.json")
    if source_signal_path and os.path.exists(source_signal_path):
        signal = read_json(source_signal_path)
    else:
        signal = {"regions": [], "bad_contributors": [], "good_contributors": [], "low_evidence_regions": []}
    synthesized = _synthesize_softpatch_regions_from_groups(group_summary)
    if synthesized["regions"]:
        signal["regions"] = synthesized["regions"]
        signal["bad_contributors"] = synthesized["bad_contributors"]
        signal["good_contributors"] = synthesized["good_contributors"]
        signal["low_evidence_regions"] = synthesized["low_evidence_regions"]
        signal["pixel_feedback_by_view"] = synthesized["pixel_feedback_by_view"]
        signal["softpatch_activation_source"] = "responsible_group_stage"
        signal["softpatch_activation_region_count"] = len(synthesized["regions"])
        signal["softpatch_activation_bad_group_count"] = len(synthesized["bad_contributors"])
    else:
        signal.setdefault("pixel_feedback_by_view", [])
        signal["softpatch_activation_source"] = "none"
        signal["softpatch_activation_region_count"] = 0
        signal["softpatch_activation_bad_group_count"] = 0
    signal["feedback_mode"] = mode
    signal["group_responsibility_summary"] = group_summary
    signal["generated_by"] = "feedback_pipeline_stages.build_softpatch_feedback_stage"
    signal["gaussian_parameters_modified"] = False
    signal["uses_lidar_for_labeling"] = False
    write_json(out_path, signal)
    return {"status": "valid", "path": out_path, "feedback_mode": mode}


def _synthesize_softpatch_regions_from_groups(group_summary):
    out = {
        "regions": [],
        "bad_contributors": [],
        "good_contributors": [],
        "low_evidence_regions": [],
        "pixel_feedback_by_view": [],
    }
    if not isinstance(group_summary, dict):
        return out
    thresholds = group_summary.get("thresholds", {}) or {}
    group_dir = thresholds.get("output_dir", "")
    contribution_summary_path = thresholds.get("contribution_summary", "")
    group_path = os.path.join(group_dir, "da3_boundary_responsible_groups.json") if group_dir else ""
    if not group_path or not os.path.exists(group_path):
        return out
    try:
        groups = read_json(group_path)
    except Exception:
        return out
    if not isinstance(groups, list) or not groups:
        return out
    frames = _load_contribution_frames(contribution_summary_path)
    frame_lookup = {
        f"{frame.get('stem')}:region{frame.get('region_id')}": frame
        for frame in frames
        if isinstance(frame, dict)
    }
    groups_by_region = {}
    for group in groups:
        key = group.get("region_key")
        if not key:
            continue
        groups_by_region.setdefault(key, []).append(group)

    for region_key, region_groups in sorted(groups_by_region.items()):
        frame = frame_lookup.get(region_key, {})
        selected = _load_frame_selected_pixels(frame, contribution_summary_path)
        bbox = _bbox_from_pixels(selected)
        view_id = str(region_groups[0].get("view_id") or frame.get("stem") or region_key.split(":region", 1)[0])
        region_id = str(region_groups[0].get("region_id") or frame.get("region_id") or region_key.split(":region", 1)[-1])
        bad_groups = [
            group for group in region_groups
            if str(group.get("group_label", "")).startswith("bad_")
            or str(group.get("future_action_tag", "")) in {"shrink_candidate", "opacity_regularization_candidate", "split_candidate"}
        ]
        protect_groups = [
            group for group in region_groups
            if str(group.get("future_action_tag", "")) == "protect"
            or str(group.get("group_label", "")) in {"good_boundary_support_group", "rgb_protect_group"}
        ]
        low_groups = [group for group in region_groups if str(group.get("group_label", "")) == "low_evidence_group"]
        if not bad_groups or bbox is None:
            out["low_evidence_regions"].append({
                "region_key": region_key,
                "view_id": view_id,
                "region_id": region_id,
                "reason": "no_bad_group_or_missing_selected_pixel_bbox",
                "group_count": len(region_groups),
                "low_evidence_group_count": len(low_groups),
            })
            continue
        score = float(max(float(group.get("group_risk_weighted_contribution", 0.0) or 0.0) for group in bad_groups))
        region_record = {
            "region_key": region_key,
            "view_id": view_id,
            "region_id": region_id,
            "region_type": "responsible_group_softpatch",
            "bbox": bbox,
            "selected_pixel_count": int(selected.shape[0]),
            "risk_score": score,
            "evidence_status": "ok",
            "bad_group_count": len(bad_groups),
            "protect_group_count": len(protect_groups),
        }
        out["regions"].append(region_record)
        for group in bad_groups:
            out["bad_contributors"].append({
                "region_key": region_key,
                "view_id": view_id,
                "region_id": region_id,
                "group_id": group.get("group_id"),
                "group_label": group.get("group_label"),
                "future_action_tag": group.get("future_action_tag"),
                "stable_gaussian_ids": group.get("stable_gaussian_ids", []),
                "score": group.get("group_risk_weighted_contribution", 0.0),
            })
        for group in protect_groups:
            out["good_contributors"].append({
                "region_key": region_key,
                "view_id": view_id,
                "region_id": region_id,
                "group_id": group.get("group_id"),
                "group_label": group.get("group_label"),
                "future_action_tag": group.get("future_action_tag"),
                "stable_gaussian_ids": group.get("stable_gaussian_ids", []),
                "score": group.get("group_risk_weighted_contribution", 0.0),
            })
        if selected.size:
            bad_pixels = [[int(x), int(y), 3.0] for x, y in selected[:512]]
            out["pixel_feedback_by_view"].append({
                "view_id": view_id,
                "region_key": region_key,
                "bad_pixels": bad_pixels,
                "good_pixels": [],
            })
    return out


def _load_contribution_frames(path):
    if not path or not os.path.exists(path):
        return []
    try:
        payload = read_json(path)
    except Exception:
        return []
    return payload.get("frames", []) if isinstance(payload, dict) else []


def _load_frame_selected_pixels(frame, contribution_summary_path=""):
    npz_path = ((frame or {}).get("paths") or {}).get("npz", "")
    if npz_path and not os.path.exists(npz_path) and contribution_summary_path:
        local_candidate = os.path.join(os.path.dirname(contribution_summary_path), os.path.basename(npz_path))
        if os.path.exists(local_candidate):
            npz_path = local_candidate
    if not npz_path or not os.path.exists(npz_path):
        return np.zeros((0, 2), dtype=np.int64)
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "selected_pixels" not in data:
                return np.zeros((0, 2), dtype=np.int64)
            return np.asarray(data["selected_pixels"], dtype=np.int64).reshape(-1, 2)
    except Exception:
        return np.zeros((0, 2), dtype=np.int64)


def _bbox_from_pixels(pixels, pad=6):
    if pixels.size == 0:
        return None
    xs = pixels[:, 0]
    ys = pixels[:, 1]
    return [
        int(max(0, xs.min() - pad)),
        int(max(0, ys.min() - pad)),
        int(xs.max() + pad + 1),
        int(ys.max() + pad + 1),
    ]


def run_group_counterfactual_dryrun_stage(dryrun_scorer_path, contribution_summary_path, signal_path, out_dir, max_regions=1, extra_args=None):
    ensure_dir(out_dir)
    if not dryrun_scorer_path:
        payload = {"status": "skipped", "reason": "dryrun scorer path is empty"}
        write_json(os.path.join(out_dir, "group_counterfactual_summary.json"), payload)
        return payload
    script = Path(dryrun_scorer_path)
    if not script.exists():
        payload = {"status": "failed", "reason": f"missing scorer: {script}"}
        write_json(os.path.join(out_dir, "group_counterfactual_summary.json"), payload)
        return payload
    cmd = [sys.executable, str(script), "--output-dir", out_dir, "--top-regions", str(max_regions)]
    if contribution_summary_path:
        cmd += ["--contribution-summary", contribution_summary_path]
    if signal_path:
        cmd += ["--softpatch-signal", signal_path]
    if extra_args:
        cmd += [str(v) for v in extra_args]
    subprocess.run(cmd, cwd=str(script.parent.parent), check=True)
    summary_path = os.path.join(out_dir, "counterfactual_summary.json")
    return read_json(summary_path) if os.path.exists(summary_path) else {"status": "valid", "path": out_dir}


def tag_repair_candidates_stage(counterfactual_dir, out_dir):
    ensure_dir(out_dir)
    script = Path("script/tag_pruning_candidates_from_counterfactual.py")
    if not script.exists() or not counterfactual_dir:
        payload = {"status": "skipped", "reason": "missing tag script or counterfactual dir"}
        write_json(os.path.join(out_dir, "candidate_tag_summary.json"), payload)
        return payload
    cmd = [sys.executable, str(script), "--counterfactual-dir", counterfactual_dir, "--output-dir", out_dir]
    subprocess.run(cmd, cwd=str(script.parent.parent), check=True)
    summary_path = os.path.join(out_dir, "pruning_candidate_summary.json")
    payload = read_json(summary_path) if os.path.exists(summary_path) else {"status": "valid"}
    payload["path"] = out_dir
    return payload
