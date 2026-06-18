import json
import os
import time
import zlib
from collections import defaultdict

import numpy as np
import torch

from lib.config import cfg
from lib.utils.camera_utils import make_rasterizer
from lib.utils.sh_utils import eval_sh

try:
    from diff_gaussian_rasterization import _C as rasterizer_C
except Exception:
    rasterizer_C = None


STABLE_ID_STRIDE = 10_000_000


def cuda_contribution_available():
    return rasterizer_C is not None and hasattr(rasterizer_C, "rasterize_gaussians_contrib")


def stable_namespace_id(model_name):
    if model_name == "background":
        return 0
    if model_name.startswith("obj_"):
        try:
            return 1 + int(model_name.split("_", 1)[1])
        except ValueError:
            pass
    return 1_000_000 + (zlib.crc32(str(model_name).encode("utf-8")) % 8_000_000)


def build_stable_id_map(model):
    count = int(model.get_xyz.shape[0])
    stable_ids = np.full(count, -1, dtype=np.int64)
    model_names = np.full(count, "unknown", dtype="<U64")
    model_local = np.full(count, -1, dtype=np.int64)
    for model_name, span in getattr(model, "graph_gaussian_range", {}).items():
        start, end = int(span[0]), int(span[1])
        local = np.arange(max(0, end - start), dtype=np.int64)
        stable_ids[start:end] = stable_namespace_id(model_name) * STABLE_ID_STRIDE + local
        model_names[start:end] = str(model_name)
        model_local[start:end] = local
    fallback = stable_ids < 0
    if np.any(fallback):
        rows = np.flatnonzero(fallback).astype(np.int64)
        stable_ids[fallback] = 9_999_999 * STABLE_ID_STRIDE + rows
        model_local[fallback] = rows
    return stable_ids, model_names, model_local


def _to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _empty_like_if_none(tensor, device):
    if tensor is not None:
        return tensor
    return torch.Tensor([]).to(device)


def _prepare_raster_inputs(camera, model, opacity_override=None, semantics=None):
    if hasattr(model, "set_visibility"):
        model.set_visibility(list(model.model_name_id.keys()))
    if hasattr(model, "parse_camera"):
        model.parse_camera(camera)
    means3D = model.get_xyz
    opacity = model.get_opacity if opacity_override is None else opacity_override
    scales = None
    rotations = None
    cov3D_precomp = None
    if cfg.render.compute_cov3D_python:
        cov3D_precomp = model.get_covariance(cfg.render.scaling_modifier)
    else:
        scales = model.get_scaling
        rotations = model.get_rotation

    shs = None
    colors_precomp = None
    if cfg.render.convert_SHs_python:
        shs_view = model.get_features.transpose(1, 2).view(-1, 3, (model.max_sh_degree + 1) ** 2)
        dir_pp = model.get_xyz - camera.camera_center.repeat(model.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        colors_precomp = torch.clamp_min(eval_sh(model.active_sh_degree, shs_view, dir_pp_normalized) + 0.5, 0.0)
    else:
        shs = model.get_features
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=False, device=means3D.device)[:, :2]
    rasterizer = make_rasterizer(camera, model.active_sh_degree)
    kwargs = dict(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=shs,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    if semantics is not None:
        kwargs["semantics"] = semantics
    return rasterizer, kwargs


def capture_contributions_cuda_live(
    model,
    camera,
    renderer,
    selected_pixels,
    top_k=16,
    stable_id_map=None,
    render_kwargs=None,
    device="cuda",
):
    """Capture selected-pixel top-K Gaussian contributions from the current in-memory model.

    This is debug/evaluation only. It uses torch.no_grad, does not mutate
    Gaussian tensors, and does not save checkpoints.
    """
    t0 = time.perf_counter()
    pixels = np.asarray(selected_pixels, dtype=np.int64).reshape(-1, 2)
    if pixels.size == 0:
        return {
            "status": "low_evidence",
            "reason": "no selected pixels",
            "selected_pixels": pixels,
            "live_cuda_contribution": False,
        }
    if not cuda_contribution_available():
        return {
            "status": "failed",
            "reason": "rasterize_gaussians_contrib is unavailable",
            "selected_pixels": pixels,
            "live_cuda_contribution": False,
        }

    with torch.no_grad():
        rasterizer, kwargs = _prepare_raster_inputs(camera, model, semantics=None)
        stable_ids, model_names, model_local = stable_id_map or build_stable_id_map(model)
        settings = rasterizer.raster_settings
        means3D = kwargs["means3D"]
        dev = means3D.device if device == "cuda" else torch.device(device)
        selected_tensor = torch.as_tensor(pixels, device=dev, dtype=torch.int32).contiguous()
        shs = _empty_like_if_none(kwargs.get("shs"), dev)
        colors_precomp = _empty_like_if_none(kwargs.get("colors_precomp"), dev)
        semantics = torch.zeros((means3D.shape[0], 0), device=dev, dtype=torch.float32)
        scales = _empty_like_if_none(kwargs.get("scales"), dev)
        rotations = _empty_like_if_none(kwargs.get("rotations"), dev)
        cov3D_precomp = _empty_like_if_none(kwargs.get("cov3D_precomp"), dev)
        ids, alpha, transmittance, weight, depth, depth_order = rasterizer_C.rasterize_gaussians_contrib(
            settings.bg,
            means3D,
            colors_precomp,
            semantics,
            kwargs["opacities"],
            scales,
            rotations,
            settings.scale_modifier,
            cov3D_precomp,
            settings.viewmatrix,
            settings.projmatrix,
            settings.tanfovx,
            settings.tanfovy,
            settings.image_height,
            settings.image_width,
            shs,
            settings.sh_degree,
            settings.campos,
            settings.prefiltered,
            settings.debug,
            selected_tensor,
            int(top_k),
        )

    ids_np = _to_numpy(ids).astype(np.int64)
    alpha_np = _to_numpy(alpha).astype(np.float32)
    trans_np = _to_numpy(transmittance).astype(np.float32)
    weight_np = _to_numpy(weight).astype(np.float32)
    depth_np = _to_numpy(depth).astype(np.float32)
    order_np = _to_numpy(depth_order).astype(np.int32)
    stable_out = np.full_like(ids_np, -1, dtype=np.int64)
    model_name_out = np.full(ids_np.shape, "unknown", dtype="<U64")
    model_local_out = np.full_like(ids_np, -1, dtype=np.int64)
    unmapped = 0
    for index, row in np.ndenumerate(ids_np):
        if row < 0:
            continue
        if row >= len(stable_ids):
            unmapped += 1
            continue
        stable_out[index] = int(stable_ids[row])
        model_name_out[index] = str(model_names[row])
        model_local_out[index] = int(model_local[row])

    support = defaultdict(int)
    for p_idx in range(ids_np.shape[0]):
        for k in range(ids_np.shape[1]):
            if ids_np[p_idx, k] >= 0 and weight_np[p_idx, k] > 0:
                support[int(stable_out[p_idx, k])] += 1
    return {
        "status": "valid",
        "selected_pixels": pixels,
        "view_local_ids": ids_np,
        "stable_gaussian_ids": stable_out,
        "model_names": model_name_out,
        "model_local_indices": model_local_out,
        "alpha": alpha_np,
        "transmittance": trans_np,
        "contribution_weight": weight_np,
        "depth": depth_np,
        "depth_order": order_np,
        "support_count_by_stable_id": dict(support),
        "stable_id_map_available": True,
        "unmapped_id_count": int(unmapped),
        "background_object_namespace_consistent": bool(np.all(stable_out[ids_np >= 0] >= 0)) if np.any(ids_np >= 0) else True,
        "live_cuda_contribution": True,
        "uses_cached_contribution": False,
        "runtime_sec": float(time.perf_counter() - t0),
    }


def write_live_contribution_outputs(result, out_dir, view_id="live", region_id="live"):
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{view_id}_region{region_id}_live_contribution.npz")
    np.savez_compressed(
        npz_path,
        selected_pixels=result.get("selected_pixels", np.zeros((0, 2), dtype=np.int64)),
        cuda_contribution_ids=result.get("view_local_ids", np.zeros((0, 0), dtype=np.int64)),
        stable_gaussian_ids=result.get("stable_gaussian_ids", np.zeros((0, 0), dtype=np.int64)),
        model_local_indices=result.get("model_local_indices", np.zeros((0, 0), dtype=np.int64)),
        model_names=result.get("model_names", np.zeros((0, 0), dtype="<U64")),
        da3_risk=result.get("selected_risk_scores", np.zeros((0,), dtype=np.float32)),
        contribution_weights=result.get("contribution_weight", np.zeros((0, 0), dtype=np.float32)),
        cuda_alpha=result.get("alpha", np.zeros((0, 0), dtype=np.float32)),
        cuda_transmittance=result.get("transmittance", np.zeros((0, 0), dtype=np.float32)),
        cuda_depth=result.get("depth", np.zeros((0, 0), dtype=np.float32)),
        cuda_depth_order=result.get("depth_order", np.zeros((0, 0), dtype=np.int32)),
    )
    summary = {
        "status": result.get("status"),
        "stem": view_id,
        "region_id": region_id,
        "selected_pixel_count": int(len(result.get("selected_pixels", []))),
        "paths": {"npz": npz_path},
        "live_cuda_contribution": bool(result.get("live_cuda_contribution", False)),
        "uses_cached_contribution": bool(result.get("uses_cached_contribution", False)),
        "stable_id_map_available": bool(result.get("stable_id_map_available", False)),
        "unmapped_id_count": int(result.get("unmapped_id_count", 0) or 0),
        "runtime_sec": float(result.get("runtime_sec", 0.0) or 0.0),
    }
    summary_path = os.path.join(out_dir, "live_contribution_summary.json")
    payload = {"frames": [summary], "aggregate": summary}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return summary_path, npz_path
