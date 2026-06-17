#python train.py --config configs/default.yaml
#CUDA_VISIBLE_DEVICES=0 python train.py --config configs/default.yaml
# conda activate "/tsinghuaData/hanchanghao/envs/sg_env"

import json
import os
import csv
import numpy as np
import torch
import cv2
from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.general_utils import safe_state
from lib.utils.camera_utils import Camera
from lib.utils.cfg_utils import save_cfg
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.config import cfg
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lib.utils.system_utils import searchForMaxIteration
from lib.utils.guided_feedback_utils import make_guided_feedback_controller
from lib.utils.feedback_controller import make_periodic_feedback_controller
from lib.utils.gaussian_control_manager import GaussianControlManager
from lib.utils.gaussian_repair_operator import GaussianRepairOperator
from lib.utils.da3_structure_feedback_utils import make_da3_bridge, da3_structure_loss
import time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


SCALAR_TRACE_FIELDS = [
    "iteration",
    "loss",
    "train_psnr",
    "l1_loss",
    "sky_loss",
    "obj_acc_loss",
    "guided_feedback_lidar_loss",
    "guided_feedback_da3_structure_loss",
    "da3_structure_loss",
    "da3_edge_loss",
    "da3_ranking_loss",
    "da3_side_loss",
    "guided_feedback_valid_pixels",
    "guided_feedback_region_count",
    "guided_feedback_region_pixels",
    "guided_feedback_da3_skipped_low_support",
    "feedback_controller_triggered",
    "feedback_controller_active_updated",
    "feedback_controller_status_valid",
    "gaussian_control_opacity_reg_loss",
    "gaussian_control_opacity_reg_loss_count",
    "gaussian_control_opacity_mean",
    "gaussian_control_opacity_min",
    "gaussian_control_opacity_max",
    "save_iteration",
    "checkpoint_iteration",
    "opacity_reset_iteration",
    "densification_iteration",
]


def _print_cuda_diagnostics(stage):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"[CUDA][{stage}] CUDA_VISIBLE_DEVICES={visible}")
    print(
        f"[CUDA][{stage}] torch.cuda.is_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()}"
    )
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(
            f"[CUDA][{stage}] current_device={current_device} "
            f"name={torch.cuda.get_device_name(current_device)}"
        )


def _require_cuda(stage):
    _print_cuda_diagnostics(stage)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Street Gaussian training, but torch reports "
            "cuda.is_available()=False. Check the A100 server driver, CUDA "
            "PyTorch build, container GPU passthrough, and CUDA_VISIBLE_DEVICES."
        )


def compute_lidar_guided_loss(
    lidar_depth,
    mask,
    depth,
    render_pkg,
    viewpoint_cam,
    guided_feedback,
    optim_args,
    scalar_dict,
):
    if lidar_depth is None:
        return depth.sum() * 0.0
    if not (optim_args.lambda_depth_lidar > 0 or (guided_feedback.enabled and guided_feedback.use_lidar_depth)):
        return depth.sum() * 0.0

    depth_mask = torch.logical_and((lidar_depth > 0.0), mask)
    depth_valid_count = int(torch.count_nonzero(depth_mask).item())
    if depth_valid_count == 0:
        scalar_dict["lidar_depth_loss_skipped_empty"] = 1
        return depth.sum() * 0.0

    expected_depth = depth / (render_pkg["acc"] + 1e-10)
    depth_error_map = torch.abs(expected_depth - lidar_depth)
    guided_weight_map = None
    guided_stats = {}
    guided_valid_count = 0

    if guided_feedback.enabled and guided_feedback.use_lidar_depth:
        guided_weight_map, guided_stats = guided_feedback.make_region_weight_map(
            viewpoint_cam, depth_error_map.shape, depth_error_map.device
        )

    if guided_weight_map is not None:
        guided_valid = depth_mask & (guided_weight_map > 1.0)
        guided_valid_count = int(torch.count_nonzero(guided_valid).item())
        if guided_valid_count >= guided_feedback.min_valid_pixels:
            weights = guided_weight_map[depth_mask]
            depth_error = depth_error_map[depth_mask] * weights
            scalar_dict["guided_feedback_valid_pixels"] = guided_valid_count
            scalar_dict["guided_feedback_region_count"] = guided_stats.get("guided_region_count", 0)
        else:
            depth_error = depth_error_map[depth_mask]
            scalar_dict["guided_feedback_skipped_low_support"] = guided_valid_count
    else:
        depth_error = depth_error_map[depth_mask]

    k = max(1, int(0.95 * depth_error.size(0)))
    depth_error, _ = torch.topk(depth_error, k, largest=False)
    lidar_depth_loss = depth_error.mean()
    scalar_dict["lidar_depth_loss"] = lidar_depth_loss
    if guided_weight_map is not None and guided_valid_count >= guided_feedback.min_valid_pixels:
        scalar_dict["guided_feedback_lidar_loss"] = lidar_depth_loss.item()
        return (optim_args.lambda_depth_lidar + guided_feedback.geometry_weight) * lidar_depth_loss
    return optim_args.lambda_depth_lidar * lidar_depth_loss


def compute_da3_structure_guided_loss(
    depth,
    render_pkg,
    viewpoint_cam,
    guided_feedback,
    da3_bridge,
    scalar_dict,
):
    if not (guided_feedback.enabled and guided_feedback.use_da3_structure and da3_bridge is not None):
        return depth.sum() * 0.0

    guided_weight_map, guided_stats = guided_feedback.make_region_weight_map(
        viewpoint_cam, depth.shape, depth.device
    )
    if guided_weight_map is None:
        if getattr(guided_feedback, "feedback_mode", "") != "global":
            return depth.sum() * 0.0
        guided_weight_map = torch.ones_like(depth, dtype=torch.float32, device=depth.device)
        guided_stats = {
            "guided_region_count": 0,
            "guided_region_pixels": int(depth.numel()),
            "guided_view": str(getattr(viewpoint_cam, "image_name", "")),
        }

    da3_guidance = da3_bridge(viewpoint_cam)
    da3_depth = da3_guidance["relative_depth"]
    da3_loss, da3_logs = da3_structure_loss(
        depth,
        render_pkg["acc"],
        da3_depth,
        guided_weight_map,
        guided_feedback,
    )
    if da3_logs.get("da3_structure_valid_pixels", 0) < guided_feedback.min_valid_pixels:
        scalar_dict["guided_feedback_da3_skipped_low_support"] = da3_logs.get("da3_structure_valid_pixels", 0)
        return depth.sum() * 0.0

    scalar_dict["guided_feedback_da3_structure_loss"] = float(da3_loss.detach().item())
    scalar_dict.update(da3_logs)
    scalar_dict["guided_feedback_region_count"] = guided_stats.get("guided_region_count", 0)
    return guided_feedback.geometry_weight * da3_loss


def compute_guided_feedback_loss(
    lidar_depth,
    mask,
    depth,
    render_pkg,
    viewpoint_cam,
    guided_feedback,
    da3_bridge,
    optim_args,
    scalar_dict,
):
    if not guided_feedback.enabled:
        return compute_lidar_guided_loss(
            lidar_depth, mask, depth, render_pkg, viewpoint_cam, guided_feedback, optim_args, scalar_dict
        )

    mode = guided_feedback.supervision_mode
    if mode == "da3_unsupervised":
        return compute_da3_structure_guided_loss(depth, render_pkg, viewpoint_cam, guided_feedback, da3_bridge, scalar_dict)
    if mode == "lidar_supervised":
        return compute_lidar_guided_loss(
            lidar_depth, mask, depth, render_pkg, viewpoint_cam, guided_feedback, optim_args, scalar_dict
        )
    if mode == "hybrid_reference":
        return compute_lidar_guided_loss(
            lidar_depth, mask, depth, render_pkg, viewpoint_cam, guided_feedback, optim_args, scalar_dict
        ) + compute_da3_structure_guided_loss(depth, render_pkg, viewpoint_cam, guided_feedback, da3_bridge, scalar_dict)
    raise ValueError(f"unknown guided feedback supervision mode: {mode}")


def _append_gaussian_control_to_manifest(manifest_path, control_summary):
    if not manifest_path or not os.path.exists(manifest_path):
        return
    try:
        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        payload["gaussian_control_summary"] = control_summary
        payload["gaussian_parameters_modified"] = False
        payload["real_prune_enabled"] = False
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        audit_path = os.path.join(os.path.dirname(manifest_path), "audit_summary.json")
        if os.path.exists(audit_path):
            with open(audit_path, "r", encoding="utf-8-sig") as f:
                audit = json.load(f)
            audit["gaussian_control_summary"] = control_summary
            audit["gaussian_parameters_modified"] = False
            audit["real_prune_enabled"] = False
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(audit, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[GaussianControl][WARN] failed to append manifest: {exc}")


def _append_initialization_to_manifest(manifest_path, scene):
    if not manifest_path or not os.path.exists(manifest_path):
        return
    init = getattr(getattr(scene, "metadata", None), "get", lambda *_: {})("initialization_manifest", {})
    if not init:
        init_path = os.path.join(cfg.model_path, "input_ply", "initialization_manifest.json")
        if os.path.exists(init_path):
            with open(init_path, "r", encoding="utf-8-sig") as f:
                init = json.load(f)
    if not init:
        return
    try:
        for path in [manifest_path, os.path.join(os.path.dirname(manifest_path), "audit_summary.json")]:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8-sig") as f:
                payload = json.load(f)
            payload["uses_lidar_initialization"] = init.get("uses_lidar_initialization", "")
            payload["initialization_source"] = init.get("initialization_source", "")
            payload["pointcloud_source"] = init.get("pointcloud_source", "")
            payload["colmap_binary"] = init.get("colmap_binary", "")
            payload["colmap_point_count"] = init.get("colmap_point_count", "")
            payload["lidar_point_count_used_for_init"] = init.get("lidar_point_count_used_for_init", "")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[InitAudit][WARN] failed to append initialization manifest: {exc}")


def _write_scalar_trace(trace_path, iteration, scalar_dict):
    if not trace_path:
        return
    try:
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        payload = {"iteration": int(iteration)}
        for key, value in scalar_dict.items():
            try:
                if torch.is_tensor(value):
                    value = float(value.detach().cpu().item())
                elif hasattr(value, "item"):
                    value = float(value.item())
                elif isinstance(value, (int, float, bool)):
                    value = value
                else:
                    value = float(value)
            except Exception:
                value = str(value)
            payload[key] = value
        jsonl_path = trace_path
        if jsonl_path.endswith(".csv"):
            jsonl_path = jsonl_path[:-4] + ".jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        csv_path = trace_path if trace_path.endswith(".csv") else trace_path + ".csv"
        csv_row = {key: payload.get(key, "") for key in SCALAR_TRACE_FIELDS}
        extra = {
            key: value
            for key, value in payload.items()
            if key not in SCALAR_TRACE_FIELDS
        }
        csv_row["extra_json"] = json.dumps(extra, ensure_ascii=False, sort_keys=True)
        fieldnames = SCALAR_TRACE_FIELDS + ["extra_json"]
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(csv_row)
    except Exception as exc:
        print(f"[Trace][WARN] failed to write scalar trace: {exc}")


def _edge_mean(depth_tensor):
    d = depth_tensor.detach()
    dx = torch.abs(d[..., :, 1:] - d[..., :, :-1]).mean()
    dy = torch.abs(d[..., 1:, :] - d[..., :-1, :]).mean()
    return float((dx + dy).item() * 0.5)


def _sky_scale_for_camera(optim_args, viewpoint_cam):
    scales = list(getattr(optim_args, "lambda_sky_scale", []) or [])
    if not scales:
        return 1.0
    cam_id = int(viewpoint_cam.meta.get("cam", 0))
    if cam_id < 0 or cam_id >= len(scales):
        raise ValueError(
            "optim.lambda_sky_scale does not cover the current camera id: "
            f"cam={cam_id}, len(lambda_sky_scale)={len(scales)}. "
            "For five Waymo cameras [0,1,2,3,4], set "
            "optim.lambda_sky_scale: [1, 1, 0, 0, 0]."
        )
    return float(scales[cam_id])


def _training_include_list(gaussians):
    return list(set(gaussians.model_name_id.keys()) - set(["sky"]))


def _restore_training_render_state(gaussians, viewpoint_cam):
    gaussians.set_visibility(include_list=_training_include_list(gaussians))
    gaussians.parse_camera(viewpoint_cam)


def _assert_training_render_shapes(iteration, viewpoint_cam, gaussians, radii, visibility_filter, viewspace_point_tensor=None):
    expected = 0
    range_summary = {}
    for model_name, (start, end) in gaussians.graph_gaussian_range.items():
        model = getattr(gaussians, model_name)
        model_count = int(model.max_radii2D.shape[0])
        expected = max(expected, int(end))
        range_summary[model_name] = {
            "range": [int(start), int(end)],
            "max_radii2D_shape": model_count,
        }
        if int(end - start) != model_count:
            raise ValueError(
                "[Densification] graph range/model shape mismatch before set_max_radii2D: "
                f"iter={iteration}, cam={viewpoint_cam.meta.get('cam', None)}, view={getattr(viewpoint_cam, 'image_name', '')}, "
                f"model={model_name}, range=({start},{end}), max_radii2D_shape={model_count}. "
                "A non-training render may have changed StreetGaussianModel graph state."
            )
    radii_len = int(radii.shape[0])
    visibility_len = int(visibility_filter.shape[0])
    viewspace_len = int(viewspace_point_tensor.shape[0]) if viewspace_point_tensor is not None else radii_len
    if radii_len != expected or visibility_len != expected or viewspace_len != expected:
        raise ValueError(
            "[Densification] training render tensors do not match current StreetGaussianModel graph state: "
            f"iter={iteration}, cam={viewpoint_cam.meta.get('cam', None)}, view={getattr(viewpoint_cam, 'image_name', '')}, "
            f"radii_shape={tuple(radii.shape)}, visibility_filter_shape={tuple(visibility_filter.shape)}, "
            f"viewspace_shape={tuple(viewspace_point_tensor.shape) if viewspace_point_tensor is not None else None}, "
            f"expected_graph_gaussians={expected}, graph_ranges={range_summary}. "
            "Ensure periodic_eval/log/validation renders restore the training camera state before densification."
        )


def _tensor_rgb_u8(tensor):
    x = torch.clamp(tensor.detach(), 0.0, 1.0).cpu().numpy()
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=-1)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return (x * 255.0).astype(np.uint8)


def _put_title(image, title):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, str(title), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _resize_like_panel(image, shape_hw):
    h, w = shape_hw
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def _depth_stats(depth_tensor):
    x = depth_tensor.detach().cpu().numpy().astype(np.float32)
    finite = np.isfinite(x)
    positive = finite & (x > 0)
    stats = {
        "finite_depth_count": int(np.count_nonzero(finite)),
        "positive_depth_count": int(np.count_nonzero(positive)),
        "depth_min": None,
        "depth_max": None,
    }
    vals = x[positive]
    if vals.size:
        stats["depth_min"] = float(np.min(vals))
        stats["depth_max"] = float(np.max(vals))
    return stats


def _acc_stats(acc_tensor):
    x = acc_tensor.detach().cpu().numpy().astype(np.float32)
    finite = np.isfinite(x)
    vals = x[finite]
    if vals.size == 0:
        return {"acc_min": None, "acc_max": None, "acc_mean": None, "acc_finite_count": 0}
    return {
        "acc_min": float(np.min(vals)),
        "acc_max": float(np.max(vals)),
        "acc_mean": float(np.mean(vals)),
        "acc_finite_count": int(vals.size),
    }


def _append_jsonl(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _select_eval_cameras(scene, train_args):
    cameras = scene.getTrainCameras()
    explicit = list(getattr(train_args, "periodic_eval_view_ids", []) or [])
    max_views = int(getattr(train_args, "periodic_eval_max_views", 15))
    selected = []
    if explicit:
        wanted = {str(v) for v in explicit}
        selected = [cam for cam in cameras if str(getattr(cam, "image_name", "")) in wanted]
    if not selected:
        views_per_camera = int(getattr(train_args, "periodic_eval_views_per_camera", 3) or 1)
        per_cam_counts = {}
        for cam in cameras:
            cam_id = int(getattr(cam, "meta", {}).get("cam", len(per_cam_counts)))
            count = per_cam_counts.get(cam_id, 0)
            if count >= views_per_camera:
                continue
            selected.append(cam)
            per_cam_counts[cam_id] = count + 1
            if len(selected) >= max_views:
                break
    if not selected and cameras:
        selected = cameras[: min(len(cameras), max_views)]
    return selected


def _full_snapshot_iterations(train_args):
    return {int(v) for v in list(getattr(train_args, "full_snapshot_iterations", []) or [])}


def _periodic_eval_camera_records(iteration, scene, train_args, sampled_cameras):
    snapshot_iterations = _full_snapshot_iterations(train_args)
    if int(iteration) not in snapshot_iterations:
        return [("train", cam) for cam in sampled_cameras], False
    splits = {str(v).lower() for v in list(getattr(train_args, "full_snapshot_splits", ["train", "test"]) or [])}
    records = []
    if "train" in splits:
        records.extend(("train", cam) for cam in _sorted_cameras_for_eval(scene.getTrainCameras()))
    if "test" in splits:
        records.extend(("test", cam) for cam in _sorted_cameras_for_eval(scene.getTestCameras()))
    return records, True


def _safe_depth_color(depth_tensor):
    try:
        depth_np = np.squeeze(depth_tensor.detach().cpu().numpy())
        depth_colored, _ = visualize_depth_numpy(depth_np)
        return depth_colored[..., [2, 1, 0]]
    except Exception as exc:
        print(f"[Visual][WARN] depth visualization failed: {exc}")
        shape = depth_tensor.shape[-2:]
        return np.zeros((int(shape[0]), int(shape[1]), 3), dtype=np.uint8)


def _edge_vis_from_depth(depth_tensor):
    x = depth_tensor.detach().cpu().numpy().squeeze().astype(np.float32)
    finite = np.isfinite(x)
    valid = finite & (x > 0)
    if not np.any(valid):
        return np.zeros((*x.shape, 3), dtype=np.uint8)
    filled = x.copy()
    filled[~valid] = float(np.median(filled[valid]))
    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad[~valid] = 0
    if np.max(grad) > 0:
        grad = grad / np.max(grad)
    edge = (grad * 255).astype(np.uint8)
    return cv2.applyColorMap(edge, cv2.COLORMAP_TURBO)


def _mask_vis(mask_tensor, shape_hw):
    if mask_tensor is None:
        return np.zeros((shape_hw[0], shape_hw[1], 3), dtype=np.uint8)
    x = mask_tensor.detach().cpu().numpy().squeeze()
    if x.shape != tuple(shape_hw):
        x = cv2.resize(x.astype(np.float32), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((shape_hw[0], shape_hw[1], 3), dtype=np.uint8)
    out[x > 0] = (255, 64, 64)
    return out


def _write_panel(path, tiles):
    rows = []
    h, w = tiles[0][0][1].shape[:2]
    for row in tiles:
        row_imgs = [_resize_like_panel(_put_title(img, title), (h, w)) for title, img in row]
        rows.append(np.concatenate(row_imgs, axis=1))
    panel = np.concatenate(rows, axis=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, panel[..., ::-1])


def _save_rgb_image(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image[..., ::-1])
    return path


def _write_training_log_image(iteration, viewpoint_cam, gt_image, image, depth, acc, scene, renderer, train_args):
    interval = int(getattr(train_args, "log_image_interval", 0) or 0)
    if not bool(getattr(train_args, "save_visuals", False)) or interval <= 0 or iteration % interval != 0:
        return
    try:
        depth_colored = _safe_depth_color(depth)
        depth_colored = torch.from_numpy(depth_colored / 255.0).permute(2, 0, 1).float().cuda()
        row0 = torch.cat([gt_image, image, depth_colored], dim=2)
        acc_vis = acc.repeat(3, 1, 1)
        with torch.no_grad():
            render_pkg_obj = renderer.render_object(viewpoint_cam, scene.gaussians)
            image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj["acc"]
        acc_obj = acc_obj.repeat(3, 1, 1)
        row1 = torch.cat([acc_vis, image_obj, acc_obj], dim=2)
        image_to_show = torch.clamp(torch.cat([row0, row1], dim=1), 0.0, 1.0)
        out_dir = os.path.join(cfg.model_path, "log_images")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{iteration}.jpg")
        save_img_torch(image_to_show, out_path)
        _append_jsonl(os.path.join(out_dir, "manifest.jsonl"), {
            "iteration": int(iteration),
            "cam_id": int(viewpoint_cam.meta.get("cam", -1)),
            "image_name": str(getattr(viewpoint_cam, "image_name", "")),
            "path": out_path,
            **_depth_stats(depth),
            **_acc_stats(acc),
        })
    except Exception as exc:
        print(f"[Visual][WARN] log image failed at iter {iteration}: {exc}")


def _write_periodic_eval(iteration, eval_cameras, scene, renderer, da3_bridge, guided_feedback, train_args, previous_stats):
    if not bool(getattr(train_args, "periodic_eval_enabled", True)):
        return
    interval = int(getattr(train_args, "eval_panel_interval", getattr(train_args, "periodic_eval_interval", 1000)) or 0)
    if interval <= 0 or iteration % interval != 0:
        return
    iter_dir = os.path.join(cfg.model_path, "periodic_eval", f"iter_{iteration:06d}")
    panel_dir = os.path.join(iter_dir, "panels")
    asset_dir = os.path.join(iter_dir, "assets")
    os.makedirs(panel_dir, exist_ok=True)
    os.makedirs(asset_dir, exist_ok=True)
    camera_records, is_full_snapshot = _periodic_eval_camera_records(iteration, scene, train_args, eval_cameras)
    feedback_enabled = bool(getattr(cfg.train.feedback_controller, "enabled", False))
    manifest = {
        "iteration": int(iteration),
        "experiment": str(getattr(cfg, "exp_name", "")),
        "model_path": str(getattr(cfg, "model_path", "")),
        "snapshot_type": "full_train_test" if is_full_snapshot else "sampled_periodic",
        "is_forced_full_snapshot": bool(is_full_snapshot),
        "requested_full_snapshot_splits": list(getattr(train_args, "full_snapshot_splits", []) or []),
        "view_count": len(camera_records),
        "uses_lidar_initialization": bool(getattr(cfg.data, "allow_lidar_initialization", False)) and not bool(getattr(cfg.data, "require_no_lidar_initialization", False)),
        "requires_no_lidar_initialization": bool(getattr(cfg.data, "require_no_lidar_initialization", False)),
        "uses_lidar_supervision": bool(getattr(cfg.train.guided_feedback, "use_lidar_depth", False)) or float(getattr(cfg.optim, "lambda_depth_lidar", 0.0)) > 0.0,
        "feedback_controller_enabled": feedback_enabled,
        "active_feedback_signal_path": str(getattr(guided_feedback, "signal_path", "") or "") if guided_feedback.enabled else "",
        "feedback_note": "risk/responsibility maps are active only for feedback groups" if feedback_enabled else "no-feedback baseline/control; risk and responsibility overlays are intentionally not claimed",
        "views": [],
    }
    for split_name, cam in camera_records:
        try:
            gt = cam.original_image.cuda(non_blocking=True) if not cam.original_image.is_cuda else cam.original_image
            mask = cam.guidance["mask"].cuda(non_blocking=True).bool() if "mask" in cam.guidance else torch.ones_like(gt[0:1]).bool()
            pkg = renderer.render(cam, scene.gaussians)
            rgb, depth, acc = pkg["rgb"], pkg["depth"], pkg["acc"]
            rgb_error = torch.clamp(torch.abs(rgb - gt) * 4.0, 0.0, 1.0)
            l1_value = float(l1_loss(rgb, gt, mask).detach().item())
            psnr_value = float(psnr(rgb, gt, mask).mean().detach().item())
            depth_info = _depth_stats(depth)
            acc_info = _acc_stats(acc)
            h, w = int(rgb.shape[-2]), int(rgb.shape[-1])
            rendered_depth = _safe_depth_color(depth)
            da3_or_edge = _edge_vis_from_depth(depth)
            if da3_bridge is not None:
                try:
                    da3_depth = da3_bridge(cam)["relative_depth"]
                    da3_or_edge = _safe_depth_color(da3_depth)
                except Exception as exc:
                    print(f"[PeriodicEval][WARN] DA3 depth panel failed for {cam.image_name}: {exc}")
            lidar_overlay = _mask_vis(cam.guidance.get("lidar_depth", None), (h, w)) if hasattr(cam, "guidance") else np.zeros((h, w, 3), dtype=np.uint8)
            risk_vis = _edge_vis_from_depth(depth)
            selected_vis = np.zeros((h, w, 3), dtype=np.uint8)
            softpatch_vis = np.zeros((h, w, 3), dtype=np.uint8)
            if guided_feedback.enabled:
                weight_map, _ = guided_feedback.make_region_weight_map(cam, depth.shape, depth.device)
                if weight_map is not None:
                    selected_vis = _mask_vis(weight_map > 1.0, (h, w))
                    softpatch_u8 = np.clip(weight_map.detach().cpu().numpy().squeeze(), 0, 4) * 60
                    softpatch_vis = cv2.applyColorMap(softpatch_u8.astype(np.uint8), cv2.COLORMAP_TURBO)
            acc_vis = _tensor_rgb_u8(acc.repeat(3, 1, 1))
            contribution_vis = np.zeros((h, w, 3), dtype=np.uint8)
            group_vis = selected_vis.copy()
            rows = [
                [("GT RGB", _tensor_rgb_u8(gt)), ("Rendered RGB", _tensor_rgb_u8(rgb)), ("RGB Error x4", _tensor_rgb_u8(rgb_error))],
                [("Rendered Depth", rendered_depth), ("DA3 Depth / Edge", da3_or_edge), ("LiDAR Sparse Overlay", lidar_overlay)],
                [("DA3 Boundary Risk", risk_vis), ("Selected Risk / Regions", selected_vis), ("Accumulation / Alpha", acc_vis)],
            ]
            if feedback_enabled:
                rows.append([
                    ("Contribution Top-K Overlay", contribution_vis),
                    ("Responsible Group Overlay", group_vis),
                    ("Active Softpatch Mask", softpatch_vis),
                ])
            cam_id = int(cam.meta.get("cam", -1))
            image_name = str(getattr(cam, "image_name", "view")).replace(os.sep, "_")
            stem = f"iter_{iteration:06d}_{split_name}_cam{cam_id}_{image_name}"
            paths = {
                "gt_rgb_path": _save_rgb_image(os.path.join(asset_dir, f"{stem}_gt_rgb.jpg"), _tensor_rgb_u8(gt)),
                "rendered_rgb_path": _save_rgb_image(os.path.join(asset_dir, f"{stem}_rendered_rgb.jpg"), _tensor_rgb_u8(rgb)),
                "rgb_error_path": _save_rgb_image(os.path.join(asset_dir, f"{stem}_rgb_error_x4.jpg"), _tensor_rgb_u8(rgb_error)),
                "depth_path": _save_rgb_image(os.path.join(asset_dir, f"{stem}_rendered_depth.jpg"), rendered_depth),
                "da3_depth_or_edge_path": _save_rgb_image(os.path.join(asset_dir, f"{stem}_da3_depth_or_edge.jpg"), da3_or_edge),
                "lidar_sparse_overlay_path": _save_rgb_image(os.path.join(asset_dir, f"{stem}_lidar_sparse_overlay.jpg"), lidar_overlay),
                "risk_path": _save_rgb_image(os.path.join(asset_dir, f"{stem}_da3_boundary_risk.jpg"), risk_vis),
                "selected_risk_path": _save_rgb_image(os.path.join(asset_dir, f"{stem}_selected_risk_regions.jpg"), selected_vis),
                "accumulation_path": _save_rgb_image(os.path.join(asset_dir, f"{stem}_accumulation.jpg"), acc_vis),
                "softpatch_mask_path": _save_rgb_image(os.path.join(asset_dir, f"{stem}_active_softpatch_mask.jpg"), softpatch_vis),
            }
            if feedback_enabled:
                paths["risk_region_path"] = paths["selected_risk_path"]
                paths["responsible_gaussian_path"] = paths["softpatch_mask_path"]
            panel_path = os.path.join(panel_dir, f"{stem}_comparison_panel.jpg")
            _write_panel(panel_path, rows)
            view_key = f"{split_name}:{image_name}"
            prev = previous_stats.get(view_key, {})
            prev_psnr = prev.get("psnr")
            warnings = []
            if prev_psnr is not None and psnr_value < prev_psnr - float(getattr(train_args, "psnr_drop_warn_threshold", 5.0)):
                warnings.append(f"psnr_drop_from_{prev_psnr:.3f}_to_{psnr_value:.3f}")
            if depth_info["positive_depth_count"] == 0:
                warnings.append("empty_positive_depth")
            if acc_info["acc_mean"] is None or acc_info["acc_mean"] <= 1e-6 or acc_info["acc_mean"] >= 0.999:
                warnings.append("acc_saturated_or_empty")
            if warnings:
                print(f"[PeriodicEval][WARN] iter={iteration} view={image_name}: {', '.join(warnings)}")
            previous_stats[view_key] = {"psnr": psnr_value}
            manifest["views"].append({
                "split": split_name,
                "cam_id": cam_id,
                "image_name": image_name,
                "panel_path": panel_path,
                **paths,
                "psnr": psnr_value,
                "l1": l1_value,
                "warnings": warnings,
                **depth_info,
                **acc_info,
            })
        except Exception as exc:
            print(f"[PeriodicEval][WARN] failed at iter={iteration} view={getattr(cam, 'image_name', '')}: {exc}")
            manifest["views"].append({
                "split": split_name,
                "cam_id": int(getattr(cam, "meta", {}).get("cam", -1)),
                "image_name": str(getattr(cam, "image_name", "")),
                "status": "failed",
                "error": str(exc),
            })
    with open(os.path.join(iter_dir, "panel_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if is_full_snapshot:
        with open(os.path.join(iter_dir, "snapshot_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)


def _selected_bbox_from_trigger(trigger_dir, width, height):
    risk_dir = os.path.join(trigger_dir, "risk_stage")
    selected_files = [p for p in os.listdir(risk_dir)] if os.path.isdir(risk_dir) else []
    selected_files = [p for p in selected_files if p.endswith("_selected_pixels.npy")]
    if not selected_files:
        return [0, 0, int(width), int(height)], 0
    pts = np.load(os.path.join(risk_dir, selected_files[0]))
    if pts.size == 0:
        return [0, 0, int(width), int(height)], 0
    xs = pts[:, 0]
    ys = pts[:, 1]
    pad = 16
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(int(width), int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(int(height), int(ys.max()) + pad + 1)
    return [x0, y0, x1, y1], int(len(pts))


def _maybe_write_opacity_decay_light_eval(
    trigger_dir,
    viewpoint_cam,
    gaussians,
    renderer,
    gt_image,
    image_before,
    depth_before,
    acc_before,
    image_after,
    depth_after,
    acc_after,
    da3_bridge,
    guided_feedback,
):
    try:
        out_dir = os.path.join(trigger_dir, "opacity_decay_apply")
        os.makedirs(out_dir, exist_ok=True)
        h, w = int(image_before.shape[-2]), int(image_before.shape[-1])
        bbox, selected_count = _selected_bbox_from_trigger(trigger_dir, w, h)
        x0, y0, x1, y1 = bbox
        rgb_before = torch.mean(torch.abs(image_before[:, y0:y1, x0:x1] - gt_image[:, y0:y1, x0:x1]))
        rgb_after = torch.mean(torch.abs(image_after[:, y0:y1, x0:x1] - gt_image[:, y0:y1, x0:x1]))
        full_rgb_before = torch.mean(torch.abs(image_before - gt_image))
        full_rgb_after = torch.mean(torch.abs(image_after - gt_image))
        depth_edge_before = _edge_mean(depth_before / (acc_before + 1e-10))
        depth_edge_after = _edge_mean(depth_after / (acc_after + 1e-10))
        da3_before = None
        da3_after = None
        if da3_bridge is not None and guided_feedback.enabled and guided_feedback.use_da3_structure:
            weight_map, _ = guided_feedback.make_region_weight_map(viewpoint_cam, depth_before.shape, depth_before.device)
            if weight_map is not None:
                da3_guidance = da3_bridge(viewpoint_cam)
                da3_depth = da3_guidance["relative_depth"]
                loss_before, logs_before = da3_structure_loss(depth_before, acc_before, da3_depth, weight_map, guided_feedback)
                loss_after, logs_after = da3_structure_loss(depth_after, acc_after, da3_depth, weight_map, guided_feedback)
                da3_before = float(loss_before.detach().item())
                da3_after = float(loss_after.detach().item())
            else:
                logs_before = {}
                logs_after = {}
        else:
            logs_before = {}
            logs_after = {}
        panel = torch.cat([
            torch.clamp(gt_image, 0, 1),
            torch.clamp(image_before, 0, 1),
            torch.clamp(image_after, 0, 1),
            torch.clamp(torch.abs(image_after - image_before) * 10.0, 0, 1),
        ], dim=2)
        save_img_torch(panel, os.path.join(out_dir, "before_after_selected_panel.jpg"))
        payload = {
            "view_id": getattr(viewpoint_cam, "image_name", ""),
            "bbox": bbox,
            "selected_pixel_count": selected_count,
            "rgb_patch_mae_before": float(rgb_before.detach().item()),
            "rgb_patch_mae_after": float(rgb_after.detach().item()),
            "rgb_patch_mae_delta": float((rgb_after - rgb_before).detach().item()),
            "rgb_full_mae_before": float(full_rgb_before.detach().item()),
            "rgb_full_mae_after": float(full_rgb_after.detach().item()),
            "rgb_full_mae_delta": float((full_rgb_after - full_rgb_before).detach().item()),
            "rendered_depth_edge_mean_before": depth_edge_before,
            "rendered_depth_edge_mean_after": depth_edge_after,
            "rendered_depth_edge_mean_delta": depth_edge_after - depth_edge_before,
            "da3_structure_loss_before": da3_before,
            "da3_structure_loss_after": da3_after,
            "da3_structure_loss_delta": None if da3_before is None or da3_after is None else da3_after - da3_before,
            "da3_logs_before": logs_before,
            "da3_logs_after": logs_after,
            "panel_path": os.path.join(out_dir, "before_after_selected_panel.jpg"),
            "lidar_used_for_labeling": False,
            "lidar_metrics_note": "LiDAR is not used for DA3-unsupervised opacity decay labeling; run external evaluation for sparse LiDAR metrics.",
        }
        with open(os.path.join(out_dir, "opacity_decay_light_eval.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return payload
    except Exception as exc:
        print(f"[GaussianControl][WARN] opacity decay light eval failed: {exc}")
        return {"status": "failed", "error": str(exc)}


def training():
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data

    start_iter = 0
    tb_writer = prepare_output_and_logger()
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)

    gaussians.training_setup()
    _print_cuda_diagnostics("after_model_setup")
    try:
        xyz = gaussians.get_xyz
        print(f"[CUDA][after_model_setup] gaussians.get_xyz device={xyz.device} shape={tuple(xyz.shape)}")
    except Exception as exc:
        print(f"[CUDA][after_model_setup][WARN] failed to inspect gaussian tensor device: {exc}")
    if cfg.resume:
        try:
            if cfg.loaded_iter == -1:
                loaded_iter = searchForMaxIteration(cfg.trained_model_dir)
            else:
                loaded_iter = cfg.loaded_iter
            ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{loaded_iter}.pth')
            state_dict = torch.load(ckpt_path)
            start_iter = state_dict['iter']
            print(f'Loading model from {ckpt_path}')
            gaussians.load_state_dict(state_dict)
        except (FileNotFoundError, AssertionError, ValueError) as e:
            raise RuntimeError(
                f"resume=True but no readable checkpoint was found in {cfg.trained_model_dir}. "
                "Set resume=false for from-scratch training."
            ) from e
    else:
        print("[Checkpoint] resume=false; starting from scratch without checkpoint search.")

    print(f'Starting from {start_iter}')
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    gaussians_renderer = StreetGaussianRenderer()
    guided_feedback = make_guided_feedback_controller(cfg.train.guided_feedback)
    if guided_feedback.enabled:
        guided_feedback.validate_supervision(optim_args)
    else:
        print("[GuidedFeedback] disabled; skipping signal loading and supervision checks.")
    if guided_feedback.enabled:
        print(f"[GuidedFeedback] {guided_feedback.summary}")
    feedback_controller = make_periodic_feedback_controller(cfg.train.feedback_controller, model_path=cfg.model_path)
    feedback_controller.validate_supervision(optim_args, guided_feedback=guided_feedback)
    if feedback_controller.enabled:
        print(f"[FeedbackController] {feedback_controller.summary}")
    gaussian_control = GaussianControlManager(cfg.train.gaussian_control)
    repair_operator = GaussianRepairOperator(cfg.train.gaussian_control)
    if gaussian_control.enabled:
        print(f"[GaussianControl] {gaussian_control.get_control_summary()}")
    else:
        print("[GaussianControl] disabled; skipping group evidence checks and control logic.")
    da3_bridge = None
    if guided_feedback.enabled and guided_feedback.use_da3_structure:
        da3_bridge = make_da3_bridge(cfg.geovit)
        print("[GuidedFeedback] DA3 boundary-structure feedback enabled")
    periodic_eval_cameras = _select_eval_cameras(scene, training_args)
    print("[PeriodicEval] fixed views: " + ", ".join(
        f"cam{cam.meta.get('cam', '?')}:{getattr(cam, 'image_name', '')}" for cam in periodic_eval_cameras
    ))
    periodic_eval_previous_stats = {}

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_dict = {}
    progress_bar = tqdm(range(start_iter, training_args.iterations))
    start_iter += 1

    viewpoint_stack = None
    for iteration in range(start_iter, training_args.iterations + 1):
    
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        gt_image = viewpoint_cam.original_image
        mask = viewpoint_cam.guidance['mask'] if 'mask' in viewpoint_cam.guidance else torch.ones_like(gt_image[0:1]).bool()
        gt_image = gt_image.cuda(non_blocking=True) if not gt_image.is_cuda else gt_image
        mask = mask.cuda(non_blocking=True) if not mask.is_cuda else mask
        lidar_depth = None
        sky_mask = None
        obj_bound = None
        if 'lidar_depth' in viewpoint_cam.guidance:
            lidar_depth = viewpoint_cam.guidance['lidar_depth']
            lidar_depth = lidar_depth.cuda(non_blocking=True) if not lidar_depth.is_cuda else lidar_depth
        if 'sky_mask' in viewpoint_cam.guidance:
            sky_mask = viewpoint_cam.guidance['sky_mask']
            sky_mask = sky_mask.cuda(non_blocking=True) if not sky_mask.is_cuda else sky_mask
        if 'obj_bound' in viewpoint_cam.guidance:
            obj_bound = viewpoint_cam.guidance['obj_bound']
            obj_bound = obj_bound.cuda(non_blocking=True) if not obj_bound.is_cuda else obj_bound
        
            
        render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)
        image, acc, viewspace_point_tensor, visibility_filter, radii = render_pkg["rgb"], render_pkg['acc'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg['depth'] # [1, H, W]

        scalar_dict = dict()
        if feedback_controller.should_trigger(iteration):
            with torch.no_grad():
                trigger_result = feedback_controller.run_trigger(
                    iteration,
                    model=gaussians,
                    scene=scene,
                    renderer=gaussians_renderer,
                    current_outputs={
                        "view_id": viewpoint_cam.image_name,
                        "camera": viewpoint_cam,
                        "rgb": image.detach(),
                        "depth": depth.detach(),
                        "acc": acc.detach(),
                    },
                    selected_views=[viewpoint_cam.image_name],
                    extra={
                        "current_view": viewpoint_cam.image_name,
                        "frame": viewpoint_cam.meta.get("frame", None),
                        "cam": viewpoint_cam.meta.get("cam", None),
                    },
                )
            if trigger_result.feedback_mask and trigger_result.feedback_mask.get("signal_path"):
                guided_feedback.update_signal_path(trigger_result.feedback_mask["signal_path"])
                scalar_dict["feedback_controller_active_updated"] = 1
            scalar_dict["feedback_controller_triggered"] = 1
            scalar_dict["feedback_controller_status_valid"] = 1 if trigger_result.status == "valid" else 0
            _append_initialization_to_manifest(trigger_result.manifest_path, scene)
            if gaussian_control.enabled and trigger_result.manifest_path:
                trigger_dir = os.path.dirname(trigger_result.manifest_path)
                group_path = os.path.join(
                    trigger_dir,
                    "responsible_group_stage",
                    "da3_boundary_responsible_groups.json",
                )
                control_summary = gaussian_control.update_from_group_evidence_file(group_path)
                control_dir = os.path.join(trigger_dir, "gaussian_control")
                if gaussian_control.control_mode == "opacity_decay_apply":
                    decay_summary = gaussian_control.apply_opacity_decay(
                        gaussians,
                        os.path.join(trigger_dir, "opacity_decay_apply"),
                        iteration=iteration,
                    )
                    if decay_summary.get("modified_count", 0) > 0:
                        with torch.no_grad():
                            render_after_decay = gaussians_renderer.render(viewpoint_cam, gaussians)
                            light_eval = _maybe_write_opacity_decay_light_eval(
                                trigger_dir,
                                viewpoint_cam,
                                gaussians,
                                gaussians_renderer,
                                gt_image,
                                image.detach(),
                                depth.detach(),
                                acc.detach(),
                                render_after_decay["rgb"].detach(),
                                render_after_decay["depth"].detach(),
                                render_after_decay["acc"].detach(),
                                da3_bridge,
                                guided_feedback,
                            )
                        decay_summary["light_eval"] = light_eval
                        with open(
                            os.path.join(trigger_dir, "opacity_decay_apply", "opacity_decay_apply_manifest.json"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(decay_summary, f, indent=2, ensure_ascii=False)
                    control_summary = gaussian_control.get_control_summary()
                    control_summary["opacity_decay_apply"] = decay_summary
                repair_manifest = None
                if gaussian_control.control_mode == "repair_dryrun":
                    repair_manifest = repair_operator.run_dryrun(
                        gaussians,
                        gaussian_control.group_evidence,
                        gaussian_control.protected_ids,
                        os.path.join(trigger_dir, "gaussian_repair_operator"),
                        iteration=iteration,
                    )
                    control_summary["repair_operator"] = repair_manifest
                gaussian_control.write_manifest(control_dir)
                _append_gaussian_control_to_manifest(trigger_result.manifest_path, control_summary)
                scalar_dict["gaussian_control_group_evidence_count"] = control_summary["group_evidence_count"]
                scalar_dict["gaussian_control_protected_count"] = control_summary["protected_gaussian_count"]
                scalar_dict["gaussian_control_opacity_reg_count"] = control_summary["opacity_regularized_gaussian_count"]
                scalar_dict["gaussian_control_opacity_decay_candidate_count"] = control_summary["opacity_decay_candidate_count"]
                scalar_dict["gaussian_control_opacity_decay_modified_count"] = control_summary["last_opacity_decay_modified_count"]
                scalar_dict["gaussian_control_dryrun_candidate_count"] = control_summary["dryrun_repair_candidate_count"]
                if repair_manifest is not None:
                    scalar_dict["gaussian_repair_dryrun_candidate_count"] = repair_manifest.get("candidate_count", 0)
        
        # rgb loss
        Ll1 = l1_loss(image, gt_image, mask)
        scalar_dict['l1_loss'] = Ll1.item()
        loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask))
    
        # sky loss
        if optim_args.lambda_sky > 0 and gaussians.include_sky and sky_mask is not None:
            acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
            sky_loss *= _sky_scale_for_camera(optim_args, viewpoint_cam)
            scalar_dict['sky_loss'] = sky_loss.item()
            loss += optim_args.lambda_sky * sky_loss
        
        if optim_args.lambda_reg > 0 and gaussians.include_obj and iteration >= optim_args.densify_until_iter:
            render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, parse_camera_again=False)
            image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = torch.clamp(acc_obj, min=1e-6, max=1.-1e-6)
            obj_acc_loss = torch.where(obj_bound, 
                -(acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj)), 
                -torch.log(1. - acc_obj)).mean()
            scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            loss += optim_args.lambda_reg * obj_acc_loss

        loss += compute_guided_feedback_loss(
            lidar_depth,
            mask,
            depth,
            render_pkg,
            viewpoint_cam,
            guided_feedback,
            da3_bridge,
            optim_args,
            scalar_dict,
        )

        control_loss = gaussian_control.compute_opacity_regularization_loss(gaussians)
        if control_loss.count > 0:
            loss += control_loss.loss
        scalar_dict["gaussian_control_opacity_reg_loss"] = control_loss.scalar
        scalar_dict["gaussian_control_opacity_reg_loss_count"] = control_loss.count
        scalar_dict["gaussian_control_opacity_mean"] = control_loss.opacity_mean
        scalar_dict["gaussian_control_opacity_min"] = control_loss.opacity_min
        scalar_dict["gaussian_control_opacity_max"] = control_loss.opacity_max
                    
        # color correction loss
        if optim_args.lambda_color_correction > 0 and gaussians.use_color_correction:
            color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)
            scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()
            loss += optim_args.lambda_color_correction * color_correction_reg_loss
                    
        scalar_dict['loss'] = loss.item()
        with torch.no_grad():
            scalar_dict["train_psnr"] = float(psnr(image, gt_image, mask).mean().detach().item())
        _write_scalar_trace(getattr(training_args, "scalar_trace_path", ""), iteration, scalar_dict)
        
        loss.backward()
        
        iter_end.record()
                
        _write_training_log_image(
            iteration,
            viewpoint_cam,
            gt_image,
            image,
            depth,
            acc,
            scene,
            gaussians_renderer,
            training_args,
        )
        _restore_training_render_state(gaussians, viewpoint_cam)
        with torch.no_grad():
            _write_periodic_eval(
                iteration,
                periodic_eval_cameras,
                scene,
                gaussians_renderer,
                da3_bridge,
                guided_feedback,
                training_args,
                periodic_eval_previous_stats,
            )
        _restore_training_render_state(gaussians, viewpoint_cam)
        
        with torch.no_grad():
            
            # Log
            tensor_dict = dict()

            if iteration % 10 == 0:                    
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_psnr_for_log = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * ema_psnr_for_log
                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}", 
                                          "Loss": f"{ema_loss_for_log:.{7}f},", 
                                          "PSNR": f"{ema_psnr_for_log:.{4}f}"})
            progress_bar.update(1)
            # if iteration == training_args.iterations:
            #     progress_bar.close()

            # Save ply
            if (iteration in training_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            structure_updates_enabled = not bool(getattr(training_args, "disable_structure_updates", False))
            scalar_dict["save_iteration"] = int(iteration in training_args.save_iterations)
            scalar_dict["checkpoint_iteration"] = int(iteration in training_args.checkpoint_iterations)
            scalar_dict["densification_iteration"] = int(
                structure_updates_enabled
                and iteration < optim_args.densify_until_iter
                and iteration > optim_args.densify_from_iter
                and iteration % optim_args.densification_interval == 0
            )
            scalar_dict["opacity_reset_interval"] = int(optim_args.opacity_reset_interval)
            scalar_dict["opacity_reset_iteration"] = int(
                structure_updates_enabled
                and iteration < optim_args.densify_until_iter
                and iteration % optim_args.opacity_reset_interval == 0
            )

            # Densification
            if structure_updates_enabled and iteration < optim_args.densify_until_iter:
                _restore_training_render_state(gaussians, viewpoint_cam)
                _assert_training_render_shapes(iteration, viewpoint_cam, gaussians, radii, visibility_filter, viewspace_point_tensor)
                gaussians.set_visibility(include_list=_training_include_list(gaussians))
                gaussians.set_max_radii2D(radii, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                prune_big_points = iteration > optim_args.opacity_reset_interval

                if iteration > optim_args.densify_from_iter:
                    if iteration % optim_args.densification_interval == 0:
                        scalars, tensors = gaussians.densify_and_prune(
                            max_grad=optim_args.densify_grad_threshold,
                            min_opacity=optim_args.min_opacity,
                            prune_big_points=prune_big_points,
                        )

                        scalar_dict.update(scalars)
                        tensor_dict.update(tensors)

            training_report(
                tb_writer,
                iteration,
                scalar_dict,
                tensor_dict,
                training_args.test_iterations,
                scene,
                gaussians_renderer,
                current_viewpoint=viewpoint_cam,
            )
                        
            # Reset opacity
            if structure_updates_enabled and iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                if data_args.white_background and iteration == optim_args.densify_from_iter:
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < training_args.iterations:
                gaussians.update_optimizer()

            if (iteration in training_args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                state_dict = gaussians.save_state_dict(is_final=(iteration == training_args.iterations))
                state_dict['iter'] = iteration
                ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                torch.save(state_dict, ckpt_path)



def prepare_output_and_logger():
    
    # if cfg.model_path == '':
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str = os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     cfg.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))
        os.system('rm -rf {}/*'.format(cfg.trained_model_dir))

    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree
        viewer_arg['white_background'] = cfg.data.white_background
        viewer_arg['source_path'] = cfg.source_path
        viewer_arg['model_path']= cfg.model_path
        cfg_log_f.write(str(Namespace(**viewer_arg)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def _tensor_stats(prefix, tensor):
    if tensor is None:
        return {
            f"{prefix}_min": "",
            f"{prefix}_max": "",
            f"{prefix}_mean": "",
        }
    values = tensor.detach()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {
            f"{prefix}_min": "",
            f"{prefix}_max": "",
            f"{prefix}_mean": "",
        }
    return {
        f"{prefix}_min": float(finite.min().item()),
        f"{prefix}_max": float(finite.max().item()),
        f"{prefix}_mean": float(finite.mean().item()),
    }


def _eval_metric_row(iteration, split_name, idx, viewpoint, render_pkg, image, gt_image, mask, eval_protocol=""):
    acc = render_pkg.get("acc")
    depth = render_pkg.get("depth")
    mask_bool = mask.bool()
    valid_pixel_count = int(torch.count_nonzero(mask_bool).item())
    l1_value = float(l1_loss(image, gt_image, mask_bool).mean().detach().cpu().item()) if valid_pixel_count > 0 else float("nan")
    psnr_value = float(psnr(image, gt_image, mask_bool).mean().detach().cpu().item()) if valid_pixel_count > 0 else float("nan")
    warnings = []
    if valid_pixel_count == 0:
        warnings.append("empty_mask")
    if tuple(image.shape) != tuple(gt_image.shape):
        warnings.append("rgb_shape_mismatch")
    if psnr_value < 10.0:
        warnings.append("low_psnr")
    if l1_value > 0.25:
        warnings.append("high_l1")

    acc_nonzero_ratio = ""
    if acc is not None:
        acc_nonzero_ratio = float(torch.count_nonzero(acc.detach() > 1e-6).item()) / float(acc.numel())
        acc_mean = float(acc.detach().mean().item())
        if acc_nonzero_ratio < 0.01:
            warnings.append("near_empty_acc")
        if acc_mean > 0.99:
            warnings.append("saturated_acc")

    depth_positive_count = ""
    depth_finite_count = ""
    if depth is not None:
        depth_detached = depth.detach()
        finite = torch.isfinite(depth_detached)
        depth_finite_count = int(torch.count_nonzero(finite).item())
        depth_positive_count = int(torch.count_nonzero(finite & (depth_detached > 0)).item())
        if depth_positive_count == 0:
            warnings.append("empty_positive_depth")

    row = {
        "iteration": int(iteration),
        "eval_protocol": eval_protocol,
        "split": split_name,
        "view_index": int(idx),
        "cam_id": viewpoint.meta.get("cam", "") if hasattr(viewpoint, "meta") else "",
        "image_name": getattr(viewpoint, "image_name", ""),
        "uid": getattr(viewpoint, "uid", ""),
        "l1": l1_value,
        "psnr": psnr_value,
        "valid_pixel_count": valid_pixel_count,
        "rgb_shape": "x".join(str(v) for v in tuple(image.shape)),
        "gt_shape": "x".join(str(v) for v in tuple(gt_image.shape)),
        "acc_nonzero_ratio": acc_nonzero_ratio,
        "depth_positive_count": depth_positive_count,
        "depth_finite_count": depth_finite_count,
        "warnings": ";".join(sorted(set(warnings))),
    }
    row.update(_tensor_stats("render_rgb", image))
    row.update(_tensor_stats("gt_rgb", gt_image))
    row.update(_tensor_stats("acc", acc))
    row.update(_tensor_stats("depth", depth))
    return row


def _write_eval_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_eval_summary(path, rows, scalar_stats):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    by_split = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)
    fieldnames = [
        "iteration", "eval_protocol", "split", "view_count", "l1_mean", "l1_median", "l1_min", "l1_max",
        "psnr_mean", "psnr_median", "psnr_min", "psnr_max", "outlier_count",
        "outlier_views", "opacity_reset_interval", "opacity_reset_iteration",
        "densification_iteration", "checkpoint_iteration", "save_iteration",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for split_name, split_rows in by_split.items():
            l1_values = np.array([r["l1"] for r in split_rows if np.isfinite(r["l1"])], dtype=np.float64)
            psnr_values = np.array([r["psnr"] for r in split_rows if np.isfinite(r["psnr"])], dtype=np.float64)
            psnr_median = float(np.median(psnr_values)) if psnr_values.size else float("nan")
            outliers = [
                r for r in split_rows
                if np.isfinite(r["psnr"]) and (r["psnr"] < 10.0 or r["psnr"] < psnr_median - 5.0)
            ]
            writer.writerow({
                "iteration": split_rows[0]["iteration"],
                "eval_protocol": split_rows[0].get("eval_protocol", ""),
                "split": split_name,
                "view_count": len(split_rows),
                "l1_mean": float(np.mean(l1_values)) if l1_values.size else "",
                "l1_median": float(np.median(l1_values)) if l1_values.size else "",
                "l1_min": float(np.min(l1_values)) if l1_values.size else "",
                "l1_max": float(np.max(l1_values)) if l1_values.size else "",
                "psnr_mean": float(np.mean(psnr_values)) if psnr_values.size else "",
                "psnr_median": psnr_median if psnr_values.size else "",
                "psnr_min": float(np.min(psnr_values)) if psnr_values.size else "",
                "psnr_max": float(np.max(psnr_values)) if psnr_values.size else "",
                "outlier_count": len(outliers),
                "outlier_views": ";".join(f"cam{r['cam_id']}:{r['image_name']}" for r in outliers),
                "opacity_reset_interval": scalar_stats.get("opacity_reset_interval", ""),
                "opacity_reset_iteration": scalar_stats.get("opacity_reset_iteration", 0),
                "densification_iteration": scalar_stats.get("densification_iteration", 0),
                "checkpoint_iteration": scalar_stats.get("checkpoint_iteration", 0),
                "save_iteration": scalar_stats.get("save_iteration", 0),
            })


def _sorted_cameras_for_eval(cameras):
    return sorted(
        list(cameras),
        key=lambda cam: (
            int(getattr(cam, "meta", {}).get("frame_idx", getattr(cam, "meta", {}).get("frame", 0))),
            int(getattr(cam, "meta", {}).get("cam", 0)),
            str(getattr(cam, "image_name", "")),
        ),
    )


def _sample_cameras_for_eval(cameras, count):
    cameras = _sorted_cameras_for_eval(cameras)
    count = int(count or 0)
    if count <= 0 or len(cameras) <= count:
        return cameras
    if count == 1:
        return [cameras[len(cameras) // 2]]
    indices = np.linspace(0, len(cameras) - 1, count).round().astype(np.int64)
    return [cameras[int(idx)] for idx in indices]


def _run_eval_configs(tb_writer, iteration, scalar_stats, scene, renderer, current_viewpoint, validation_configs, eval_protocol):
    per_view_rows = []
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            split_rows = []
            for idx, viewpoint in enumerate(config['cameras']):
                render_pkg = renderer.render(viewpoint, scene.gaussians)
                image = torch.clamp(render_pkg["rgb"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if tb_writer and (idx < 5):
                    tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    if iteration == cfg.train.test_iterations[0]:
                        tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                if hasattr(viewpoint, 'original_mask'):
                    mask = viewpoint.original_mask.cuda().bool()
                else:
                    mask = torch.ones_like(gt_image[0]).bool()
                row = _eval_metric_row(iteration, config["name"], idx, viewpoint, render_pkg, image, gt_image, mask, eval_protocol)
                split_rows.append(row)
                per_view_rows.append(row)
                if current_viewpoint is not None:
                    _restore_training_render_state(scene.gaussians, current_viewpoint)

            l1_values = np.array([row["l1"] for row in split_rows if np.isfinite(row["l1"])], dtype=np.float64)
            psnr_values = np.array([row["psnr"] for row in split_rows if np.isfinite(row["psnr"])], dtype=np.float64)
            l1_test = float(np.mean(l1_values)) if l1_values.size else float("nan")
            psnr_test = float(np.mean(psnr_values)) if psnr_values.size else float("nan")
            psnr_median = float(np.median(psnr_values)) if psnr_values.size else float("nan")
            low_rows = [row for row in split_rows if np.isfinite(row["psnr"]) and row["psnr"] < 10.0]
            print(
                "\n[ITER {}] Evaluating {} [{}]: L1 {} PSNR {} "
                "median_psnr {} outliers {} views {}".format(
                    iteration, config['name'], eval_protocol, l1_test, psnr_test,
                    psnr_median, len(low_rows), len(split_rows)
                )
            )
            if tb_writer:
                tb_writer.add_scalar(config['name'] + f'/{eval_protocol} - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + f'/{eval_protocol} - psnr', psnr_test, iteration)
                tb_writer.add_scalar(config['name'] + f'/{eval_protocol} - psnr_median', psnr_median, iteration)
    return per_view_rows


def training_report(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene, renderer: StreetGaussianRenderer, current_viewpoint=None):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')
            
            
    sampled_interval = int(getattr(cfg.train, "eval_sampled_interval", 0) or 0)
    full_interval = int(getattr(cfg.train, "eval_full_interval", 0) or 0)
    do_sampled = sampled_interval > 0 and iteration % sampled_interval == 0
    do_full = full_interval > 0 and iteration % full_interval == 0

    if do_sampled or do_full:
        torch.cuda.empty_cache()

        metrics_dir = os.path.join(cfg.model_path, "metrics")
        if do_sampled:
            sampled_configs = (
                {
                    'name': 'test/test_view',
                    'cameras': _sample_cameras_for_eval(scene.getTestCameras(), getattr(cfg.train, "eval_sampled_test_view_count", 15)),
                },
                {
                    'name': 'test/train_view',
                    'cameras': _sample_cameras_for_eval(scene.getTrainCameras(), getattr(cfg.train, "eval_sampled_train_view_count", 15)),
                },
            )
            sampled_rows = _run_eval_configs(tb_writer, iteration, scalar_stats, scene, renderer, current_viewpoint, sampled_configs, "sampled_diagnostic_eval")
            _write_eval_csv(os.path.join(metrics_dir, f"eval_sampled_iter_{iteration:06d}_per_view.csv"), sampled_rows)
            _append_eval_summary(os.path.join(metrics_dir, "eval_summary_sampled.csv"), sampled_rows, scalar_stats)
        if do_full:
            full_configs = []
            if bool(getattr(cfg.train, "eval_full_test_enabled", True)):
                full_configs.append({'name': 'test/test_view', 'cameras': _sorted_cameras_for_eval(scene.getTestCameras())})
            if bool(getattr(cfg.train, "eval_full_train_enabled", True)):
                full_configs.append({'name': 'test/train_view', 'cameras': _sorted_cameras_for_eval(scene.getTrainCameras())})
            full_rows = _run_eval_configs(tb_writer, iteration, scalar_stats, scene, renderer, current_viewpoint, full_configs, "full_split_training_eval")
            _write_eval_csv(os.path.join(metrics_dir, f"eval_full_iter_{iteration:06d}_per_view.csv"), full_rows)
            _append_eval_summary(os.path.join(metrics_dir, "eval_summary_full.csv"), full_rows, scalar_stats)
        if current_viewpoint is not None:
            _restore_training_render_state(scene.gaussians, current_viewpoint)

        #if tb_writer:
            #tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #    tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Optimizing " + cfg.model_path)
    _require_cuda("startup")

    # Initialize system state (RNG)
    safe_state(cfg.train.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    training()

    # All done
    print("\nTraining complete.")
