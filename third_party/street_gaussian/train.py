#python train.py --config configs/default.yaml
#CUDA_VISIBLE_DEVICES=0 python train.py --config configs/default.yaml
# conda activate "/tsinghuaData/hanchanghao/envs/sg_env"

import json
import os
import numpy as np
import torch
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
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
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
        _write_scalar_trace(getattr(training_args, "scalar_trace_path", ""), iteration, scalar_dict)
        
        loss.backward()
        
        iter_end.record()
                
        is_save_images = True
        if is_save_images and (iteration % 1000 == 0):
            # row0: gt_image, image, depth
            # row1: acc, image_obj, acc_obj
            depth_colored, _ = visualize_depth_numpy(depth.detach().cpu().numpy().squeeze(0))
            depth_colored = depth_colored[..., [2, 1, 0]] / 255.
            depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()
            row0 = torch.cat([gt_image, image, depth_colored], dim=2)
            acc = acc.repeat(3, 1, 1)
            with torch.no_grad():
                render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
                image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = acc_obj.repeat(3, 1, 1)
            row1 = torch.cat([acc, image_obj, acc_obj], dim=2)
            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
            os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
            save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")
        
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

            # Densification
            if structure_updates_enabled and iteration < optim_args.densify_until_iter:
                gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
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
                        
            # Reset opacity
            if structure_updates_enabled and iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                if data_args.white_background and iteration == optim_args.densify_from_iter:
                    gaussians.reset_opacity()

            training_report(tb_writer, iteration, scalar_dict, tensor_dict, training_args.test_iterations, scene, gaussians_renderer)

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

def training_report(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene, renderer: StreetGaussianRenderer):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')
            
            
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 'cameras' : scene.getTestCameras()},
                              {'name': 'test/train_view', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.cuda().bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        #if tb_writer:
            #tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #    tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Optimizing " + cfg.model_path)

    # Initialize system state (RNG)
    safe_state(cfg.train.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    training()

    # All done
    print("\nTraining complete.")
