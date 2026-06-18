import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from lib.utils.feedback_pipeline_stages import (
    build_da3_boundary_risk_stage,
    build_lidar_error_risk_stage,
    run_cuda_contribution_stage,
    select_da3_responsible_group_stage,
    build_softpatch_feedback_stage,
    run_group_counterfactual_dryrun_stage,
    tag_repair_candidates_stage,
)


VALID_MODES = {"off", "diagnose_only", "feedback_update", "repair_dryrun", "repair_apply"}
VALID_RISK_SOURCES = {"lidar_error", "da3_boundary", "da3_structure"}
VALID_SUPERVISION = {"lidar_supervised", "da3_unsupervised", "hybrid_reference"}
VALID_REPAIR = {"none", "dryrun", "opacity_regularization", "prune"}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [value]


def _json_write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


@dataclass
class FeedbackTriggerResult:
    status: str
    manifest_path: str
    feedback_mask: object = None


class PeriodicFeedbackController:
    """Debug-oriented scheduler for periodic responsibility diagnosis.

    This controller intentionally does not implement pruning or parameter edits.
    It records a manifest for each trigger and can optionally call existing
    offline debug scripts. Training receives only a feedback-mask placeholder.
    """

    def __init__(self, cfg_node, model_path=None):
        self.cfg = cfg_node
        self.enabled = bool(getattr(cfg_node, "enabled", False))
        self.mode = str(getattr(cfg_node, "mode", "off"))
        self.risk_source = str(getattr(cfg_node, "risk_source", "da3_boundary"))
        self.supervision_mode = str(getattr(cfg_node, "supervision_mode", "da3_unsupervised"))
        self.repair_mode = str(getattr(cfg_node, "repair_mode", "none"))
        self.start_iter = int(getattr(cfg_node, "start_iter", 0))
        self.interval = max(1, int(getattr(cfg_node, "interval", 1000)))
        self.selected_views = _as_list(getattr(cfg_node, "selected_views", []))
        self.max_regions = int(getattr(cfg_node, "max_regions", 30))
        self.max_triggers = int(getattr(cfg_node, "max_triggers", -1))
        self.max_pixels_per_region = int(getattr(cfg_node, "max_pixels_per_region", 64))
        self.top_contributors = int(getattr(cfg_node, "top_contributors", 5))
        self.spatial_cell_size = int(getattr(cfg_node, "spatial_cell_size", 96))
        self.feedback_mode = str(getattr(cfg_node, "feedback_mode", "none"))
        self.signal_path = str(getattr(cfg_node, "signal_path", ""))
        self.contribution_summary_path = str(getattr(cfg_node, "contribution_summary_path", ""))
        self.contribution_source = str(getattr(cfg_node, "contribution_source", "cached_summary"))
        self.dryrun_scorer_path = str(getattr(cfg_node, "dryrun_scorer_path", ""))
        self.dryrun_extra_args = [str(v) for v in _as_list(getattr(cfg_node, "dryrun_extra_args", []))]
        self.run_dryrun_scorer = bool(getattr(cfg_node, "run_dryrun_scorer", False))
        self.use_cached_if_exists = bool(getattr(cfg_node, "use_cached_if_exists", True))
        self.recompute_risk = bool(getattr(cfg_node, "recompute_risk", False))
        self.recompute_contribution = bool(getattr(cfg_node, "recompute_contribution", False))
        self.recompute_softpatch = bool(getattr(cfg_node, "recompute_softpatch", False))
        self.recompute_responsible_groups = bool(getattr(cfg_node, "recompute_responsible_groups", False))
        self.run_counterfactual = bool(getattr(cfg_node, "run_counterfactual", False))
        self.run_candidate_tagging = bool(getattr(cfg_node, "run_candidate_tagging", False))
        self.run_group_counterfactual_every_n_triggers = int(getattr(cfg_node, "run_group_counterfactual_every_n_triggers", 0))
        self.run_candidate_tagging_every_n_triggers = int(getattr(cfg_node, "run_candidate_tagging_every_n_triggers", 0))
        self.fail_policy = str(getattr(cfg_node, "fail_policy", "warn"))
        self.allow_parameter_modification = bool(getattr(cfg_node, "allow_parameter_modification", False))
        self.skip_existing = bool(getattr(cfg_node, "skip_existing", True))
        out = str(getattr(cfg_node, "output_dir", ""))
        if not out:
            out = os.path.join(model_path or "output", "feedback_controller")
        self.output_dir = out
        self.da3_risk_cache_dir = os.path.join(self.output_dir, "da3_risk_cache")
        self.trigger_count = 0
        self.active_feedback = None
        self._validate()

    def _validate(self):
        if self.mode not in VALID_MODES:
            raise ValueError(f"unknown train.feedback_controller.mode={self.mode!r}")
        if self.risk_source not in VALID_RISK_SOURCES:
            raise ValueError(f"unknown train.feedback_controller.risk_source={self.risk_source!r}")
        if self.supervision_mode not in VALID_SUPERVISION:
            raise ValueError(f"unknown train.feedback_controller.supervision_mode={self.supervision_mode!r}")
        if self.repair_mode not in VALID_REPAIR:
            raise ValueError(f"unknown train.feedback_controller.repair_mode={self.repair_mode!r}")
        if not self.enabled:
            return
        if self.mode == "off":
            raise ValueError("train.feedback_controller.enabled=True requires mode != 'off'")
        if self.mode != "repair_apply" and self.repair_mode in {"opacity_regularization", "prune"}:
            raise ValueError("destructive repair modes require mode='repair_apply'")
        if self.mode == "repair_apply" and not self.allow_parameter_modification:
            raise ValueError("repair_apply requires allow_parameter_modification=True")
        if self.repair_mode == "prune":
            raise NotImplementedError("feedback_controller prune is intentionally not implemented")
        if self.fail_policy not in {"skip", "warn", "stop"}:
            raise ValueError("train.feedback_controller.fail_policy must be skip, warn, or stop")

    @property
    def summary(self):
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "risk_source": self.risk_source,
            "supervision_mode": self.supervision_mode,
            "repair_mode": self.repair_mode,
            "feedback_mode": self.feedback_mode,
            "start_iter": self.start_iter,
            "interval": self.interval,
            "max_triggers": self.max_triggers,
            "recompute_risk": self.recompute_risk,
            "recompute_contribution": self.recompute_contribution,
            "recompute_responsible_groups": self.recompute_responsible_groups,
            "recompute_softpatch": self.recompute_softpatch,
            "output_dir": self.output_dir,
            "da3_risk_cache_dir": self.da3_risk_cache_dir,
            "contribution_source": self.contribution_source,
            "max_pixels_per_region": self.max_pixels_per_region,
            "top_contributors": self.top_contributors,
            "spatial_cell_size": self.spatial_cell_size,
        }

    def should_trigger(self, iteration):
        if not self.enabled or self.mode == "off":
            return False
        if iteration < self.start_iter:
            return False
        if self.max_triggers >= 0 and self.trigger_count >= self.max_triggers:
            return False
        return (iteration - self.start_iter) % self.interval == 0

    def trigger(self, iteration, selected_views=None, selected_regions=None, extra=None):
        return self.run_trigger(iteration, selected_views=selected_views, selected_regions=selected_regions, extra=extra)

    def run_trigger(self, iteration, model=None, scene=None, renderer=None, current_outputs=None, selected_views=None, selected_regions=None, extra=None):
        trigger_dir = os.path.join(self.output_dir, f"iter_{int(iteration):06d}")
        manifest_path = os.path.join(trigger_dir, "feedback_controller_manifest.json")
        if self.skip_existing and os.path.exists(manifest_path):
            return FeedbackTriggerResult(status="skipped", manifest_path=manifest_path)

        views = self.select_views(iteration, scene, selected_views=selected_views)
        regions = _as_list(selected_regions)
        status = "valid"
        errors = []
        dryrun_output = ""
        candidate_output = ""
        contribution_out = ""
        feedback_signal_path = ""
        active_summary = {}
        risk_summary = {}
        contribution_stage = {}
        group_summary = {}
        overlay_summary = {}
        t0 = time.perf_counter()

        try:
            rendered_outputs = self.render_diagnostic_views(model, renderer, views, current_outputs=current_outputs)
            risk_summary = self.build_risk_map(rendered_outputs, views, trigger_dir=trigger_dir)
            regions = regions or self.select_risk_regions(risk_summary)
            contribution_stage = self.run_contribution_dump(
                risk_summary,
                trigger_dir=trigger_dir,
                model=model,
                camera=(current_outputs or {}).get("camera"),
                renderer=renderer,
            )
            contribution_out = contribution_stage.get("path", contribution_stage if isinstance(contribution_stage, str) else "")
            group_summary = self.select_responsible_groups(contribution_out, trigger_dir)
            feedback_stage = self.build_feedback_signal(contribution_out, group_summary, trigger_dir)
            feedback_signal_path = feedback_stage.get("path", feedback_stage if isinstance(feedback_stage, str) else "")
            overlay_summary = self.write_feedback_visual_overlays(
                trigger_dir,
                rendered_outputs,
                risk_summary,
                group_summary,
                feedback_signal_path,
            )
            if self.mode in {"feedback_update", "repair_dryrun"}:
                self.update_active_feedback(feedback_signal_path)
                active_summary = self._write_active_feedback_summary(trigger_dir)
            run_slow_counterfactual = (
                self.mode == "repair_dryrun"
                or self.run_counterfactual
                or self.run_dryrun_scorer
                or (self.run_group_counterfactual_every_n_triggers > 0 and self.trigger_count % self.run_group_counterfactual_every_n_triggers == 0)
            )
            if run_slow_counterfactual:
                dryrun_output = self.run_counterfactual_dryrun(feedback_signal_path, trigger_dir, iteration)
            run_slow_tagging = self.run_candidate_tagging or (
                self.run_candidate_tagging_every_n_triggers > 0 and self.trigger_count % self.run_candidate_tagging_every_n_triggers == 0
            )
            if run_slow_tagging and dryrun_output:
                candidate_stage = self.run_candidate_tagging_from_dryrun(dryrun_output, trigger_dir)
                candidate_output = candidate_stage.get("path", candidate_stage if isinstance(candidate_stage, str) else "")
        except Exception as exc:
            status = "failed"
            errors.append(str(exc))
            if self.fail_policy == "stop":
                raise
            if self.fail_policy == "skip":
                status = "skipped"
            elif self.fail_policy == "warn":
                print(f"[FeedbackController][WARN] trigger failed at iter {iteration}: {exc}")

        payload = {
            "iteration": int(iteration),
            "start_checkpoint": self._checkpoint_hint(iteration),
            "current_checkpoint": self._checkpoint_hint(iteration),
            "risk_source": self.risk_source,
            "supervision_mode": self.supervision_mode,
            "feedback_mode": self.mode,
            "active_feedback_mode": self.feedback_mode,
            "repair_mode": self.repair_mode,
            "selected_views": views,
            "selected_regions": regions[: self.max_regions] if regions else [],
            "selected_pixels_count": int(risk_summary.get("selected_pixels_count", 0)) if isinstance(risk_summary, dict) else 0,
            "gaussian_group_count": int(group_summary.get("counts", {}).get("group_count", 0)) if isinstance(group_summary, dict) else 0,
            "cuda_ok_count": int(contribution_stage.get("cuda_ok_count", risk_summary.get("cuda_ok_count", 0))) if isinstance(contribution_stage, dict) else 0,
            "low_evidence_count": int(contribution_stage.get("low_evidence_count", risk_summary.get("low_evidence_count", 0))) if isinstance(contribution_stage, dict) else 0,
            "signal_path": self.signal_path,
            "active_feedback_signal_path": feedback_signal_path,
            "contribution_summary_path": self.contribution_summary_path,
            "contribution_output_path": contribution_out,
            "contribution_source": self.contribution_source,
            "live_cuda_contribution": bool(contribution_stage.get("live_cuda_contribution", False)) if isinstance(contribution_stage, dict) else False,
            "uses_cached_contribution": bool(contribution_stage.get("uses_cached_contribution", self.contribution_source != "live_current_model")) if isinstance(contribution_stage, dict) else True,
            "stable_id_map_available": bool(contribution_stage.get("stable_id_map_available", False)) if isinstance(contribution_stage, dict) else False,
            "unmapped_id_count": int(contribution_stage.get("unmapped_id_count", 0)) if isinstance(contribution_stage, dict) else 0,
            "cuda_runtime_sec": float(contribution_stage.get("cuda_runtime_sec", 0.0)) if isinstance(contribution_stage, dict) else 0.0,
            "dryrun_scorer_path": self.dryrun_scorer_path,
            "dryrun_output_dir": dryrun_output,
            "candidate_tags_path": candidate_output,
            "feedback_mask_coverage": active_summary.get("feedback_mask_coverage", None),
            "uses_lidar_supervision": self.supervision_mode in {"lidar_supervised", "hybrid_reference"},
            "uses_lidar_for_evaluation_only": self.supervision_mode == "da3_unsupervised",
            "selected_pixel_source": self._selected_pixel_source(),
            "uses_lidar_selected_pixels": self.risk_source == "lidar_error",
            "gaussian_parameters_modified": False,
            "real_repair_enabled": False,
            "allow_parameter_modification": self.allow_parameter_modification,
            "recompute_risk": self.recompute_risk,
            "recompute_contribution": self.recompute_contribution,
            "recompute_responsible_groups": self.recompute_responsible_groups,
            "recompute_softpatch": self.recompute_softpatch,
            "visual_overlay_summary": overlay_summary,
            "status": status,
            "errors": errors,
            "runtime_sec": float(time.perf_counter() - t0),
            "extra": extra or {},
        }
        _json_write(manifest_path, payload)
        audit_path = os.path.join(trigger_dir, "feedback_controller_audit.json")
        _json_write(audit_path, payload)
        _json_write(os.path.join(trigger_dir, "audit_summary.json"), payload)
        _json_write(os.path.join(trigger_dir, "pipeline_stage_manifest.json"), self._stage_manifest(payload, risk_summary, active_summary))
        self.trigger_count += 1
        return FeedbackTriggerResult(status=status, manifest_path=manifest_path, feedback_mask=self.active_feedback)

    def validate_supervision(self, optim_args=None, guided_feedback=None):
        if not self.enabled:
            return
        if self.supervision_mode == "da3_unsupervised":
            violations = []
            lidar_weight = float(getattr(optim_args, "lambda_depth_lidar", 0.0)) if optim_args is not None else 0.0
            if abs(lidar_weight) > 1e-12:
                violations.append(f"optim.lambda_depth_lidar must be 0, got {lidar_weight}")
            if guided_feedback is not None:
                if bool(getattr(guided_feedback, "use_lidar_depth", False)):
                    violations.append("train.guided_feedback.use_lidar_depth must be False")
                if not bool(getattr(guided_feedback, "use_da3_structure", False)):
                    violations.append("train.guided_feedback.use_da3_structure must be True")
            if self.risk_source == "lidar_error":
                violations.append("da3_unsupervised cannot use risk_source=lidar_error")
            if violations:
                raise ValueError("Invalid DA3-unsupervised feedback_controller configuration: " + "; ".join(violations))
            print("[FeedbackController] LiDAR supervision disabled; LiDAR is used for evaluation only.")

    def select_views(self, iteration, scene=None, selected_views=None):
        explicit = _as_list(selected_views) or _as_list(getattr(self.cfg, "trigger_views", [])) or self.selected_views
        if explicit:
            return [str(v) for v in explicit]
        if scene is None:
            return []
        cameras = scene.getTrainCameras()
        if not cameras:
            return []
        idx = int(iteration) % len(cameras)
        return [str(cameras[idx].image_name)]

    def render_diagnostic_views(self, model, renderer, views, current_outputs=None):
        if current_outputs is None:
            return {}
        return {
            "current_view": current_outputs.get("view_id"),
            "view_id": current_outputs.get("view_id"),
            "camera": current_outputs.get("camera"),
            "rgb": current_outputs.get("rgb"),
            "depth": current_outputs.get("depth"),
            "acc": current_outputs.get("acc"),
            "has_rgb": current_outputs.get("rgb") is not None,
            "has_depth": current_outputs.get("depth") is not None,
            "has_acc": current_outputs.get("acc") is not None,
            "view_count": len(views),
        }

    def build_risk_map(self, rendered_outputs, views, trigger_dir=None):
        if self.recompute_risk and trigger_dir:
            stage_dir = os.path.join(trigger_dir, "risk_stage")
            if self.risk_source == "lidar_error":
                return build_lidar_error_risk_stage(
                    rendered_outputs,
                    views,
                    stage_dir,
                    max_pixels_per_region=self.max_pixels_per_region,
                )
            return build_da3_boundary_risk_stage(
                rendered_outputs,
                views,
                stage_dir,
                max_pixels_per_region=self.max_pixels_per_region,
                cache_dir=self.da3_risk_cache_dir,
            )
        selected_pixels = 0
        low_evidence = 0
        cuda_ok = 0
        if self.contribution_summary_path and os.path.exists(self.contribution_summary_path):
            try:
                with open(self.contribution_summary_path, "r", encoding="utf-8-sig") as f:
                    payload = json.load(f)
                frames = payload.get("frames", [])[: self.max_regions]
                for frame in frames:
                    if frame.get("status") == "ok":
                        selected_pixels += int(frame.get("selected_pixel_count", 0) or 0)
                        cuda_ok += 1
                    else:
                        low_evidence += 1
            except Exception:
                pass
        return {
            "risk_source": self.risk_source,
            "selected_pixel_source": self._selected_pixel_source(),
            "uses_lidar_selected_pixels": self.risk_source == "lidar_error",
            "views": views,
            "rendered_outputs": {
                "current_view": rendered_outputs.get("current_view"),
                "has_rgb": bool(rendered_outputs.get("has_rgb", False)),
                "has_depth": bool(rendered_outputs.get("has_depth", False)),
                "has_acc": bool(rendered_outputs.get("has_acc", False)),
                "view_count": rendered_outputs.get("view_count", len(views)),
            },
            "selected_pixels_count": selected_pixels,
            "cuda_ok_count": cuda_ok,
            "low_evidence_count": low_evidence,
            "risk_map_mode": "cached_summary" if selected_pixels else "manifest_only",
            "recompute_risk": self.recompute_risk,
        }

    def select_risk_regions(self, risk_map):
        regions = []
        if self.contribution_summary_path and os.path.exists(self.contribution_summary_path):
            try:
                with open(self.contribution_summary_path, "r", encoding="utf-8-sig") as f:
                    payload = json.load(f)
                for frame in payload.get("frames", [])[: self.max_regions]:
                    regions.append(f"{frame.get('stem')}:region{frame.get('region_id')}")
            except Exception:
                pass
        return regions

    def run_contribution_dump(self, selected_pixels, trigger_dir=None, model=None, camera=None, renderer=None):
        if trigger_dir:
            return run_cuda_contribution_stage(
                selected_pixels,
                self.contribution_summary_path,
                os.path.join(trigger_dir, "contribution_stage"),
                use_cached=(self.use_cached_if_exists or not self.recompute_contribution),
                contribution_source=self.contribution_source,
                model=model,
                camera=camera,
                renderer=renderer,
                top_k=self.top_contributors,
            )
        return {"path": self.contribution_summary_path if self.contribution_summary_path else ""}

    def select_responsible_groups(self, contribution_summary_path, trigger_dir):
        if self.recompute_responsible_groups and contribution_summary_path:
            return select_da3_responsible_group_stage(
                contribution_summary_path,
                os.path.join(trigger_dir, "responsible_group_stage"),
                max_regions=self.max_regions,
                spatial_cell_size=self.spatial_cell_size,
            )
        summary = {"status": "skipped", "reason": "recompute_responsible_groups is false", "counts": {"group_count": 0}}
        _json_write(os.path.join(trigger_dir, "responsible_group_summary.json"), summary)
        return summary

    def build_feedback_signal(self, contribution_results, group_summary, trigger_dir):
        if self.recompute_softpatch:
            return build_softpatch_feedback_stage(
                self.signal_path,
                group_summary,
                trigger_dir,
                mode=self.feedback_mode,
            )
        os.makedirs(trigger_dir, exist_ok=True)
        out_path = os.path.join(trigger_dir, "feedback_signal.json")
        if self.signal_path and os.path.exists(self.signal_path):
            shutil.copyfile(self.signal_path, out_path)
        else:
            _json_write(out_path, {
                "regions": [],
                "bad_contributors": [],
                "good_contributors": [],
                "low_evidence_regions": [],
                "feedback_mode": self.feedback_mode,
                "risk_source": self.risk_source,
                "supervision_mode": self.supervision_mode,
                "generated_by": "PeriodicFeedbackController",
                "note": "empty feedback signal because no signal_path was available",
            })
        return {"status": "valid", "path": out_path, "feedback_mode": self.feedback_mode}

    def write_feedback_visual_overlays(self, trigger_dir, rendered_outputs, risk_summary, group_summary, feedback_signal_path):
        overlay_dir = os.path.join(trigger_dir, "visual_overlays")
        summary = {
            "status": "skipped",
            "reason": "",
            "output_dir": overlay_dir,
            "paths": {},
            "view_id": risk_summary.get("view_id", "") if isinstance(risk_summary, dict) else "",
            "selected_pixel_count": int(risk_summary.get("selected_pixels_count", 0)) if isinstance(risk_summary, dict) else 0,
            "responsible_group_count": 0,
        }
        if cv2 is None:
            summary["reason"] = "cv2 is not available"
            return summary
        rgb = self._tensor_to_rgb_u8(rendered_outputs.get("rgb") if isinstance(rendered_outputs, dict) else None)
        depth_vis = self._depth_to_rgb_u8(rendered_outputs.get("depth") if isinstance(rendered_outputs, dict) else None)
        if rgb is None and depth_vis is None:
            summary["reason"] = "missing current rendered rgb/depth"
            return summary
        base = rgb if rgb is not None else depth_vis
        h, w = base.shape[:2]
        if rgb is None:
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
        if depth_vis is None:
            depth_vis = np.zeros((h, w, 3), dtype=np.uint8)

        selected = self._load_selected_pixels(risk_summary)
        groups = self._load_responsible_groups(trigger_dir)
        summary["responsible_group_count"] = len(groups)
        risk_mask = np.zeros((h, w), dtype=np.uint8)
        bbox = None
        if selected.size:
            xs = np.clip(selected[:, 0].astype(np.int64), 0, w - 1)
            ys = np.clip(selected[:, 1].astype(np.int64), 0, h - 1)
            risk_mask[ys, xs] = 255
            risk_mask = cv2.dilate(risk_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

        rgb_risk = self._overlay_mask(rgb, risk_mask, (255, 48, 48), alpha=0.55)
        depth_risk = self._overlay_mask(depth_vis, risk_mask, (255, 48, 48), alpha=0.55)
        rgb_group = rgb.copy()
        depth_group = depth_vis.copy()
        combined = rgb_risk.copy()
        self._draw_responsible_group_annotation(rgb_group, bbox, groups)
        self._draw_responsible_group_annotation(depth_group, bbox, groups)
        self._draw_responsible_group_annotation(combined, bbox, groups)
        self._draw_overlay_header(rgb_risk, "risk regions from selected pixels")
        self._draw_overlay_header(depth_risk, "risk regions from selected pixels")
        self._draw_overlay_header(rgb_group, "responsible Gaussian groups")
        self._draw_overlay_header(depth_group, "responsible Gaussian groups")
        self._draw_overlay_header(combined, "risk regions + responsible Gaussian groups")

        view_id = str(summary["view_id"] or "view").replace(os.sep, "_")
        os.makedirs(overlay_dir, exist_ok=True)
        outputs = {
            "rgb_risk_regions": os.path.join(overlay_dir, f"{view_id}_rgb_risk_regions.jpg"),
            "depth_risk_regions": os.path.join(overlay_dir, f"{view_id}_depth_risk_regions.jpg"),
            "rgb_responsible_groups": os.path.join(overlay_dir, f"{view_id}_rgb_responsible_groups.jpg"),
            "depth_responsible_groups": os.path.join(overlay_dir, f"{view_id}_depth_responsible_groups.jpg"),
            "rgb_risk_and_responsible_groups": os.path.join(overlay_dir, f"{view_id}_rgb_risk_and_responsible_groups.jpg"),
        }
        cv2.imwrite(outputs["rgb_risk_regions"], rgb_risk[..., ::-1])
        cv2.imwrite(outputs["depth_risk_regions"], depth_risk[..., ::-1])
        cv2.imwrite(outputs["rgb_responsible_groups"], rgb_group[..., ::-1])
        cv2.imwrite(outputs["depth_responsible_groups"], depth_group[..., ::-1])
        cv2.imwrite(outputs["rgb_risk_and_responsible_groups"], combined[..., ::-1])
        summary.update({
            "status": "valid",
            "reason": "",
            "paths": outputs,
            "feedback_signal_path": feedback_signal_path,
            "note": "Red marks selected risk pixels/region. Cyan boxes/text identify the final responsible Gaussian groups attached to this risk region; empty groups are explicitly marked.",
        })
        _json_write(os.path.join(overlay_dir, "visual_overlay_manifest.json"), summary)
        return summary

    def _tensor_to_rgb_u8(self, value):
        if value is None or not hasattr(value, "detach"):
            return None
        arr = value.detach().float().cpu().numpy()
        arr = np.squeeze(arr)
        if arr.ndim == 3 and arr.shape[0] in {1, 3, 4}:
            arr = np.moveaxis(arr[:3], 0, -1)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim != 3:
            return None
        return (np.clip(arr[..., :3], 0.0, 1.0) * 255).astype(np.uint8)

    def _depth_to_rgb_u8(self, value):
        if value is None or not hasattr(value, "detach"):
            return None
        depth = np.squeeze(value.detach().float().cpu().numpy()).astype(np.float32)
        finite = np.isfinite(depth) & (depth > 0)
        out = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(finite):
            lo, hi = np.percentile(depth[finite], [5, 95])
            norm = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0, 1)
            out = (norm * 255).astype(np.uint8)
        return cv2.applyColorMap(out, cv2.COLORMAP_TURBO)[..., ::-1]

    def _overlay_mask(self, image, mask, color, alpha=0.5):
        out = image.copy()
        active = mask > 0
        if np.any(active):
            color_arr = np.asarray(color, dtype=np.float32)
            out[active] = np.clip((1.0 - alpha) * out[active].astype(np.float32) + alpha * color_arr, 0, 255).astype(np.uint8)
        return out

    def _load_selected_pixels(self, risk_summary):
        if not isinstance(risk_summary, dict):
            return np.zeros((0, 2), dtype=np.int64)
        path = risk_summary.get("selected_pixels_path", "")
        if not path or not os.path.exists(path):
            return np.zeros((0, 2), dtype=np.int64)
        try:
            arr = np.asarray(np.load(path), dtype=np.int64).reshape(-1, 2)
            return arr
        except Exception:
            return np.zeros((0, 2), dtype=np.int64)

    def _load_responsible_groups(self, trigger_dir):
        path = os.path.join(trigger_dir, "responsible_group_stage", "da3_boundary_responsible_groups.json")
        if not os.path.exists(path):
            return []
        try:
            payload = read_json(path)
        except Exception:
            return []
        return payload if isinstance(payload, list) else []

    def _draw_overlay_header(self, image, text):
        cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_responsible_group_annotation(self, image, bbox, groups):
        h, w = image.shape[:2]
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            x0, y0 = max(0, x0), max(30, y0)
            x1, y1 = min(w - 1, x1), min(h - 1, y1)
            cv2.rectangle(image, (x0, y0), (x1, y1), (32, 220, 255), 2)
            text_x, text_y = x0, min(h - 8, y1 + 22)
        else:
            text_x, text_y = 8, 54
        if not groups:
            cv2.putText(image, "no responsible group selected", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (32, 220, 255), 2, cv2.LINE_AA)
            return
        for idx, group in enumerate(groups[:5]):
            label = str(group.get("group_label", "group"))
            score = group.get("group_risk_weighted_contribution", 0.0)
            member_count = group.get("member_count", len(group.get("stable_gaussian_ids", [])))
            text = f"g{group.get('group_id', idx)} {label} n={member_count} s={float(score):.2f}"
            y = min(h - 8, text_y + idx * 22)
            cv2.putText(image, text[:96], (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (32, 220, 255), 2, cv2.LINE_AA)

    def update_active_feedback(self, feedback_signal):
        self.active_feedback = {
            "signal_path": feedback_signal,
            "mode": self.feedback_mode,
            "supervision_mode": self.supervision_mode,
            "metadata": {
                "risk_source": self.risk_source,
                "uses_lidar_selected_pixels": self.risk_source == "lidar_error",
            },
        }
        return self.active_feedback

    def run_counterfactual_dryrun(self, feedback_signal, trigger_dir, iteration=None):
        if not self.dryrun_scorer_path:
            return ""
        summary = run_group_counterfactual_dryrun_stage(
            self.dryrun_scorer_path,
            self.contribution_summary_path,
            feedback_signal,
            os.path.join(trigger_dir, "group_counterfactual_dryrun"),
            max_regions=self.max_regions,
            extra_args=self.dryrun_extra_args,
        )
        return os.path.join(trigger_dir, "group_counterfactual_dryrun") if summary else ""

    def run_candidate_tagging_from_dryrun(self, dryrun_output, trigger_dir):
        out_dir = os.path.join(trigger_dir, "candidate_tags")
        return tag_repair_candidates_stage(dryrun_output, out_dir)

    def write_manifest(self):
        return self.summary

    def get_active_feedback(self):
        return self.active_feedback

    def _selected_pixel_source(self):
        if self.risk_source == "lidar_error":
            return "lidar_error_map"
        if self.risk_source == "da3_boundary":
            return "da3_boundary_risk_map"
        return "da3_structure_risk_map"

    def _checkpoint_hint(self, iteration):
        return f"current_training_state_iter_{int(iteration):06d}"

    def _write_active_feedback_summary(self, trigger_dir):
        summary = {
            "active_feedback_path": self.active_feedback.get("signal_path") if self.active_feedback else "",
            "active_feedback_mode": self.feedback_mode,
            "supervision_mode": self.supervision_mode,
            "feedback_mask_coverage": None,
            "note": "Coverage is computed by guided loss per view; controller stores signal-level active feedback.",
        }
        _json_write(os.path.join(trigger_dir, "active_feedback_summary.json"), summary)
        return summary

    def _stage_manifest(self, manifest, risk_summary, active_summary):
        return {
            "iteration": manifest["iteration"],
            "stages": {
                "render_diagnostic_views": risk_summary.get("rendered_outputs", {}),
                "build_risk_map": risk_summary,
                "run_contribution_dump": {
                    "path": manifest.get("contribution_output_path"),
                    "contribution_source": manifest.get("contribution_source"),
                    "live_cuda_contribution": manifest.get("live_cuda_contribution"),
                    "uses_cached_contribution": manifest.get("uses_cached_contribution"),
                },
                "build_feedback_signal": {"path": manifest.get("active_feedback_signal_path")},
                "update_active_feedback": active_summary,
                "counterfactual_dryrun": {"path": manifest.get("dryrun_output_dir")},
                "candidate_tagging": {"path": manifest.get("candidate_tags_path")},
            },
            "gaussian_parameters_modified": False,
        }

    def _run_dryrun_scorer(self, iteration, trigger_dir, feedback_signal=None):
        if not self.dryrun_scorer_path:
            raise ValueError("dryrun_scorer_path is empty")
        scorer = Path(self.dryrun_scorer_path)
        if not scorer.exists():
            raise FileNotFoundError(str(scorer))
        dryrun_out = os.path.join(trigger_dir, "repair_dryrun")
        cmd = [
            sys.executable,
            str(scorer),
            "--output-dir",
            dryrun_out,
            "--top-regions",
            str(self.max_regions),
        ]
        if self.contribution_summary_path:
            cmd += ["--contribution-summary", self.contribution_summary_path]
        signal = feedback_signal or self.signal_path
        if signal:
            cmd += ["--softpatch-signal", signal]
        cmd += self.dryrun_extra_args
        subprocess.run(cmd, cwd=str(scorer.parent.parent), check=True)
        return dryrun_out


def make_periodic_feedback_controller(cfg_node, model_path=None):
    return PeriodicFeedbackController(cfg_node, model_path=model_path)
