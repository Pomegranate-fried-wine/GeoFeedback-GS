import json
import os
from collections import defaultdict

import torch


class GuidedFeedbackController:
    def __init__(self, cfg_node):
        self.enabled = bool(getattr(cfg_node, "enabled", False))
        self.signal_path = str(getattr(cfg_node, "signal_path", "") or "")
        self.region_weight = float(getattr(cfg_node, "region_weight", 2.0))
        self.geometry_weight = float(getattr(cfg_node, "geometry_weight", 1.0))
        self.min_valid_pixels = int(getattr(cfg_node, "min_valid_pixels", 1))
        self.supervision_mode = str(getattr(cfg_node, "supervision_mode", "lidar_supervised") or "lidar_supervised")
        self.use_lidar_depth = bool(getattr(cfg_node, "use_lidar_depth", True))
        self.feedback_mode = str(getattr(cfg_node, "feedback_mode", "region") or "region")
        self.bad_pixel_weight = float(getattr(cfg_node, "bad_pixel_weight", 3.0))
        self.good_pixel_weight = float(getattr(cfg_node, "good_pixel_weight", 0.25))
        self.pixel_radius = int(getattr(cfg_node, "pixel_radius", 2))
        self.use_da3_structure = bool(getattr(cfg_node, "use_da3_structure", False))
        self.assert_no_lidar_supervision = bool(getattr(cfg_node, "assert_no_lidar_supervision", False))
        self.da3_edge_weight = float(getattr(cfg_node, "da3_edge_weight", 1.0))
        self.da3_ranking_weight = float(getattr(cfg_node, "da3_ranking_weight", 1.0))
        self.da3_side_weight = float(getattr(cfg_node, "da3_side_weight", 0.5))
        self.da3_edge_margin = float(getattr(cfg_node, "da3_edge_margin", 0.05))
        self.da3_ranking_margin = float(getattr(cfg_node, "da3_ranking_margin", 0.02))
        self.regions_by_view = defaultdict(list)
        self.pixel_feedback_by_view = defaultdict(list)
        self.summary = {
            "enabled": self.enabled,
            "signal_path": self.signal_path,
            "supervision_mode": self.supervision_mode,
            "feedback_mode": self.feedback_mode,
            "use_lidar_depth": self.use_lidar_depth,
            "use_da3_structure": self.use_da3_structure,
            "bad_region_count": 0,
            "bad_contributor_count": 0,
            "good_contributor_count": 0,
            "low_evidence_region_count": 0,
            "bad_pixel_count": 0,
            "good_pixel_count": 0,
            "loaded": False,
            "waiting_for_signal": False,
        }
        if self.enabled and self.feedback_mode != "global" and self.signal_path:
            self._load()
        elif self.enabled and self.feedback_mode != "global":
            self.summary["waiting_for_signal"] = True

    def validate_supervision(self, optim_args):
        if not self.enabled:
            return
        mode = self.supervision_mode
        allowed = {"lidar_supervised", "da3_unsupervised", "hybrid_reference"}
        if mode not in allowed:
            raise ValueError(f"unknown guided_feedback.supervision_mode={mode!r}; expected one of {sorted(allowed)}")

        lidar_weight = float(getattr(optim_args, "lambda_depth_lidar", 0.0))
        status = {
            "supervision_mode": mode,
            "lambda_depth_lidar": lidar_weight,
            "use_lidar_depth": self.use_lidar_depth,
            "use_da3_structure": self.use_da3_structure,
            "assert_no_lidar_supervision": self.assert_no_lidar_supervision,
        }

        if mode == "da3_unsupervised":
            violations = []
            if abs(lidar_weight) > 1e-12:
                violations.append(f"optim.lambda_depth_lidar must be 0, got {lidar_weight}")
            if self.use_lidar_depth:
                violations.append("train.guided_feedback.use_lidar_depth must be False")
            if not self.use_da3_structure:
                violations.append("train.guided_feedback.use_da3_structure must be True")
            if violations:
                raise ValueError(
                    "Invalid DA3-unsupervised guided feedback configuration: "
                    + "; ".join(violations)
                )
            print("[GuidedFeedback] LiDAR supervision disabled; LiDAR is used for evaluation only.")
        elif mode == "lidar_supervised":
            if self.use_da3_structure:
                print("[GuidedFeedback] lidar_supervised mode: DA3 structure is enabled only if explicitly configured.")
        elif mode == "hybrid_reference":
            print("[GuidedFeedback] hybrid_reference mode: LiDAR and DA3 feedback may both be active; use as reference only.")

        if self.assert_no_lidar_supervision and (abs(lidar_weight) > 1e-12 or self.use_lidar_depth):
            raise ValueError("assert_no_lidar_supervision=True but LiDAR depth supervision is enabled.")
        print(f"[GuidedFeedback] supervision check: {status}")

    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load(self):
        if not self.signal_path:
            self.summary["loaded"] = False
            self.summary["waiting_for_signal"] = True
            return
        if not os.path.exists(self.signal_path):
            raise FileNotFoundError(f"guided feedback signal not found: {self.signal_path}")
        self.regions_by_view = defaultdict(list)
        self.pixel_feedback_by_view = defaultdict(list)
        self.summary["waiting_for_signal"] = False
        signal = self._load_json(self.signal_path)
        self.summary["bad_contributor_count"] = len(signal.get("bad_contributors", []))
        self.summary["good_contributor_count"] = len(signal.get("good_contributors", []))
        self.summary["low_evidence_region_count"] = len(signal.get("low_evidence_regions", []))
        self._load_pixel_feedback(signal)

        bad_region_keys = {
            item.get("region_key")
            for item in signal.get("bad_contributors", [])
            if item.get("region_key")
        }
        if not bad_region_keys:
            bad_region_keys = {
                item.get("region_key")
                for item in signal.get("regions", [])
                if item.get("region_key")
                and item.get("evidence_status", "ok") != "low_evidence"
            }
        summary_path = os.path.join(os.path.dirname(self.signal_path), "contribution_responsibility_all_views_summary.json")
        if os.path.exists(summary_path):
            all_views = self._load_json(summary_path)
            for frame in all_views.get("frames", []):
                region_key = f"{frame.get('stem')}:region{frame.get('region_id')}"
                if region_key not in bad_region_keys or frame.get("status") != "ok":
                    continue
                input_region = frame.get("input_region", {}) or {}
                bbox = input_region.get("bbox") or frame.get("pixel_bbox")
                if not bbox or len(bbox) != 4:
                    continue
                self.regions_by_view[str(frame.get("stem"))].append(
                    {
                        "region_key": region_key,
                        "bbox": [int(v) for v in bbox],
                        "region_type": frame.get("region_type"),
                        "selected_pixel_count": int(frame.get("selected_pixel_count", 0)),
                        "bad_count": int(frame.get("counterfactual_label_counts", {}).get("bad_contributor", 0)),
                    }
                )
        else:
            for item in signal.get("regions", []):
                region_key = item.get("region_key")
                if region_key not in bad_region_keys:
                    continue
                bbox = item.get("bbox")
                view_id = item.get("view_id")
                if not bbox or len(bbox) != 4 or not view_id:
                    continue
                self.regions_by_view[str(view_id)].append(
                    {
                        "region_key": region_key,
                        "bbox": [int(v) for v in bbox],
                        "region_type": item.get("region_type"),
                        "selected_pixel_count": int(item.get("valid_lidar_high_error_pixels", 0)),
                        "bad_count": 1,
                    }
                )
        self.summary["bad_region_count"] = int(sum(len(v) for v in self.regions_by_view.values()))
        self.summary["loaded"] = True

    def update_signal_path(self, signal_path):
        self.signal_path = str(signal_path or "")
        self.summary["signal_path"] = self.signal_path
        if self.feedback_mode != "global":
            self._load()
        return self.summary

    def _load_pixel_feedback(self, signal):
        for view_record in signal.get("pixel_feedback_by_view", []):
            view_id = str(view_record.get("view_id", ""))
            if not view_id:
                continue
            for item in view_record.get("bad_pixels", []):
                if len(item) < 2:
                    continue
                item_weight = float(item[2]) if len(item) >= 3 else self.bad_pixel_weight
                self.pixel_feedback_by_view[view_id].append(
                    {"xy": [int(item[0]), int(item[1])], "kind": "bad", "weight": item_weight}
                )
            for item in view_record.get("good_pixels", []):
                if len(item) < 2:
                    continue
                item_weight = float(item[2]) if len(item) >= 3 else self.good_pixel_weight
                self.pixel_feedback_by_view[view_id].append(
                    {"xy": [int(item[0]), int(item[1])], "kind": "good", "weight": item_weight}
                )
        self.summary["bad_pixel_count"] = int(
            sum(1 for records in self.pixel_feedback_by_view.values() for item in records if item["kind"] == "bad")
        )
        self.summary["good_pixel_count"] = int(
            sum(1 for records in self.pixel_feedback_by_view.values() for item in records if item["kind"] == "good")
        )

    def make_region_weight_map(self, camera, image_shape, device):
        if not self.enabled:
            return None, {}
        if self.feedback_mode == "global":
            h, w = int(image_shape[-2]), int(image_shape[-1])
            weight = torch.ones((1, h, w), device=device, dtype=torch.float32)
            return weight, {
                "guided_region_count": 0,
                "guided_region_pixels": int(h * w),
                "guided_view": str(getattr(camera, "image_name", "")),
            }
        if self.feedback_mode in {"contribution", "contribution_specific", "pixel"}:
            return self._make_pixel_weight_map(camera, image_shape, device)
        view_id = str(getattr(camera, "image_name", ""))
        regions = self.regions_by_view.get(view_id, [])
        if not regions:
            return None, {"guided_region_count": 0, "guided_region_pixels": 0}
        h, w = int(image_shape[-2]), int(image_shape[-1])
        weight = torch.ones((1, h, w), device=device, dtype=torch.float32)
        for region in regions:
            x0, y0, x1, y1 = region["bbox"]
            x0 = max(0, min(w, x0))
            x1 = max(0, min(w, x1))
            y0 = max(0, min(h, y0))
            y1 = max(0, min(h, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            weight[:, y0:y1, x0:x1] = torch.maximum(
                weight[:, y0:y1, x0:x1],
                torch.tensor(1.0 + self.region_weight, device=device, dtype=torch.float32),
            )
        guided_pixels = int(torch.count_nonzero(weight > 1.0).item())
        return weight, {
            "guided_region_count": int(len(regions)),
            "guided_region_pixels": guided_pixels,
            "guided_view": view_id,
        }

    def _make_pixel_weight_map(self, camera, image_shape, device):
        view_id = str(getattr(camera, "image_name", ""))
        records = self.pixel_feedback_by_view.get(view_id, [])
        if not records:
            return None, {"guided_region_count": 0, "guided_region_pixels": 0}
        h, w = int(image_shape[-2]), int(image_shape[-1])
        weight = torch.ones((1, h, w), device=device, dtype=torch.float32)
        radius = max(0, self.pixel_radius)
        for record in records:
            x, y = record["xy"]
            x0 = max(0, min(w, x - radius))
            x1 = max(0, min(w, x + radius + 1))
            y0 = max(0, min(h, y - radius))
            y1 = max(0, min(h, y + radius + 1))
            if x1 <= x0 or y1 <= y0:
                continue
            if record["kind"] == "good":
                good_weight = max(0.0, min(1.0, float(record.get("weight", self.good_pixel_weight))))
                weight[:, y0:y1, x0:x1] = torch.minimum(
                    weight[:, y0:y1, x0:x1],
                    torch.tensor(good_weight, device=device, dtype=torch.float32),
                )
            else:
                bad_weight = max(0.0, float(record.get("weight", self.bad_pixel_weight)))
                weight[:, y0:y1, x0:x1] = torch.maximum(
                    weight[:, y0:y1, x0:x1],
                    torch.tensor(1.0 + bad_weight, device=device, dtype=torch.float32),
                )
        guided_pixels = int(torch.count_nonzero(weight > 1.0).item())
        protected_pixels = int(torch.count_nonzero(weight < 1.0).item())
        return weight, {
            "guided_region_count": int(len(records)),
            "guided_region_pixels": guided_pixels,
            "protected_pixels": protected_pixels,
            "guided_view": view_id,
        }


def make_guided_feedback_controller(cfg_node):
    return GuidedFeedbackController(cfg_node)
