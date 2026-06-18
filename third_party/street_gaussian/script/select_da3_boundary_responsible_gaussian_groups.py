import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def stable_namespace_id(model_name):
    import zlib

    if model_name == "background":
        return 0
    if model_name.startswith("obj_"):
        try:
            return 1 + int(model_name.split("_", 1)[1])
        except ValueError:
            pass
    return 1_000_000 + (zlib.crc32(model_name.encode("utf-8")) % 8_000_000)


def load_single_gaussian_dryrun(path):
    if not path or not os.path.exists(path):
        return {}
    payload = read_json(path)
    lookup = defaultdict(list)
    for item in payload:
        lookup[(str(item.get("view_id")), str(item.get("region_id")), int(item.get("stable_gaussian_id", -1)))].append(item)
    return lookup


def load_npz_frame(frame):
    npz_path = frame.get("paths", {}).get("npz", "")
    if not npz_path or not os.path.exists(npz_path):
        return None
    return np.load(npz_path, allow_pickle=True)


def frame_key(frame):
    return f"{frame.get('stem')}:region{frame.get('region_id')}"


def aggregate_gaussian_records(npz, frame, top_n=None):
    ids = np.asarray(npz["cuda_contribution_ids"], dtype=np.int64)
    weights = np.asarray(npz["contribution_weights"], dtype=np.float32)
    alpha = np.asarray(npz["cuda_alpha"], dtype=np.float32) if "cuda_alpha" in npz else weights
    trans = np.asarray(npz["cuda_transmittance"], dtype=np.float32) if "cuda_transmittance" in npz else np.ones_like(weights)
    depth = np.asarray(npz["cuda_depth"], dtype=np.float32) if "cuda_depth" in npz else np.zeros_like(weights)
    order = np.asarray(npz["cuda_depth_order"], dtype=np.int32) if "cuda_depth_order" in npz else np.zeros_like(ids)
    selected = np.asarray(npz["selected_pixels"], dtype=np.int64)
    stable_matrix = np.asarray(npz["stable_gaussian_ids"], dtype=np.int64) if "stable_gaussian_ids" in npz else None
    model_names_matrix = np.asarray(npz["model_names"]) if "model_names" in npz else None
    stable_by_row = {}
    if "candidate_view_local_indices" in npz and "candidate_gaussian_ids" in npz:
        local_rows = np.asarray(npz["candidate_view_local_indices"], dtype=np.int64).reshape(-1)
        stable_ids = np.asarray(npz["candidate_gaussian_ids"], dtype=np.int64).reshape(-1)
        stable_by_row = {int(r): int(g) for r, g in zip(local_rows, stable_ids)}
    risk = np.ones((ids.shape[0],), dtype=np.float32)
    if "da3_risk" in npz:
        risk = np.asarray(npz["da3_risk"], dtype=np.float32).reshape(-1)[: ids.shape[0]]
    elif "geometry_errors" in npz:
        raw = np.asarray(npz["geometry_errors"], dtype=np.float32).reshape(-1)[: ids.shape[0]]
        if np.any(np.isfinite(raw)):
            denom = max(float(np.nanpercentile(raw, 95)), 1e-6)
            risk = np.clip(raw / denom, 0.0, 1.0).astype(np.float32)
    rows = {}
    for p_idx in range(ids.shape[0]):
        for k in range(ids.shape[1]):
            row = int(ids[p_idx, k])
            if row < 0:
                continue
            stable_gid = int(stable_matrix[p_idx, k]) if stable_matrix is not None and stable_matrix.shape == ids.shape else int(stable_by_row.get(row, row))
            w = float(weights[p_idx, k])
            if w <= 0:
                continue
            rec = rows.setdefault(
                row,
                {
                    "view_local_index": row,
                    "stable_gaussian_id": stable_gid,
                    "model_name": "unknown",
                    "model_local_index": int(row),
                    "pixel_indices": [],
                    "pixel_x_values": [],
                    "pixel_y_values": [],
                    "raw_weight_sum": 0.0,
                    "risk_weighted_sum": 0.0,
                    "max_talpha": 0.0,
                    "alpha_values": [],
                    "transmittance_values": [],
                    "depth_values": [],
                    "depth_orders": [],
                },
            )
            rec["pixel_indices"].append(int(p_idx))
            if selected.ndim == 2 and p_idx < selected.shape[0]:
                rec["pixel_x_values"].append(int(selected[p_idx, 0]))
                rec["pixel_y_values"].append(int(selected[p_idx, 1]))
            if model_names_matrix is not None and model_names_matrix.shape == ids.shape:
                rec["model_name"] = str(model_names_matrix[p_idx, k])
            rec["raw_weight_sum"] += w
            rec["risk_weighted_sum"] += w * float(risk[p_idx])
            rec["max_talpha"] = max(rec["max_talpha"], w)
            rec["alpha_values"].append(float(alpha[p_idx, k]))
            rec["transmittance_values"].append(float(trans[p_idx, k]))
            rec["depth_values"].append(float(depth[p_idx, k]))
            rec["depth_orders"].append(int(order[p_idx, k]))
    # Existing CUDA npz stores row ids, not stable ids. Try to use per-pixel json names if present is not cheap here,
    # so make a deterministic row namespace. Downstream should prefer stable ids from newer dumps when available.
    for rec in rows.values():
        rec["support_count"] = int(len(set(rec["pixel_indices"])))
        rec["mean_talpha"] = float(rec["raw_weight_sum"] / max(rec["support_count"], 1))
        rec["mean_risk_weighted"] = float(rec["risk_weighted_sum"] / max(rec["support_count"], 1))
        rec["support_factor"] = float(math.log1p(rec["support_count"]))
        rec["score"] = float(rec["risk_weighted_sum"] * rec["support_factor"])
        rec["mean_depth_order"] = float(np.mean(rec["depth_orders"])) if rec["depth_orders"] else None
        rec["min_depth_order"] = int(np.min(rec["depth_orders"])) if rec["depth_orders"] else None
        rec["max_depth_order"] = int(np.max(rec["depth_orders"])) if rec["depth_orders"] else None
        rec["mean_depth"] = float(np.mean(rec["depth_values"])) if rec["depth_values"] else None
        rec["mean_x"] = float(np.mean(rec["pixel_x_values"])) if rec["pixel_x_values"] else None
        rec["mean_y"] = float(np.mean(rec["pixel_y_values"])) if rec["pixel_y_values"] else None
        rec["pixel_bbox"] = [
            int(np.min(rec["pixel_x_values"])),
            int(np.min(rec["pixel_y_values"])),
            int(np.max(rec["pixel_x_values"])) + 1,
            int(np.max(rec["pixel_y_values"])) + 1,
        ] if rec["pixel_x_values"] and rec["pixel_y_values"] else None
        rec["mean_alpha"] = float(np.mean(rec["alpha_values"])) if rec["alpha_values"] else None
        rec["mean_transmittance"] = float(np.mean(rec["transmittance_values"])) if rec["transmittance_values"] else None
        rec["selected_pixel_count"] = int(len(selected))
        rec["view_id"] = str(frame.get("stem"))
        rec["region_id"] = str(frame.get("region_id"))
    records = sorted(rows.values(), key=lambda r: r["score"], reverse=True)
    return records[:top_n] if top_n else records


def make_groups(records, args):
    if not records:
        return []
    groups = defaultdict(list)
    for rec in records:
        order = rec.get("mean_depth_order")
        if order is None:
            depth_bucket = "unknown_order"
        else:
            depth_bucket = f"order_{int(order // max(args.depth_order_bin, 1))}"
        if rec.get("mean_x") is None or rec.get("mean_y") is None:
            spatial_bucket = "unknown_cell"
        else:
            cell = max(int(args.spatial_cell_size), 1)
            spatial_bucket = f"cell_{int(rec['mean_x'] // cell)}_{int(rec['mean_y'] // cell)}"
        model_bucket = str(rec.get("model_name") or "unknown")
        bucket = f"{model_bucket}|{spatial_bucket}|{depth_bucket}"
        groups[bucket].append(rec)
    out = []
    for idx, (bucket, members) in enumerate(groups.items()):
        pixel_sets = [set(m["pixel_indices"]) for m in members]
        shared = len(set.intersection(*pixel_sets)) if pixel_sets else 0
        union = len(set.union(*pixel_sets)) if pixel_sets else 0
        order_values = [v for m in members for v in m["depth_orders"]]
        order_min = int(np.min(order_values)) if order_values else None
        order_max = int(np.max(order_values)) if order_values else None
        cross_boundary = bool(order_min is not None and order_max is not None and (order_max - order_min) >= args.cross_boundary_order_span)
        risk_weighted = float(sum(m["risk_weighted_sum"] for m in members))
        raw = float(sum(m["raw_weight_sum"] for m in members))
        support = int(union)
        side_mixing = cross_boundary or len(members) >= args.min_group_size_for_mixing
        label = "neutral_group"
        if support < args.min_group_support:
            label = "low_evidence_group"
        elif side_mixing and risk_weighted >= args.bad_group_score:
            label = "bad_boundary_mixing_group"
        elif risk_weighted >= args.bad_group_score:
            label = "bad_edge_blurring_group"
        elif raw >= args.good_support_score and risk_weighted < args.bad_group_score:
            label = "good_boundary_support_group"
        out.append(
            {
                "group_id": idx,
                "group_key": bucket,
                "stable_gaussian_ids": [int(m["stable_gaussian_id"]) for m in members],
                "view_local_indices": [int(m["view_local_index"]) for m in members],
                "member_count": len(members),
                "aggregation_rule": "model_namespace + image_spatial_cell + depth_order_bin",
                "spatial_cell_size": int(args.spatial_cell_size),
                "mean_screen_xy": [
                    float(np.mean([m["mean_x"] for m in members if m.get("mean_x") is not None])) if any(m.get("mean_x") is not None for m in members) else None,
                    float(np.mean([m["mean_y"] for m in members if m.get("mean_y") is not None])) if any(m.get("mean_y") is not None for m in members) else None,
                ],
                "pixel_bbox": _merge_bboxes([m.get("pixel_bbox") for m in members]),
                "support_pixels": support,
                "shared_pixels": int(shared),
                "group_raw_talpha_sum": raw,
                "group_risk_weighted_contribution": risk_weighted,
                "group_max_talpha": float(max(m["max_talpha"] for m in members)),
                "group_mean_talpha": float(np.mean([m["mean_talpha"] for m in members])),
                "depth_order_min": order_min,
                "depth_order_max": order_max,
                "depth_order_range": int(order_max - order_min) if order_min is not None and order_max is not None else None,
                "crosses_da3_boundary": cross_boundary,
                "possible_side_mixing": bool(side_mixing),
                "rgb_edge_support": None,
                "lidar_jump_support_reference_only": None,
                "group_label": label,
            }
        )
    out.sort(key=lambda g: g["group_risk_weighted_contribution"], reverse=True)
    return out


def _merge_bboxes(bboxes):
    bboxes = [b for b in bboxes if b and len(b) == 4]
    if not bboxes:
        return None
    return [
        int(min(b[0] for b in bboxes)),
        int(min(b[1] for b in bboxes)),
        int(max(b[2] for b in bboxes)),
        int(max(b[3] for b in bboxes)),
    ]


def attach_group_counterfactual(groups, frame, dryrun_lookup):
    for group in groups:
        deltas = []
        rgb_deltas = []
        tags = []
        for gid in group["stable_gaussian_ids"]:
            for item in dryrun_lookup.get((str(frame.get("stem")), str(frame.get("region_id")), int(gid)), []):
                if item.get("da3_structure_delta") is not None:
                    deltas.append(float(item["da3_structure_delta"]))
                if item.get("rgb_patch_mae_delta") is not None:
                    rgb_deltas.append(float(item["rgb_patch_mae_delta"]))
                tags.append(str(item.get("tag")))
        group["counterfactual_mode"] = "aggregated_single_gaussian_dryrun" if tags else "not_available"
        group["da3_structure_delta_mean"] = float(np.mean(deltas)) if deltas else None
        group["rgb_patch_mae_delta_mean"] = float(np.mean(rgb_deltas)) if rgb_deltas else None
        group["single_gaussian_counterfactual_tags"] = dict(Counter(tags))
        if tags and "good_contributor" in tags:
            group["group_label"] = "rgb_protect_group"
        elif tags and "bad_contributor" in tags and group["possible_side_mixing"]:
            group["group_label"] = "bad_boundary_mixing_group"
        elif tags and "bad_contributor" in tags:
            group["group_label"] = "bad_edge_blurring_group"
    return groups


def future_action(group):
    label = group["group_label"]
    if label in {"good_boundary_support_group", "rgb_protect_group"}:
        return "protect"
    if label == "bad_boundary_mixing_group":
        return "shrink_candidate"
    if label == "bad_edge_blurring_group":
        return "opacity_regularization_candidate"
    if label == "bad_ranking_conflict_group":
        return "split_candidate"
    return "skip"


def write_csv(path, rows):
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_overlay(path, frame, groups):
    img = np.zeros((320, 640, 3), dtype=np.uint8)
    cv2.putText(img, str(frame_key(frame)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y = 80
    for group in groups[:8]:
        txt = f"g{group['group_id']} {group['group_label']} n={group['member_count']} s={group['group_risk_weighted_contribution']:.2f}"
        cv2.putText(img, txt[:90], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        y += 28
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contribution-summary", default="output/local_feedback/da3_boundary_contribution_debug_A5000_top30_regionpixels/contribution_responsibility_all_views_summary.json")
    parser.add_argument("--single-gaussian-dryrun", default="output/local_feedback/da3_structure_counterfactual_dryrun_A5000_top30/counterfactual_candidates.json")
    parser.add_argument("--output-dir", default="output/local_feedback/da3_boundary_responsible_groups_A5000_top30")
    parser.add_argument("--max-regions", type=int, default=30)
    parser.add_argument("--top-gaussians-per-region", type=int, default=50)
    parser.add_argument("--depth-order-bin", type=int, default=2)
    parser.add_argument("--spatial-cell-size", type=int, default=96)
    parser.add_argument("--cross-boundary-order-span", type=int, default=3)
    parser.add_argument("--min-group-support", type=int, default=8)
    parser.add_argument("--min-group-size-for-mixing", type=int, default=3)
    parser.add_argument("--bad-group-score", type=float, default=1.0)
    parser.add_argument("--good-support-score", type=float, default=5.0)
    parser.add_argument("--save-visuals", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    summary = read_json(args.contribution_summary)
    dryrun_lookup = load_single_gaussian_dryrun(args.single_gaussian_dryrun)
    all_groups = []
    low = []
    for frame in summary.get("frames", [])[: args.max_regions]:
        if frame.get("status") not in {"ok", "valid"}:
            low.append({"region_key": frame_key(frame), "reason": frame.get("status", "not_ok")})
            continue
        npz = load_npz_frame(frame)
        if npz is None:
            low.append({"region_key": frame_key(frame), "reason": "missing_contribution_npz"})
            continue
        records = aggregate_gaussian_records(npz, frame, top_n=args.top_gaussians_per_region)
        groups = make_groups(records, args)
        groups = attach_group_counterfactual(groups, frame, dryrun_lookup)
        for g in groups:
            g["view_id"] = str(frame.get("stem"))
            g["region_id"] = str(frame.get("region_id"))
            g["region_key"] = frame_key(frame)
            g["future_action_tag"] = future_action(g)
            g["uses_lidar_for_labeling"] = False
            g["uses_lidar_for_evaluation_only"] = True
            all_groups.append(g)
        if args.save_visuals:
            save_overlay(str(out_dir / "overlays" / f"{frame.get('stem')}_region{frame.get('region_id')}.png"), frame, groups)

    by_gid = defaultdict(list)
    for g in all_groups:
        for gid in g["stable_gaussian_ids"]:
            by_gid[int(gid)].append(g)
    multiview_rows = []
    for gid, groups in by_gid.items():
        labels = Counter(g["group_label"] for g in groups)
        actions = Counter(g["future_action_tag"] for g in groups)
        risk_scores = [float(g["group_risk_weighted_contribution"]) for g in groups]
        final_conf = min(1.0, math.log1p(sum(risk_scores)) / 5.0 + 0.1 * len(set(g["view_id"] for g in groups)))
        if labels.get("rgb_protect_group", 0) or actions.get("protect", 0):
            action = "protect"
        elif labels.get("bad_boundary_mixing_group", 0):
            action = "shrink_candidate"
        elif labels.get("bad_edge_blurring_group", 0):
            action = "opacity_regularization_candidate"
        else:
            action = "skip"
        multiview_rows.append(
            {
                "stable_gaussian_id": gid,
                "view_count": len(set(g["view_id"] for g in groups)),
                "region_count": len(set(g["region_key"] for g in groups)),
                "risk_weighted_contribution_mean": float(np.mean(risk_scores)),
                "risk_weighted_contribution_max": float(np.max(risk_scores)),
                "bad_group_evidence_count": int(sum(v for k, v in labels.items() if k.startswith("bad_"))),
                "protect_evidence_count": int(labels.get("rgb_protect_group", 0) + labels.get("good_boundary_support_group", 0)),
                "rgb_risky_count": int(labels.get("rgb_protect_group", 0)),
                "final_confidence": float(final_conf),
                "future_action_tag": action,
            }
        )
    multiview_rows.sort(key=lambda r: (r["final_confidence"], r["risk_weighted_contribution_max"]), reverse=True)

    label_counts = Counter(g["group_label"] for g in all_groups)
    action_counts = Counter(g["future_action_tag"] for g in all_groups)
    payload = {
        "method": "DA3 boundary-aware Gaussian group responsibility",
        "counterfactual_mode": "group labels use contribution grouping plus optional aggregated single-Gaussian dry-run evidence",
        "uses_lidar_for_labeling": False,
        "uses_lidar_for_evaluation_only": True,
        "gaussian_parameters_modified": False,
        "counts": {
            "group_count": len(all_groups),
            "low_evidence_groups": len(low),
            "labels": dict(label_counts),
            "future_actions": dict(action_counts),
            "stable_gaussian_count": len(multiview_rows),
        },
        "thresholds": vars(args),
    }
    write_json(str(out_dir / "da3_boundary_responsible_groups.json"), all_groups)
    write_csv(str(out_dir / "da3_boundary_responsible_groups.csv"), all_groups)
    write_csv(str(out_dir / "stable_gaussian_group_multiview_table.csv"), multiview_rows)
    write_json(str(out_dir / "group_counterfactual_summary.json"), payload)
    write_json(str(out_dir / "low_evidence_groups.json"), low)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
