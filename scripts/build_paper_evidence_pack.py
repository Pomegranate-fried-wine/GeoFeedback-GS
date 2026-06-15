#!/usr/bin/env python3
"""Build a paper-ready evidence directory from GeoGuardGS experiment outputs.

The script is intentionally conservative: it never fabricates metrics. Missing
files are recorded in missing_evidence reports so a finished run can be audited
before writing paper claims.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path


EXPERIMENT_ORDER = [
    "streetgs_original_baseline",
    "baseline_streetgs",
    "baseline_streetgs_colmap_5000",
    "da3_only_full_scene_lidar_init",
    "da3_only",
    "da3_only_colmap_5000",
    "da3_periodic_group_softpatch_full_scene_lidar_init",
    "da3_periodic_group_softpatch",
    "da3_periodic_group_softpatch_colmap_5000",
    "da3_periodic_group_softpatch_opacity_reg",
    "da3_periodic_group_softpatch_opacity_decay",
    "lidar_init_streetgs_reference",
    "lidar_supervised_reference",
    "hybrid_reference",
]

METRIC_KEYS = ["AbsRel", "RMSE", "MAE", "delta_lt_1_25", "PSNR", "SSIM", "LPIPS", "rgb_l1", "rgb_mae"]
REGION_KEYS = [
    "all_valid",
    "boundary_band",
    "rendered_depth_edge_band",
    "prior_depth_edge_band",
    "thin_structure_band",
    "stable_non_boundary",
]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def maybe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def latest_file(paths):
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def flatten_metrics(prefix, payload):
    out = {}
    if not isinstance(payload, dict):
        return out
    for key, value in payload.items():
        name = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_metrics(f"{name}.", value))
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[name] = value
    return out


def summarize_metric_rows(rows):
    summary = {}
    if not rows:
        return summary
    for key in rows[0].keys():
        values = [maybe_float(row.get(key)) for row in rows]
        values = [v for v in values if v is not None]
        if values:
            summary[f"{key}_final"] = values[-1]
            summary[f"{key}_mean"] = sum(values) / len(values)
            summary[f"{key}_best_min"] = min(values)
            summary[f"{key}_best_max"] = max(values)
    return summary


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def find_experiment_dirs(root):
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    order = {name: idx for idx, name in enumerate(EXPERIMENT_ORDER)}
    return sorted(dirs, key=lambda p: (order.get(p.name, 999), p.name))


def collect_final_metrics(exp_dir):
    row = {}
    sources = []
    candidates = [
        exp_dir / "final_eval" / "metrics.json",
        exp_dir / "metrics" / "summary.json",
        exp_dir / "metrics.json",
    ]
    for path in candidates:
        if path.exists():
            payload = read_json(path)
            row.update(flatten_metrics("", payload))
            sources.append(str(path))
    for name in ["rgb_metrics.csv", "lidar_geometry_metrics.csv", "da3_structure_metrics.csv", "boundary_metrics.csv"]:
        path = exp_dir / "metrics" / name
        if path.exists():
            rows = read_csv_rows(path)
            row.update({f"{name}.{k}": v for k, v in summarize_metric_rows(rows).items()})
            sources.append(str(path))
    row["metric_sources"] = ";".join(sources)
    return row


def collect_eval_rows(exp_name, exp_dir):
    rows = []
    summary_path = exp_dir / "metrics" / "eval_summary.csv"
    if summary_path.exists():
        for row in read_csv_rows(summary_path):
            rows.append({"experiment": exp_name, "source": str(summary_path), **row})
    return rows


def collect_final_evaluation_rows(final_eval_root):
    rows = []
    summary_main = final_eval_root / "summary_main.csv"
    summary_by_scope = final_eval_root / "summary_by_scope.csv"
    if summary_main.exists():
        for row in read_csv_rows(summary_main):
            rows.append({"source": str(summary_main), "table": "summary_main", **row})
    if summary_by_scope.exists():
        for row in read_csv_rows(summary_by_scope):
            rows.append({"source": str(summary_by_scope), "table": "summary_by_scope", **row})
    return rows


def collect_latest_per_view_rows(exp_name, exp_dir):
    latest = latest_file(list((exp_dir / "metrics").glob("eval_iter_*_per_view.csv")))
    rows = []
    if latest:
        for row in read_csv_rows(latest):
            rows.append({"experiment": exp_name, "source": str(latest), **row})
    return rows


def collect_scalar_trace_rows(exp_name, exp_dir):
    path = exp_dir / "metrics" / "train_loss_trace.csv"
    rows = []
    if path.exists():
        for row in read_csv_rows(path):
            rows.append({"experiment": exp_name, "source": str(path), **row})
    return rows


def collect_initialization_row(exp_name, exp_dir):
    manifest = exp_dir / "input_ply" / "initialization_manifest.json"
    row = {
        "experiment": exp_name,
        "manifest": str(manifest) if manifest.exists() else "",
        "status": "present" if manifest.exists() else "missing",
        "uses_lidar_training_supervision": "",
        "uses_lidar_initialization": "",
        "initialization_source": "",
        "pointcloud_source": "",
        "colmap_binary": "",
        "colmap_point_count": "",
        "lidar_point_count_used_for_init": "",
        "no_lidar_leakage": "unknown",
    }
    if manifest.exists():
        payload = read_json(manifest)
        for key in [
            "uses_lidar_training_supervision",
            "uses_lidar_initialization",
            "initialization_source",
            "pointcloud_source",
            "colmap_binary",
            "colmap_point_count",
            "lidar_point_count_used_for_init",
        ]:
            row[key] = payload.get(key, "")
        row["no_lidar_leakage"] = "fail" if payload.get("uses_lidar_initialization", False) else "pass"
    return row


def collect_region_rows(exp_name, exp_dir):
    rows = []
    paths = list((exp_dir / "metrics").glob("*region*.json")) + list((exp_dir / "final_eval").glob("*region*.json"))
    for path in paths:
        payload = read_json(path)
        records = payload.get("regions", payload if isinstance(payload, list) else [])
        if isinstance(records, dict):
            records = [{"region": key, **value} for key, value in records.items() if isinstance(value, dict)]
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            rows.append({
                "experiment": exp_name,
                "source": str(path),
                "region": record.get("region") or record.get("name") or record.get("region_name") or record.get("region_type") or "",
                "valid_lidar_count": record.get("valid_lidar_count", ""),
                "confidence": record.get("confidence", ""),
                "MAE": record.get("MAE", ""),
                "RMSE": record.get("RMSE", ""),
                "AbsRel": record.get("AbsRel", ""),
                "delta_lt_1_25": record.get("delta_lt_1_25", ""),
            })
    return rows


def collect_feedback_rows(exp_name, exp_dir, manifest_out):
    rows = []
    fc_dir = exp_dir / "feedback_controller"
    if not fc_dir.exists():
        return [{
            "experiment": exp_name,
            "iteration": "",
            "status": "not_applicable",
            "risk_source": "",
            "supervision_mode": "",
            "selected_pixels_count": "",
            "gaussian_group_count": "",
            "cuda_ok_count": "",
            "low_evidence_count": "",
            "live_cuda_contribution": "",
            "uses_lidar_supervision": "",
            "uses_lidar_selected_pixels": "",
            "uses_lidar_initialization": "",
            "initialization_source": "",
            "gaussian_parameters_modified": "",
            "real_repair_enabled": "",
            "manifest": "",
        }]
    for manifest in sorted(fc_dir.glob("iter_*/feedback_controller_manifest.json")):
        payload = read_json(manifest)
        iter_name = manifest.parent.name
        copy_file(manifest, manifest_out / exp_name / iter_name / manifest.name)
        for sibling in [
            "feedback_controller_audit.json",
            "audit_summary.json",
            "pipeline_stage_manifest.json",
            "responsible_group_summary.json",
            "active_feedback_summary.json",
        ]:
            p = manifest.parent / sibling
            if p.exists():
                copy_file(p, manifest_out / exp_name / iter_name / sibling)
        rows.append({
            "experiment": exp_name,
            "iteration": payload.get("iteration", ""),
            "status": payload.get("status", ""),
            "risk_source": payload.get("risk_source", ""),
            "supervision_mode": payload.get("supervision_mode", ""),
            "selected_pixels_count": payload.get("selected_pixels_count", ""),
            "gaussian_group_count": payload.get("gaussian_group_count", ""),
            "cuda_ok_count": payload.get("cuda_ok_count", ""),
            "low_evidence_count": payload.get("low_evidence_count", ""),
            "live_cuda_contribution": payload.get("live_cuda_contribution", ""),
            "uses_lidar_supervision": payload.get("uses_lidar_supervision", ""),
            "uses_lidar_selected_pixels": payload.get("uses_lidar_selected_pixels", ""),
            "uses_lidar_initialization": payload.get("uses_lidar_initialization", ""),
            "initialization_source": payload.get("initialization_source", ""),
            "gaussian_parameters_modified": payload.get("gaussian_parameters_modified", ""),
            "real_repair_enabled": payload.get("real_repair_enabled", ""),
            "manifest": str(manifest),
        })
    return rows


def collect_safety_rows(exp_name, exp_dir):
    rows = []
    for path in sorted(exp_dir.glob("feedback_controller/iter_*/*audit*.json")):
        payload = read_json(path)
        rows.append({
            "experiment": exp_name,
            "iteration": payload.get("iteration", path.parent.name.replace("iter_", "")),
            "source": str(path),
            "status": payload.get("status", ""),
            "uses_lidar_supervision": payload.get("uses_lidar_supervision", ""),
            "uses_lidar_selected_pixels": payload.get("uses_lidar_selected_pixels", ""),
            "gaussian_parameters_modified": payload.get("gaussian_parameters_modified", ""),
            "real_repair_enabled": payload.get("real_repair_enabled", ""),
            "allow_parameter_modification": payload.get("allow_parameter_modification", ""),
        })
    for path in sorted(exp_dir.glob("feedback_controller/iter_*/gaussian_control/*audit*.json")):
        payload = read_json(path)
        checks = payload.get("safety_checks", {})
        rows.append({
            "experiment": exp_name,
            "iteration": path.parents[1].name.replace("iter_", ""),
            "source": str(path),
            "status": payload.get("status", ""),
            "uses_lidar_supervision": "",
            "uses_lidar_selected_pixels": "",
            "gaussian_parameters_modified": payload.get("gaussian_parameters_modified", ""),
            "real_repair_enabled": payload.get("real_repair_enabled", ""),
            "allow_parameter_modification": payload.get("allow_parameter_modification", ""),
            "safety_checks": json.dumps(checks, ensure_ascii=False),
        })
    return rows


def collect_repair_rows(exp_name, exp_dir):
    rows = []
    for path in sorted(exp_dir.glob("feedback_controller/iter_*/gaussian_control/opacity_decay_apply/opacity_decay_apply_manifest.json")):
        payload = read_json(path)
        rows.append({
            "experiment": exp_name,
            "iteration": path.parents[2].name.replace("iter_", ""),
            "mode": "opacity_decay_apply",
            "source": str(path),
            "status": payload.get("status", ""),
            "modified_count": payload.get("modified_count", payload.get("decayed_count", "")),
            "protected_count": payload.get("protected_count", ""),
            "skipped_count": payload.get("skipped_count", ""),
            "rgb_delta": payload.get("rgb_delta", ""),
        })
    for path in sorted(exp_dir.glob("feedback_controller/iter_*/gaussian_control/repair_operator_manifest.json")):
        payload = read_json(path)
        rows.append({
            "experiment": exp_name,
            "iteration": path.parents[1].name.replace("iter_", ""),
            "mode": "repair_dryrun",
            "source": str(path),
            "status": payload.get("status", ""),
            "modified_count": 0,
            "protected_count": payload.get("protected_count", ""),
            "skipped_count": payload.get("skipped_count", ""),
            "candidate_count": payload.get("candidate_count", ""),
        })
    return rows


def collect_figures(exp_name, exp_dir, figure_out, max_per_category):
    copied = []
    categories = {
        "panels": ["*panel*.png", "*panel*.jpg", "panels/*.png", "panels/*.jpg"],
        "risk_maps": ["*risk*.png", "risk_stage/*.png", "risk_maps/*.png"],
        "contribution": ["*contribution*.png", "contribution/*.png"],
        "group_responsibility": ["*responsib*.png", "*group*.png"],
        "opacity_decay": ["opacity_decay_apply/*.png", "opacity_decay_apply/*.jpg"],
    }
    search_roots = [exp_dir / "final_eval", exp_dir / "periodic_eval", exp_dir / "feedback_controller"]
    for category, patterns in categories.items():
        seen = set()
        count = 0
        for root in search_roots:
            if not root.exists():
                continue
            for pattern in patterns:
                for src in sorted(root.glob(f"**/{pattern}")):
                    if src.suffix.lower() not in IMAGE_EXTS or src in seen:
                        continue
                    seen.add(src)
                    dst = figure_out / category / exp_name / src.name
                    if dst.exists():
                        dst = figure_out / category / exp_name / f"{src.parent.name}_{src.name}"
                    copied.append({"experiment": exp_name, "category": category, "source": str(src), "copied_to": copy_file(src, dst)})
                    count += 1
                    if count >= max_per_category:
                        break
                if count >= max_per_category:
                    break
            if count >= max_per_category:
                break
    return copied


def missing_items(exp_name, exp_dir, feedback_rows):
    rows = []
    required = [
        ("final_eval/metrics.json", exp_dir / "final_eval" / "metrics.json"),
        ("metrics/rgb_metrics.csv", exp_dir / "metrics" / "rgb_metrics.csv"),
        ("metrics/lidar_geometry_metrics.csv", exp_dir / "metrics" / "lidar_geometry_metrics.csv"),
    ]
    for label, path in required:
        if not path.exists():
            rows.append({"experiment": exp_name, "missing": label, "severity": "paper_table_gap"})
    if exp_name not in {"baseline_streetgs", "da3_only"} and not [r for r in feedback_rows if r.get("status") != "not_applicable"]:
        rows.append({"experiment": exp_name, "missing": "feedback_controller/iter_*/feedback_controller_manifest.json", "severity": "method_evidence_gap"})
    if not any((exp_dir / "final_eval").glob("**/*panel*.*")) and not any((exp_dir / "feedback_controller").glob("**/*panel*.*")):
        rows.append({"experiment": exp_name, "missing": "representative panels", "severity": "figure_gap"})
    if not (exp_dir / "input_ply" / "initialization_manifest.json").exists():
        rows.append({"experiment": exp_name, "missing": "input_ply/initialization_manifest.json", "severity": "initialization_audit_gap"})
    if not (exp_dir / "metrics" / "eval_summary.csv").exists():
        rows.append({"experiment": exp_name, "missing": "metrics/eval_summary.csv", "severity": "paper_table_gap"})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/a100_main_experiments")
    parser.add_argument("--paper-dir", default="outputs/paper_evidence_full_scene_v2")
    parser.add_argument("--final-eval-root", default="outputs/final_evaluation_full_scene_v2")
    parser.add_argument("--max-figures-per-category", type=int, default=12)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    paper_dir = Path(args.paper_dir)
    final_eval_root = Path(args.final_eval_root)
    table_dir = paper_dir / "tables"
    figure_dir = paper_dir / "figures"
    manifest_dir = paper_dir / "manifests"
    summary_dir = paper_dir / "summaries"

    experiment_rows = []
    final_rows = []
    region_rows = []
    feedback_rows_all = []
    safety_rows = []
    repair_rows = []
    figure_rows = []
    missing_rows = []
    eval_rows = []
    eval_per_view_rows = []
    scalar_trace_rows = []
    initialization_rows = []

    for exp_dir in find_experiment_dirs(output_root):
        exp_name = exp_dir.name
        feedback_rows = collect_feedback_rows(exp_name, exp_dir, manifest_dir)
        metrics = collect_final_metrics(exp_dir)
        init_row = collect_initialization_row(exp_name, exp_dir)
        final_row = {"experiment": exp_name, **metrics, **{f"init.{k}": v for k, v in init_row.items() if k != "experiment"}}
        final_rows.append(final_row)
        initialization_rows.append(init_row)
        eval_rows.extend(collect_eval_rows(exp_name, exp_dir))
        eval_per_view_rows.extend(collect_latest_per_view_rows(exp_name, exp_dir))
        scalar_trace_rows.extend(collect_scalar_trace_rows(exp_name, exp_dir))
        region_rows.extend(collect_region_rows(exp_name, exp_dir))
        feedback_rows_all.extend(feedback_rows)
        safety_rows.extend(collect_safety_rows(exp_name, exp_dir))
        repair_rows.extend(collect_repair_rows(exp_name, exp_dir))
        figure_rows.extend(collect_figures(exp_name, exp_dir, figure_dir, args.max_figures_per_category))
        missing_rows.extend(missing_items(exp_name, exp_dir, feedback_rows))
        feedback_trigger_count = len([row for row in feedback_rows if row.get("status") != "not_applicable"])
        experiment_rows.append({
            "experiment": exp_name,
            "path": str(exp_dir),
            "has_final_metrics": bool(metrics),
            "initialization_status": init_row.get("status", ""),
            "uses_lidar_initialization": init_row.get("uses_lidar_initialization", ""),
            "initialization_source": init_row.get("initialization_source", ""),
            "feedback_trigger_count": feedback_trigger_count,
            "figure_count": len([row for row in figure_rows if row["experiment"] == exp_name]),
        })

    final_fields = sorted({key for row in final_rows for key in row.keys()} | {"experiment"})
    write_csv(table_dir / "main_final_metrics.csv", final_rows, final_fields)
    write_csv(table_dir / "region_lidar_geometry_metrics.csv", region_rows, ["experiment", "source", "region", "valid_lidar_count", "confidence", "MAE", "RMSE", "AbsRel", "delta_lt_1_25"])
    eval_fields = sorted({key for row in eval_rows for key in row.keys()} | {"experiment", "source"})
    per_view_fields = sorted({key for row in eval_per_view_rows for key in row.keys()} | {"experiment", "source"})
    write_csv(table_dir / "eval_summary.csv", eval_rows, eval_fields)
    write_csv(table_dir / "eval_latest_per_view.csv", eval_per_view_rows, per_view_fields)
    scalar_fields = sorted({key for row in scalar_trace_rows for key in row.keys()} | {"experiment", "source", "iteration"})
    write_csv(table_dir / "train_scalar_trace.csv", scalar_trace_rows, scalar_fields)
    write_csv(table_dir / "initialization_summary.csv", initialization_rows, ["experiment", "manifest", "status", "uses_lidar_training_supervision", "uses_lidar_initialization", "initialization_source", "pointcloud_source", "colmap_binary", "colmap_point_count", "lidar_point_count_used_for_init", "no_lidar_leakage"])
    write_csv(table_dir / "feedback_trigger_summary.csv", feedback_rows_all, ["experiment", "iteration", "status", "risk_source", "supervision_mode", "selected_pixels_count", "gaussian_group_count", "cuda_ok_count", "low_evidence_count", "live_cuda_contribution", "uses_lidar_supervision", "uses_lidar_selected_pixels", "uses_lidar_initialization", "initialization_source", "gaussian_parameters_modified", "real_repair_enabled", "manifest"])
    write_csv(table_dir / "safety_audit_summary.csv", safety_rows, ["experiment", "iteration", "source", "status", "uses_lidar_supervision", "uses_lidar_selected_pixels", "gaussian_parameters_modified", "real_repair_enabled", "allow_parameter_modification", "safety_checks"])
    write_csv(table_dir / "repair_candidate_summary.csv", repair_rows, ["experiment", "iteration", "mode", "source", "status", "modified_count", "protected_count", "skipped_count", "candidate_count", "rgb_delta"])
    write_csv(table_dir / "figure_index.csv", figure_rows, ["experiment", "category", "source", "copied_to"])
    write_csv(table_dir / "missing_evidence_report.csv", missing_rows, ["experiment", "missing", "severity"])
    write_csv(table_dir / "experiment_inventory.csv", experiment_rows, ["experiment", "path", "has_final_metrics", "initialization_status", "uses_lidar_initialization", "initialization_source", "feedback_trigger_count", "figure_count"])
    final_eval_rows = collect_final_evaluation_rows(final_eval_root)
    final_eval_fields = sorted({key for row in final_eval_rows for key in row.keys()} | {"experiment", "eval_protocol", "scope", "split"})
    write_csv(table_dir / "final_full_evaluation_summary.csv", final_eval_rows, final_eval_fields)

    summary = {
        "output_root": str(output_root),
        "paper_dir": str(paper_dir),
        "experiment_count": len(experiment_rows),
        "tables": {
            "main_final_metrics": str(table_dir / "main_final_metrics.csv"),
            "region_lidar_geometry_metrics": str(table_dir / "region_lidar_geometry_metrics.csv"),
            "feedback_trigger_summary": str(table_dir / "feedback_trigger_summary.csv"),
            "initialization_summary": str(table_dir / "initialization_summary.csv"),
            "eval_summary": str(table_dir / "eval_summary.csv"),
            "eval_latest_per_view": str(table_dir / "eval_latest_per_view.csv"),
            "train_scalar_trace": str(table_dir / "train_scalar_trace.csv"),
            "final_full_evaluation_summary": str(table_dir / "final_full_evaluation_summary.csv"),
            "safety_audit_summary": str(table_dir / "safety_audit_summary.csv"),
            "repair_candidate_summary": str(table_dir / "repair_candidate_summary.csv"),
            "figure_index": str(table_dir / "figure_index.csv"),
            "missing_evidence_report": str(table_dir / "missing_evidence_report.csv"),
        },
        "missing_evidence_count": len(missing_rows),
        "notes": [
            "Metrics are copied or summarized only from existing run outputs.",
            "Training-time eval_summary.csv is sampled periodic diagnostic eval, not the paper-grade main result.",
            "Paper main results should use final_full_evaluation_summary.csv when available.",
            "DA3-unsupervised paper claims should require uses_lidar_supervision=false and uses_lidar_selected_pixels=false in safety/feedback tables.",
            "No-LiDAR initialization claims should require uses_lidar_initialization=false in initialization_summary.csv.",
            "Region geometry rows must be interpreted with valid_lidar_count and confidence.",
        ],
    }
    write_json(summary_dir / "paper_evidence_summary.json", summary)
    write_json(summary_dir / "missing_evidence_report.json", {"missing": missing_rows})

    readme = paper_dir / "README.md"
    readme.write_text(
        "# GeoGuardGS Paper Evidence Pack\n\n"
        "Generated by `scripts/build_paper_evidence_pack.py`.\n\n"
        "## Tables\n\n"
        "- `tables/main_final_metrics.csv`: final/global experiment metrics.\n"
        "- `tables/final_full_evaluation_summary.csv`: paper-grade full final evaluation summary when available.\n"
        "- `tables/initialization_summary.csv`: initialization source and LiDAR-init leakage audit.\n"
        "- `tables/eval_summary.csv`: sampled periodic diagnostic training evaluation mean/median/outlier summary.\n"
        "- `tables/eval_latest_per_view.csv`: latest per-view evaluation diagnostics.\n"
        "- `tables/train_scalar_trace.csv`: per-iteration training losses and diagnostic scalars.\n"
        "- `tables/region_lidar_geometry_metrics.csv`: region-local geometry metrics with `valid_lidar_count` and confidence.\n"
        "- `tables/feedback_trigger_summary.csv`: periodic feedback trigger evidence.\n"
        "- `tables/safety_audit_summary.csv`: LiDAR leakage and repair safety audit rows.\n"
        "- `tables/repair_candidate_summary.csv`: opacity decay and repair dry-run summaries.\n"
        "- `tables/figure_index.csv`: copied figure assets and their sources.\n"
        "- `tables/missing_evidence_report.csv`: gaps that must be resolved before strong paper claims.\n\n"
        "## Figures\n\n"
        "`figures/` contains copied panels, risk maps, contribution overlays, group responsibility figures, and opacity-decay panels when they exist in experiment outputs.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
