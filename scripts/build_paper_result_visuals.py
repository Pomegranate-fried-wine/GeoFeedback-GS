#!/usr/bin/env python3
"""Render paper-facing tables and plots from a GeoFeedback-GS evidence pack."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


FORMAL_EXPERIMENTS = [
    "A_streetgs_lidar_init_lidar_sup",
    "B_lidar_init_no_lidar_sup",
    "C_lidar_init_da3_feedback",
    "PVC_no_lidar_init_da3_feedback",
]

FINAL_TO_TRAIN_EXPERIMENT = {
    "a100_baseline_streetgs": "A_streetgs_lidar_init_lidar_sup",
    "a100_no_lidar_supervision_control": "B_lidar_init_no_lidar_sup",
    "a100_da3_periodic_group_softpatch": "C_lidar_init_da3_feedback",
    "a100_pv_da3_feedback_obj": "PVC_no_lidar_init_da3_feedback",
}

EXPERIMENT_LABELS = {
    "a100_baseline_streetgs": "A StreetGS",
    "A_streetgs_lidar_init_lidar_sup": "A StreetGS",
    "streetgs_original_baseline": "A StreetGS",
    "baseline_streetgs": "A StreetGS",
    "baseline_streetgs_colmap_5000": "A StreetGS",
    "a100_no_lidar_supervision_control": "B No-LiDAR-Sup",
    "B_lidar_init_no_lidar_sup": "B No-LiDAR-Sup",
    "no_lidar_supervision_control": "B No-LiDAR-Sup",
    "no_lidar_supervision_control": "B No-LiDAR-Sup",
    "no_lidar_supervision_control_colmap_5000": "B No-LiDAR-Sup",
    "a100_da3_periodic_group_softpatch": "C DA3+Feedback",
    "C_lidar_init_da3_feedback": "C DA3+Feedback",
    "da3_periodic_group_softpatch_full_scene_lidar_init": "C DA3+Feedback",
    "da3_periodic_group_softpatch": "C DA3+Feedback",
    "da3_periodic_group_softpatch_colmap_5000": "C DA3+Feedback",
    "da3_periodic_group_softpatch_opacity_reg": "D +Opacity Reg",
    "da3_periodic_group_softpatch_opacity_decay": "E +Opacity Decay",
    "lidar_init_streetgs_reference": "D LiDAR-init Ref",
    "lidar_supervised_reference": "E LiDAR-supervised Ref",
    "hybrid_reference": "Hybrid Ref",
    "a100_pv_da3_feedback_obj": "PV-C Pure-Vision",
    "PVC_no_lidar_init_da3_feedback": "PV-C Pure-Vision",
    "pv_da3_feedback_obj": "PV-C Pure-Vision",
}


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def label_for(exp):
    return EXPERIMENT_LABELS.get(exp, exp)


def train_exp_for(exp):
    return FINAL_TO_TRAIN_EXPERIMENT.get(exp, exp)


def markdown_table(rows, fields):
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def latex_escape(text):
    return str(text).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def latex_table(rows, fields, caption, label):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{" + "l" * len(fields) + "}",
        "\\hline",
        " & ".join(latex_escape(f) for f in fields) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(field, "")) for field in fields) + " \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def choose_last_eval_rows(eval_rows):
    by_exp_split = {}
    for row in eval_rows:
        exp = row.get("experiment", "")
        split = row.get("split", "")
        iteration = to_float(row.get("iteration")) or -1
        key = (exp, split)
        if key not in by_exp_split or iteration > (to_float(by_exp_split[key].get("iteration")) or -1):
            by_exp_split[key] = row
    return list(by_exp_split.values())


def build_main_result_table(tables_dir):
    final_rows = read_csv(tables_dir / "final_full_evaluation_summary.csv")
    init_rows = {row.get("experiment", ""): row for row in read_csv(tables_dir / "initialization_summary.csv")}
    feedback_rows = read_csv(tables_dir / "feedback_trigger_summary.csv")
    feedback_count = {}
    feedback_valid = {}
    for feedback in feedback_rows:
        feedback_exp = feedback.get("experiment", "")
        if feedback.get("status") == "not_applicable":
            continue
        feedback_count[feedback_exp] = feedback_count.get(feedback_exp, 0) + 1
        if feedback.get("status") == "valid":
            feedback_valid[feedback_exp] = feedback_valid.get(feedback_exp, 0) + 1
    summary_main_rows = [
        row for row in final_rows
        if row.get("table") == "summary_main"
    ]
    final_main = summary_main_rows or [
        row for row in final_rows
        if row.get("scope") == "full_image" and row.get("split") == "test"
    ]
    if final_main:
        rows = []
        seen = set()
        for row in final_main:
            exp = row.get("experiment", "")
            if exp in seen:
                continue
            seen.add(exp)
            train_exp = train_exp_for(exp)
            init = init_rows.get(train_exp, init_rows.get(exp, {}))
            rows.append({
                "experiment": exp,
                "label": label_for(exp),
                "iteration": "final",
                "eval_protocol": row.get("eval_protocol", "full_final_evaluation"),
                "psnr_mean": row.get("psnr_mean", ""),
                "psnr_median": row.get("psnr_median", ""),
                "l1_mean": row.get("l1_mean", ""),
                "ssim_mean": row.get("ssim_mean", ""),
                "lpips_mean": row.get("lpips_mean", ""),
                "outlier_count": "",
                "uses_lidar_init": init.get("uses_lidar_initialization", ""),
                "init_source": init.get("initialization_source", ""),
                "feedback_valid": feedback_valid.get(train_exp, ""),
                "feedback_total": feedback_count.get(train_exp, ""),
            })
        return sorted(rows, key=lambda r: r["label"])

    all_eval_rows = read_csv(tables_dir / "eval_summary.csv")
    full_eval_rows = [row for row in all_eval_rows if row.get("eval_protocol") == "full_split_training_eval"]
    sampled_eval_rows = [row for row in all_eval_rows if row.get("eval_protocol") == "sampled_diagnostic_eval"]
    eval_protocol = "full_split_training_eval" if full_eval_rows else "sampled_diagnostic_eval"
    eval_rows = choose_last_eval_rows(full_eval_rows or sampled_eval_rows or all_eval_rows)
    init_rows = {row.get("experiment", ""): row for row in read_csv(tables_dir / "initialization_summary.csv")}
    feedback_rows = read_csv(tables_dir / "feedback_trigger_summary.csv")
    feedback_count = {}
    feedback_valid = {}
    for row in feedback_rows:
        exp = row.get("experiment", "")
        if row.get("status") == "not_applicable":
            continue
        feedback_count[exp] = feedback_count.get(exp, 0) + 1
        if row.get("status") == "valid":
            feedback_valid[exp] = feedback_valid.get(exp, 0) + 1

    rows = []
    for row in eval_rows:
        if row.get("split") != "test/train_view":
            continue
        exp = row.get("experiment", "")
        init = init_rows.get(exp, {})
        rows.append({
            "experiment": exp,
            "label": label_for(exp),
            "iteration": row.get("iteration", ""),
            "eval_protocol": eval_protocol,
            "psnr_mean": row.get("psnr_mean", ""),
            "psnr_median": row.get("psnr_median", ""),
            "l1_mean": row.get("l1_mean", ""),
            "ssim_mean": "",
            "lpips_mean": "",
            "outlier_count": row.get("outlier_count", ""),
            "uses_lidar_init": init.get("uses_lidar_initialization", ""),
            "init_source": init.get("initialization_source", ""),
            "feedback_valid": feedback_valid.get(exp, 0),
            "feedback_total": feedback_count.get(exp, 0),
        })
    return sorted(rows, key=lambda r: r["label"])


def build_no_lidar_audit_table(tables_dir):
    init_rows = read_csv(tables_dir / "initialization_summary.csv")
    safety_rows = read_csv(tables_dir / "safety_audit_summary.csv")
    feedback_rows = read_csv(tables_dir / "feedback_trigger_summary.csv")
    lidar_supervision = {}
    lidar_selected = {}
    modified = {}
    for row in safety_rows + feedback_rows:
        exp = row.get("experiment", "")
        values = lidar_supervision.setdefault(exp, [])
        selected = lidar_selected.setdefault(exp, [])
        mods = modified.setdefault(exp, [])
        if row.get("uses_lidar_supervision") != "":
            values.append(truthy(row.get("uses_lidar_supervision")))
        if row.get("uses_lidar_selected_pixels") != "":
            selected.append(truthy(row.get("uses_lidar_selected_pixels")))
        if row.get("gaussian_parameters_modified") != "":
            mods.append(truthy(row.get("gaussian_parameters_modified")))

    rows = []
    for init in init_rows:
        exp = init.get("experiment", "")
        uses_init = truthy(init.get("uses_lidar_initialization"))
        uses_train = any(lidar_supervision.get(exp, []))
        uses_selected = any(lidar_selected.get(exp, []))
        any_modified = any(modified.get(exp, []))
        rows.append({
            "experiment": exp,
            "label": label_for(exp),
            "uses_lidar_init": init.get("uses_lidar_initialization", ""),
            "uses_lidar_training": str(uses_train).lower() if lidar_supervision.get(exp) else "",
            "uses_lidar_selected": str(uses_selected).lower() if lidar_selected.get(exp) else "",
            "gaussian_modified": str(any_modified).lower() if modified.get(exp) else "",
            "claim_safe_no_lidar": str((not uses_init) and (not uses_train) and (not uses_selected)).lower(),
            "init_source": init.get("initialization_source", ""),
            "colmap_points": init.get("colmap_point_count", ""),
        })
    return sorted(rows, key=lambda r: r["label"])


def build_feedback_table(tables_dir):
    rows = []
    grouped = {}
    for row in read_csv(tables_dir / "feedback_trigger_summary.csv"):
        exp = row.get("experiment", "")
        grouped.setdefault(exp, []).append(row)
    for exp, items in grouped.items():
        applicable = [r for r in items if r.get("status") != "not_applicable"]
        if not applicable:
            rows.append({
                "experiment": exp,
                "label": label_for(exp),
                "status": "not_applicable",
                "trigger_count": 0,
                "valid_count": 0,
                "selected_pixels_total": "",
                "gaussian_group_total": "",
                "live_cuda_count": "",
            })
            continue
        rows.append({
            "experiment": exp,
            "label": label_for(exp),
            "status": "present",
            "trigger_count": len(applicable),
            "valid_count": sum(1 for r in applicable if r.get("status") == "valid"),
            "selected_pixels_total": sum(int(to_float(r.get("selected_pixels_count")) or 0) for r in applicable),
            "gaussian_group_total": sum(int(to_float(r.get("gaussian_group_count")) or 0) for r in applicable),
            "live_cuda_count": sum(1 for r in applicable if truthy(r.get("live_cuda_contribution"))),
        })
    return sorted(rows, key=lambda r: r["label"])


def load_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt, None
    except Exception as exc:
        return None, str(exc)


def plot_eval_curves(eval_rows, out_dir, metric, ylabel, protocol="", title="", filename_prefix=""):
    plt, error = load_matplotlib()
    if plt is None:
        return {"path": "", "status": "skipped", "error": error}
    by_exp = {}
    for row in eval_rows:
        if row.get("split") != "test/train_view":
            continue
        if protocol and row.get("eval_protocol") != protocol:
            continue
        x = to_float(row.get("iteration"))
        y = to_float(row.get(metric))
        if x is None or y is None:
            continue
        by_exp.setdefault(row.get("experiment", ""), []).append((x, y))
    if not by_exp:
        return {"path": "", "status": "missing_data", "error": ""}
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for exp, points in sorted(by_exp.items(), key=lambda item: label_for(item[0])):
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=1.8, label=label_for(exp))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title or protocol or "Training eval")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    stem = out_dir / f"{prefix}{metric}_curve"
    path = stem.with_suffix(".png")
    fig.savefig(path, dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)
    return {"path": str(path), "status": "ok", "error": ""}


def plot_scalar_curves(scalar_rows, out_dir, metric, ylabel, every=100):
    plt, error = load_matplotlib()
    if plt is None:
        return {"path": "", "status": "skipped", "error": error, "metric": metric}
    by_exp = {}
    for row in scalar_rows:
        x = to_float(row.get("iteration"))
        y = to_float(row.get(metric))
        if x is None or y is None:
            continue
        if every > 1 and int(x) % every != 0:
            continue
        by_exp.setdefault(row.get("experiment", ""), []).append((x, y))
    if not by_exp:
        return {"path": "", "status": "missing_data", "error": "", "metric": metric}
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for exp, points in sorted(by_exp.items(), key=lambda item: label_for(item[0])):
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], linewidth=1.3, label=label_for(exp))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    stem = out_dir / f"{metric}_curve"
    path = stem.with_suffix(".png")
    fig.savefig(path, dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)
    return {"path": str(path), "status": "ok", "error": "", "metric": metric}


def plot_feedback_timeline(feedback_rows, out_dir):
    plt, error = load_matplotlib()
    if plt is None:
        return {"path": "", "status": "skipped", "error": error}
    points = []
    for row in feedback_rows:
        if row.get("status") == "not_applicable":
            continue
        iteration = to_float(row.get("iteration"))
        if iteration is None:
            continue
        points.append((row.get("experiment", ""), iteration, row.get("status", "")))
    if not points:
        return {"path": "", "status": "missing_data", "error": ""}
    out_dir.mkdir(parents=True, exist_ok=True)
    experiments = sorted({p[0] for p in points}, key=label_for)
    exp_to_y = {exp: idx for idx, exp in enumerate(experiments)}
    fig, ax = plt.subplots(figsize=(7.2, max(2.8, 0.45 * len(experiments) + 1.2)), dpi=160)
    for exp, iteration, status in points:
        ax.scatter(iteration, exp_to_y[exp], s=42, c="#2f7ed8" if status == "valid" else "#d64f4f")
    ax.set_yticks(list(exp_to_y.values()))
    ax.set_yticklabels([label_for(exp) for exp in experiments])
    ax.set_xlabel("Iteration")
    ax.set_title("Feedback Trigger Timeline")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    stem = out_dir / "feedback_trigger_timeline"
    path = stem.with_suffix(".png")
    fig.savefig(path, dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)
    return {"path": str(path), "status": "ok", "error": ""}


def plot_initialization_audit(init_rows, out_dir):
    plt, error = load_matplotlib()
    if plt is None:
        return {"path": "", "status": "skipped", "error": error}
    if not init_rows:
        return {"path": "", "status": "missing_data", "error": ""}
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [label_for(r.get("experiment", "")) for r in init_rows]
    values = [1 if truthy(r.get("uses_lidar_initialization")) else 0 for r in init_rows]
    colors = ["#d64f4f" if v else "#3a9d5d" for v in values]
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.38 * len(labels) + 1.4)), dpi=160)
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No LiDAR init", "LiDAR init"])
    ax.set_title("Initialization Leakage Audit")
    fig.tight_layout()
    stem = out_dir / "initialization_audit"
    path = stem.with_suffix(".png")
    fig.savefig(path, dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)
    return {"path": str(path), "status": "ok", "error": ""}


def copy_selected_figures(paper_dir, out_dir, max_per_category):
    rows = read_csv(paper_dir / "tables" / "figure_index.csv")
    copied = []
    by_category = {}
    for row in rows:
        by_category.setdefault(row.get("category", "misc"), []).append(row)
    for category, items in by_category.items():
        count = 0
        for row in items:
            src = Path(row.get("copied_to") or row.get("source") or "")
            if not src.exists():
                continue
            dst = out_dir / "selected_panels" / category / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst = dst.with_name(f"{src.parent.name}_{src.name}")
            shutil.copy2(src, dst)
            copied.append({"category": category, "source": str(src), "copied_to": str(dst)})
            count += 1
            if count >= max_per_category:
                break
    return copied


def _local_periodic_asset(raw_root, exp, iteration, server_path):
    if not server_path:
        return None
    name = Path(server_path).name
    for subdir in ["assets", "panels"]:
        candidate = raw_root / exp / "periodic_eval" / iteration / subdir / name
        if candidate.exists():
            return candidate
    return None


def _load_periodic_manifest(raw_root, exp, iteration):
    path = raw_root / exp / "periodic_eval" / iteration / "panel_manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    views = {}
    for view in payload.get("views", []):
        image_name = view.get("image_name", "")
        if image_name:
            views[image_name] = view
    return views


def _image_with_title(image, title, width=260, title_h=30):
    from PIL import Image, ImageDraw, ImageFont

    image = image.convert("RGB")
    scale = width / max(1, image.width)
    height = max(1, int(image.height * scale))
    image = image.resize((width, height), Image.BICUBIC)
    out = Image.new("RGB", (width, height + title_h), "white")
    out.paste(image, (0, title_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 7), title, fill=(0, 0, 0), font=font)
    return out


def _blank_tile(width=260, height=176, title=""):
    from PIL import Image
    return _image_with_title(Image.new("RGB", (width, height), (245, 245, 245)), title, width=width)


def build_periodic_rgb_depth_comparisons(raw_root, out_dir, experiments, iterations, max_views):
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        return [{"status": "skipped", "error": f"pillow_missing:{exc}"}]
    if not raw_root.exists():
        return [{"status": "missing_data", "error": f"raw_output_root_not_found:{raw_root}"}]

    generated = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for iteration in iterations:
        manifests = {exp: _load_periodic_manifest(raw_root, exp, iteration) for exp in experiments}
        available = [set(v.keys()) for v in manifests.values() if v]
        if not available:
            generated.append({"iteration": iteration, "status": "missing_data", "error": "no_periodic_manifests"})
            continue
        common_views = sorted(set.intersection(*available))[:max_views]
        for image_name in common_views:
            first_view = next((m[image_name] for m in manifests.values() if image_name in m), {})
            cam_id = first_view.get("cam_id", "")
            row_rgb = []
            row_depth = []
            gt_path = _local_periodic_asset(raw_root, experiments[0], iteration, first_view.get("gt_rgb_path", ""))
            if gt_path and gt_path.exists():
                row_rgb.append(_image_with_title(Image.open(gt_path), f"GT cam{cam_id} {image_name}"))
            else:
                row_rgb.append(_blank_tile(title=f"GT cam{cam_id} {image_name}"))
            row_depth.append(_blank_tile(title="Depth"))
            for exp in experiments:
                view = manifests.get(exp, {}).get(image_name, {})
                rgb_path = _local_periodic_asset(raw_root, exp, iteration, view.get("rendered_rgb_path", ""))
                depth_path = _local_periodic_asset(raw_root, exp, iteration, view.get("depth_path", ""))
                label = label_for(exp)
                row_rgb.append(_image_with_title(Image.open(rgb_path), f"{label} RGB") if rgb_path and rgb_path.exists() else _blank_tile(title=f"{label} RGB"))
                row_depth.append(_image_with_title(Image.open(depth_path), f"{label} Depth") if depth_path and depth_path.exists() else _blank_tile(title=f"{label} Depth"))
            tiles = row_rgb + row_depth
            cols = len(row_rgb)
            tile_w = max(tile.width for tile in tiles)
            tile_h = max(tile.height for tile in tiles)
            canvas = Image.new("RGB", (cols * tile_w, 2 * tile_h + 42), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 10), f"{iteration} | cam{cam_id} | view {image_name} | RGB and rendered depth comparison", fill=(0, 0, 0))
            for idx, tile in enumerate(row_rgb):
                canvas.paste(tile, (idx * tile_w, 42))
            for idx, tile in enumerate(row_depth):
                canvas.paste(tile, (idx * tile_w, 42 + tile_h))
            out_path = out_dir / iteration / f"{iteration}_cam{cam_id}_{image_name}_rgb_depth_comparison.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(out_path, quality=92)
            tiff_path = out_path.with_suffix(".tiff")
            canvas.save(tiff_path, compression="tiff_lzw")
            generated.append({
                "iteration": iteration,
                "image_name": image_name,
                "cam_id": cam_id,
                "path": str(out_path),
                "tiff_path": str(tiff_path),
                "status": "ok",
            })
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", default="outputs/paper_evidence_full_scene_v2")
    parser.add_argument("--out-dir", default="outputs/paper_results_full_scene_v2")
    parser.add_argument("--raw-output-root", default="")
    parser.add_argument("--formal-experiments", nargs="*", default=FORMAL_EXPERIMENTS)
    parser.add_argument("--comparison-iterations", nargs="*", default=["iter_005000", "iter_010000", "iter_015000", "iter_020000", "iter_025000", "iter_030000"])
    parser.add_argument("--comparison-max-views", type=int, default=15)
    parser.add_argument("--max-selected-figures-per-category", type=int, default=6)
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    out_dir = Path(args.out_dir)
    tables_dir = paper_dir / "tables"
    out_tables = out_dir / "tables"
    out_plots = out_dir / "plots"
    out_latex = out_dir / "latex"

    main_rows = build_main_result_table(tables_dir)
    audit_rows = build_no_lidar_audit_table(tables_dir)
    feedback_rows = build_feedback_table(tables_dir)

    main_fields = ["label", "iteration", "eval_protocol", "psnr_mean", "psnr_median", "l1_mean", "ssim_mean", "lpips_mean", "outlier_count", "uses_lidar_init", "init_source", "feedback_valid", "feedback_total"]
    audit_fields = ["label", "uses_lidar_init", "uses_lidar_training", "uses_lidar_selected", "gaussian_modified", "claim_safe_no_lidar", "init_source", "colmap_points"]
    feedback_fields = ["label", "status", "trigger_count", "valid_count", "selected_pixels_total", "gaussian_group_total", "live_cuda_count"]

    write_csv(out_tables / "table_main_results.csv", main_rows, ["experiment"] + main_fields)
    write_csv(out_tables / "table_no_lidar_audit.csv", audit_rows, ["experiment"] + audit_fields)
    write_csv(out_tables / "table_feedback_summary.csv", feedback_rows, ["experiment"] + feedback_fields)
    (out_tables / "table_main_results.md").write_text(markdown_table(main_rows, main_fields), encoding="utf-8")
    (out_tables / "table_no_lidar_audit.md").write_text(markdown_table(audit_rows, audit_fields), encoding="utf-8")
    (out_tables / "table_feedback_summary.md").write_text(markdown_table(feedback_rows, feedback_fields), encoding="utf-8")
    out_latex.mkdir(parents=True, exist_ok=True)
    (out_latex / "table_main_results.tex").write_text(latex_table(main_rows, main_fields, "Main quantitative results.", "tab:main_results"), encoding="utf-8")
    (out_latex / "table_no_lidar_audit.tex").write_text(latex_table(audit_rows, audit_fields, "No-LiDAR leakage audit.", "tab:no_lidar_audit"), encoding="utf-8")
    (out_latex / "table_feedback_summary.tex").write_text(latex_table(feedback_rows, feedback_fields, "Feedback trigger summary.", "tab:feedback_summary"), encoding="utf-8")

    eval_rows = read_csv(tables_dir / "eval_summary.csv")
    scalar_rows = read_csv(tables_dir / "train_scalar_trace.csv")
    plot_rows = [
        plot_eval_curves(eval_rows, out_plots, "psnr_mean", "PSNR mean", "sampled_diagnostic_eval", "Sampled diagnostic eval", "sampled"),
        plot_eval_curves(eval_rows, out_plots, "psnr_median", "PSNR median", "sampled_diagnostic_eval", "Sampled diagnostic eval", "sampled"),
        plot_eval_curves(eval_rows, out_plots, "l1_mean", "L1 mean", "sampled_diagnostic_eval", "Sampled diagnostic eval", "sampled"),
        plot_eval_curves(eval_rows, out_plots, "psnr_mean", "PSNR mean", "full_split_training_eval", "Full-split training eval", "full_split"),
        plot_eval_curves(eval_rows, out_plots, "psnr_median", "PSNR median", "full_split_training_eval", "Full-split training eval", "full_split"),
        plot_eval_curves(eval_rows, out_plots, "l1_mean", "L1 mean", "full_split_training_eval", "Full-split training eval", "full_split"),
        plot_scalar_curves(scalar_rows, out_plots, "loss", "Training loss"),
        plot_scalar_curves(scalar_rows, out_plots, "l1_loss", "Training RGB L1 loss"),
        plot_scalar_curves(scalar_rows, out_plots, "guided_feedback_da3_structure_loss", "DA3 structure loss"),
        plot_scalar_curves(scalar_rows, out_plots, "lidar_depth_loss", "LiDAR depth loss"),
        plot_feedback_timeline(read_csv(tables_dir / "feedback_trigger_summary.csv"), out_plots),
        plot_initialization_audit(read_csv(tables_dir / "initialization_summary.csv"), out_plots),
    ]
    copied_figures = copy_selected_figures(paper_dir, out_dir / "figures", args.max_selected_figures_per_category)
    comparison_figures = []
    if args.raw_output_root:
        comparison_figures = build_periodic_rgb_depth_comparisons(
            Path(args.raw_output_root),
            out_dir / "figures" / "formal_rgb_depth_comparisons",
            args.formal_experiments,
            args.comparison_iterations,
            args.comparison_max_views,
        )

    missing = read_csv(tables_dir / "missing_evidence_report.csv")
    manifest = {
        "paper_dir": str(paper_dir),
        "out_dir": str(out_dir),
        "tables": {
            "main_results": str(out_tables / "table_main_results.csv"),
            "no_lidar_audit": str(out_tables / "table_no_lidar_audit.csv"),
            "feedback_summary": str(out_tables / "table_feedback_summary.csv"),
        },
        "latex": {
            "main_results": str(out_latex / "table_main_results.tex"),
            "no_lidar_audit": str(out_latex / "table_no_lidar_audit.tex"),
            "feedback_summary": str(out_latex / "table_feedback_summary.tex"),
        },
        "plots": plot_rows,
        "selected_figures": copied_figures,
        "comparison_figures": comparison_figures,
        "missing_evidence_count": len(missing),
        "claim_gate": {
            "has_no_lidar_safe_rows": any(row.get("claim_safe_no_lidar") == "true" for row in audit_rows),
            "missing_evidence_report_empty": len(missing) == 0,
        },
    }
    write_json(out_dir / "paper_result_manifest.json", manifest)

    readme = (
        "# GeoFeedback-GS Paper Results\n\n"
        "Generated by `scripts/build_paper_result_visuals.py` from an existing evidence pack.\n\n"
        "## Tables\n\n"
        "- `tables/table_main_results.md`: compact result table for paper drafting; prefers full final evaluation, then full-split training eval, and uses sampled diagnostic only as a last-resort fallback.\n"
        "- `tables/table_no_lidar_audit.md`: no-LiDAR initialization/supervision audit.\n"
        "- `tables/table_feedback_summary.md`: feedback trigger evidence summary.\n"
        "- `latex/*.tex`: LaTeX table drafts.\n\n"
        "## Plots\n\n"
        "- `plots/sampled_psnr_mean_curve.png` (sampled diagnostic eval)\n"
        "- `plots/sampled_psnr_median_curve.png` (sampled diagnostic eval)\n"
        "- `plots/sampled_l1_mean_curve.png` (sampled diagnostic eval)\n"
        "- `plots/full_split_psnr_mean_curve.png` (full-split training eval)\n"
        "- `plots/full_split_psnr_median_curve.png` (full-split training eval)\n"
        "- `plots/full_split_l1_mean_curve.png` (full-split training eval)\n"
        "- `plots/loss_curve.png`\n"
        "- `plots/l1_loss_curve.png`\n"
        "- `plots/guided_feedback_da3_structure_loss_curve.png`\n"
        "- `plots/lidar_depth_loss_curve.png`\n"
        "- `plots/feedback_trigger_timeline.png`\n"
        "- `plots/initialization_audit.png`\n\n"
        "## Formal RGB/Depth Comparisons\n\n"
        "`figures/formal_rgb_depth_comparisons/` contains same-view A/B/C/PV-C RGB and rendered-depth panels when `--raw-output-root` is provided.\n\n"
        "## Gate\n\n"
        "Do not claim no-LiDAR results unless `table_no_lidar_audit` has `claim_safe_no_lidar=true` for the formal rows and the evidence pack missing report is acceptable.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

