#!/usr/bin/env python3
"""Build cross-experiment fixed-view training galleries from periodic panels.

This script consumes the per-experiment `periodic_eval/iter_XXXXXX` outputs
written during training and creates a paper-facing directory organized by
iteration and fixed view. It does not render new images or fabricate missing
panels; it indexes and copies what training already produced.
"""

import argparse
import csv
import html
import json
import shutil
from pathlib import Path


DEFAULT_EXPERIMENTS = [
    "streetgs_original_baseline",
    "da3_only_full_scene_lidar_init",
    "da3_periodic_group_softpatch_full_scene_lidar_init",
    "da3_periodic_group_softpatch_opacity_reg",
    "da3_periodic_group_softpatch_opacity_decay",
    "lidar_supervised_reference",
    "hybrid_reference",
]


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_iteration(path):
    try:
        return int(path.name.replace("iter_", ""))
    except ValueError:
        return -1


def safe_name(value):
    text = str(value).replace("\\", "_").replace("/", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def rel(path, root):
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def discover_experiment_dirs(output_root, names):
    if names:
        return [output_root / name for name in names]
    return [output_root / name for name in DEFAULT_EXPERIMENTS if (output_root / name).exists()]


def collect_panel_rows(output_root, exp_dirs, out_dir, copy_assets):
    rows = []
    missing_rows = []
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        periodic_root = exp_dir / "periodic_eval"
        if not periodic_root.exists():
            missing_rows.append({
                "experiment": exp_name,
                "iteration": "",
                "view_key": "",
                "missing": "periodic_eval directory",
            })
            continue
        for iter_dir in sorted(periodic_root.glob("iter_*"), key=parse_iteration):
            manifest_path = iter_dir / "panel_manifest.json"
            iteration = parse_iteration(iter_dir)
            if not manifest_path.exists():
                missing_rows.append({
                    "experiment": exp_name,
                    "iteration": iteration,
                    "view_key": "",
                    "missing": "panel_manifest.json",
                })
                continue
            manifest = read_json(manifest_path)
            for view in manifest.get("views", []):
                cam_id = view.get("cam_id", "")
                image_name = view.get("image_name", "")
                view_key = f"cam{cam_id}_{safe_name(image_name)}"
                panel_path = Path(view.get("panel_path", ""))
                if not panel_path.is_absolute():
                    panel_path = output_root.parent / panel_path
                if not panel_path.exists():
                    missing_rows.append({
                        "experiment": exp_name,
                        "iteration": iteration,
                        "view_key": view_key,
                        "missing": "comparison panel",
                    })
                    continue
                target_dir = out_dir / "by_iteration" / f"iter_{iteration:06d}" / view_key
                target = target_dir / f"{safe_name(exp_name)}_{panel_path.name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(panel_path, target)
                asset_targets = {}
                if copy_assets:
                    for key in [
                        "gt_rgb_path",
                        "rendered_rgb_path",
                        "rgb_error_path",
                        "depth_path",
                        "da3_depth_or_edge_path",
                        "lidar_sparse_overlay_path",
                        "risk_path",
                        "selected_risk_path",
                        "accumulation_path",
                        "softpatch_mask_path",
                    ]:
                        src_value = view.get(key, "")
                        if not src_value:
                            continue
                        src = Path(src_value)
                        if not src.is_absolute():
                            src = output_root.parent / src
                        if not src.exists():
                            continue
                        dst = out_dir / "assets" / safe_name(exp_name) / f"iter_{iteration:06d}" / view_key / src.name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                        asset_targets[key] = rel(dst, out_dir)
                rows.append({
                    "experiment": exp_name,
                    "iteration": iteration,
                    "cam_id": cam_id,
                    "image_name": image_name,
                    "view_key": view_key,
                    "panel_path": rel(target, out_dir),
                    "source_panel_path": str(panel_path),
                    "psnr": view.get("psnr", ""),
                    "l1": view.get("l1", ""),
                    "positive_depth_count": view.get("positive_depth_count", ""),
                    "finite_depth_count": view.get("finite_depth_count", ""),
                    "depth_min": view.get("depth_min", ""),
                    "depth_max": view.get("depth_max", ""),
                    "acc_min": view.get("acc_min", ""),
                    "acc_max": view.get("acc_max", ""),
                    "acc_mean": view.get("acc_mean", ""),
                    "warnings": ";".join(view.get("warnings", [])) if isinstance(view.get("warnings", []), list) else view.get("warnings", ""),
                    **asset_targets,
                })
    return rows, missing_rows


def build_html(out_dir, rows):
    grouped = {}
    for row in rows:
        key = (int(row["iteration"]), row["view_key"])
        grouped.setdefault(key, []).append(row)
    parts = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>GeoGuardGS Training Gallery</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px} img{max-width:420px;border:1px solid #ddd} "
        ".grid{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:32px}.card{max-width:440px}.meta{font-size:12px;color:#444}</style>",
        "<h1>GeoGuardGS Training Gallery</h1>",
    ]
    for (iteration, view_key), items in sorted(grouped.items()):
        parts.append(f"<h2>iter_{iteration:06d} / {html.escape(view_key)}</h2>")
        parts.append("<div class='grid'>")
        for row in sorted(items, key=lambda r: r["experiment"]):
            parts.append("<div class='card'>")
            parts.append(f"<h3>{html.escape(row['experiment'])}</h3>")
            parts.append(f"<a href='{html.escape(row['panel_path'])}'><img src='{html.escape(row['panel_path'])}'></a>")
            parts.append(
                "<div class='meta'>"
                f"cam={html.escape(str(row['cam_id']))}, image={html.escape(str(row['image_name']))}<br>"
                f"PSNR={html.escape(str(row['psnr']))}, L1={html.escape(str(row['l1']))}<br>"
                f"depth+={html.escape(str(row['positive_depth_count']))}, warnings={html.escape(str(row['warnings']))}"
                "</div>"
            )
            parts.append("</div>")
        parts.append("</div>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/a100_main_experiments")
    parser.add_argument("--out-dir", default="outputs/paper_results/training_gallery")
    parser.add_argument("--experiments", nargs="*", default=None, help="Experiment directory names under output-root. Defaults to known formal full-scene names.")
    parser.add_argument("--copy-assets", action="store_true", help="Also copy individual RGB/depth/risk assets in addition to comparison panels.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir)
    exp_dirs = discover_experiment_dirs(output_root, args.experiments)
    rows, missing_rows = collect_panel_rows(output_root, exp_dirs, out_dir, args.copy_assets)

    fields = sorted({key for row in rows for key in row.keys()} | {
        "experiment",
        "iteration",
        "cam_id",
        "image_name",
        "view_key",
        "panel_path",
        "source_panel_path",
        "psnr",
        "l1",
        "positive_depth_count",
        "finite_depth_count",
        "warnings",
    })
    write_csv(out_dir / "training_gallery_index.csv", rows, fields)
    write_csv(out_dir / "training_gallery_missing.csv", missing_rows, ["experiment", "iteration", "view_key", "missing"])
    build_html(out_dir, rows)
    manifest = {
        "output_root": str(output_root),
        "out_dir": str(out_dir),
        "experiment_count": len(exp_dirs),
        "panel_count": len(rows),
        "missing_count": len(missing_rows),
        "index": str(out_dir / "training_gallery_index.csv"),
        "missing": str(out_dir / "training_gallery_missing.csv"),
        "html": str(out_dir / "index.html"),
    }
    (out_dir / "training_gallery_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
