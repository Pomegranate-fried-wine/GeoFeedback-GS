#!/usr/bin/env python3
"""Build final GeoGuardGS visual evaluation assets from completed outputs."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


SNAPSHOT_ITERS = [1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000]
EXPERIMENTS = {
    "A": "A_streetgs_lidar_init_lidar_sup",
    "B": "B_lidar_init_no_lidar_sup",
    "C": "C_lidar_init_da3_feedback",
    "PV-C": "PVC_no_lidar_init_da3_feedback",
}


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def parse_iter_dir(path):
    try:
        return int(path.name.replace("iter_", ""))
    except ValueError:
        return -1


def rel(path, root):
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def load_image(path, missing, label):
    if not path or not Path(path).exists():
        missing.append({"missing": label, "path": str(path)})
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        missing.append({"missing": f"unreadable_{label}", "path": str(path)})
    return img


def resize_to_height(img, height):
    if img is None or img.shape[0] == height:
        return img
    width = max(1, int(round(img.shape[1] * height / img.shape[0])))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def label_image(img, label):
    out = img.copy()
    bar = np.zeros((34, out.shape[1], 3), dtype=np.uint8)
    bar[:, :] = (20, 20, 20)
    cv2.putText(bar, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, out])


def hstack_labeled(items):
    valid = [(label, img) for label, img in items if img is not None]
    if not valid:
        return None
    height = min(img.shape[0] for _, img in valid)
    panels = [label_image(resize_to_height(img, height), label) for label, img in valid]
    return np.hstack(panels)


def collect_snapshots(output_root, missing):
    rows = []
    by_key = {}
    for group, dirname in EXPERIMENTS.items():
        exp_dir = output_root / dirname
        for iteration in SNAPSHOT_ITERS:
            manifest_path = exp_dir / "periodic_eval" / f"iter_{iteration:06d}" / "snapshot_manifest.json"
            if not manifest_path.exists():
                missing.append({"experiment": group, "iteration": iteration, "missing": "snapshot_manifest.json", "path": str(manifest_path)})
                continue
            manifest = read_json(manifest_path)
            for view in manifest.get("views", []):
                if view.get("status") == "failed":
                    missing.append({"experiment": group, "iteration": iteration, "missing": "failed_view", "path": str(manifest_path), "view": view.get("image_name", "")})
                    continue
                split = view.get("split", "")
                cam_id = view.get("cam_id", "")
                image_name = view.get("image_name", "")
                view_key = f"{split}_cam{cam_id}_{safe_name(image_name)}"
                row = {
                    "group": group,
                    "experiment_dir": dirname,
                    "iteration": iteration,
                    "split": split,
                    "cam_id": cam_id,
                    "image_name": image_name,
                    "view_key": view_key,
                    "gt": view.get("snapshot_gt_path") or view.get("gt_rgb_path", ""),
                    "rgb": view.get("snapshot_rgb_path") or view.get("rendered_rgb_path", ""),
                    "depth": view.get("snapshot_depth_path") or view.get("depth_path", ""),
                    "feedback_risk_score": view.get("snapshot_feedback_risk_score_path", ""),
                    "feedback_responsible_groups": view.get("snapshot_feedback_responsible_groups_path", ""),
                    "feedback_risk_and_groups": view.get("snapshot_feedback_risk_and_groups_path", ""),
                    "psnr": view.get("psnr", ""),
                    "l1": view.get("l1", ""),
                }
                rows.append(row)
                by_key.setdefault((iteration, view_key), {})[group] = row
    return rows, by_key


def build_rgb_montages(out_dir, output_root, by_key, missing):
    rows = []
    for (iteration, view_key), groups in sorted(by_key.items()):
        gt_src = (groups.get("A") or groups.get("PV-C") or groups.get("C") or groups.get("B") or {}).get("gt", "")
        gt = load_image(gt_src, missing, f"gt_{iteration}_{view_key}")
        a = load_image((groups.get("A") or {}).get("rgb", ""), missing, f"A_rgb_{iteration}_{view_key}")
        pvc = load_image((groups.get("PV-C") or {}).get("rgb", ""), missing, f"PVC_rgb_{iteration}_{view_key}")
        main = hstack_labeled([("GT", gt), ("A", a), ("PV-C", pvc)])
        if main is not None:
            target = out_dir / "rgb_gt_a_pvc" / f"iter_{iteration:06d}" / f"{view_key}_gt_a_pvc_rgb.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), main)
            rows.append({"iteration": iteration, "view_key": view_key, "type": "gt_a_pvc_rgb", "path": rel(target, output_root)})
        imgs = [("GT", gt)]
        for label in ["A", "B", "C", "PV-C"]:
            imgs.append((label, load_image((groups.get(label) or {}).get("rgb", ""), missing, f"{label}_rgb_{iteration}_{view_key}")))
        full = hstack_labeled(imgs)
        if full is not None:
            target = out_dir / "rgb_gt_abcpvc" / f"iter_{iteration:06d}" / f"{view_key}_gt_abcpvc_rgb.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), full)
            rows.append({"iteration": iteration, "view_key": view_key, "type": "gt_abcpvc_rgb", "path": rel(target, output_root)})
    return rows


def copy_feedback_visuals(out_dir, output_root, snapshot_rows, missing):
    rows = []
    for row in snapshot_rows:
        if row["group"] not in {"C", "PV-C"}:
            continue
        for key in ["feedback_risk_score", "feedback_responsible_groups", "feedback_risk_and_groups"]:
            src = Path(row.get(key, ""))
            if not src.exists():
                missing.append({"experiment": row["group"], "iteration": row["iteration"], "view_key": row["view_key"], "missing": key, "path": str(src)})
                continue
            dst = out_dir / "feedback_matrices" / row["group"] / f"iter_{int(row['iteration']):06d}" / f"{row['view_key']}_{key}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                missing.append({"experiment": row["group"], "iteration": row["iteration"], "view_key": row["view_key"], "missing": f"unreadable_{key}", "path": str(src)})
                continue
            cv2.imwrite(str(dst), img)
            rows.append({"group": row["group"], "iteration": row["iteration"], "view_key": row["view_key"], "visual_type": key, "path": rel(dst, output_root)})
    return rows


def read_scalar_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def plot_training_curves(output_root, out_dir, missing):
    if plt is None:
        missing.append({"missing": "matplotlib_unavailable", "path": "training_curves"})
        return []
    curve_rows = []
    series = {
        "loss": "loss",
        "l1_loss": "l1_loss",
        "guided_feedback_da3_structure_loss": "guided_feedback_da3_structure_loss",
        "da3_structure_loss": "da3_structure_loss",
        "train_psnr": "train_psnr",
    }
    traces = {}
    for group, dirname in EXPERIMENTS.items():
        path = output_root / dirname / "metrics" / "train_loss_trace.csv"
        rows = read_scalar_csv(path)
        if not rows:
            missing.append({"experiment": group, "missing": "metrics/train_loss_trace.csv", "path": str(path)})
            continue
        traces[group] = rows
    for curve_name, field in series.items():
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        has_data = False
        for group, rows in traces.items():
            xs, ys = [], []
            for row in rows:
                x = to_float(row.get("iteration"))
                y = to_float(row.get(field))
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            if xs:
                ax.plot(xs, ys, label=group, linewidth=1.2)
                has_data = True
        if has_data:
            ax.set_xlabel("Iteration")
            ax.set_ylabel(field)
            ax.set_title(field)
            ax.grid(True, alpha=0.25)
            ax.legend()
            target = out_dir / "training_curves" / f"{curve_name}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(target)
            curve_rows.append({"curve": curve_name, "path": rel(target, output_root)})
        plt.close(fig)
    return curve_rows


def collect_eval_summaries(output_root, missing):
    rows = []
    for group, dirname in EXPERIMENTS.items():
        path = output_root / dirname / "metrics" / "eval_summary_full.csv"
        eval_rows = read_scalar_csv(path)
        if not eval_rows:
            missing.append({"experiment": group, "missing": "metrics/eval_summary_full.csv", "path": str(path)})
            continue
        for row in eval_rows:
            row = dict(row)
            row["group"] = group
            rows.append(row)
    return rows


def write_completeness(output_root, out_dir, snapshot_rows, montage_rows, feedback_rows, curve_rows, eval_rows, missing):
    required = [
        ("snapshots", bool(snapshot_rows)),
        ("gt_a_pvc_rgb_montages", any(r["type"] == "gt_a_pvc_rgb" for r in montage_rows)),
        ("four_group_rgb_montages", any(r["type"] == "gt_abcpvc_rgb" for r in montage_rows)),
        ("feedback_visuals", bool(feedback_rows)),
        ("training_curves", bool(curve_rows)),
        ("full_eval_metrics", bool(eval_rows)),
    ]
    rows = [{"item": item, "status": "present" if ok else "missing"} for item, ok in required]
    rows.extend({"item": f"missing:{m.get('missing', '')}", "status": "missing", "path": m.get("path", "")} for m in missing)
    write_csv(output_root / "final_eval" / "completeness_report.csv", rows, ["item", "status", "path"])
    write_json(output_root / "final_eval" / "completeness_report.json", {
        "required": dict(required),
        "missing_count": len(missing),
        "missing": missing,
        "outputs": {
            "final_visual_eval": str(out_dir),
            "completeness_report_csv": str(output_root / "final_eval" / "completeness_report.csv"),
        },
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--out-dir", default="outputs/final_visual_eval")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir)
    missing = []
    snapshot_rows, by_key = collect_snapshots(output_root, missing)
    montage_rows = build_rgb_montages(out_dir, output_root, by_key, missing)
    feedback_rows = copy_feedback_visuals(out_dir, output_root, snapshot_rows, missing)
    curve_rows = plot_training_curves(output_root, out_dir, missing)
    eval_rows = collect_eval_summaries(output_root, missing)

    write_csv(out_dir / "snapshot_assets_index.csv", snapshot_rows, sorted({k for r in snapshot_rows for k in r.keys()} or {"group"}))
    write_csv(out_dir / "rgb_montage_index.csv", montage_rows, ["iteration", "view_key", "type", "path"])
    write_csv(out_dir / "feedback_visual_index.csv", feedback_rows, ["group", "iteration", "view_key", "visual_type", "path"])
    write_csv(out_dir / "training_curve_index.csv", curve_rows, ["curve", "path"])
    write_csv(out_dir / "eval_summary_full_combined.csv", eval_rows, sorted({k for r in eval_rows for k in r.keys()} or {"group"}))
    write_csv(out_dir / "missing_assets.csv", missing, sorted({k for r in missing for k in r.keys()} or {"missing"}))
    manifest = {
        "output_root": str(output_root),
        "out_dir": str(out_dir),
        "snapshot_iterations": SNAPSHOT_ITERS,
        "experiments": EXPERIMENTS,
        "counts": {
            "snapshot_assets": len(snapshot_rows),
            "rgb_montages": len(montage_rows),
            "feedback_visuals": len(feedback_rows),
            "training_curves": len(curve_rows),
            "eval_rows": len(eval_rows),
            "missing": len(missing),
        },
        "missing": missing,
    }
    write_json(out_dir / "final_visual_eval_manifest.json", manifest)
    write_completeness(output_root, out_dir, snapshot_rows, montage_rows, feedback_rows, curve_rows, eval_rows, missing)
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
