#!/usr/bin/env python3
"""Collect compact metrics and manifests from GeoFeedback-GS experiment outputs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--out", default="outputs/collected_summary.json")
    args = parser.parse_args()
    root = Path(args.output_root)
    rows = []
    for exp in sorted(p for p in root.iterdir() if p.is_dir()):
        row = {"experiment": exp.name}
        for name in ["experiment_manifest.json", "final_eval/metrics.json", "metrics/rgb_metrics.csv", "metrics/lidar_geometry_metrics.csv"]:
            p = exp / name
            row[name] = str(p) if p.exists() else ""
        fc = exp / "feedback_controller"
        row["feedback_trigger_count"] = len(list(fc.glob("iter_*/feedback_controller_manifest.json"))) if fc.exists() else 0
        rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"experiments": rows}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
