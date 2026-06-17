#!/usr/bin/env python3
"""Recover training and evaluation curves from saved StreetGS tqdm logs.

The real scalar trace is ``metrics/train_loss_trace.csv``.  When that file was
not preserved, the tqdm log still contains the displayed total loss and training
PSNR plus evaluation summaries.  This script converts those log lines into
auditable CSV files and diagnostic plots.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "recovered_training_curves_from_logs"

LOG_SPECS = {
    "main_A_lidar_init_lidar_sup_gpu4.log": ("A StreetGS", "a100_baseline_streetgs"),
    "main_B_lidar_init_no_lidar_supervision_control_nolidar_sup_gpu5.log": ("B No-LiDAR-Supervision Control", "a100_no_lidar_supervision_control"),
    "main_C_lidar_init_da3_feedback_nolidar_sup_gpu6.log": (
        "C DA3+Feedback",
        "a100_da3_periodic_group_softpatch",
    ),
    "main_PVC_purevision_da3_feedback_obj_gpu7.log": ("PV-C LiDAR-free GeoFeedback-GS", "a100_pv_da3_feedback_obj"),
}

PROGRESS_RE = re.compile(
    r"(?P<iteration>\d+)/(?P<total>\d+).*?"
    r"Exp=(?P<exp>[^,\]]+),\s*Loss=(?P<loss>[-+0-9.eE]+),,\s*PSNR=(?P<psnr>[-+0-9.eE]+)"
)

EVAL_RE = re.compile(
    r"\[ITER\s+(?P<iteration>\d+)\]\s+Evaluating\s+"
    r"(?P<name>[^\[]+)\[(?P<protocol>[^\]]+)\]:\s+"
    r"L1\s+(?P<l1>[-+0-9.eE]+)\s+PSNR\s+(?P<psnr>[-+0-9.eE]+)\s+"
    r"median_psnr\s+(?P<median_psnr>[-+0-9.eE]+)\s+"
    r"outliers\s+(?P<outliers>\d+)\s+views\s+(?P<views>\d+)"
)


def _to_float(value: str) -> float:
    out = float(value)
    if math.isfinite(out):
        return out
    return float("nan")


def _read_log_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_progress(log_path: Path, label: str, experiment: str) -> list[dict[str, object]]:
    text = _read_log_text(log_path).replace("\r", "\n")
    rows: list[dict[str, object]] = []
    last_pair: tuple[float, float] | None = None
    for line in text.splitlines():
        match = PROGRESS_RE.search(line)
        if not match:
            continue
        loss = _to_float(match.group("loss"))
        psnr = _to_float(match.group("psnr"))
        pair = (loss, psnr)
        # tqdm repeats the latest postfix on subsequent progress refreshes.
        # Keep only points where the displayed scalar changed.
        if pair == last_pair:
            continue
        last_pair = pair
        rows.append(
            {
                "label": label,
                "experiment": experiment,
                "log_file": log_path.name,
                "iteration": int(match.group("iteration")),
                "total_iterations": int(match.group("total")),
                "displayed_loss": loss,
                "displayed_train_psnr": psnr,
                "source": "tqdm_postfix_recovered",
            }
        )
    return rows


def parse_eval(log_path: Path, label: str, experiment: str) -> list[dict[str, object]]:
    text = _read_log_text(log_path).replace("\r", "\n")
    rows: list[dict[str, object]] = []
    for match in EVAL_RE.finditer(text):
        rows.append(
            {
                "label": label,
                "experiment": experiment,
                "log_file": log_path.name,
                "iteration": int(match.group("iteration")),
                "split": match.group("name").strip(),
                "eval_protocol": match.group("protocol").strip(),
                "l1_mean": _to_float(match.group("l1")),
                "psnr_mean": _to_float(match.group("psnr")),
                "psnr_median": _to_float(match.group("median_psnr")),
                "outlier_count": int(match.group("outliers")),
                "view_count": int(match.group("views")),
                "source": "training_log_eval_summary_recovered",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def maybe_plot(progress_rows: list[dict[str, object]], eval_rows: list[dict[str, object]], out_dir: Path) -> None:
    warning_path = out_dir / "plot_warning.txt"
    if warning_path.exists():
        warning_path.unlink()
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        warning_path.write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "A StreetGS": "#2762B4",
        "B No-LiDAR-Supervision Control": "#2E8B57",
        "C DA3+Feedback": "#D98032",
        "PV-C LiDAR-free GeoFeedback-GS": "#B94842",
    }

    def save_all(fig, stem: str) -> None:
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(plots_dir / f"{stem}.{suffix}", bbox_inches="tight", dpi=220)
        plt.close(fig)

    for metric, ylabel, stem in [
        ("displayed_loss", "Displayed total loss", "recovered_tqdm_training_loss"),
        ("displayed_train_psnr", "Displayed training PSNR", "recovered_tqdm_training_psnr"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for label in LOG_SPECS.values():
            display_label = label[0]
            rows = [r for r in progress_rows if r["label"] == display_label]
            rows.sort(key=lambda r: int(r["iteration"]))
            if not rows:
                continue
            ax.plot(
                [int(r["iteration"]) for r in rows],
                [float(r[metric]) for r in rows],
                label=display_label,
                linewidth=1.4,
                color=colors.get(display_label),
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} recovered from tqdm logs")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        save_all(fig, stem)

    for protocol in ("sampled_diagnostic_eval", "full_split_training_eval"):
        for split in ("test/test_view", "test/train_view"):
            subset = [r for r in eval_rows if r["eval_protocol"] == protocol and r["split"] == split]
            if not subset:
                continue
            for metric, ylabel in [("psnr_mean", "PSNR mean"), ("l1_mean", "L1 mean")]:
                fig, ax = plt.subplots(figsize=(7.2, 4.2))
                for display_label, _exp in LOG_SPECS.values():
                    rows = [r for r in subset if r["label"] == display_label]
                    rows.sort(key=lambda r: int(r["iteration"]))
                    if not rows:
                        continue
                    ax.plot(
                        [int(r["iteration"]) for r in rows],
                        [float(r[metric]) for r in rows],
                        marker="o",
                        markersize=3,
                        label=display_label,
                        linewidth=1.5,
                        color=colors.get(display_label),
                    )
                ax.set_xlabel("Iteration")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{ylabel} - {protocol} - {split}")
                ax.grid(True, alpha=0.25)
                ax.legend(frameon=False, fontsize=8)
                safe_split = split.replace("/", "_")
                save_all(fig, f"recovered_{protocol}_{safe_split}_{metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=ROOT, help="Directory containing main_*.log files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    progress_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    missing = []
    for filename, (label, experiment) in LOG_SPECS.items():
        path = args.log_root / filename
        if not path.exists():
            missing.append(str(path))
            continue
        progress_rows.extend(parse_progress(path, label, experiment))
        eval_rows.extend(parse_eval(path, label, experiment))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "tables" / "recovered_tqdm_training_progress.csv",
        progress_rows,
        [
            "label",
            "experiment",
            "log_file",
            "iteration",
            "total_iterations",
            "displayed_loss",
            "displayed_train_psnr",
            "source",
        ],
    )
    write_csv(
        args.out_dir / "tables" / "recovered_eval_summary.csv",
        eval_rows,
        [
            "label",
            "experiment",
            "log_file",
            "iteration",
            "split",
            "eval_protocol",
            "l1_mean",
            "psnr_mean",
            "psnr_median",
            "outlier_count",
            "view_count",
            "source",
        ],
    )
    maybe_plot(progress_rows, eval_rows, args.out_dir)

    report = [
        "# Recovered Training Curves",
        "",
        f"- Progress rows: {len(progress_rows)}",
        f"- Evaluation rows: {len(eval_rows)}",
        f"- Missing logs: {len(missing)}",
        "",
        "## Scope",
        "",
        "- `displayed_loss` and `displayed_train_psnr` are recovered from tqdm postfix lines.",
        "- They are not a replacement for the full `metrics/train_loss_trace.csv` scalar trace.",
        "- Evaluation rows are recovered from `[ITER ...] Evaluating ...` summary log lines.",
        "- DA3/feedback/sky/object sub-losses cannot be recovered unless they were printed in the log.",
    ]
    if missing:
        report.extend(["", "## Missing", "", *[f"- {item}" for item in missing]])
    (args.out_dir / "RECOVERY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[RecoverLogs] progress_rows={len(progress_rows)} eval_rows={len(eval_rows)} out={args.out_dir}")


if __name__ == "__main__":
    main()

