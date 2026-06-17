# 输出规范

最终实验应能展示以下内容：

1. 不同分支 RGB 渲染对比图。
2. 不同分支 depth map。
3. depth error heatmap。
4. DA3 boundary-risk overlay。
5. LiDAR sparse depth overlay。
6. high-error / high-risk selected region panels。
7. contribution top-K Gaussian 可视化。
8. group responsibility 可视化。
9. opacity decay 前后 panel。
10. controlled Gaussian opacity trace。
11. global RGB metrics 表。
12. LiDAR-valid geometry metrics 表。
13. DA3 top-risk region metrics 表。
14. boundary / depth-edge metrics 表。
15. repair candidate 统计表。
16. safety audit 表。
17. 每 500 轮训练趋势曲线。
18. final comparison table。

推荐目录：

```text
outputs/<exp_name>/
  checkpoints/
  periodic_eval/
  feedback_controller/
  metrics/
  final_eval/
```

`log_images/` is only a lightweight optional training log controlled by
`train.save_visuals` and `train.log_image_interval`. Paper figures and
diagnostics should use `periodic_eval/` and `feedback_controller/`.

Formal full-scene v2 training should write fixed-view panels every 1000
iterations:

```text
outputs/<group_output_dir>/
  periodic_eval/
    iter_000500/
      panel_manifest.json
      assets/
        iter_000500_cam0_<image_name>_gt_rgb.jpg
        iter_000500_cam0_<image_name>_rendered_rgb.jpg
        iter_000500_cam0_<image_name>_rendered_depth.jpg
        iter_000500_cam0_<image_name>_da3_boundary_risk.jpg
      panels/
        iter_000500_cam0_<image_name>_comparison_panel.jpg
```

Each `panel_manifest.json` records the source view, `cam_id`, `image_name`,
panel path, individual asset paths, PSNR, L1, depth finite/positive counts,
depth min/max, accumulation statistics, and warnings such as PSNR drops, empty
positive depth, or saturated accumulation.

For the required A/B/C 5000-iteration diagnostic comparison, the same structure
is written under:

```text
outputs/a100_short_5000/
  baseline_streetgs/
  no_lidar_supervision_control/
  da3_periodic_group_softpatch/
```

## 论文证据包

正式实验完成后运行：

```bash
python scripts/build_paper_evidence_pack.py \
  --output-root outputs \
  --paper-dir outputs/paper_evidence
```

生成目录：

```text
outputs/paper_evidence/
  README.md
  summaries/
    paper_evidence_summary.json
    missing_evidence_report.json
  tables/
    experiment_inventory.csv
    main_final_metrics.csv
    region_lidar_geometry_metrics.csv
    feedback_trigger_summary.csv
    safety_audit_summary.csv
    repair_candidate_summary.csv
    figure_index.csv
    missing_evidence_report.csv
  figures/
    panels/
    risk_maps/
    contribution/
    group_responsibility/
    opacity_decay/
  manifests/
```

`missing_evidence_report.csv` 必须作为论文写作前检查项。如果该文件中仍有 `paper_table_gap`、`method_evidence_gap` 或 `figure_gap`，对应结论不能强写。

DA3-unsupervised 论文结论必须同时检查：

- `feedback_trigger_summary.csv` 中 `uses_lidar_supervision=false`；
- `feedback_trigger_summary.csv` 中 `uses_lidar_selected_pixels=false`；
- `safety_audit_summary.csv` 中没有 real prune / split / shrink；
- `region_lidar_geometry_metrics.csv` 中每个 region 都带 `valid_lidar_count` 和 `confidence`。

## Paper Result Visuals

After building the evidence pack, render paper-facing result tables and plots:

```bash
python scripts/build_paper_result_visuals.py \
  --paper-dir outputs/paper_evidence \
  --out-dir outputs/paper_results
```

This creates:

```text
outputs/paper_results/
  README.md
  paper_result_manifest.json
  tables/
    table_main_results.csv
    table_main_results.md
    table_no_lidar_audit.csv
    table_no_lidar_audit.md
    table_feedback_summary.csv
    table_feedback_summary.md
  latex/
    table_main_results.tex
    table_no_lidar_audit.tex
    table_feedback_summary.tex
  plots/
    psnr_mean_curve.png
    psnr_median_curve.png
    l1_mean_curve.png
    feedback_trigger_timeline.png
    initialization_audit.png
  figures/
    selected_panels/
  training_gallery/
    index.html
    training_gallery_index.csv
    training_gallery_missing.csv
    training_gallery_manifest.json
```

The result renderer does not fabricate missing metrics. If source CSV files are
missing, the corresponding table or plot is empty or marked skipped in
`paper_result_manifest.json`.

For fixed-view cross-group visual comparison after training, run:

```bash
python scripts/build_paper_training_gallery.py \
  --output-root outputs \
  --out-dir outputs/paper_results/training_gallery \
  --copy-assets
```

This indexes and copies the every-1000-iteration `periodic_eval` panels by
`iteration + cam_id + image_name`, so the same selected frame/view can be
compared directly across experiment groups.

## Full-scene v2 paper output

Use the v2 output names for the next full-scene rerun:

```bash
python scripts/build_paper_evidence_pack.py \
  --output-root outputs \
  --paper-dir outputs/paper_evidence_full_scene_v2

python scripts/build_paper_result_visuals.py \
  --paper-dir outputs/paper_evidence_full_scene_v2 \
  --out-dir outputs/paper_results_full_scene_v2

python scripts/build_paper_training_gallery.py \
  --output-root outputs \
  --out-dir outputs/paper_results_full_scene_v2/training_gallery \
  --copy-assets
```

Required outputs:

- `outputs/paper_results_full_scene_v2/plots/psnr_mean_curve.png`
- `outputs/paper_results_full_scene_v2/plots/psnr_median_curve.png`
- `outputs/paper_results_full_scene_v2/plots/l1_mean_curve.png`
- `outputs/paper_results_full_scene_v2/plots/loss_curve.png`
- `outputs/paper_results_full_scene_v2/plots/l1_loss_curve.png`
- `outputs/paper_results_full_scene_v2/plots/guided_feedback_da3_structure_loss_curve.png`
- `outputs/paper_results_full_scene_v2/plots/lidar_depth_loss_curve.png`
- `outputs/paper_results_full_scene_v2/training_gallery/index.html`
- `outputs/paper_results_full_scene_v2/training_gallery/training_gallery_index.csv`

Training defaults now save 15 fixed views every 1000 iterations: 5 cameras x 3
frames. Set `train.periodic_eval_view_ids` only when manually pinning exact
image names.


