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

## 论文证据包

正式实验完成后运行：

```bash
python scripts/build_paper_evidence_pack.py \
  --output-root outputs/a100_main_experiments \
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
