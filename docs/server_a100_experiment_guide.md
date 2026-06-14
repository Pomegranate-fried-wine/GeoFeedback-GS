# GeoGuardGS A100 正式实验指导说明

本文档用于把 `GitHub_main` 迁移到 A100 Linux 服务器后，完成环境安装、数据/权重放置、CUDA 扩展编译、配置安全检查、并行训练、每 500 轮输出和结果收集。

## 1. 实验目标

正式实验验证的是闭环机制，而不是只看单一 RGB 指标：

```text
Street Gaussian training
-> DA3 / LiDAR risk source
-> live CUDA T*alpha contribution dump
-> boundary-aware Gaussian group responsibility
-> periodic feedback controller
-> group softpatch / opacity regularization / safe opacity decay
-> evaluation every 500 iterations
```

核心问题：

1. DA3-unsupervised 主线在不使用 LiDAR 训练监督时，是否能改善几何结构风险区域？
2. live CUDA contribution 是否能在训练态稳定产生责任 Gaussian group？
3. 周期性 group softpatch feedback 是否比 plain training 更稳定？
4. opacity regularization 是否能作为安全训练项运行？
5. conservative opacity decay 是否在不明显破坏 RGB 的前提下改善局部几何？
6. LiDAR-supervised reference 相比 DA3 主线提供怎样的上界参考？

## 2. 服务器目录建议

推荐：

```text
/data/hch/GeoGuardGS/
  configs/
  docs/
  geoguardgs/
  scripts/
  third_party/
  data/waymo/
  weights/da3/
  weights/streetgs/
  outputs/
```

不要把 Waymo 数据、DA3 权重、大 checkpoint 提交到 GitHub。它们只在服务器本地放置。

## 3. 环境安装

```bash
cd /data/hch/GeoGuardGS
conda env create -f environment.yml
conda activate geoguardgs
pip install -r requirements.txt
```

建议确认：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

## 4. 迁移后必须重新编译 / 安装的内容

Windows 本地编译产物不能用于 A100 Linux。以下组件必须在服务器重新编译或安装：

| 组件 | 路径 | 安装命令 | 必要性 |
| --- | --- | --- | --- |
| diff-gaussian-rasterization | `third_party/diff_gaussian_rasterization` | `pip install --no-build-isolation -e third_party/diff_gaussian_rasterization` | 必须；包含 Gaussian rasterizer 和 selected-pixel contribution debug 接口 |
| simple-knn | `third_party/simple_knn` | `pip install --no-build-isolation -e third_party/simple_knn` | 必须；StreetGS / 3DGS KNN CUDA 扩展 |
| nvdiffrast | `third_party/nvdiffrast` | `pip install --no-build-isolation -e third_party/nvdiffrast` | 按需；若配置或依赖调用则安装 |
| simple-waymo-open-dataset-reader | `third_party/simple_waymo_open_dataset_reader` | `pip install -e third_party/simple_waymo_open_dataset_reader` | 按需；Waymo reader |
| Depth Anything 3 依赖 | `third_party/depth_anything_3` | 根据其 README 和本项目 requirements 安装 | 必须用于 DA3 主线 |

推荐一键执行：

```bash
bash scripts/install_server_extensions.sh
python scripts/check_imports.py
```

如果编译失败，优先检查：

- PyTorch CUDA 版本；
- 系统 CUDA toolkit；
- GCC/G++ 版本；
- `CUDA_HOME`；
- `LD_LIBRARY_PATH`；
- A100 驱动版本。

不要把服务器编译出的 `.so`、`build/`、`*.egg-info/` 提交到 GitHub。

## 5. 数据与权重放置

### 5.1 Waymo

```text
data/waymo/002/
```

要求：

- 图像、相机、LiDAR 文件能被 StreetGS loader 读取；
- `data/waymo/002` 可以是指向 `/data/hch/GeoGuardGS_runtime/data/waymo/002` 的软链接；
- 正式配置使用原版 Waymo 图像范围：`selected_frames: [0, 198]`，`cameras: [0, 1, 2, 3, 4]`；
- 正式配置默认 `data.use_colmap=false`、`data.filter_colmap=false`，不依赖服务器系统 COLMAP；
- sparse LiDAR 必须保留 valid mask；
- invalid LiDAR 不能当作 depth 0；
- LiDAR 指标只在 valid pixels 上计算。

### 5.2 DA3 权重

```text
weights/da3/DA3-LARGE-1.1/
```

配置示例：

```yaml
geovit:
  enabled: true
  model_dir: weights/da3/DA3-LARGE-1.1
  local_files_only: true
```

### 5.3 Checkpoint

如果从已有模型继续训练：

```text
weights/streetgs/<experiment_name>/iteration_xxxx.pth
```

并在配置中设置：

```yaml
train:
  start_checkpoint: weights/streetgs/<experiment_name>/iteration_xxxx.pth
```

## 6. 实验组

| 组别 | 配置 | 作用 |
| --- | --- | --- |
| A | `a100_baseline_streetgs.yaml` | 原始 StreetGS baseline |
| B | `a100_da3_only.yaml` | DA3-only baseline，不启用 periodic controller |
| C | `a100_da3_periodic_group_softpatch.yaml` | DA3 主方法：周期性 group softpatch |
| D | `a100_da3_periodic_group_softpatch_opacity_reg.yaml` | C + opacity regularization loss |
| E | `a100_da3_periodic_group_softpatch_opacity_decay.yaml` | C + conservative opacity decay |
| F | `a100_lidar_supervised_reference.yaml` | LiDAR-supervised upper-bound reference |
| G | `a100_hybrid_reference.yaml` | hybrid diagnostic reference |

主论文/主实验建议重点比较 A/B/C/D/E。F 只作为 supervised upper-bound。G 只作为诊断参考。

## 6.1 配置层级

正式配置层级为：

```text
configs/base/geoguardgs_base.yaml
  -> configs/experiments/a100_*.yaml
  -> configs/smoke/a100_*_smoke.yaml
```

`scripts/train.py` 会在启动前递归展开 `_BASE_`，并把 `workspace` 固定为仓库根目录，生成到 `.geoguardgs_merged_configs/` 后再交给 StreetGS。这样 StreetGS 最终拿到的配置会包含：

```yaml
source_path: data/waymo/002
data:
  type: Waymo
  selected_frames: [0, 198]
  cameras: [0, 1, 2, 3, 4]
  split_train: 1
  split_test: -1
  white_background: false
  extent: 10
  use_colmap: false
  filter_colmap: false
resume: false
```

DA3-unsupervised 主线保持：

```yaml
optim:
  lambda_depth_lidar: 0.0
train:
  guided_feedback:
    use_lidar_depth: false
  feedback_controller:
    risk_source: da3_boundary
```

## 7. 周期性模块推荐参数

正式初版推荐：

```yaml
train:
  feedback_controller:
    enabled: true
    mode: feedback_update
    start_iter: 1000
    interval: 500
    max_triggers: -1
    risk_source: da3_boundary
    supervision_mode: da3_unsupervised
    feedback_mode: group_softpatch
    repair_mode: none
    contribution_source: live_current_model
    recompute_risk: true
    recompute_contribution: true
    recompute_responsible_groups: true
    recompute_softpatch: true
    max_regions: 30
    max_pixels_per_region: 64
    top_contributors: 8
    run_counterfactual: false
    run_candidate_tagging: true
    fail_policy: warn
```

参数解释：

- `start_iter=1000`：训练初期 Gaussian 过不稳定，过早反馈容易噪声大。
- `interval=500`：满足每 500 轮观察渲染图和指标。
- `max_regions=30`：正式实验比 smoke 更充分，但控制开销。
- `max_pixels_per_region=64`：限制 selected-pixel CUDA dump 成本。
- `top_contributors=8`：记录主要贡献 Gaussian，不做全图 contributor map。
- `run_counterfactual=false`：长训默认不做高成本 counterfactual。
- `run_candidate_tagging=true`：保留 repair dry-run 统计。

## 8. 每 500 轮输出要求

每个实验应输出到：

```text
outputs/a100_main_experiments/<experiment_name>/feedback_controller/iter_XXXXXX/
```

至少包含：

```text
rgb/
  rendered_rgb_<view>.png
  gt_rgb_<view>.png
  rgb_error_<view>.png

depth/
  rendered_depth_<view>.png
  da3_depth_<view>.png
  lidar_sparse_overlay_<view>.png
  depth_error_lidar_valid_<view>.png

risk_maps/
  da3_boundary_risk_<view>.png
  rendered_depth_edge_<view>.png
  edge_mismatch_<view>.png
  selected_risk_regions_<view>.png

contribution/
  top_contributor_overlay_<view>.png
  contribution_summary_<view>.json

gaussian_control/
  group_responsibility_overlay_<view>.png
  gaussian_control_manifest.json
  gaussian_repair_manifest.json
  safety_audit.json

panels/
  comparison_panel_<view>.png

metrics.json
```

如果 I/O 压力过大，至少保留：

- selected views 的 panels；
- 每 500 轮 compact metrics；
- feedback_controller manifest；
- safety audit；
- final full evaluation。

## 9. 每 500 轮指标

### 9.1 RGB

- RGB MAE / L1；
- PSNR；
- SSIM，可选；
- LPIPS，可选。

### 9.2 LiDAR-valid geometry

必须报告 `valid_lidar_count`：

- MAE；
- RMSE；
- AbsRel；
- delta < 1.25；
- valid_lidar_count。

区域：

- all_valid；
- boundary_band；
- canny_band；
- rendered_depth_edge_band；
- thin_structure_band；
- stable_non_boundary。

### 9.3 DA3 structure

DA3 不作为 metric depth truth，只看结构：

- DA3 / rendered edge mismatch；
- relative ranking violation；
- boundary-side consistency；
- DA3 top-risk local score；
- rendered-depth edge strength。

### 9.4 Responsibility / control

- selected risk pixel count；
- live CUDA valid count；
- low-evidence count；
- stable-id unmapped count；
- group count；
- protected Gaussian count；
- opacity regularized Gaussian count；
- opacity decayed Gaussian count；
- dry-run prune candidate count；
- dry-run shrink candidate count；
- dry-run split candidate count；
- RGB safety skipped count。

## 10. 安全检查

每个配置启动前：

```bash
python scripts/check_closed_loop_config.py \
  --config configs/experiments/a100_da3_periodic_group_softpatch.yaml

python scripts/validate_no_lidar_leakage.py \
  configs/experiments/a100_da3_periodic_group_softpatch.yaml

python scripts/validate_repair_safety.py \
  configs/experiments/a100_da3_periodic_group_softpatch_opacity_decay.yaml
```

必须确认：

- DA3-unsupervised 不使用 LiDAR training loss；
- LiDAR selected pixels 不用于 DA3 labeling；
- real prune/shrink/split 关闭；
- opacity_decay_apply 必须显式允许参数修改；
- 输出目录有效；
- data / weights 路径存在；
- live CUDA contribution 可用；
- checkpoint save interval 合理。

## 11. 并行启动

先 dry-run：

```bash
cd /data/projects/GeoGuardGS

python scripts/launch_a100_experiments.py \
  --configs \
    configs/experiments/a100_baseline_streetgs.yaml \
    configs/experiments/a100_da3_only.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch_opacity_reg.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch_opacity_decay.yaml \
    configs/experiments/a100_lidar_supervised_reference.yaml \
  --gpus 0,1,2,3 \
  --output-root outputs/a100_main_experiments \
  --dry-run
```

确认命令正确后正式运行：

```bash
python scripts/launch_a100_experiments.py \
  --configs \
    configs/experiments/a100_baseline_streetgs.yaml \
    configs/experiments/a100_da3_only.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch_opacity_reg.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch_opacity_decay.yaml \
    configs/experiments/a100_lidar_supervised_reference.yaml \
  --gpus 0,1,2,3 \
  --output-root outputs/a100_main_experiments
```

`scripts/train.py` 是 wrapper，会进入 `third_party/street_gaussian/train.py`，并把 `GitHub_main` 和 `third_party/street_gaussian` 加入 `PYTHONPATH`。

## 12. 断点续训

若训练中断：

1. 检查 `outputs/<exp_name>/checkpoints/` 或 StreetGS checkpoint 目录；
2. 设置配置中的 `train.start_checkpoint`；
3. 或使用 launcher 的 `--resume` 参数；
4. 确认 feedback controller 的 `skip_existing` 策略是否符合预期。

建议保留关键 checkpoint：

- 5000；
- 10000；
- 20000；
- 30000。

## 12.1 A100 smoke 命令

先确认 safety check：

```bash
python scripts/check_closed_loop_config.py --config configs/smoke/a100_baseline_streetgs_smoke.yaml
python scripts/check_closed_loop_config.py --config configs/smoke/a100_da3_only_smoke.yaml
python scripts/check_closed_loop_config.py --config configs/smoke/a100_da3_periodic_group_softpatch_smoke.yaml
python scripts/check_closed_loop_config.py --config configs/smoke/a100_da3_periodic_group_softpatch_opacity_reg_smoke.yaml
python scripts/check_closed_loop_config.py --config configs/smoke/a100_da3_periodic_group_softpatch_opacity_decay_smoke.yaml
python scripts/validate_no_lidar_leakage.py configs/smoke/a100_da3_only_smoke.yaml
python scripts/validate_no_lidar_leakage.py configs/smoke/a100_da3_periodic_group_softpatch_smoke.yaml
```

逐个启动 100 iter smoke：

```bash
python scripts/train.py --config configs/smoke/a100_baseline_streetgs_smoke.yaml
python scripts/train.py --config configs/smoke/a100_da3_only_smoke.yaml
python scripts/train.py --config configs/smoke/a100_da3_periodic_group_softpatch_smoke.yaml
python scripts/train.py --config configs/smoke/a100_da3_periodic_group_softpatch_opacity_reg_smoke.yaml
python scripts/train.py --config configs/smoke/a100_da3_periodic_group_softpatch_opacity_decay_smoke.yaml
```

正常日志应包含：

```text
[Checkpoint] resume=false; starting from scratch without checkpoint search.
[GuidedFeedback] disabled; skipping signal loading and supervision checks.
[FeedbackController] LiDAR supervision disabled; LiDAR is used for evaluation only.
[GaussianControl] disabled; skipping group evidence checks and control logic.
```

对于 DA3-only，正常现象是 `feedback_mode=global`，不要求 `signal_path`。
对于 periodic group softpatch，50/100 iter smoke 会触发 controller，并在输出目录生成 `feedback_controller/iter_000050/`、`feedback_controller/iter_000100/`。

## 12.2 正式实验命令

正式 30000 iter 使用：

```bash
python scripts/launch_a100_experiments.py \
  --configs \
    configs/experiments/a100_baseline_streetgs.yaml \
    configs/experiments/a100_da3_only.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch_opacity_reg.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch_opacity_decay.yaml \
    configs/experiments/a100_lidar_supervised_reference.yaml \
    configs/experiments/a100_hybrid_reference.yaml \
  --gpus 0,1,2,3 \
  --output-root outputs/a100_main_experiments
```

正式配置默认：

- `source_path=data/waymo/002`；
- `selected_frames=[0, 198]`；
- `cameras=[0, 1, 2, 3, 4]`；
- `resume=false`；
- `use_colmap=false`；
- `filter_colmap=false`；
- DA3-unsupervised 主线 `lambda_depth_lidar=0.0` 且 `use_lidar_depth=false`。

## 13. 结果收集

训练结束或中途检查：

```bash
python scripts/collect_experiment_outputs.py \
  --output-root outputs/a100_main_experiments \
  --out outputs/a100_main_experiments/collected_summary.json
```

需要重点查看：

- `metrics/rgb_metrics.csv`
- `metrics/lidar_geometry_metrics.csv`
- `metrics/da3_structure_metrics.csv`
- `feedback_controller/iter_XXXXXX/feedback_controller_manifest.json`
- `feedback_controller/iter_XXXXXX/responsible_group_summary.json`
- `feedback_controller/iter_XXXXXX/gaussian_control/gaussian_control_manifest.json`
- `feedback_controller/iter_XXXXXX/*safety_audit*.json`

## 13.1 论文证据包

训练与 final evaluation 完成后，一键整理论文需要的表格、manifest、audit 和可用图像：

```bash
python scripts/build_paper_evidence_pack.py \
  --output-root outputs/a100_main_experiments \
  --paper-dir outputs/paper_evidence
```

重点输出：

```text
outputs/paper_evidence/tables/main_final_metrics.csv
outputs/paper_evidence/tables/region_lidar_geometry_metrics.csv
outputs/paper_evidence/tables/feedback_trigger_summary.csv
outputs/paper_evidence/tables/safety_audit_summary.csv
outputs/paper_evidence/tables/repair_candidate_summary.csv
outputs/paper_evidence/tables/figure_index.csv
outputs/paper_evidence/tables/missing_evidence_report.csv
outputs/paper_evidence/figures/
outputs/paper_evidence/manifests/
```

论文写作前先看：

```bash
cat outputs/paper_evidence/tables/missing_evidence_report.csv
```

若仍存在 `paper_table_gap`、`method_evidence_gap` 或 `figure_gap`，先补对应 evaluation / panel / final metrics，再写强结论。

## 14. 常见问题

### CUDA extension import failed

重新编译：

```bash
pip install --no-build-isolation -e third_party/diff_gaussian_rasterization
pip install --no-build-isolation -e third_party/simple_knn
```

### DA3 权重找不到

确认：

```text
weights/da3/DA3-LARGE-1.1/
```

并设置：

```yaml
geovit:
  model_dir: weights/da3/DA3-LARGE-1.1
  local_files_only: true
```

### DA3-unsupervised 检查失败

检查：

```yaml
train.guided_feedback.use_lidar_depth: false
optim.lambda_depth_lidar: 0.0
train.feedback_controller.risk_source: da3_boundary
```

### repair safety 检查失败

确认：

```yaml
allow_real_prune: false
allow_real_split: false
allow_real_shrink: false
```

只有 `opacity_decay_apply` 可以设置：

```yaml
allow_parameter_modification: true
```

### 每 500 轮 I/O 太大

保留 numeric metrics 和 selected-view panels；不要全量保存所有 view 的 overlay。最终 evaluation 再全量生成。

## 15. 禁止事项

当前正式实验禁止：

- real prune；
- real shrink；
- real split；
- real surface-align；
- DA3-unsupervised 使用 LiDAR training loss；
- DA3 depth 当 metric ground truth；
- invalid LiDAR 当 depth 0；
- 提交数据、权重、checkpoint、outputs、编译产物。

当前唯一允许的真实 Gaussian 参数处理是：

```text
opacity_decay_apply
```

并且必须显式开启：

```yaml
train.gaussian_control.control_mode: opacity_decay_apply
train.gaussian_control.allow_parameter_modification: true
train.gaussian_control.allow_real_prune: false
train.gaussian_control.allow_real_split: false
train.gaussian_control.allow_real_shrink: false
```
