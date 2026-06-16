# GeoFeedback-GS

**GeoFeedback-GS: LiDAR-Reduced Dynamic Street Gaussian Reconstruction with Responsible Gaussian Feedback**

**GeoFeedback-GS：基于责任高斯反馈的低 LiDAR 依赖动态街景高斯重建**

GeoFeedback-GS 是一个研究型代码库，用于研究如何在保留动态目标建模能力的前提下，降低面向自动驾驶街景重建的 object-aware dynamic street Gaussian reconstruction 对 LiDAR 的依赖。它不是从零重新实现的新渲染器，而是在 Street Gaussians 风格框架的基础上，加入经过审计的实验设置、DA3 相对结构引导、责任高斯反馈、保留测试集评估以及面向论文/汇报的结果整理流程。

当前项目的研究边界是有意收敛的：在保留动态车辆建模的同时，减少并审计 LiDAR 的使用。我们将 LiDAR 的使用拆分为三个不同角色：

1. **LiDAR 初始化**：背景/目标高斯是否由 LiDAR 点云初始化。
2. **LiDAR 训练监督**：训练过程中是否将 LiDAR 深度作为 loss 或选点风险来源。
3. **LiDAR 评估参考**：训练结束后是否仅使用保留 LiDAR 作为几何评估参考。

这个区分很重要。一个方法可以去掉训练阶段的 LiDAR 监督，但仍然使用 LiDAR 初始化；一个无 LiDAR 训练设置也可能仍然依赖相机位姿、SfM/COLMAP 以及目标跟踪框。

## 项目概览

GeoFeedback-GS 当前围绕四组正式实验展开：

| 组别 | 名称 | 初始化方式 | 训练监督 | 反馈机制 | 作用 |
| --- | --- | --- | --- | --- | --- |
| A | LiDAR-supervised StreetGS Reference | COLMAP + LiDAR 背景/目标初始化 | RGB + LiDAR 深度监督 | 无 | 接近原 StreetGS 风格 LiDAR 辅助流程的参考设置 |
| B | No-LiDAR-Supervision Control | COLMAP + LiDAR 初始化 | RGB/目标训练，不使用 LiDAR 深度监督 | 无周期反馈 | 控制去除 LiDAR 训练监督后的影响 |
| C | LiDAR-init GeoFeedback-GS | COLMAP + LiDAR 初始化 | 不使用 LiDAR 深度监督；由 softpatch 反馈激活 DA3 相对结构损失 | 有 | 在 LiDAR 初始化条件下测试责任高斯反馈 |
| PV-C | LiDAR-free GeoFeedback-GS | COLMAP 背景初始化 + random-box 目标初始化 | 不使用 LiDAR 初始化，不使用 LiDAR 深度监督；由 softpatch 反馈激活 DA3 相对结构损失 | 有 | 在 pose-and-box supervision 条件下测试无 LiDAR 的 object-aware 重建 |

关于 B 组的重要实现说明：当前 `global` guided-feedback 模式可能不会真正激活 DA3 structure pixels，因为已经实现的 DA3 structure loss 只选择 `feedback_weight > 1.0` 的像素，而 global 模式返回的是全 1 权重。因此，除非训练日志确认 `guided_feedback_da3_structure_loss` 非零，否则 B 组应描述为 no-LiDAR-supervision control，而不是过度表述为完全生效的 DA3-only 方法。

## 核心思想

GeoFeedback-GS 引入了一个 responsible Gaussian feedback loop：

```text
risk region selection
-> responsible Gaussian group attribution
-> softpatch region weight map
-> DA3 relative-structure loss on selected regions
-> periodic audit outputs
```

DA3 不被视为 metric depth ground truth。训练信号被定义为相对结构先验，主要包括：

- 边缘一致性；
- 局部深度排序一致性；
- 边界两侧一致性。

反馈损失不是一个独立的新 loss。在当前实现路径中，feedback 生成区域/像素权重图。被选中的 softpatch 区域获得大于 1 的权重，这些权重进一步激活并调制 DA3 structure loss。

## 方法流程

主训练路径如下：

```text
StreetGS-style object-aware rendering
-> RGB/object/regularization losses
-> optional LiDAR depth loss for A only
-> DA3 bridge for relative depth structure
-> periodic feedback controller for C and PV-C
-> feedback_signal.json
-> GuidedFeedbackController.update_signal_path(...)
-> softpatch region weight map
-> da3_edge_loss + da3_ranking_loss + da3_side_loss
-> guided_feedback_da3_structure_loss
```

相关实现文件包括：

- `third_party/street_gaussian/train.py`
  - `compute_guided_feedback_loss`
  - `compute_da3_structure_guided_loss`
- `third_party/street_gaussian/lib/utils/da3_structure_feedback_utils.py`
  - `make_da3_bridge`
  - `da3_structure_loss`
  - `da3_edge_loss`
  - `da3_ranking_loss`
  - `da3_side_loss`
- `third_party/street_gaussian/lib/utils/guided_feedback_utils.py`
  - `GuidedFeedbackController`
  - `make_region_weight_map`
  - `feedback_weight > 1.0` 激活行为
- `third_party/street_gaussian/lib/utils/feedback_controller.py`
  - 周期触发与 feedback signal 加载
- `third_party/street_gaussian/lib/models/gaussian_model_actor.py`
  - PV-C random-box actor Gaussian initialization fallback

## 实验设计

正式实验配置位于 `configs/experiments/`：

```text
configs/experiments/a100_baseline_streetgs.yaml
configs/experiments/a100_da3_only.yaml
configs/experiments/a100_da3_periodic_group_softpatch.yaml
configs/experiments/a100_pv_da3_feedback_obj.yaml
```

当前推荐解释如下：

- **A** 是 LiDAR-supervised StreetGS reference。
- **B** 去除了训练阶段 LiDAR 监督，但仍保留 LiDAR 初始化；除非训练日志确认 DA3 loss 已激活，否则不要将其过度表述为完全生效的 DA3-only 方法。
- **C** 使用 LiDAR 初始化，不使用 LiDAR 训练监督，并引入 feedback-activated DA3 relative-structure loss。
- **PV-C** 使用 COLMAP 背景初始化、random-box 目标初始化，不使用 LiDAR 初始化，不使用 LiDAR 训练监督，并引入 feedback-activated DA3 relative-structure loss。

PV-C 应表述为 **LiDAR-free under pose-and-box supervision**，而不是完全无约束的单目重建。它仍然使用相机位姿、COLMAP/SfM 背景点以及目标跟踪框。

## 安装

推荐环境：带 CUDA 的 Linux 服务器，建议使用 A100 级别 GPU。

```bash
git clone <repo-url> GeoFeedback-GS
cd GeoFeedback-GS
conda env create -f environment.yml
conda activate geoguardgs
pip install -r requirements.txt
```

在目标服务器上重新编译 CUDA/C++ 扩展：

```bash
bash scripts/install_server_extensions.sh
python scripts/check_imports.py
python scripts/verify_migration_package.py
```

可能需要本地编译的主要扩展包括：

- `third_party/street_gaussian/submodules/diff-gaussian-rasterization`，或当前 checkout 使用的迁移版 rasterizer 路径；
- `third_party/simple_knn`；
- 若启用，则包括 `third_party/nvdiffrast`；
- 若服务器设置需要，则包括 Waymo reader 相关依赖。

不要提交已编译的 `.so`、`.pyd`、`.dll`、`build/` 或 `*.egg-info` 产物。

## 数据准备

本仓库不包含 Waymo 数据、DA3 权重、COLMAP 输出、大型 checkpoint 或训练结果。

期望的本地目录结构如下：

```text
data/
  waymo/
    002/
weights/
  da3/
    DA3-LARGE-1.1/
  streetgs/
outputs/
```

对于 COLMAP 初始化，需要通过配置或环境变量提供可用的 COLMAP binary：

```bash
export COLMAP_BIN=/path/to/colmap
python scripts/check_colmap_environment.py --config configs/experiments/a100_pv_da3_feedback_obj.yaml
```

## 训练命令

以下命令都会将日志实时输出到终端，并通过 `tee` 同步保存日志文件。请通过 `CUDA_VISIBLE_DEVICES` 选择空闲 GPU。

### A：LiDAR-supervised StreetGS Reference

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train.py \
  --config configs/experiments/a100_baseline_streetgs.yaml \
  2>&1 | tee logs/A_lidar_supervised_streetgs.log
```

输出目录：

```text
outputs/a100_main_experiments/baseline_streetgs/
```

### B：No-LiDAR-Supervision Control

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/train.py \
  --config configs/experiments/a100_da3_only.yaml \
  2>&1 | tee logs/B_no_lidar_supervision_control.log
```

输出目录：

```text
outputs/a100_main_experiments/da3_only/
```

### C：LiDAR-init GeoFeedback-GS

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/train.py \
  --config configs/experiments/a100_da3_periodic_group_softpatch.yaml \
  2>&1 | tee logs/C_lidar_init_geofeedback_gs.log
```

输出目录：

```text
outputs/a100_main_experiments/da3_periodic_group_softpatch/
```

### PV-C：LiDAR-free GeoFeedback-GS

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/train.py \
  --config configs/experiments/a100_pv_da3_feedback_obj.yaml \
  2>&1 | tee logs/PVC_lidar_free_geofeedback_gs.log
```

输出目录：

```text
outputs/a100_main_experiments/pv_da3_feedback_obj/
```

长时间训练前，建议先验证配置：

```bash
python scripts/check_closed_loop_config.py --config configs/experiments/a100_baseline_streetgs.yaml
python scripts/check_closed_loop_config.py --config configs/experiments/a100_da3_only.yaml
python scripts/check_closed_loop_config.py --config configs/experiments/a100_da3_periodic_group_softpatch.yaml
python scripts/check_closed_loop_config.py --config configs/experiments/a100_pv_da3_feedback_obj.yaml
```

## 评估命令

### 最终保留测试集 RGB 评估

使用 held-out test split 作为论文主表的主要来源：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/final_evaluate_experiments.py \
  --configs \
    configs/experiments/a100_baseline_streetgs.yaml \
    configs/experiments/a100_da3_only.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch.yaml \
    configs/experiments/a100_pv_da3_feedback_obj.yaml \
  --output-root outputs/final_evaluation_test_only_v2 \
  --loaded-iter 30000 \
  --splits test \
  2>&1 | tee logs/final_eval_test_only_v2.log
```

预期输出：

```text
outputs/final_evaluation_test_only_v2/
  summary_main.csv
  summary_by_scope.csv
  <experiment>/
    metrics_full_image.csv
    metrics_object_region.csv
    metrics_background_region.csv
    figures/
```

### 几何一致性评估

该评估仅将 held-out LiDAR 作为评估参考，不应与训练监督声明混淆。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_geometry_consistency.py \
  --configs \
    configs/experiments/a100_baseline_streetgs.yaml \
    configs/experiments/a100_da3_only.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch.yaml \
    configs/experiments/a100_pv_da3_feedback_obj.yaml \
  --output-root outputs/geometry_eval_test_only_v1 \
  --loaded-iter 30000 \
  --split test \
  2>&1 | tee logs/geometry_eval_test_only_v1.log
```

预期输出：

```text
outputs/geometry_eval_test_only_v1/
  compare_geometry_summary.csv
  <experiment>/
    per_view_geometry_metrics.csv
    summary_geometry_metrics.csv
    visualization_panels/
```

### 论文证据包

完成最终评估和几何评估后，运行：

```bash
python scripts/build_paper_evidence_pack.py \
  --output-root outputs/a100_main_experiments \
  --final-eval-root outputs/final_evaluation_test_only_v2 \
  --geometry-eval-root outputs/geometry_eval_test_only_v1 \
  --paper-dir outputs/paper_evidence_geofeedback_gs

python scripts/build_paper_result_visuals.py \
  --paper-dir outputs/paper_evidence_geofeedback_gs
```

### 训练过程图集

固定视角周期性面板属于训练诊断输出，不是最终论文指标协议：

```bash
python scripts/build_paper_training_gallery.py \
  --output-root outputs/paper_training_gallery_geofeedback_gs
```

## 仓库结构

```text
GeoFeedback-GS/
  assets/                         # 仅存放轻量静态资源
  configs/
    base/                         # 共享配置默认项
    experiments/                  # 正式 A/B/C/PV-C 实验配置
    experiments_pure_vision/      # PV-C 配置链
    short_5000/                   # 短程诊断配置
    smoke/                        # smoke-test 配置
  data/                           # 本地数据挂载点，git 忽略
  docs/                           # 协议说明、审计记录、草稿、服务器指南
  geoguardgs/                     # 项目辅助模块，保留历史包名
  scripts/                        # 官方入口脚本与结果打包工具
  third_party/                    # StreetGS 风格框架与相关依赖
  weights/                        # 本地权重挂载点，git 忽略
  outputs/                        # 本地输出目录，git 忽略
```

部分内部 package 名称和历史文件路径仍包含 `geoguardgs`。这些名称被保留是为了避免破坏已有配置、import 和服务器脚本。仓库对外展示的项目名称是 GeoFeedback-GS。

## 结果摘要

当前单场景 Waymo held-out 实验支持一个有边界的结论：

- PV-C 可以在不使用 LiDAR 初始化、也不使用 LiDAR 训练监督的情况下，完成 object-aware dynamic reconstruction 训练。
- 在当前 held-out test split 上，PV-C 的 RGB PSNR/SSIM 接近或略高于 LiDAR-supervised StreetGS reference。
- C 和 PV-C 成功运行了 periodic responsible-feedback chain，并生成了审计产物。

不要过度表述结果：

- 这不能证明 GeoFeedback-GS 全面优于 StreetGS。
- 这不能证明绝对 metric geometry 已经被解决。
- 这不能证明跨场景泛化能力。
- 全图 RGB 指标不足以证明几何可靠性；几何相关结论仍需要 held-out LiDAR geometry evaluation 支撑。

## 可视化输出

训练与评估脚本会生成几类互补输出：

- `periodic_eval/`：训练过程中的固定视角 RGB/depth 诊断面板。
- `feedback_controller/`：risk、contribution、responsible group、softpatch signal 和 audit manifests。
- `final_evaluation_test_only_v2/`：面向论文的 held-out RGB/object/background 指标。
- `geometry_eval_test_only_v1/`：held-out geometry consistency 指标和面板。
- `paper_evidence_geofeedback_gs/`：用于论文/PPT 写作的精简表格和图像。

周期性面板适合调试训练动态。主结果表应使用最终 held-out evaluation；几何相关声明应由 geometry evaluation script 支撑。

## 局限性

- 当前正式证据基于有限 Waymo 设置，后续需要扩展到更多场景。
- 由于当前 global-weight 激活行为，B 组可能不是完全生效的 DA3-only loss 设置。
- PV-C 避免了 LiDAR 初始化和 LiDAR 监督，但仍依赖相机位姿、COLMAP/SfM 和目标跟踪框。
- 当前 feedback 提供的是可审计机制和局部结构引导；在缺少更多消融前，不应宣称它能普遍提升最终指标。
- DA3 相对结构不能替代 metric depth evaluation。

## 引用与致谢

本项目继承并修改了 Street Gaussians 风格的动态街景重建框架，并使用了 Gaussian rasterization、Waymo 数据读取器、COLMAP/SfM 工具和 Depth Anything 3 等第三方组件。在重新发布或论文投稿前，请核查上游许可证和引用要求。

`CITATION.cff` 中的 citation metadata 目前仍是占位内容，公开发布前应更新。
