# GeoFeedback-GS

**GeoFeedback-GS: LiDAR-Reduced Dynamic Street Gaussian Reconstruction with Responsible Gaussian Feedback**

**GeoFeedback-GS：基于责任高斯反馈的低 LiDAR 依赖动态街景高斯重建**

GeoFeedback-GS is a research codebase for studying how LiDAR dependency can be reduced in object-aware dynamic street Gaussian reconstruction. It is not a new renderer from scratch. The project builds on a Street Gaussians-style framework and adds audited experiment settings, DA3 relative-structure guidance, responsible Gaussian feedback, held-out evaluation, and paper-facing result packaging.

The current project focus is deliberately bounded: reduce and audit LiDAR usage while preserving dynamic vehicle modeling. We separate LiDAR usage into three roles:

1. **LiDAR initialization**: whether background/object Gaussians are initialized from LiDAR point clouds.
2. **LiDAR training supervision**: whether LiDAR depth is used as a training loss or selected-pixel risk source.
3. **LiDAR evaluation reference**: whether held-out LiDAR is used only after training for geometry evaluation.

This distinction is important. A method can remove LiDAR training supervision while still using LiDAR initialization, and a LiDAR-free training setup may still rely on camera poses, SfM/COLMAP, and object track boxes.

## Overview

GeoFeedback-GS studies four formal experiment groups:

| Group | Name | Initialization | Training supervision | Feedback | Role |
| --- | --- | --- | --- | --- | --- |
| A | LiDAR-supervised StreetGS Reference | COLMAP + LiDAR background/object initialization | RGB + LiDAR depth supervision | No | Reference setting close to the original StreetGS-style LiDAR-assisted pipeline |
| B | No-LiDAR-Supervision Control | COLMAP + LiDAR initialization | RGB/object training without LiDAR depth supervision | No periodic feedback | Controls the effect of removing LiDAR training supervision |
| C | LiDAR-init GeoFeedback-GS | COLMAP + LiDAR initialization | No LiDAR depth supervision; DA3 relative-structure loss activated by softpatch feedback | Yes | Tests responsible Gaussian feedback under LiDAR initialization |
| PV-C | LiDAR-free GeoFeedback-GS | COLMAP background + random-box object initialization | No LiDAR initialization and no LiDAR depth supervision; DA3 relative-structure loss activated by softpatch feedback | Yes | Tests LiDAR-free object-aware reconstruction under pose-and-box supervision |

Important implementation note for B: the current `global` guided-feedback mode may not activate DA3 structure pixels because the implemented DA3 structure loss selects pixels with `feedback_weight > 1.0`, while global mode returns all-one weights. Therefore B should be described as a no-LiDAR-supervision control unless logs confirm non-zero `guided_feedback_da3_structure_loss`.

## Key Idea

GeoFeedback-GS adds a responsible Gaussian feedback loop:

```text
risk region selection
-> responsible Gaussian group attribution
-> softpatch region weight map
-> DA3 relative-structure loss on selected regions
-> periodic audit outputs
```

DA3 is not treated as metric depth ground truth. The training signal is a relative structure prior:

- edge consistency,
- local depth ranking consistency,
- boundary-side consistency.

The feedback loss is not an independent new loss. In the implemented training path, feedback creates a region/pixel weight map. The selected softpatch regions receive weights greater than one, and these weights activate/modulate the DA3 structure loss.

Periodic feedback is stateful between controller triggers. When a trigger writes a new `feedback_signal.json`, training calls `guided_feedback.update_signal_path(...)`; subsequent iterations keep using that loaded signal until the next trigger replaces it.

## Method Pipeline

The main training path is:

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

The relevant implementation files are:

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
  - `feedback_weight > 1.0` activation behavior
- `third_party/street_gaussian/lib/utils/feedback_controller.py`
  - periodic trigger and feedback signal loading
- `third_party/street_gaussian/lib/models/gaussian_model_actor.py`
  - PV-C random-box actor Gaussian initialization fallback

## Experiment Design

The formal experiments are configured under `configs/experiments/`:

```text
configs/experiments/a100_baseline_streetgs.yaml
configs/experiments/a100_no_lidar_supervision_control.yaml
configs/experiments/a100_da3_periodic_group_softpatch.yaml
configs/experiments/a100_pv_da3_feedback_obj.yaml
```

The current recommended interpretation is:

- **A** is the LiDAR-supervised StreetGS reference.
- **B** removes training-time LiDAR supervision while retaining LiDAR initialization; it is not the old DA3-only setting.
- **C** uses LiDAR initialization, no LiDAR training supervision, and feedback-activated DA3 relative-structure loss.
- **PV-C** uses COLMAP background initialization, random-box object initialization, no LiDAR initialization, no LiDAR training supervision, and feedback-activated DA3 relative-structure loss.

PV-C should be described as **LiDAR-free under pose-and-box supervision**, not as fully unconstrained monocular reconstruction. It still uses camera poses, COLMAP/SfM background points, and object track boxes.

The four official training output roots are fixed by each config:

```text
outputs/A_streetgs_lidar_init_lidar_sup/
outputs/B_lidar_init_no_lidar_sup/
outputs/C_lidar_init_da3_feedback/
outputs/PVC_no_lidar_init_da3_feedback/
```

All model checkpoints, saved Gaussians, RGB/depth snapshots, metrics, logs, and feedback-controller artifacts for a group are written under that group's directory.

## Installation

Recommended environment: Linux server with CUDA and A100-class GPUs.

```bash
git clone <repo-url> GeoFeedback-GS
cd GeoFeedback-GS
conda env create -f environment.yml
conda activate geoguardgs
pip install -r requirements.txt
```

Rebuild CUDA/C++ extensions on the target server:

```bash
bash scripts/install_server_extensions.sh
python scripts/check_imports.py
python scripts/verify_migration_package.py
```

The main extensions that may require local compilation are:

- `third_party/street_gaussian/submodules/diff-gaussian-rasterization` or the migrated rasterizer path used by the checkout,
- `third_party/simple_knn`,
- `third_party/nvdiffrast` if enabled,
- Waymo reader dependencies if required by the server setup.

Do not commit compiled `.so`, `.pyd`, `.dll`, `build/`, or `*.egg-info` artifacts.

## Data Preparation

This repository does not include Waymo data, DA3 weights, COLMAP outputs, large checkpoints, or training results.

Expected local layout:

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

For COLMAP initialization, provide a working COLMAP binary either through config or environment:

```bash
export COLMAP_BIN=/path/to/colmap
python scripts/check_colmap_environment.py --config configs/experiments/a100_pv_da3_feedback_obj.yaml
```

## Training Commands

All commands below stream logs to the console and save them with `tee`. Choose idle GPUs through `CUDA_VISIBLE_DEVICES`.

### A: LiDAR-supervised StreetGS Reference

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train.py \
  --config configs/experiments/a100_baseline_streetgs.yaml \
  2>&1 | tee logs/A_lidar_supervised_streetgs.log
```

Output:

```text
outputs/A_streetgs_lidar_init_lidar_sup/
```

### B: No-LiDAR-Supervision Control

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/train.py \
  --config configs/experiments/a100_no_lidar_supervision_control.yaml \
  2>&1 | tee logs/B_no_lidar_supervision_control.log
```

Output:

```text
outputs/B_lidar_init_no_lidar_sup/
```

### C: LiDAR-init GeoFeedback-GS

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/train.py \
  --config configs/experiments/a100_da3_periodic_group_softpatch.yaml \
  2>&1 | tee logs/C_lidar_init_geofeedback_gs.log
```

Output:

```text
outputs/C_lidar_init_da3_feedback/
```

### PV-C: LiDAR-free GeoFeedback-GS

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/train.py \
  --config configs/experiments/a100_pv_da3_feedback_obj.yaml \
  2>&1 | tee logs/PVC_lidar_free_geofeedback_gs.log
```

Output:

```text
outputs/PVC_no_lidar_init_da3_feedback/
```

Before long runs, validate configs:

```bash
python scripts/check_closed_loop_config.py --config configs/experiments/a100_baseline_streetgs.yaml
python scripts/check_closed_loop_config.py --config configs/experiments/a100_no_lidar_supervision_control.yaml
python scripts/check_closed_loop_config.py --config configs/experiments/a100_da3_periodic_group_softpatch.yaml
python scripts/check_closed_loop_config.py --config configs/experiments/a100_pv_da3_feedback_obj.yaml
python scripts/verify_four_group_configs.py
```

## Evaluation Commands

### Final held-out RGB evaluation

Use the held-out test split as the main paper table source:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/final_evaluate_experiments.py \
  --configs \
    configs/experiments/a100_baseline_streetgs.yaml \
    configs/experiments/a100_no_lidar_supervision_control.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch.yaml \
    configs/experiments/a100_pv_da3_feedback_obj.yaml \
  --output-root outputs/final_evaluation_test_only_v2 \
  --loaded-iter 30000 \
  --splits test \
  2>&1 | tee logs/final_eval_test_only_v2.log
```

Expected outputs:

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

### Geometry consistency evaluation

This evaluation uses held-out LiDAR only as an evaluation reference. It should not be mixed with training supervision claims.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_geometry_consistency.py \
  --configs \
    configs/experiments/a100_baseline_streetgs.yaml \
    configs/experiments/a100_no_lidar_supervision_control.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch.yaml \
    configs/experiments/a100_pv_da3_feedback_obj.yaml \
  --output-root outputs/geometry_eval_test_only_v1 \
  --loaded-iter 30000 \
  --split test \
  2>&1 | tee logs/geometry_eval_test_only_v1.log
```

Expected outputs:

```text
outputs/geometry_eval_test_only_v1/
  compare_geometry_summary.csv
  <experiment>/
    per_view_geometry_metrics.csv
    summary_geometry_metrics.csv
    visualization_panels/
```

### Paper evidence pack

After final evaluation and geometry evaluation:

```bash
python scripts/build_paper_evidence_pack.py \
  --output-root outputs \
  --final-eval-root outputs/final_evaluation_test_only_v2 \
  --geometry-eval-root outputs/geometry_eval_test_only_v1 \
  --paper-dir outputs/paper_evidence_geofeedback_gs

python scripts/build_paper_result_visuals.py \
  --paper-dir outputs/paper_evidence_geofeedback_gs
```

### Training galleries

The fixed-view periodic panels are training diagnostics, not the final paper metric protocol:

```bash
python scripts/build_paper_training_gallery.py \
  --output-root outputs/paper_training_gallery_geofeedback_gs
```

## Repository Structure

```text
GeoFeedback-GS/
  assets/                         # lightweight static assets only
  configs/
    base/                         # shared config defaults
    experiments/                  # formal A/B/C/PV-C configs
    experiments_pure_vision/      # PV-C config chain
    short_5000/                   # short diagnostic configs
    smoke/                        # smoke-test configs
  data/                           # local data mount point, ignored by git
  docs/                           # protocol notes, audits, drafts, server guide
  geoguardgs/                     # project helper modules retained under historical package name
  scripts/                        # official entrypoints and packaging utilities
  third_party/                    # StreetGS-style framework and dependencies
  weights/                        # local weights mount point, ignored by git
  outputs/                        # local outputs, ignored by git
```

Some internal package names and historical file paths still contain `geoguardgs`. They are retained to avoid breaking existing configs, imports, and server scripts. The repository-facing project name is GeoFeedback-GS.

## Results Summary

Current single-scene Waymo held-out experiments support a bounded claim:

- PV-C can train with object-aware dynamic reconstruction while using no LiDAR initialization and no LiDAR training supervision.
- On the current held-out test split, PV-C reaches RGB PSNR/SSIM close to or slightly above the LiDAR-supervised StreetGS reference.
- C and PV-C successfully exercise the periodic responsible-feedback chain and produce audit artifacts.

Do not overstate the result:

- It does not prove that GeoFeedback-GS comprehensively outperforms StreetGS.
- It does not prove absolute metric geometry is solved.
- It does not establish cross-scene generalization.
- Full-image RGB metrics are insufficient to prove geometry reliability; held-out LiDAR geometry evaluation remains necessary.

## Visualization Outputs

Training and evaluation scripts produce several complementary output types:

- `metrics/train_loss_trace.csv` and `metrics/train_loss_trace.jsonl`: per-iteration loss, training PSNR, guided feedback, DA3, and control scalars.
- `metrics/eval_summary_full.csv`: full train/test evaluation summary every 5000 iterations.
- `periodic_eval/`: fixed-view RGB/depth diagnostic panels during training. At iterations `1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000`, it switches to train+test all-view snapshots, writes `snapshot_manifest.json`, and stores single-view files under `gt/`, `rgb/`, `depth/`, plus C/PV-C `feedback_risk_score/`, `feedback_responsible_groups/`, and `feedback_risk_and_groups/`.
- `feedback_controller/`: risk, contribution, responsible group, softpatch signal, and audit manifests. C/PV-C cache original-view DA3 priors under `da3_risk_cache/`, write dynamic risk matrices under each trigger's `risk_stage/` (`*_risk_score_matrix.npy`, heatmaps, risk-band masks, rendered-RGB quality risk maps), per-risk-pixel Gaussian contribution dumps under `contribution_stage/`, grouped responsibility tables under `responsible_group_stage/`, and original-resolution overlays under `visual_overlays/`.
- `final_visual_eval/`: generated after training by `python scripts/build_final_experiment_visuals.py`; contains `GT | A | PV-C` RGB montages, optional `GT | A | B | C | PV-C` RGB montages, feedback matrix visualizations, loss curves, PSNR curves, and `final_visual_eval_manifest.json`.
- `final_evaluation_test_only_v2/`: paper-grade held-out RGB/object/background metrics.
- `geometry_eval_test_only_v1/`: held-out geometry consistency metrics and panels.
- `paper_evidence_geofeedback_gs/`: compact tables and figures for paper/PPT writing.

Periodic panels are useful for debugging training dynamics. The main result table should use final held-out evaluation, with geometry claims supported by the geometry evaluation script.

## Limitations

- Current formal evidence is based on a limited Waymo setting and should be expanded to more scenes.
- B is the no-LiDAR-supervision control with LiDAR initialization; it should not be described as DA3-only.
- PV-C avoids LiDAR initialization and LiDAR supervision, but still relies on camera poses, COLMAP/SfM, and object track boxes.
- Feedback currently provides an auditable mechanism and local structure guidance; it should not be claimed as a universal final-metric improvement without additional ablations.
- DA3 relative structure does not replace metric depth evaluation.

## Citation / Acknowledgement

This project inherits and modifies a Street Gaussians-style dynamic street reconstruction framework and uses third-party components such as Gaussian rasterization, Waymo data readers, COLMAP/SfM tooling, and Depth Anything 3. Please verify upstream licenses and citations before redistribution or paper submission.

Citation metadata is currently a placeholder in `CITATION.cff` and should be updated before public release.
