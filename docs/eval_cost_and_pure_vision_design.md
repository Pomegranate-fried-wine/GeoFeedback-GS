# Eval Cost and Pure-Vision DA3 Design

## 1000-iteration full-split eval cost audit

Current full-scene data defaults:

```yaml
data.selected_frames: [0, 198]
data.cameras: [0, 1, 2, 3, 4]
data.split_train: 1
data.split_test: -1
```

This means approximately `199 x 5 = 995` train cameras and no held-out test
cameras under the current `split_train=1` protocol. The previous
`7cc33a8` behavior evaluated the full train split every 1000 iterations:

```text
995 renders/eval x 30 evals = 29,850 extra renders per experiment
29,850 x 3 groups = 89,550 extra renders for A/B/C
```

That is too expensive for formal 30000-iteration training, especially because
evaluation render calls also produce depth/acc tensors and write per-view CSV.

## New training-time eval policy

The training loop now separates sampled diagnostics from full-split training
evaluation:

```yaml
train.eval_sampled_interval: 1000
train.eval_full_interval: 5000
train.eval_panel_interval: 1000
train.eval_sampled_train_view_count: 15
train.eval_sampled_test_view_count: 15
train.eval_full_train_enabled: true
train.eval_full_test_enabled: true
```

Outputs:

```text
metrics/eval_sampled_iter_XXXXXX_per_view.csv
metrics/eval_summary_sampled.csv
metrics/eval_full_iter_XXXXXX_per_view.csv
metrics/eval_summary_full.csv
periodic_eval/iter_XXXXXX/
```

For 30000 iterations, full split eval now runs at 5000, 10000, 15000, 20000,
25000, and 30000: about `995 x 6 = 5,970` extra renders per experiment, or
17,910 for A/B/C. Sampled diagnostics add only `15 x 30 = 450` train renders
per experiment, plus test samples if a held-out split exists.

Final paper tables should still use:

```bash
python scripts/final_evaluate_experiments.py \
  --configs configs/experiments/a100_baseline_streetgs.yaml configs/experiments/a100_da3_only.yaml configs/experiments/a100_da3_periodic_group_softpatch.yaml \
  --output-root outputs/final_evaluation_full_scene_v2 \
  --loaded-iter 30000
```

## Pure-vision + DA3 object-aware prototype

Goal:

- no LiDAR pointcloud initialization;
- no LiDAR depth supervision;
- object branch retained with `include_obj=true`;
- background initialized from COLMAP/SfM;
- object Gaussians initialized from track-box random sampling;
- DA3 provides unsupervised structure/risk signals.

Current code support:

- Background no-LiDAR initialization already supports COLMAP-only pointclouds.
- `GaussianModelActor.create_from_pcd()` already falls back to random points in
  the object 3D box when no `points3D_obj_XXX.ply` exists or the object ply is
  too small.
- The manifest now records separate background/object LiDAR usage fields.

Prototype configs:

```text
configs/experiments_pure_vision/a100_pv_baseline_colmap_obj.yaml
configs/experiments_pure_vision/a100_pv_da3_pseudo_depth_obj.yaml
configs/experiments_pure_vision/a100_pv_da3_feedback_obj.yaml
```

These are smoke/prototype configs, not replacements for the current A/B/C
formal experiments. They use `selected_frames: [0, 14]`, all five cameras, and
3000 iterations.

Expected manifest fields:

```json
{
  "uses_lidar_initialization": false,
  "uses_lidar_background_initialization": false,
  "uses_lidar_object_initialization": false,
  "uses_lidar_supervision": false,
  "background_init_source": "colmap",
  "object_init_source": "random_box",
  "da3_pseudo_point_count": 0,
  "scale_alignment_source": "colmap_sfm",
  "lidar_point_count_used_for_init": 0,
  "lidar_object_point_count_used_for_init": 0
}
```

Main open risks:

- COLMAP may be sparse on dynamic objects.
- Random-box object initialization may need stronger box regularization or
  DA3/mask-guided object losses to avoid transparent or drifting vehicles.
- DA3 monocular depth scale must be aligned without LiDAR; the current smoke
  uses COLMAP/SfM scale only. Future work can add DA3 pseudo pointclouds with
  confidence filtering and multi-view consistency.

Recommended short-term sequence:

1. Run A/B/C formal full-scene experiments with LiDAR initialization.
2. Run pure-vision smoke after A/B/C has started or completed:

```bash
python scripts/check_closed_loop_config.py --config configs/experiments_pure_vision/a100_pv_baseline_colmap_obj.yaml
python scripts/check_closed_loop_config.py --config configs/experiments_pure_vision/a100_pv_da3_pseudo_depth_obj.yaml
python scripts/check_closed_loop_config.py --config configs/experiments_pure_vision/a100_pv_da3_feedback_obj.yaml

python scripts/train.py --config configs/experiments_pure_vision/a100_pv_baseline_colmap_obj.yaml
python scripts/train.py --config configs/experiments_pure_vision/a100_pv_da3_pseudo_depth_obj.yaml
python scripts/train.py --config configs/experiments_pure_vision/a100_pv_da3_feedback_obj.yaml
```

Success criteria for the smoke:

- initialization manifest reports zero LiDAR usage;
- object branch is active and vehicles are visible in panels;
- RGB/depth panels do not collapse by 3000 iterations;
- DA3 feedback manifests report no LiDAR selected pixels or supervision.
