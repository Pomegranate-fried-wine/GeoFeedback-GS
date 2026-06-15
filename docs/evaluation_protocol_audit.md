# Evaluation Protocol Audit

## Current training-time evaluation

`third_party/street_gaussian/train.py` contains two different diagnostic
paths:

1. `training_report()` writes `metrics/eval_iter_XXXXXX_per_view.csv` and
   appends `metrics/eval_summary.csv`.
2. `_write_periodic_eval()` writes fixed-view visual panels under
   `periodic_eval/iter_XXXXXX/`.

`training_report()` evaluates:

- `test/test_view`: `scene.getTestCameras()`, i.e. the dataset test split.
- `test/train_view`: five sampled training cameras:
  `scene.getTrainCameras()[idx % len(train)]` for `idx in range(5, 30, 5)`.

Therefore `test/train_view` is not the full training split. It is a sampled
training-view diagnostic. `eval_summary.csv` stores mean/median/min/max L1 and
PSNR over those evaluated views for each training checkpoint. The matching
`eval_iter_XXXXXX_per_view.csv` stores per-view diagnostics for that iteration,
including camera id, image name, valid pixels, render/GT statistics,
accumulation/depth statistics, and warnings.

`_write_periodic_eval()` is a visual diagnostic. It now defaults to 15 fixed
views, 5 cameras x 3 frames, and saves comparison panels every 500 iterations.
It is not a full split metric.

## Why current PSNR curves can shake

The existing `paper_results` PSNR/L1 curves are built from
`metrics/eval_summary.csv`. This means they reflect sampled periodic diagnostic
evaluation rather than paper-grade full evaluation. They can shake because:

- `test/train_view` contains only five training views;
- `test/test_view` may be empty or much smaller depending on `data.split_test`;
- per-view difficulty and opacity/densification state can dominate a small
  sample;
- train-time eval is interleaved with checkpoint/save/densification events.

These curves are useful for training dynamics and failure diagnosis, but they
must not be presented as the main quantitative paper result.

## Original Street Gaussians reference

The Street Gaussians paper evaluates rendering quality with PSNR, SSIM, and
LPIPS, and reports a moving-object-region PSNR variant in the Waymo table.
For Waymo, the paper states that every fourth image is selected as test and
the remaining images are used for training. The released `metrics.py` computes
PSNR, SSIM, and LPIPS from rendered image folders for train/test splits.

Local code references:

- `third_party/street_gaussian/metrics.py`: full-image PSNR/SSIM/LPIPS over
  rendered train/test folders.
- `third_party/street_gaussian/render.py`: renders `scene.getTrainCameras()`
  and `scene.getTestCameras()` in evaluate mode.
- `third_party/street_gaussian/lib/datasets/waymo_full_readers.py`: builds
  train/test cameras using `split_train` / `split_test`; object masks are
  stored as `guidance["obj_bound"]`.

## GeoGuardGS formal evaluation plan

Training-time `periodic_eval` and `eval_summary.csv` remain diagnostics.
Main paper results must use final full evaluation:

```bash
python scripts/final_evaluate_experiments.py \
  --configs \
    configs/experiments/a100_baseline_streetgs.yaml \
    configs/experiments/a100_da3_only.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch.yaml \
  --output-root outputs/final_evaluation_full_scene_v2 \
  --loaded-iter 30000
```

Outputs:

```text
outputs/final_evaluation_full_scene_v2/
  <experiment>/
    metrics_full_image.csv
    metrics_object_region.csv
    metrics_background_region.csv
    summary_by_scope.csv
    figures/final_comparison_panels/
  summary_main.csv
  summary_by_scope.csv
```

Required scopes:

- `full_image`: all valid pixels.
- `object_region`: pixels from `obj_bound`; if object branch is disabled or no
  object pixels are present, rows are marked `not_applicable`.
- `background_region`: valid pixels outside object masks.

Paper evidence scripts now prefer `final_full_evaluation_summary.csv` for main
tables and treat `eval_summary.csv` as `sampled_periodic_diagnostic_eval`.
