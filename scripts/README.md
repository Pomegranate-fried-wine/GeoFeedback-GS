# GeoGuardGS scripts

Official migration-safe entrypoints:

- `train.py`: wrapper around `third_party/street_gaussian/train.py`.
- `launch_a100_experiments.py`: launch multiple configs on selected GPUs.
- `check_closed_loop_config.py`: validate config safety gates.
- `validate_no_lidar_leakage.py`: assert DA3-unsupervised branch does not use LiDAR training supervision.
- `validate_repair_safety.py`: assert real prune/shrink/split are disabled.
- `install_server_extensions.sh`: rebuild server-side CUDA/C++ extensions.
- `check_imports.py`: verify key imports after installation.
- `verify_migration_package.py`: verify GitHub package completeness and absence of large artifacts.
- `collect_experiment_outputs.py`: collect compact output manifests.
- `build_paper_evidence_pack.py`: collect metrics, manifests, safety audits, repair summaries, and figure assets into `outputs/paper_evidence/` for paper writing.
- `build_paper_result_visuals.py`: render paper-facing tables, plots, LaTeX table drafts, and selected figures from `outputs/paper_evidence/`.
- `evaluate.py` / `evaluate_geometry_metrics.py`: evaluation launchers.
- `render_periodic_panels.py`: periodic panel helper.
- `test_depth_visualization.py`: smoke-test robust depth visualization on empty/invalid depth maps.

Historical local research scripts are kept in `scripts/research_archive/` and
may contain local defaults such as A5000 or p15 paths. They are not official
A100 entrypoints.
