# Current Formal Experiment Scheme

## Hard rule

All formal street-scene experiments after this revision must keep vehicles:

```yaml
model.nsg.include_obj: true
```

Configs with `include_obj=false` are static-background ablations only and must
not be used as the main autonomous-driving street-scene results.

## Main full-scene groups

| Group | Config | Object branch | Initialization | LiDAR training supervision | Supported conclusion |
| --- | --- | --- | --- | --- | --- |
| A | `configs/experiments/a100_baseline_streetgs.yaml` | on | StreetGS-style COLMAP + LiDAR/object init | yes, `lambda_depth_lidar=0.1` | Original StreetGS baseline reproduction |
| B | `configs/experiments/a100_da3_only.yaml` | on | StreetGS-style COLMAP + LiDAR/object init | no | DA3-only unsupervised structure signal under the original full-scene initialization |
| C | `configs/experiments/a100_da3_periodic_group_softpatch.yaml` | on | StreetGS-style COLMAP + LiDAR/object init | no | DA3 + periodic group softpatch feedback under the original full-scene initialization |

Use A as the baseline. Compare B/C against A to decide whether the method can
replace LiDAR training supervision while preserving vehicles.

## Optional strict no-LiDAR-initialization ablation

These groups are retained only as static-background ablations because the
current no-LiDAR initialization path disables object Gaussians:

| Group | Config | Object branch | Initialization | LiDAR training supervision |
| --- | --- | --- | --- | --- |
| A-static | `configs/experiments/a100_static_bg_baseline_no_lidar_init.yaml` | off | COLMAP only | no |
| B-static | `configs/experiments/a100_static_bg_da3_only_no_lidar_init.yaml` | off | COLMAP only | no |
| C-static | `configs/experiments/a100_static_bg_da3_periodic_group_softpatch_no_lidar_init.yaml` | off | COLMAP only | no |

If these static-background results are strong, they support only the narrower
claim that COLMAP-only background reconstruction can work without LiDAR
initialization. They do not prove full autonomous-driving street-scene
reconstruction because vehicles are disabled.

## Recommended server commands

Run config checks first:

```bash
python scripts/check_closed_loop_config.py --config configs/experiments/a100_baseline_streetgs.yaml
python scripts/check_closed_loop_config.py --config configs/experiments/a100_da3_only.yaml
python scripts/check_closed_loop_config.py --config configs/experiments/a100_da3_periodic_group_softpatch.yaml
```

Then train:

```bash
python scripts/train.py --config configs/experiments/a100_baseline_streetgs.yaml
python scripts/train.py --config configs/experiments/a100_da3_only.yaml
python scripts/train.py --config configs/experiments/a100_da3_periodic_group_softpatch.yaml
```

Build paper evidence after training:

```bash
python scripts/build_paper_evidence_pack.py --output-root outputs/a100_main_experiments --paper-dir outputs/paper_evidence
python scripts/build_paper_result_visuals.py --paper-dir outputs/paper_evidence --out-dir outputs/paper_results
```

## Paper framing decision

If B/C approach A while using no LiDAR training supervision, the clean claim is:

```text
The method removes LiDAR training supervision while retaining the original
StreetGS full-scene initialization protocol.
```

Do not claim strict no-LiDAR initialization for B/C, because vehicles currently
depend on the StreetGS-style LiDAR/object initialization path.
