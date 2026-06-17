# 实验计划

## 主实验

1. Baseline StreetGS。
2. No-LiDAR-supervision control（历史上曾称 DA3-only baseline）。
3. DA3 periodic group softpatch。
4. DA3 periodic group softpatch + opacity regularization。
5. DA3 periodic group softpatch + conservative opacity decay。
6. LiDAR-supervised reference。

## 观察重点

- DA3 主线是否在不使用 LiDAR 训练监督时改善 boundary / edge / thin-structure 区域。
- live CUDA contribution 是否稳定产生 group responsibility。
- opacity regularization 是否稳定且不破坏 RGB。
- conservative opacity decay 是否带来局部几何改善。
- LiDAR reference 只作为上界，不作为主方法。

## 每 500 轮输出

每 500 轮保存可视化和指标，便于判断闭环训练是否稳定。
# Historical experiment-plan draft

This file is retained as a historical planning note. The canonical current GeoFeedback-GS four-group scheme is in `README.md` and `docs/project_overview.md`.

