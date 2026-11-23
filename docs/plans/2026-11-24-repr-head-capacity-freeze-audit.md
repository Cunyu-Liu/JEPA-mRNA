# 三种子表示/结构头容量路线封存审计

日期：2026-11-24  
范围：bpRNA VL0 model-dev；不是外部测试、严格 OOD 或 SOTA 证据。

## 裁决

`STOP_REPRESENTATION_HEAD_CAPACITY_THREE_SEED`。简单容量扩张不是稳健生产路线；raw 排序改善不等于最终结构改善。表示、训练目标与优化的因果仍未分离。不得选择冠军 seed/checkpoint，不增加 `blocks/hidden/d_z`，不开放外部评估或严格 OOD。

## 完整性与独立性

- seed1 三个 run 的代码 HEAD：`3b3d1bb32d2d4c9162162483b8bd48a4ec6ef755`；seed2：`64be8a035f4b6d26143c8d22999fab31d015268c`。
- 六个确认 run 均为 `completed/step_cap`、9,600 steps、385 条训练日志、有限 loss/gnorm、fresh head、每 seed 共享初始化、control 重训、196 条 VL0 raw 导出；checkpoint 实际 SHA256 与导出元数据一致。
- 冻结合同 SHA256：`ce2372f2836b59cd44dc8b029428eafdd72a84ea7aa36acb4b08c59e13f14bf6`。
- 本审计不读取官方比较值来计算结果，而从各 seed 的 raw-score 独立重算指标重新求 delta 与门；官方值仅用于事后交叉核对。seed0/1/2 raw 重算 SHA256 分别为 `5f1da4b2…cb1e07`、`97bd0f0a…c9a797`、`421913a2…9090`。
- seed0 历史记录含 9 个方向门；后续确认合同增加 macro pair-F1 严格正门。审计保留历史记录，并对所有 seed 另行计算现行全部门；未改写冻结科学门。

## 逐 seed delta（候选 − 同 seed control）

| Seed | Arm | micro pair-F1 | macro pair-F1 | micro stem-F1 | macro stem-F1 | micro AUPRC | r@P0.8 | 失败门 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | head_deep | +0.016533 | +0.014087 | +0.027663 | +0.004924 | +0.077490 | +0.043742 | 无（仅单 seed 信号） |
| 0 | repr_head | +0.020258 | +0.007725 | +0.031021 | +0.008609 | +0.109335 | +0.112603 | 无（仅单 seed 信号） |
| 1 | head_deep | −0.005988 | −0.009554 | +0.004356 | −0.008725 | +0.032395 | +0.000650 | pair-F1 至少一项严格为正；micro/macro pair-F1 非劣；macro pair-F1 严格为正；macro stem 保持 |
| 1 | repr_head | −0.057850 | −0.078484 | −0.042310 | −0.059177 | +0.075638 | +0.083586 | 同上，另 micro stem 保持失败 |
| 2 | head_deep | −0.022362 | −0.034249 | −0.027736 | −0.041105 | +0.079271 | −0.012993 | pair-F1 至少一项严格为正；micro/macro pair-F1 非劣；macro pair-F1 严格为正；macro precision 保持；micro/macro stem 保持；r@P0.8 严格为正 |
| 2 | repr_head | −0.065952 | −0.085021 | −0.088749 | −0.118673 | +0.180848 | +0.194023 | pair-F1 至少一项严格为正；micro/macro pair-F1 非劣；macro pair-F1 严格为正；micro/macro stem 保持 |

precision 的完整数值保留在 JSON；除 seed2/head_deep 的 macro precision 门失败外，其余逐 seed precision 门通过。AUPRC/r@P0.8 不得抵消 pair/stem 失败。

## 三种子均值与方向不一致

| Arm | micro pair-F1 | macro pair-F1 | micro stem-F1 | macro stem-F1 | micro AUPRC | r@P0.8 | 均值失败门 |
|---|---:|---:|---:|---:|---:|---:|---|
| head_deep | −0.0039389391 | −0.0099052807 | +0.0014277310 | −0.0149689482 | +0.0630519273 | +0.0104662913 | pair-F1 至少一项严格为正；micro/macro pair-F1 非劣；macro pair-F1 严格为正；macro stem 保持 |
| repr_head | −0.0345147191 | −0.0519265689 | −0.0333460085 | −0.0564139810 | +0.1219403702 | +0.1300707377 | 同上，另 micro stem 保持失败 |

两臂均同时存在“均值为负”和“逐 seed 方向不一致”：seed0 为正，seeds 1/2 的多项结构终点为负。`repr_head − head_deep` 只有 AUPRC、r@P0.8 和 precision 跨 seed 为正，不能据此选择更复杂模型。

## 验证与 artifact

- 测试：`/root/miniconda3/envs/lucaone_cu128/bin/python -m pytest -q tests/test_audit_repr_head_capacity_freeze.py tests/test_repr_head_three_seed.py` → `9 passed`。
- 独立封存：`tools/audit_repr_head_capacity_freeze.py`。
- 输出：`/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/representation_head_mechanism_4arm_v1/capacity_freeze_audit.json`。
- 输出 SHA256：`6f435ceaca547e814a7ecbc689512a85ec56e97cb21aba249795da40a202071d`。
- 旧聚合文件中的“capacity is a robust contributing factor”不受三种子证据支持；生成脚本已更正。历史 artifact 未覆盖，本文件与 `capacity_freeze_audit.json` 明确取代该叙事，不改变数值或裁决。
