# 当前证据与 baseline 边界（训练期间只读盘点）

日期：2026-11-24。本文只汇总已有注册，不运行外部数据、不改变任何模型或 SOTA 裁决。

## 注册覆盖

- `baseline_registry.json`：63 行；历史 25、主榜 13、CHANRG 29 的 core union=54，补充 9 行。类别为 structured decoder 14、direct neural 16、foundation model 28、project 5。
- `data_registry.json`：31 行，27 `FROZEN`、4 `BLOCKED`。
- `prediction_registry.json`：276 行，162 `FROZEN`、18 `BLOCKED`、96 `SUMMARY_ONLY`。
- baseline 状态包括 27 `LEGACY_TABLE_ONLY`、7 `FAMILY_PROVENANCE_ONLY`、21 个具有 frozen raw prediction scope 的条目，以及其他部分覆盖/阻塞条目；36 个 baseline 当前无预测，2 个预测 scope 为 `BLOCKED`。
- 62/63 baseline 的 `provenance_status=BLOCKED_PROVENANCE`，只有 1 个为 `PARTIAL_OR_VERIFIED`。缺权重、版本、训练集、许可或逐序列预测者不得填 0，也不得当作统一可比排名。

## 科学边界

历史表格中的 6/8 是特定预注册去污染参照口径，且旧主表明确含身份混用纠错；它不是当前候选的统一完整 SOTA。当前容量路线失败，SRD 未超过 MEA，新 stacking 形式仅处于 seed0 pilot；这些路线均未获外部评估授权，统一完整 SOTA 为 **0/8**。raw、own-clean、matched-clean 与 CHANRG 协议必须分栏，`LEGACY_TABLE_ONLY` 不能充当 frozen prediction。

## 注册哈希

- baseline：`51878f149f7521ab714b8ed7e1db9cee8dab11f32639aa38a0f6a0fb65711c8a`
- data：`4fe3f5376498b64ffcce758300067f3bcf1489868dbe3156d84b886f0c16fd69`
- prediction：`da2b6fadaf994496dc004375be96cda9fdaa024dba4ce0e465738f048010d1c9`

完整 baseline/benchmark 放行仍需新形式三种子全部门通过，或另行冻结的合格分支申请；当前没有这样的分支。
