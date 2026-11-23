# Structured Re-Decoding v1：VL0 负结果与停止裁决

日期：2026-11-24  
范围：冻结 `control_matched_seed0_step9600`，bpRNA VL0 196 条；单 seed model-dev，不是外部敏感性、严格 OOD 或 SOTA。

## 裁决

`STOP_SRD_NOT_BETTER_THAN_CURRENT_MEA_OR_CONSTRAINT_FAILURE`。nested SRD stacking 未超过当前 MEA，按冻结合同立即停止；不运行多模型候选重组，不申请外部评估，不搜索 stacking bonus、temperature、gamma 或阈值。

## 完整 VL0 结果

| Decoder | micro pair-F1 | macro pair-F1 | micro P | macro P | micro stem-F1 | macro stem-F1 | 总解码秒数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nussinov MAP | 0.800210 | 0.804848 | 0.778187 | 0.805172 | 0.536771 | 0.629216 | 4.666 |
| 当前 MEA | **0.825713** | **0.805772** | **0.853478** | **0.827610** | **0.568500** | **0.634191** | 19.870 |
| nested canonical matching | 0.800210 | 0.804848 | 0.778187 | 0.805172 | 0.536771 | 0.629216 | 4.648 |
| nested SRD stacking | 0.800210 | 0.804848 | 0.778187 | 0.805172 | 0.536998 | 0.629216 | 3.989 |
| best-of-N | 0.800210 | 0.804848 | 0.778187 | 0.805172 | 0.536771 | 0.629216 | 0.034 |

SRD−MEA：micro pair-F1 **−0.025502**、macro pair-F1 **−0.000924**、micro precision **−0.075291**、macro precision **−0.022437**、micro stem-F1 **−0.031502**。失败门为 micro/macro pair-F1 严格提高、micro/macro precision 保持、micro stem-F1 保持。

所有臂 nested/canonical 满足率 100%，失败序列与 fallback 均为 0。SRD 平均每条 2 个候选结构、25.18 个候选 pairs、6.94 个候选 helices；最大求解时间 0.592 秒，进程峰值 RSS 最大 610,784 KiB。195/196 条 SRD 与 MAP 结构完全相同；best-of-N 196/196 与 MAP 相同，未产生独立收益。

相同冻结 scores 的辅助 raw 指标不因 decoder 改变：micro/macro AUPRC=0.288178/0.470915，micro/macro r@P0.8=0.035513/0.279273。它们不能抵消结构终点失败。

## 等价性与独立复核

- 合成短序列上，同候选集、同 SRD 目标，精确 helix-prefix DP 与穷举 nested support 求解器目标一致；同候选同目标下 QUBO 没有新增科学价值。
- MAP 与 nested canonical matching 使用相同分数、目标和约束，结果一致；SRD 的收益只能来自 stacking 或候选变化，而非求解器包装。
- 独立 verifier 从 980 条逐序列预测和冻结 truth 重算 pair/stem，所有预测均 canonical/nested；pair/stem 指标与官方汇总最大差值为 **0.0**。

## Artifact 与验证

- Artifact：`/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/structured_redecoding_v1/seed0_control_matched`
- 冻结合同 SHA256：`6054fb56f04bae9a3223a8dd0ba4503eb565cc528b6a10eece699b33c32b172d`
- `srd_summary.json` SHA256：`d0147bcd0d875a2938e8f187e326eec1eee9bfd230dc0ad296adc0e267b5e5bb`
- `independent_verification.json` SHA256：`10551f3d2b943cd26c228ed83c30d089cd971ffe0f34d092b5e4c3547616fd7f`
- 测试：`pytest -q tests/test_run_structured_redecoding_vl0.py tests/test_structured_redecoding.py tests/test_pair_decode_common.py`，11 passed。
- 多模型重组：`NOT_RUN_SINGLE_SOURCE_STOP`。外部评估、严格 OOD、SOTA 继续关闭。
