# Stacking Closure Form v1：seed0 负结果与停止裁决

日期：2026-11-24。范围：seed0、bpRNA VL0 196 条、9,600 steps、四臂同参数新形式探针；不是三种子确认、外部评估、严格 OOD 或 SOTA。

## 裁决

`STOP_STACKING_CLOSURE_FORM_SEED0`。候选的 micro pair-F1 和固定目标边界 recall 有局部改善，但 macro pair/stem 未保持，且未优于匹配 destroy 对照。不得启动 seeds 1/2，不得继续堆模块，不开放外部评估。

## 四臂结构终点

| Arm | micro pair-F1 | macro pair-F1 | micro P | macro P | micro stem-F1 | macro stem-F1 | 目标边界 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| local_control | 0.842909 | **0.830911** | 0.837682 | **0.832462** | 0.617163 | **0.690001** | 0.948235 |
| stacking_closure | **0.846899** | 0.825271 | **0.842184** | 0.825867 | 0.616211 | 0.670570 | **0.954706** |
| destroy_offset | 0.847915 | 0.829225 | 0.845905 | 0.832417 | 0.621029 | 0.677507 | 0.949412 |
| destroy_channel_shuffle | 0.845774 | 0.823352 | 0.837867 | 0.817819 | 0.624158 | 0.681873 | 0.959412 |

目标集合为 local_control 固定的 1,700 个 helix-boundary truth pairs，所有臂分母相同。

## 候选减 local_control 与失败门

- micro pair-F1 **+0.0039898**，目标边界 recall **+0.0064706**，micro AUPRC **+0.0271899**：局部假设信号存在。
- macro pair-F1 **−0.0056406**，macro stem-F1 **−0.0194306**，macro precision **−0.0065948**：结构终点失败。
- micro stem-F1 −0.0009518，处于保持门内；micro precision +0.0045024。
- micro r@P0.8 严格正门失败，macro pair-F1 非劣/严格正门、macro precision/stem 保持门均失败。
- 对两个 destroy 对照必须在 micro/macro pair-F1 与 micro/macro stem-F1 全部占优；destroy_offset 的 micro pair-F1=0.847915 高于候选，且其余 destroy 门也未全部通过，故四项 destroy 门均失败。

局部边界改善不能抵消 macro 结构失败，也不能证明显式 stacking 是稳健机制。raw-only、micro-only 和 destroy 对照共同否定了本路线继续扩张的合理性。

## 工程与独立复核

- 四臂均完成 9,600 steps、`completed/step_cap`、fresh head、相同 batch/schedule 和 196 条 VL0 导出；终端解冻 20/33 blocks。
- 参数：每臂 head=605,711、closure=98,689；名称、形状与解析 MAC 完全一致。`zchunk=16` 仅用于显存。
- 独立 raw-score recheck：`PASS`，全部 pair/stem/raw/target 指标与官方汇总最大差值 **0.0**，重算裁决一致。
- 相关测试通过；全项目回归为 674 passed、8 failed、1 skipped。8 个失败位于既有 CL5 浮点严格相等、PairJEPA、RIPE 和旧配置测试，与本路线无关，未借本任务修改。

## Artifact 与哈希

- 根：`/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/stacking_closure_form_v1/pilot`
- 冻结合同 SHA256：`d8e77cecc69b7525e343c48ec9f479a0db1c9332e16ed9932d275ecc05c8bf5e`
- `evaluation/pilot_verdict.json`：`1e6bc6218fb12f000137c396fc7705fdcf31e958f3807068f8c246632a42c03c`
- `independent_recheck.json`：`36a77e6ce68184c84bf8083030843734bf1b8fcae771d4d2cdaad74baff1c54e`
- local_control checkpoint：`f2ef76457317c56fd9463739d37be13201ee308de61290397345b053e42a32f0`
- stacking_closure checkpoint：`c52378ea83deecdab0b15667841d8d56f79ac736c9e75cd44531bc33b8915d19`

`seed1_seed2_allowed=false`、`external_evaluation_allowed=false`、`strict_ood_allowed=false`、`sota_claim=false`。本负结果与容量路线、SRD 负结果共同保留为路线边界。
