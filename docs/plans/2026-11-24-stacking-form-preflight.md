# Stacking Closure Form v1：工程预检记录

日期：2026-11-24。范围：seed0 最小形式探针的准入检查；无外部评测、严格 OOD 或 SOTA。

## 合同与公平计算

- 独立合同：`configs/stacking_closure_form_v1.json`；旧容量合同未修改。
- 四臂：`local_control`、`stacking_closure`、`destroy_offset`、`destroy_channel_shuffle`。
- 每臂 head 参数 605,711，closure 参数 98,689；参数名称和形状完全一致。
- closure 解析 MAC（仅 input/update/head linear，B=4、L=500）均为 98,432,000,000；不包含 encoder、pair representation、GELU 和逐元素乘积，未冒称完整 FLOPs。
- 训练数据、共享初始化、decoder 均使用服务器实测 SHA256；identity preflight=`PASS`，SHA256 `29a1c8b26a25c0811372fb51237e6bb20dce60274107abd91cc3f87f23411c5d`。
- 架构与回归测试 27 passed；checkpoint scorer 加载与合同测试 10 passed；目标子群定义测试 2 passed。

## 显存与工程偏差

- 在 B=4、L=500、20/33 backbone blocks 解冻、完整 forward+backward 下，四臂 allocated 均约 10,553 MiB。
- `zchunk=64/32` 的最坏 reserved 分别为 12,494/12,114 MiB，未达到 GPU 实际 32,110.94 MiB 中保留 20,000 MiB 空闲的门；历史报告均保留。
- `zchunk=16` 最坏 reserved=11,914 MiB，满足门限；该设置只改变梯度重算分段，不改变数学、参数或科学变量。v4 报告 SHA256 `c2e44310cd4a1ba03bcb5d1774b872e7dd1d736465e9433ceed0bfbbc942d01e`。
- 首次 smoke 的训练完成，但 `xens2_eval.load_plana` 强制使用 `resnet2d`，导出失败。已修复为读取 `meta.scorer`，并用原 checkpoint 成功导出 2 条；失败目录保留，修复后在独立 `smoke_v2` 重跑。

## 目标子群与停止门

评估使用由 `local_control` 预测固定的 helix-boundary truth pairs：每个目标 pair 至少一个真实 stacking 邻居被 control 正确预测。所有臂共享同一目标集合和分母，避免通过改变邻居预测来缩小评估集合。

只有 `stacking_closure` 同时通过全部 pair/stem 方向门、目标边界 recall 严格提高，并在 micro/macro pair-F1 和 micro/macro stem-F1 上严格优于两个 destroy 对照，才允许预注册三种子确认。任何失败均停止；smoke 不产生科学裁决，外部评估始终关闭。

## 四臂 smoke 完成状态

`smoke_v2` 四臂均完成 2-step 训练与 2 条 VL0 导出，run meta 为 `completed/step_cap`，fresh head、有限 loss/gnorm、checkpoint step=2，导出 checkpoint/合同身份一致。评估 artifact 为 `smoke_v2/evaluation_v3/pilot_verdict.json`，裁决固定为 `SMOKE_ONLY_NO_SCIENTIFIC_VERDICT`。两条 smoke 没有目标边界集合，不作为性能证据；早期空目录 `evaluation/` 和因空目标集合中止的 `evaluation_v2/` 均保留。

评估器初始化及空目标处理的两处工程错误已修复，未重跑训练。相关代码提交：`c4221e5`（形式矩阵）、`7f43cb0`（预检/导出/评估）、`1ba5940`（部分恢复与独立 raw recheck）。
