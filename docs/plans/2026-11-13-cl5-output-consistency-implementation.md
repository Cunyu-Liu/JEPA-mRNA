# CL5 输出一致性项 Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** 在统一 B0 初始化与训练协议下，只给 CL4 增加一个可微的 64 维合法 pair 输出一致性项，并与 B0 比较。

**Architecture:** 复用 CL4 的双视图、困难负样本与加权 InfoNCE。CL5 仅新增两个视图在同一合法 pair 位置的输出表示一致性损失，训练期计算，部署 scorer 与推理路径不变。

**Tech Stack:** Python、PyTorch、现有 `rnajepa.contrast` 和 `train_plan_a.py`、真实 CUDA。

---

### Batch 1: 可微输出项与最小数值验证

**Files:**
- Modify: `src/rnajepa/contrast.py`
- Test: `tests/test_cl5_output_round55.py`

**Risk:** Medium — 必须避免非法 pair、零范数和错误归一化影响科学结果。

**Implementation:** 新增 `output_consistency(z1, z2, valid_mask)`，对合法 pair 的 64 维输出计算余弦一致性 `1-cos`，返回 loss、平均相似度、表示范数和通道方差。使用现有合法 mask，不引入新负样本、额外模型或部署参数。

**Minimum verification:** 测试相同输入零损失、独立输入非零损失、非法 pair 无贡献、空 mask 拒绝、有限梯度和常量表示诊断；真实 CUDA 小张量复核。

**Independent review:** No — 数值门与后续训练 QA 已覆盖核心风险。

### Batch 2: 单变量训练接入与启动器

**Files:**
- Modify: `tools/train_plan_a.py`
- Create: `tools/run_rinalmo_cl5_round54.sh`
- Create: `tools/run_rinalmo_cl5_smoke_round54.sh`

**Risk:** High — 不能把 CL5 变成 CL4 以外的额外增强或部署改动。

**Implementation:** 新增 `--cl5-output-weight`（CL5 固定 0.1，其他臂默认 0），非零时要求 CL2/CL3/CL4 配置完整。把 `output_consistency` 加入总损失并记录 `output_consistency`、范数、方差和相似度；run_meta 记录配置。启动器保持 B0 的数据、seed、batch、四目标、优化器、解冻与 best VL0 协议。

**Minimum verification:** 静态参数检查、两步真实 CUDA smoke，确认输出项梯度非零、直接任务正常、run_meta 仅新增目标项、部署不含额外参数。

**Independent review:** Yes — 真实 smoke artifact 检查后才允许三种子。

### Batch 3: B0 对照 QA 与队列恢复

**Files:**
- Modify: `tools/qa_rinalmo_cl2_training_round54.py`
- Modify: `tools/run_prism_cl2_cl5_b0_round55.sh`
- Modify: `交接文档/records/DECISION_TRAINING_LOG.md`

**Risk:** Medium — 比较基线必须是 B0，不能再回到 CL1 或 CL2 串联裁决。

**Implementation:** QA 接受 CL5 的负样本模式和输出诊断，三种子汇总只计算 `paired_delta_b0`。恢复原串行队列完成 CL5 三种子，保留负结果和 PARTIAL。

**Minimum verification:** 单 seed QA 双跑逐字节一致，三种子汇总双跑一致；检查测试集未用于选臂、推理无新增分支。

**Independent review:** Yes — 最终结论必须同时满足原始 artifact、独立复算和 B0 对照。
