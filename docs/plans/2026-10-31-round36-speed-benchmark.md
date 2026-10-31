# Round 36 Speed and Benchmark Integrity Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** 在不重训、不混用计时口径、不隐藏数据泄露的前提下，定位并优化 4090 原始序列推理吞吐，补齐公平的基线证据和汇报 PPT。

**Architecture:** 以现有冻结 checkpoint、TS0 和 strict micro-F1 为唯一主线，先建立 raw-sequence / forward-only / head-only 三种明确协议，再对原始序列路径做分段 profile。速度优化只改推理工程，所有候选必须在同一 4090 上通过全量预测与 F1 等价门；主 benchmark 与泄露审计沿用原始结果，不从速度图反推精度。

**Tech Stack:** Python 3.10、PyTorch 2.5.1+cu121、C/OpenMP、NumPy、python-pptx、LibreOffice、Git/GitHub。

---

### Batch 1: 合同、泄露感知表与 RNAformer 公平性

**Files:**
- Create: `交接文档/docs/plans/2026-10-31-round36-speed-benchmark.md`
- Modify: `交接文档/{spec.md,tasks.md,checklist.md}`、`交接文档/spec/{spec.md,tasks.md,checklist.md}`、`交接文档/spec_sync/{spec.md,tasks.md,checklist.md}`
- Create: `交接文档/source_data/appendix_leakage/leakage_aware_baseline_source.json`
- Create: `交接文档/source_data/appendix_leakage/rnaformer_protocol_audit.json`

**Risk:** High — 表格直接决定 6/8 SOTA 与 TS2 结论是否可信。

**Implementation:** 逐格记录模型、训练语料、checkpoint、测试集、重叠数、GT 和指标。原始全板继续保留；新表只用零重叠参照或 clean-181，不把未审计单元格写成干净。RNAformer 同时给出固定 bprna checkpoint、inter-family 特殊微调 checkpoint、release-full GT 与 canonical-nested GT 的边界；原文 quoted 值与项目复测值分列，不用不同口径替换胜场。

**Minimum verification:** 从原始 result/audit JSON 独立重算全部表格单元格；三遍核对行数、TSB clean-181、TS2/TS3 参照和胜场。缺证据标 `NA/not audited`，不补零、不猜值。

**Independent review:** Yes — 这是论文主结论的口径门。

### Batch 2: 速度复现、分段 profile 与无损优化

**Files:**
- Modify: `/home/rna_jev/tools/speed_end_to_end_4090.py`
- Create: `/home/rna_jev/tools/profile_speed_round36.py`
- Create: `/home/rna_jev/tests/test_speed_round36.py`
- Artifacts: `/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/speed_round36/`

**Risk:** High — 任何 padding、精度或 batch 改动都可能改变预测，且并行 GPU 作业会污染计时。

**Implementation:** 先复现历史 head-only B=256 工程点并确认其不包含编码器，再测 raw-sequence 的 encoder/head/DP/mask/transfer 分段耗时和有效 batch。按 profile 选择优化：精确长度批化的上限、长度分桶 padding、CUDA/算子与精度路径、C/OpenMP 解码。每次只改一个变量；优化不是重训，不涉及收敛提前停止。正式测速必须独占 4090，NB 队列仅在 run 边界让路，结束后立即恢复。

**Minimum verification:** CUDA 强制可用；全 TS0 三次重复；同 batch 预测逐位一致；跨配置逐序配对一致率≥99% 且 |ΔF1|≤1e-3；记录峰值显存、有效 batch 和各阶段耗时。未达到等价门的配置只作工程负结果，不入图。

**Independent review:** Yes — 需要独立复算计时与预测等价性，防止用不完整计时制造速度提升。

### Batch 3: 同机基线、4090 主 benchmark 复核与 PPT

**Files:**
- Modify: `ppt/RNA-JEV_论文逻辑重构_20261031_导师讨论_round35.pptx`
- Create: `ppt/patch_round36.py`
- Create: `交接文档/source_data/fig05_speed/speed_round36_summary.json`

**Risk:** Medium — 新页会影响后续页码和附录计数，速度文案容易越过现有证据。

**Implementation:** 优先复用已有 4090 八 split 复现结果，只有缺项才重跑。补 RNAformer/UFold 的同机 raw-sequence 完整点；资产确实不可得才保留 blocked。第 5 页解释历史峰值与新值差异，第 18 页用流程边界解释 head-only、forward-only、raw-sequence；第 17 页后新增泄露感知完整基线表。其他用户调整不改，全部文本微软雅黑、1.5 倍行距。

**Minimum verification:** 表格与源 JSON 一致；ZIP 完整性、字体、行距、越界、文本碰撞、逐页 LibreOffice 渲染和图文匹配全部通过；新页后目录/页码/附录数量同步。速度 SOTA 只有同机关键基线闭环且达到第一时才写，否则明确写“当前未证明”。

**Independent review:** Yes — 最终交付前逐页复核。

### Batch 4: 台账、源数据、归档与 GitHub

**Files:**
- Modify: `交接文档/records/DECISION_TRAINING_LOG.md`、`交接文档/source_data/SOURCE_MANIFEST.json`
- Modify: `交接文档/archive_20261031/ARCHIVE_INDEX.{md,json}`

**Risk:** Low — 主要是可追溯性，误删活跃源数据是首要风险。

**Implementation:** 每次 GPU 运行记录 commit、配置、CUDA、数据/权重哈希、原始结果、计时和结论。只移动确认废弃的文档和旧图源，活跃原始数据保留并写 SHA256；不删除用户文件。代码留在 `/home/rna_jev`，大文件留 `/teams`，本地只同步核心结论和源清单。

**Minimum verification:** 归档前后文件计数与哈希一致，manifest 无失效活跃引用；代码静态检查和相关测试通过后提交并 push，不提交秘密或大型 artifact。

**Independent review:** No — 最终交付复核覆盖即可。
