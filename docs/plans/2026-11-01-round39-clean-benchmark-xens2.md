# Round 39 Clean Benchmark and xens2 Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** 用逐模型去泄露的 25 系统、8 split、三次评测建立真实强对手板，并在不混用协议的前提下攻关 xens2 的 8/8、端到端速度和校准/置信度目标。

**Architecture:** 先冻结模型身份、训练语料和 clean 子集，再复用统一评分器完成 600 单元结果。xens2 优先做多模型融合/批化的无损工程，骨干权重不同不得伪称共享编码；只有 VL0 证明需要时才进入三种子训练或蒸馏为单骨干、单决策头学生；最终图表只读取验收后的原始 artifact。

**Tech Stack:** Python 3.10、PyTorch 2.5.1+cu121、MMseqs2、NumPy、官方基线 CLI/API、python-pptx、LibreOffice、Git/GitHub。

---

### Batch 1: 模型资产、泄露矩阵与全量复测

**Files:**
- Modify: `交接文档/{spec.md,tasks.md,checklist.md}`、`交接文档/spec/{spec.md,tasks.md,checklist.md}`、`交接文档/spec_sync/{spec.md,tasks.md,checklist.md}`
- Create: `/home/rna_jev/tools/round39_model_registry.json`
- Create: `/home/rna_jev/tools/run_round39_clean_benchmark.py`
- Artifacts: `/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark/`

**Risk:** High — 每个模型的 clean 集合和 checkpoint 不同，错绑语料会直接改变 SOTA 结论。

**Implementation:** 注册 25 系统，恢复 BPfold/UFold/E2Efold 与新增四基线的官方资产。逐模型计算精确、80% identity 和可用家族重叠，保存 clean 清单；固定权重模型独立评测 seed=0/1/2，统一 strict micro-F1 与 canonical-nested GT，输出 600 单元或明确 BLOCKED_PROVENANCE。

**Minimum verification:** 每 split 的 `raw_n=removed+clean_n`；每 cell 三次预测/计数/均值和哈希可独立重算；EternaFold bpRNA-new 亲自三次复测；任何未知语料不得标 clean。

**Independent review:** Yes — 主 benchmark 的公平性门。

### Batch 2: xens2 速度、精度与校准候选

**Files:**
- Modify: `/home/rna_jev/tools/profile_speed_round36.py`
- Modify: `/home/rna_jev/tools/speed_end_to_end_4090.py`
- Create: `/home/rna_jev/tests/test_round39_xens2_inference.py`
- Artifacts: `/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_xens2/`

**Risk:** High — 多模型融合或蒸馏可能改变预测、校准和模型身份。

**Implementation:** 骨干权重不同，不伪称无损共享；先做多模型融合/批化，如不能达到目标，再从 VL0 锁定集成蒸馏为单骨干单头学生、模型或数据候选。所有新训练三种子到 validation plateau 并独立 GPU QA。校准只由 VL0 拟合，比较 ECE/Brier/NLL、Spearman 和 risk–coverage。

**Minimum verification:** CUDA 强制；全 TS0 三次；同配置逐位一致，跨配置配对一致≥99% 且 |ΔF1|≤1e-3；8 split 只能由最终三次均值判胜；速度终测独占 4090。

**Independent review:** Yes — 检查是否用弱参照、测试集调参或缓存口径制造目标。

### Batch 3: PPT、论文叙事与源数据

**Files:**
- Create: `ppt/patch_round39.py`
- Modify: `ppt/RNA-JEV_论文逻辑重构_20261031_导师讨论_round38.pptx`
- Create: `交接文档/paper/round39_claim_evidence_audit.md`
- Modify: `交接文档/source_data/{README.md,SOURCE_MANIFEST.json}`

**Risk:** Medium — 长标题易溢出，新旧数字和待测状态容易混写。

**Implementation:** 主图 1 逐字使用指定 5/8、4/8 与三个未领先 split，只联动必要页面。每条论文叙事按现象、对照、限制检查；新图沿用统一配色、字体、分辨率和源 JSON，不改用户其他调整。

**Minimum verification:** 23 页文本/表格/图片逐项对应；微软雅黑、1.5 行距、无重叠/越界；ZIP 与 LibreOffice 全页渲染；PPT 嵌入图与源 PNG 哈希一致。

**Independent review:** Yes — 最终导师汇报稿逐页复核。

### Batch 4: 台账、归档与 GitHub 交付

**Files:**
- Modify: `交接文档/records/DECISION_TRAINING_LOG.md`
- Modify: `交接文档/archive_20261031/ARCHIVE_INDEX.{md,json}`
- Modify: `/home/rna_jev/records/DECISION_TRAINING_LOG.md`

**Risk:** Low — 主要风险是误归档活跃源数据或误提交大型未跟踪资产。

**Implementation:** 逐 run 保存配置、seed、CUDA、数据/权重哈希、best epoch、停止原因和科学结论。PPT 修复后再归档废弃交接件与旧源数据，保留所有活跃原始数据；只提交确认的代码和小型结论文件。

**Minimum verification:** 归档前后计数与哈希一致，manifest 无失效活跃引用；相关测试通过；`git status` 逐项审查后 commit+push，不提交秘密、权重或大型 artifact。

**Independent review:** No — 最终三通道验收覆盖。
