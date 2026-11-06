# Round39 ViennaRNA Native Probability Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** 为 ViennaRNA-MFE、ViennaRNA-centroid、ViennaRNA-MEA 补齐八 split 原生 BPP、三次确定性复算和与 xens2 的附加敏感性比较，不改变 own-clean 主榜。

**Architecture:** 每条序列每轮建立一个 `fold_compound`，一次 `pf()` 同时服务三种原生解码和 float64 `bpp()`。三种结果共享同一 NPZ 概率数组，但分别保存预测和结构置信度；独立验证后才进入 matched-clean 比较。

**Tech Stack:** Python 3.10、ViennaRNA 2.7.2、NumPy、项目 `valid_pair_mask` 与校准指标、Bash 持久队列。

---

### Batch 1: 原生概率计算与协议测试

**Files:**
- Create: `/home/rna_jev/tools/calibrate_vienna_round39.py`
- Create: `/home/rna_jev/tools/test_calibrate_vienna_round39.py`
- Modify: `/home/rna_jev/tools/compare_bpfold_xens2_calibration_round39.py:18-47`

**Risk:** High — BPP 的一基索引、配对掩码和浮点求和顺序会直接影响校准结论。

**Implementation:** 每 split 完整执行三次；逐条预测与 `vienna/raw_strict/<split>.json` 一致，保存共享 `names/offsets/probabilities/labels` NPZ、各解码的 TP/FP/FN、三次 F1、置信度、代码/数据/软件哈希。ViennaRNA 不拟合训练序列，因此保留完整测试集，不套用其他模型的 clean n。原生 CPU 不冒充 GPU，最终结构 one-hot 不作为概率。

**Minimum verification:** 小序列 roundtrip 测试；既有 Kirigami/BPfold/xens2 回归；真实 TS2 三次全量逐条预测、F1、BPP 和独立指标一致。已完成为 10 项测试通过及 TS2 三格验收。

**Independent review:** Yes — 独立验证器重新读取 NPZ 并复算全部数值，不能只检查输出 JSON 是否存在。

### Batch 2: 八 split 持久执行与独立复算

**Files:**
- Create: `/home/rna_jev/tools/run_vienna_calibration_round39.sh`
- Create: `/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark/vienna/calibration/independent_verification.json`

**Risk:** Medium — 长任务、环境路径和断点跳过可能造成缺格或重复推理。

**Implementation:** 固定 ViennaRNA 2.7.2 的原生模块路径，lucaone 环境运行统一指标；按小 split、TS0、ArchiveII、bpRNA-new 顺序执行，flock 防重复。独立复算每格的标签、预测、F1、概率哈希、ECE/Brier/NLL、Spearman/AURC 与 risk–coverage，24 格齐全才形成总账。

**Minimum verification:** 已完成 TS2 三格和 TS0 三次原生计算；其余队列完成后运行 `run_vienna_calibration_round39.sh` 内置独立验证，核对 24 格及哈希。不能把当前进度写成全板结果。

**Independent review:** No — 常规队列检查；最终科学结论在 Batch 3 复核。

### Batch 3: 附加敏感性、证据归档与交付

**Files:**
- Create: `/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark/calibration_comparisons/vienna_summary.json`
- Modify: `/home/rna_jev/records/DECISION_TRAINING_LOG.md`
- Modify: `/home/rna_jev/source_data/SOURCE_MANIFEST.json`
- Modify: `/home/rna_jev/spec/tasks.md`

**Risk:** High — 交集与原生主榜混淆、把单格较低 ECE 外推为全板最佳，都会造成错误科学叙事。

**Implementation:** 按名称、序列和实际评分 GT 对齐，输出 24 格 matched-clean 敏感性；空交集不写 0 分。后续源码核验已确认 EternaFold/CONTRAfold 支持 `--bpseq` 与 `--posteriors 0`，另按原生 posterior MEA 合同补测，不再标为概率不可用。大数组留 `/teams`，核心 JSON 和代码提交 GitHub，最终仅定点更新必要图件。

**Minimum verification:** 24 个独立验收单元、24 个比较单元及清单哈希全部通过；源 JSON→台账→图表一致。多 split 已存在低于 xens2 的 ECE，不宣称全板最低 ECE 或最佳置信度。并发运行时间不作为速度证据。

**Independent review:** Yes — 最终里程碑需独立数值和叙事复核；实施保持当前单一执行上下文，不额外创建代理或重复工作区。

**执行分组：** Batch 1 已完成；Batch 2 与已完成的 TS2/TS0 比较并行推进；Batch 3 只在 24 格齐套后汇总。原始数组不删除，历史失败记录保留。
