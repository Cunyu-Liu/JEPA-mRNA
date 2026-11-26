# BPfold bpRNA-new 标签协议审计

## 范围与结论

本审计基于 FoldNAR `c183df8457b2e3d3695b6e9eb78bb6b56bff5d91`，只读复算冻结的外部 Test 产物。未运行训练、推理、校准或模型选择，未启动 GPU 任务，未修改历史结果。官方 BPfold v0.2 release 预测、`BPfold_data` 参考标签、项目标签和项目 replay 预测始终作为四种不同身份处理。

核心结论不是“同一标签有 5,282 条解析差异”，而是：**项目标签由 BPfold 官方 release 的预测 BPSEQ 经项目投影规则生成，而官方 0.647 口径使用另一份 `BPfold_data/test_data/bpRNAnew` 参考标签。** 在 5,388 条同名同序列记录中，只有 106 条项目标签集合等于官方参考标签，5,282 条不同。release API 明确把 `BPfold_test_results.tar.gz` 描述为预测结构，把 `BPfold_data.tar.gz` 描述为数据集；项目构建脚本与 manifest 进一步证实了输入路径和过滤过程。

## 逐记录分类

完整 5,401 行见 `source_data/bpfold_label_protocol_audit_20261127/per_record.jsonl.gz`。每行保留序列关系、配对数与哈希、互斥分类、min-loop/crossing 删除配对明细、canonical/min-loop/crossing 标志及参考/项目配对差集计数。

| 互斥分类 | 记录数 | 可检验含义 |
|---|---:|---|
| `LABEL_SETS_EQUAL` | 106 | 项目标签集合等于官方参考标签 |
| `DIFFERENT_REFERENCE_VS_UNFILTERED_RELEASE` | 2,724 | 项目标签等于未过滤 release，但与参考标签不同 |
| `DIFFERENT_MIN_LOOP_OR_CROSSING_PROJECTION_ONLY` | 1 | 参考标签等于 release，项目差异仅来自项目投影 |
| `DIFFERENT_REFERENCE_AND_PROJECT_PROJECTION` | 2,557 | 项目经过投影，且参考标签也不同于 release |
| `OFFICIAL_ONLY_PROJECT_REJECTED_BAD_ALPHABET` | 13 | 项目因非 `ACGU` 字符硬拒绝，未进入 5,388 条 |

因此，5,282 条不同记录中不能把全部差异归因于 min-loop 或 crossing。2,724 条甚至没有项目投影造成的配对变化；其标签差异来自参考标签与 release 预测本身不同。2,557 条同时具有“参考来源不同”和“项目投影”两层差异，分类不表示二者各自贡献可以线性相加。

## 5282 差异与 13 条排除的具体来源

### 序列与索引

- 官方 release 与官方参考的 5,401 条名称一一对应，序列全部一致。
- 项目 5,388 条全部是官方记录的子集；同名序列在大小写规范化及 `T->U` 后全部一致，项目独有为 0，序列不一致为 0。
- BPSEQ 连续 1-based 索引转换为项目 zero-based inclusive 配对，不需要额外位移。
- 全部 5,388 条项目配对集合都逐条等于历史 `project_to_legal` 对同名 release 配对的投影结果，排除了未记录的序列重排或索引偏移解释。

### min-loop 与 crossing

项目 `scripts/build_ss_corpus.sh` 对 bpRNAnew 使用 `--min-loop-policy drop --pseudoknot-policy drop`。历史转换器先删除 `span<=3` 配对，再反复删除涉及最多 crossing 的配对，同数时选择跨度更大的配对。

- 219 条记录删除 min-loop 配对，共 227 对。
- 2,463 条记录删除 crossing 配对，共 10,549 对。
- 合计 2,602 条记录至少有一种删除；与历史 manifest 的 `n_records_with_drops=2602`、`n_dropped_illegal_pairs=227`、`n_dropped_crossing_pairs=10549` 完全一致。
- 这是 5,388 条项目标签相对 release 预测的过滤来源，但不能单独解释项目标签相对官方参考标签的全部 5,282 条差异。

### canonical

在 5,388 条交集中，release、官方参考和项目三套配对的非 canonical 配对数都为 0。13 条被排除记录的 release 预测另有 11 个非 canonical 配对，但其官方参考仍为 0；这些记录不属于 5,282 条交集差异。项目转换器没有单独的 canonical 删除阶段。故 canonical 过滤不是本数据集 5,282 条差异的来源；本审计也不把“交集中没有非 canonical”误写成发生过 canonical 清洗。

### 13 条官方独有记录

项目从 5,401 条变为 5,388 条是 13 条 `bad_alphabet` 硬拒绝，不是索引配对失败，也不是 crossing/min-loop 删除整条记录。13 个名称及各自异常字符见逐记录文件和 `summary.json`。

### DSSR / 文献协议

现有 release API、BPSEQ、项目转换器和 manifest 没有逐记录 DSSR 或文献人工整理谱系。它们足以证明“release 预测来源＋项目过滤”与“官方参考标签来源”不同，但不足以把差异进一步归因于 DSSR 版本、文献整理规则或实验协议。本审计将此项标为 `NOT_ESTABLISHED_BY_THESE_ARTIFACTS`，不作无证据根因推断。

## 三组审计值独立复算

严格 zero-based inclusive 精确配对匹配；计分器不额外删除 crossing、canonical 或短环配对。micro-F1 由全库 TP/FP/FN 汇总，macro-F1 是逐序列 F1 的非加权平均；真值与预测均为空时单序列 F1=1。

| 预测身份 | 标签身份 | n | TP | FP | FN | micro-F1 | macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 官方 v0.2 release | 官方 `BPfold_data` 参考 | 5,401 | 86,933 | 59,441 | 41,092 | **0.633624758107719** | **0.6470223791166072** |
| 项目六 checkpoint replay | 项目标签 | 5,388 | 106,998 | 48,176 | 28,186 | **0.7370074184282851** | 0.7447389175263477 |
| 官方 v0.2 release | 项目标签，仅误配审计 | 5,388 | 135,184 | 10,776 | 0 | **0.9616708875167174** | 0.9584497822657716 |
| 官方 v0.2 release | 官方参考的项目名称交集 | 5,388 | 86,731 | 59,229 | 40,929 | 0.6339521964768657 | 0.6472307861307366 |

官方 5,401 条 macro-F1 四舍五入为 0.647，复现 BPfold v0.2 README 值。0.9616708875167174 的 FN=0 是因为项目标签本身来自这些 release 预测经过删除后的子集；该值是标签来源敏感/循环一致性审计，**不是独立精度、官方结果或可用于选模的比较差距**。

## 推荐同协议比较口径

1. **官方 BPfold 口径**：同一官方 release 预测对全部 5,401 条官方参考标签，严格精确配对，报告上述 micro/macro-F1。与另一模型比较时，另一模型必须提供同样 5,401 条、同一参考标签、同一指标和覆盖率的预测。
2. **项目 raw 口径**：项目 replay 或其他冻结模型对项目 5,388 条标签，可复算 raw 项目内比较，但必须标记标签来源、过滤规则、raw 非去污染身份和模型形式。不得与官方 5,401 条结果混排。
3. **交集敏感性口径**：若只能取得 5,388 条交集，release 对官方参考的结果是 0.6339521964768657 / 0.6472307861307366；它不能替代全量官方 5,401 条结果。
4. **禁止口径**：不得把项目 replay 预测替换为 release 预测，不得把项目标签分数与官方参考分数放在同一排名列，不得把 0.961671 当作 BPfold 官方精度，不得利用这些外部 Test 审计值选择 checkpoint、阈值或模型。

## 产物与复现

- `tools/audit_bpfold_label_protocol_20261127.py`：生成逐记录事实表及指标摘要。
- `tools/verify_bpfold_label_protocol_20261127.py`：不导入生成器，独立实现 BPSEQ 解析、投影、配对检查和计分，逐字段核对全部 5,401 行。
- `tests/test_bpfold_label_protocol_20261127.py`：索引、过滤、冲突和指标单元测试。
- `source_data/bpfold_label_protocol_audit_20261127/summary.json`：输入哈希、分类计数和完整指标。
- `source_data/bpfold_label_protocol_audit_20261127/per_record.jsonl.gz`：确定性 gzip 的完整逐记录分类，避免将大型原始产物纳入 Git。
- `source_data/bpfold_label_protocol_audit_20261127/independent_verification.json`：独立校验结果与交付文件哈希。

```bash
cd /home/rna_jev_bpfold_protocol
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
"$PY" tools/audit_bpfold_label_protocol_20261127.py
"$PY" -m pytest -q tests/test_bpfold_label_protocol_20261127.py
"$PY" tools/verify_bpfold_label_protocol_20261127.py
```
