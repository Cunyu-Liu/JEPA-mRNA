# Round38 Own-clean Table Appendix Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** 在原 23 页 round38 PPT 后追加 8 页附录，将 13 系统×8 split 的 own-clean n、F1、TP/FP/FN、精确删除数和同源删除数完整入表。

**Architecture:** 每个 split 一页、每页 13 个系统，保留源 JSON 的 clean n 和 EMPTY 语义。新增页面采用原 deck 的蓝/青配色、微软雅黑和 1.5 倍行距，不重排或改写已有页面。

**Tech Stack:** Python 3、python-pptx、原生 PowerPoint PDF 导出、JSON/SHA256 验证。

---

### Batch 1: 表格生成与源数据绑定

**Files:**
- Create: `/home/rna_jev/tools/append_round38_ownclean_tables.py`
- Create: `ppt/.round39_work/test_append_round38_ownclean_tables.py`
- Input: `ppt/.round39_work/source_data/appendix_leakage/round39_own_clean_leaderboard.json`

**Risk:** High — 104 格跨系统 clean n 不同，EMPTY 被误写为 0 或串行错位会改变科学结论。

**Implementation:** 校验 `PASS_COMPLETE_OWN_CLEAN_LEADERBOARD`、104/104 和源哈希。每个 split 输出 13 行，字段为 system、clean n、F1 mean、TP、FP、FN、removed exact、removed homology。EMPTY 的 F1 和计数显示 `—`，不套其他系统数据。保留原生 598 nt 等特殊协议说明。

**Minimum verification:** 全 104 格与源 JSON 逐值比较；源 SHA256=`9da97573e4049122dc63f5e2323685c571dbb5e51e5e677cc7cdaaef93f7b326`；EMPTY 仅 RNAformer-intrafam 与 SPOT-RNA 的 TS3。

**Independent review:** No — 确定性字段映射由全量断言覆盖。

### Batch 2: 追加八页附录

**Files:**
- Modify: `ppt/RNA-JEV_论文逻辑重构_20261031_导师讨论_round38.pptx`
- Backup: `ppt/.round39_work/round38_before_ownclean_tables.pptx`
- Create: `ppt/.round39_work/round38_ownclean_tables_qa.json`

**Risk:** Medium — 表格溢出、字体过小或误改原页会影响导师汇报。

**Implementation:** 在原 23 页后追加 8 页，每页一个 split、13 行。保留源系统顺序，xens2 首行仅便于阅读；其余系统顺序不变。使用微软雅黑、1.5 倍行距、蓝表头、交替浅色行和页码 24–31。正文包含 clean n、F1、TP/FP/FN 与两类删除数，页脚说明 strict micro-F1 和 EMPTY 边界。

**Minimum verification:** 31 页、原 23 页文字逐项一致、表格行数/列数正确、页面边界与字体检查通过。

**Independent review:** No — 版式通过原生渲染和自动尺寸检查复核。

### Batch 3: 原生渲染、归档与提交

**Files:**
- Modify: `ppt/RNA-JEV_论文逻辑重构_20261031_导师讨论_round38.pdf`
- Modify: `交接文档/records/DECISION_TRAINING_LOG.md`
- Modify: `交接文档/source_data/SOURCE_MANIFEST.json`

**Risk:** High — 错误文档、旧内存图或缺失中文字体会造成表面完成。

**Implementation:** 只按 `_round38.pptx` 后缀定位目标，先关闭旧内存副本再打开。导出 31 页 PDF，核对第 8 页和新增 24–31 页文本身份，保存图源、QA、代码和提交。

**Minimum verification:** 原生 PDF 31 页、中文正常、无表格越界/遮挡；104 格与源表一致；两个 EMPTY 均显示 `—`；其他 22 个非附录页面及第 8 页均保持原状。

**Independent review:** Yes — 最终科学数字和视觉交付需独立复核，不能用文件存在代替验收。

**执行分组：** 在当前工作区保留备份并顺序实施，不创建额外代理或重复工作区。Batch 1 与 Batch 2 一起交付，Batch 3 在完整附录生成后一次验收。
