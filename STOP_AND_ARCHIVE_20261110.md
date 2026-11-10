# 停止实验与归档交接总结（2026-11-10）

## 停止状态

用户明确要求停止后续实验。远端 watchdog、训练、评测及等待队列均已停止，不再自动恢复。停止记录为 `source_data/round50/xens2_complex/user_stop_20261110.json`，SHA256=`d04730d3d3559c715fe03e45a5a35d1fd67dd89722390586492d7586639ebd01`。远端大型数据、权重和 artifacts 保持原位。

## 已确认的科学结果

- 旧 xens2 扩充数据 Rfam：micro-F1=**0.6455382794**，macro-F1=0.6399835804，precision=0.6637207166，recall=0.6283254843。三次原始结果逐字节一致，双 QA 一致，QA SHA256=`68003ed93f160a3fc30f636e5a037872ab29fa8094adee0c22e377d1ceec7a29`。
- 相对旧 xens2 约 0.6514，本次扩充没有提升。只使用符合旧 xens2 1024 nt 上限的 150 条扩充链，不能据此否定全部复合物数据或外推长链效果；模型仍为集成，不替代独立单模型。
- structRFM 三臂和 RIPE 七臂主榜均未达到 8/8 SOTA，负结果与独立 QA 均保留。
- Pair-JEPA 仅完成 context_v1 工程验收及 J-mlp 三种子、主 39 格。greedy/Turner=0、EMA/noEMA/random 和全臂裁决未齐套，不作效果或防坍塌结论。
- 复合物 D0/D1 和 LinearDesign 正式实验未开展。8/8 SOTA、速度 SOTA、最佳置信度和长序列优势均未达成，不能写成完成。

## 证据与绘图源

- `source_data/SOURCE_MANIFEST.json`：1045 个文件、4,011,927,576 字节，SHA256=`5aafd5eefd643e19a2a3823df8e124e7d152ac3567a1011f6c6297fe6d268261`。
- `source_data/FIGURE_SOURCE_INDEX.md`：当前关键图件、原始数据与停止边界。
- `source_data/round50/xens2_complex/`：Rfam 三次原始结果、双 QA、数据配方和停止记录。
- `source_data/round49/rna_complex_decontamination/legacy_647_359/`：旧 647/359 版本独立保留；当前版本为 805/201，不混用。
- 旧版源清单：`archive_20261110/manifest_history/SOURCE_MANIFEST.pre_round50.json`。

## 根目录整理

已归档 33 组、1987 个文件，约 750,422,015 字节，仅移动，不删除：

- 根目录旧说明、历史方案、早期草稿和旧训练/评测脚本；
- `figs/`、`records/` 旧图件与记录；
- `.pytest_cache`、`.DS_Store`；
- `ppt/.round39_work` 和已完成的 `.round49_*`、`.round50_pairjepa_work` 开发目录。

逐文件哈希：`archive_20261110/ARCHIVE_INDEX.json`，SHA256=`f037085226057f927875cb19eaa9cc0e0fcb5f2ee697543f6b3aed82bc643034`。汇总索引为 `ARCHIVE_GROUP_INDEX.json` 和 `ARCHIVE_INDEX.md`。索引中的根目录单文件路径和组级计数已复核修正，全部归档文件均通过哈希验证。

根目录现仅保留 `交接文档/`、`论文/`、`ppt/`、`tables/` 和 `archive_20261110/`。PPT 原稿及用户调整未修改。

## 代码与后续交接

代码已推送至 `dd9f033`。Rfam QA 修复为 `a2e1aff`，VL0 参考种子修复为 `dd9f033`。恢复工作时必须先阅读 `spec/tasks.md`、`spec/spec.md` 与 `records/DECISION_TRAINING_LOG.md`，不得把停止状态或未齐套结果写成完成，也不得未经用户同意重新启动实验。

## 2026-11-10 最新恢复覆盖

用户已明确批准重新执行实验，但将后续骨干固定为RiNALMo，并新增结构感知对比与LinearDesign下游扩展。最新统一合同为 `docs/plans/2026-11-10-rinalmo-full-replay-and-downstream-expansion.md`。structRFM所有结果和本停止记录均保留为历史诊断，不删除、不混入新对照。当前先完成文档与资产核对，尚未启动RiNALMo新训练。
