# Round39 ViennaRNA 原生概率补测合同

## 目标与边界

为 T-A174 补充 ViennaRNA-MFE、ViennaRNA-centroid、ViennaRNA-MEA 的原生配对概率、ECE/Brier/NLL、结构置信度和 risk–coverage。它们是附加概率基线，不改变已完成的 13 系统 own-clean 主榜，也不把 matched-clean 交集当成主排名。

## 方案选择

采用 ViennaRNA Python API：每条序列、每次 repeat 建立一个 `fold_compound`，先 `mfe()`，再执行一次 `pf()`，随后获得 `centroid()`、`MEA()` 和 `bpp()`。一次分区函数服务三种解码，保留原生 float64 BPP。

备选 CLI 方案会产生五位小数文本并重复计算分区函数，只在 API 不可用时考虑，不与 float64 结果混写。最终结构的 one-hot 不是边际概率，禁止用于 ECE。EternaFold/CONTRAfold 的现有官方 MEA 结果未保存逐对后验概率，标记 `NOT_AVAILABLE_NATIVE_PAIR_PROBABILITIES`，不伪造数值。

ViennaRNA 不拟合训练序列，保留八个完整测试集；不能套用其他模型的 clean n。原生 CPU 计算不是 GPU 静默降级，所有 GPU 模型仍执行 CUDA 硬门。

## 数据流与验收

1. 对 `vienna/raw_strict/<split>.json` 核验数据哈希、`not_sequence_fitted`、八 split 原始三次预测。
2. 每 split 完整执行三次，三种结构的预测和 BPP 逐 repeat 一致；逐条 F1 与既有原生结果一致，不能只比较汇总值。
3. 每种解码保存 `names/offsets/probabilities/labels` NPZ、原生预测、TP/FP/FN、三次 F1、结构置信度和代码/数据/软件版本哈希。候选配对仍为严格上三角 `valid_pair_mask`，10 bins。
4. 独立验证器重新读取 NPZ，逐条复算标签、F1、概率哈希、ECE/Brier/NLL、Spearman/AURC 和 risk–coverage。24 格全部通过后才形成汇总。
5. 与 xens2 按名称、序列和实际评分 GT 对齐，输出 24 格 matched-clean 附加敏感性。空交集不写 0 分。
6. 小序列单元测试及真实 TS2 三次全量对齐通过后再启动八 split 长期队列；大 split 未完成、其他 eligible 基线未齐或 provenance blocked 未解除时，不宣称全板最低 ECE 或最佳置信度。
7. 原始大数组留在 `/teams`；核心验证 JSON、比较汇总、台账、测试与代码提交 GitHub。并发运行耗时不作为速度结果。
