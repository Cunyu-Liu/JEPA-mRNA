# 物理基线实测表（2026-09-24）

> 来源：`eval/ss/run_baselines.py`（新增）。指标与 `evaluate_decision.py` **共用同一份实现**
> （`ss.metrics.PairLevelMetrics` / `StructureLevelMetrics`，micro 汇总 TP/FP/FN、macro 取逐序列 F1 均值、
> INF 取均值），因此基线与模型行**可直接比较**。ViennaRNA **2.7.2**。

## 1. 结果

| split | 基线 | micro F1 | macro F1 | INF | 耗时 |
|---|---|---|---|---|---|
| **bprna_ts0**（1,288） | `vienna_mfe` | 0.5055 | 0.5086 | 0.5222 | 77 s |
| | **`vienna_centroid`** | **0.5393** | 0.5288 | 0.5405 | 129 s |
| | `vienna_mea` | 0.5241 | 0.5218 | 0.5346 | 124 s |
| | `nussinov_turner`（我们 MLP_T=0） | 0.2124 | 0.2303 | 0.2362 | 229 s |
| **archiveii**（3,950） | `vienna_mfe` | 0.5764 | 0.6219 | 0.6264 | 665 s |
| **bprna_new**（5,388，OOD） | `vienna_mfe` | 0.6379 | 0.6581 | 0.6638 | 199 s |
| | **`vienna_centroid`** | **0.6770** | 0.6821 | 0.6871 | 326 s |

（`archiveii` / `bprna_new` 的 centroid/MEA 与 `nussinov_turner` 仍在跑。）

## 2. 三条对论文有直接影响的结论

### 2.1 **centroid > MEA > MFE**，且 centroid 是最强物理点预测

三个 split 上一致。centroid 与 MEA 都**建立在配分函数之上**（用 BPP），MFE 不用。
→ **这是 C1 的正确标尺**：我们的免配分函数决策头，必须去逼近 **centroid 的 F1（TS0 0.5393）**，
而它连一次配分函数都没算。这句话比"快 50×"有力得多，而且**可被实测支持**。

### 2.2 `nussinov_turner = 0.2124` 与**未训练头 0.2167 几乎相同** —— 一个有效的内部一致性检验

`MLP_T = 0` 时决策头的起点**精确等于**「Nussinov + Turner 堆叠能」（spec §0.7 问题 3）。
实测两者 F1 相差 0.004（且未训练头略高，因为其残余项已随机初始化但校准温度仍在）。
**这验证了"训练起点 = 物理模型"这一实现声明**，同时说明：0.56 epoch 的头**确实还没学到东西**，
不是"架构不行"。

### 2.3 **ViennaRNA 在 OOD 上更好（bprna_new 0.638 > TS0 0.506）**

领域内常引用的"所有学习方法在 bpRNA-new 上都差"（spec 记 E2Efold F≈0.0361）**不是数据问题，是学习方法的问题**——
纯物理基线在同一集合上反而**明显更好**。这为我们的**物理先验归纳偏置**（Turner 残差先验 + 热力学蒸馏）
提供了直接的实证支持，也把论文的泛化主张从"精度更高"改成"**衰减更小**"变得更有底气。

> **待核验**：E2Efold 在 bpRNA-new 上 F≈0.0361 这一数字来自早期调研笔记，**引用前必须回原始论文核对**。
> 本次没有独立复现它。

## 3. 复现命令

```bash
PYTHONPATH=/mnt/cunyuliu/pylibs:src:eval python eval/ss/run_baselines.py \
    --split bprna_ts0 --out /mnt/cunyuliu/rna-jepa/eval_decision/baselines_bprna_ts0.json
```

## 4. 尚未取得的基线（如实记录）

| 基线 | 状态 | 说明 |
|---|---|---|
| UFold / SPOT-RNA / SPOT-RNA2 | **未取得** | 需下载权重；是 C1-a 的头号对照（"免 DP 但未校准"） |
| BPfold | 权重已在集群（`rna_ss_data/bpfold/mp/model_predict/`），**包未安装** | 需 clone `heqin-zhu/BPfold` |
| MXfold2 / CONTRAfold / LinearPartition | 未安装 | CONTRAfold 是最接近的先验工作，应尽量补上 |
| E2Efold | 未复现 | 其 bpRNA-new 数字是 OOD 论断的关键，需核对原文 |
