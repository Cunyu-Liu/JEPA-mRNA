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

## 4. 基线获取状态（实测，非推测）

### 4.1 已取得

| 基线 | 状态 | 证据 |
|---|---|---|
| ViennaRNA MFE / centroid / MEA | ✅ **已测** | 见 §1，ViennaRNA 2.7.2 |
| Nussinov+Turner（我们 MLP_T=0） | ✅ **已测** | 见 §1 |
| **MXfold2** | ✅ **可运行** | `~/miniconda3/envs/rna_baselines` 已装；`python -m mxfold2 predict x.fa` 实测输出正确（`GGGAAACCCUUUAGCUAGCU → ((....))............ (-2.0)`）。**注意子命令是 `predict` 不是 `fold`**；`from mxfold2 import mxfold2` 会 ImportError（`__init__.py` 为空），必须走 CLI。训练参数随包自带（`mxfold2/models/TrainSetAB.pth`）。 |

`eval/ss/run_baselines.py` 新增 `--external-dbn`：对**外部工具产出的 dot-bracket** 打分，
复用同一套 `score_all`（同一份指标实现），并**逐条校验序列长度**以防顺序错位。

### 4.2 明确取不到（含原因，避免重复尝试）

> **本节于 2026-09-24 晚被大幅更正。** 原表把 UFold、RNAformer、EternaFold 都写成"不可得"，
> **是错的**，且已造成实际损失（C1-a 被误判为不可能完成、教师问题被搁置）。
> 更正依据与逐条证据见 `DECISION_TRAINING_LOG.md` §14.32。

| 基线 | 更正后的结论 | 证据（实测命令与结果） |
|---|---|---|
| **UFold** | ✅ **权重在集群本地**，可直接用 | `ls /home/cunyuliu/rna_baselines_src/UFold-main/models/` → `ufold_train_alldata.pt`（34.6 MB）。旧记录"仓库只有 Readme"说的是**上游仓库**的 `models/` 目录，而这里早就有一份下载好的权重，且 `ufold_run.log` 显示它**跑通过**。真正的障碍是**数据**：`data/` 只有 `Readme.md` 与 `input.txt`，`TS0.cPickle` 等缺失，必须绕开它的 `RNASSDataGenerator` 直接喂序列 |
| **RNAformer** | ✅ **权重 + 源码本轮已齐** | 3 个 checkpoint + 配置在 `/mnt/cunyuliu/rna-jepa/refmodels/models/`；源码 `git clone https://github.com/automl/RNAformer.git` **本轮成功**（`github.com` 间歇可达，本次通）→ `/mnt/cunyuliu/rna_baselines_src/RNAformer-code`，含 `evaluate_RNAformer.py`。**它是本轮新 benchmark 的来源，因此是可比性最强的已发布基线** |
| **EternaFold** | ✅ 源码在盘上，`make multi` 可编译 | `/mnt/cunyuliu/rna_baselines_src/EternaFold/`，`parameters/EternaFoldParams.v1` 在位；用法见其 README：`./src/contrafold predict <seq> --params parameters/EternaFoldParams.v1`。**同时是教师集成问题的答案**（§14.30） |
| **MoEFold2D / RiFold / Graph-Mamba** | ✅ 源码在盘上 | `/mnt/cunyuliu/{MoEFold2D-main,RiFold,Graph-Mamba-main}` |
| **mmseqs** | ✅ 在盘上 | `/mnt/cunyuliu/mmseqs`。旧记录"未安装所以 80% identity 去冗余做不了"因此作废 |
| **SPOT-RNA / SPOT-RNA2** | ⚠️ **仍未核实**（但按同样方法必须再查一遍） | 原证据是"仓库根目录只有代码、`models/` 路径 404"。鉴于 UFold 的结论就是这样被搞错的，**这条必须先在集群本地搜一遍文件名再下结论**，不得沿用 |
| E2Efold | ❌ 未复现 | 其 bpRNA-new 数字是 OOD 论断的关键，**引用前必须回原文核对** |

**教训（已固化为纪律）**：旧结论是把「在少数几台探测主机上网络不可达」外推成「资产不存在」。
这两者不是一回事——**权重可以已经在盘上，与网络无关**。今后任何"不可得"结论必须写明：
在哪几台机器上、用什么命令探测过、错误原文是什么，并**先在集群本地搜一遍文件名**。

> **对 C1-a 的影响（本节更正后重新判定）**：C1-a 要求与本项目 System-1 头做同口径 ECE 对比的
> sigmoid 概率。原结论"UFold 权重不可得→C1-a 无法完成"建立在错误前提上，**作废**。
> 现行判定：
> - **C1-b（对 ViennaRNA 精确边际）已完成**（TS0 ECE 0.0048）；
> - **C1-c（对精确边际）已完成**，分两口径报告（见 §14 表）；
> - **C1-a 可以完成**——条件是从 UFold / RNAformer 取到 **sigmoid 概率矩阵**。
>   两条任务已独立启动，产物落 `/mnt/cunyuliu/rna-jepa/eval_decision/`。
>   **在概率矩阵真的产出之前，不得声称 C1-a 已完成**；`spec/checklist.md` 的 C1-a 门已由
>   「不可完成」改为「待产物」。论文中原本要写的"缺口说明"段落改为
>   **"与 UFold / RNAformer 的 sigmoid 概率在同一 split 上做了同口径 ECE 对比"**，
>   并附权重文件哈希与 commit。

---

## 5. MXfold2 实测（唯一可得的学习型基线）

| split | micro F1 | macro F1 | INF | 备注 |
|---|---|---|---|---|
| **bpRNA TS0** | **0.5651** | **0.5698** | **0.5832** | 1,288 条，打分 23.6 s |

命令：`python -m mxfold2 predict bprna_ts0.fa > mxfold2_bprna_ts0.dbn`（`rna_baselines` env）
→ `eval/ss/run_baselines.py --split bprna_ts0 --external-dbn ... --external-name mxfold2`

### 5.1 **这改变了 P1/P2 的目标值**

之前以 ViennaRNA centroid 0.5393 作为"要超越的基线"；**MXfold2 把门槛抬到 0.5651**。
当前最好模型（`rinalmo_ff_b4_s0` @1000 步）是 **0.4554**，差距 **0.110**。
**这是当前第一风险，必须优先解决**（架构/解码/训练量三条路，见 `DECISION_TRAINING_LOG.md` 运行 14.7 末段）。

### 5.2 一个必须核实的污染问题（**尚未证实，不得当成结论**）

MXfold2 的训练集是随包自带的 `models/TrainSetAB.pth`（bpRNA-1m 的 TrainSetA+B）。
若它确实只用了 bpRNA-1m 的标准 train 划分，则与 TS0 天然不相交，0.5651 是无污染的；
但**本集群无法读取该 `.pth` 的训练序列清单**，所以：
- 论文中引用 0.5651 时**必须同时声明"未能核实其训练集与 TS0 的同源重叠"**；
- 可行的替代核验：用 CD-HIT / BLAST 对 TS0 与 bpRNA-1m train 做同源筛选（**未做**）。

### 5.3 顺带修掉的一个真 bug：`--external-dbn` 曾被静默忽略

`eval/ss/run_baselines.py` 接受 `--external-dbn` 却**从不读取它**——当初生成该补丁的脚本
定义了主流程的改动却只应用了 CLI 与读取器两处（`MAIN_OLD`/`MAIN_NEW` 定义了但没进循环）。
后果：第一次运行"成功"写出结果文件，但里面**没有 mxfold2 那一行**，表格看起来是完整的。
现已补上主流程并新增 2 个测试（打分正确性 + 顺序错位必须报错）。**这类"静默丢行"比直接报错更危险。**

