# 决策模型训练记录（结构预测线）

> 本文件记录**每一次训练运行的配置与结论**。规则：**smoke / proxy / 训练集结果一律标注为中间结论，不得当作最终科学结论**。
> 权威数据/评测口径见 `spec/benchmark_decision.md`。运行产物在 `/mnt/cunyuliu/rna-jepa/runs/`，台账 `runs/ledger.jsonl`。

---

## 环境与约定（2026-09-24 建立）

| 项 | 值 |
|---|---|
| 集群 | `ssh A100`，8×A100-40GB，无作业调度器 |
| 可用加速器 | GPU 0–5（共享）+ GPU 6（7×MIG 1g.5gb）+ GPU 7（2×MIG 3g.20gb） |
| Python | `/home/cunyuliu/miniconda3/envs/lucaone/bin/python`（torch 2.5.1+cu121） |
| ViennaRNA | **2.7.2**，装在 `/mnt/cunyuliu/pylibs`（`/home` 配额已满 200 G） |
| 必需环境变量 | `PYTHONPATH=/mnt/cunyuliu/pylibs:src`、`TMPDIR=/mnt/cunyuliu/tmp` |
| 代码位置 | `/home/cunyuliu/rna-jepa`（提交在此）；大数据在 `/mnt/cunyuliu` |
| 监控 | cron `RNAJEV_DECISION_MONITOR`（10 min）：显存快照 + 日志健康检查 + 有空闲即派发队列 |

---

## 运行 0：GPU 冒烟（真实数据）— **中间结论，不是科学结果**

- 配置：`--arm smoke_real`，35M encoder，batch 2，lr 1e-4，20 steps，数据 `bprna_vl0.jsonl`（198 条），`lambda_distill=rlcd=cal=0`
- 设备：GPU 5，`run_meta.json` 记录 `device=cuda, resolved=cuda:0, gpu_name=NVIDIA A100-PCIE-40GB, CUDA_VISIBLE_DEVICES=5`
- 参数量：**21,231,438**（35M preset：6 层 / d_model 512）
- 结果：loss 48.24 → 37.43（末值 20.03），**无 NaN**，梯度覆盖 **86/86 covered，0 missing，0 zero**
- 耗时：20 steps / 22.6 s
- **结论**：端到端通路可用；仅证明「能跑」，**不构成任何科学结论**。

### 本次冒烟暴露的 4 个真 bug（已修，均已提交）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `Expected all tensors on the same device, cuda:0 and cpu` | `collate()` 只造 CPU 张量，冻结的损失模块不会搬设备 | 新增 `_batch_to_device()`，在 `objective_terms` 入口统一搬 |
| 2 | `mat1 and mat2 shapes cannot be multiplied (38642x1536 and 2304x128)` | `build_encoder("35M")` 是 d_model=512，但 head 用 `config.d_model`（默认 768） | head 改为读**实际构建出的 encoder** 的 `d_model` |
| 3 | `nll_only` 臂 100% CPU 卡死 4+ 分钟、零 GPU 活动 | 数据加载对**每条**序列都算教师概率；无 `--teacher-dir` 时退化为 `MockTeacher` 的 O(L³) numpy DP（10,682 条 ≈ 2.5e10 次运算），而 `lambda_distill=0` 根本不会读它 | 加 `need_teacher_probs`（不读就不算）+ `strict_teacher`（真教师缺条**报错**而非静默换 mock） |
| 4 | 修 #3 后 `nll_only` 又崩：`can't convert np.ndarray of type numpy.object_` | `teacher_probs` 是 `[None, ...]`，`np.asarray(None)` 是 object 数组 | `_batch_to_device` 放行 `None`（`objective_terms` 已硬性拒绝「需要却缺失」） |

> **教训**：这 4 个问题**没有任何一个**能被 224 项 CPU 测试发现。用户要求「训练必须用 GPU」的真正价值就在这里——不是仪式，是唯一能暴露这些 bug 的路径。

---

## 运行 1/2/3：三臂并行消融（进行中）

- 启动：2026-09-24 04:42（并行，显式钉卡，避免并发抢占同一张卡）
- 数据：`bprna_tr0.jsonl`（**10,682** 条，投影后；均值 L=132.5，范围 33–498）
- 教师：`ss_data/teacher/bprna_tr0`（42 shards，`verify_teacher_labels() == True`，锁定 **ViennaRNA 2.7.2**）
- 公共配置：35M encoder，batch 2，lr 1e-4，**3000 steps**，seed 0，grad_clip 1.0

| 运行 | arm | 设备 | λ_nll | λ_distill | λ_rlcd | λ_cal | 对应消融 |
|---|---|---|---|---|---|---|---|
| 1 | `full` | GPU 1 | 1 | 1 | 1 | 1 | 完整四项目标（主结果候选） |
| 2 | `nll_distill` | GPU 5 | 1 | 1 | 0 | 0 | 去掉 RLCD 校准项 |
| 3 | `nll_only` | GPU 4 | 1 | 0 | 0 | 0 | 仅硬标签（无热力学蒸馏） |

**首轮观测（中间结论）**：

| 运行 | 已到 step | 近期 loss | 各项量级 |
|---|---|---|---|
| `full` | 140/3000 | 58.8 / 103.8（长度分桶造成批间波动） | nll≈59–104，distill≈0.30，rlcd≈−0.74，cal≈0.31 |
| `nll_distill` | 180/3000 | 82.1 / 109.2 | nll≈82–109，distill≈0.40–0.53 |
| `nll_only` | 90/3000 | 65.6 / 66.3 | 仅 nll |

- GPU 利用率：0–5 全部 **100%**（符合「每张卡显存占满」的推进策略）
- 梯度范数（**裁剪前**）：475–736，远大于 `grad_clip=1.0` → 实际每步都被裁到 1.0。**待观察**：这是否拖慢收敛；若 loss 平台化需改自适应裁剪或放宽阈值。
- **注意**：loss 绝对值随 batch 内序列长度变化，**不可跨臂直接比较**，必须等评测阶段用同一测试集算 F1/ECE。

---

## 运行 4：评测管线打通 + **未训练基线**（TS0，1,288 条）

- 命令：`eval/ss/evaluate_decision.py --checkpoint runs/smoke_real/resume.pt --data .../bprna_ts0.jsonl --exact-marginal-limit 200`
- checkpoint 是**只有 20 步**的冒烟权重 → **这是「未训练基线」，不是科学结果**

| 指标 | System-1（免 DP 决策头） | 精确边际（CRF 后验，仅作 C1-c 参照） |
|---|---|---|
| 配对 micro P / R / F1 | 0.1886 / 0.2547 / **0.2167** | — |
| 配对 macro F1 / INF | **0.2372** / 0.2433 | — |
| **ECE** | **0.5547** | **0.0057** |
| Brier | 0.3431 | 0.0071 |
| NLL | 1.1444 | 0.0356 |
| 平均预测概率 vs 实际频率 | **0.5605 vs 0.0058** | 0.0090 vs 0.0068 |
| 边际校准误差 | 0.5547 | 0.0022 |
| 样本数 | 6,015,437 对 | 777,174 对（前 200 条） |

**合法性：非法结构率 = 0，最小发夹环违规率 = 0**（G1/G2 成立，架构硬保证）

**延迟（batch=1，System-1 = 前向 + 解码；不含配分函数）**：

| 长度桶 | n | System-1 总计(ms) | 精确边际参照(ms) |
|---|---|---|---|
| <100 | 565 | 110.1 | 33.5 |
| 100–200 | 528 | 210.8 | 70.8 |
| 200–400 | 167 | 870.3 | 292.8 |
| 400–600 | 28 | 1745.1 | 999.8 |

### 三条必须记住的结论

1. **C1-c 目前 FAIL（gap = 0.549，阈值 0.02）——这正是 C1 要解决的问题，不是坏消息。**
   未训练的决策头 sigmoid 输出**严重过度自信**（预测均值 0.56，真实配对频率 0.58%），而精确边际几乎完美校准（ECE 0.0057）。
   **这说明：C1 的立论前提在真实数据上成立**——「免 DP 直接出概率」与「免 DP 且概率校准」之间确实有 0.55 的鸿沟，而 RLCD/校准训练的任务就是把它压到 0.02 以内。**等三臂训练完成后重测才有意义**。

2. **F1 ≈ 0.22 是未训练水平**（SPOT-RNA 在 TS1 上报 F1≈0.69）。**不得当作方法性能引用。**

3. **速度门 S3（L=512 ≤10 ms）当前不可能达标**，原因明确且可修：评测里 System-1 的解码用的是**精确 `nussinov_map`（O(L³)，numpy，跑在 CPU）**，而 spec §5.0.1 定义 System-1 的解码是「**带状/线性 DP**」。
   → 待办：为速度门单独实现**带状/层级级联解码路径**（§5.0.2 的 L0/L1/L2），并**分别报告**「精确解码的 F1」与「快速解码的 F1/延迟」。在补上之前，**任何加速比数字都不得引用**。

---


## 运行 5：六臂长训练（batch 4，40,000 steps，MIG 并行）— **进行中**

- 启动：2026-09-24 06:06（6 个 arm 各占一个空闲 MIG 1g.5gb 切片，`scripts/launch_long_decision.sh`）
- 为什么是 MIG：GPU 0–5 被其他用户占满（利用率 100%），GPU 7 的 3g.20gb 切片仅剩 0.14 GiB。
  GPU 6 的 7×1g.5gb 是**唯一真正空闲**的加速器（实测每片 4.3–4.6 GiB 空闲）→ 按项目规则「有空闲即占满」全部填上。
- 为什么 batch 4：实测 flat head 峰值显存（fp32，L=498 语料最长序列）**2366 MiB**，
  在每一片 MIG 上都能容纳；batch 8 在 L=498 会 OOM。
- 公共配置：35M encoder（**随机初始化**，见「运行 6」发现 B）、batch 4、lr 1e-4、**40,000 steps**（≈15 epoch）、`--save-every 4000`

| # | tag | λ_nll | λ_distill | λ_rlcd | λ_cal | seed | 用途 |
|---|---|---|---|---|---|---|---|
| 1 | `full_b4_s0` | 1 | 1 | 1 | 1 | 0 | 主结果候选 |
| 2 | `nllonly_b4_s0` | 1 | 0 | 0 | 0 | 0 | 消融：仅硬标签 |
| 3 | `nlldistill_b4_s0` | 1 | 1 | 0 | 0 | 0 | 消融：蒸馏但无显式校准（**C1/H6 的关键对照**） |
| 4 | `full_b4_s1` | 1 | 1 | 1 | 1 | 1 | 种子方差 |
| 5 | `full_b4_s2` | 1 | 1 | 1 | 1 | 2 | 种子方差 |
| 6 | `nlldistill_b4_s1` | 1 | 1 | 0 | 0 | 1 | 种子方差（消融臂） |

> **纪律**：这六臂在跑完并过评测前，**不得产生任何科学结论**。日志里的 loss 因长度分桶而跨臂不可比。

---

## 运行 6：第二轮回归审计（代码级核验）+ 显存修复

> 第一轮审计（spec §0.9.1–§0.9.4）比对的是**范式**；第二轮直接读代码与运行配置，**结论比第一轮严厉**。已写入 `spec/spec.md` §0.9.5。

### 发现 A（严重）：C2「层级决策级联」不在训练/评测路径上，且与目标函数不相容

**三条独立证据（全部实测，非推断）**：

1. **无调用点**：`train_decision.build_decision_model()` 只构造 `FlatDecisionHead`；
   `DecisionModel.forward` 向 head 传 `lengths=`，而 `HierarchicalCascade.forward(h, seq_ids)` **不接受该参数**
   → 直接 `TypeError: forward() got an unexpected keyword argument 'lengths'`。
2. **与配分函数目标不相容**（`tools/profile_step.py`，L=60）：
   `flat_nll = 6.271` vs `cascade_nll = 90017.977`，非活跃块对占比 **94.14%**。
   目标函数是 `log Z(x) − Σ s_ij`，`Z` 取自**整个非交叉结构空间**；把 94% 的候选对写成 `-inf`（驱动里替换为 `NEG_BIG=-1e4`）
   之后，`log Z` 描述的已不是模型能产出的分布。
3. **L0 召回率远低于门限**：未训练 L0 螺旋召回 = **0.1548**（min 0.0），而 P6 要求 **≥ 0.98**。
   `top_k=2` 的硬选择丢掉了约 85% 的真值螺旋，且**下游不可恢复**。

**判定**：C2 **不是「尚未验证」，而是「尚未成为可用架构」**。它需要一个分层目标函数
（L0 块对 / L1 螺旋 / L2 局部各自成项、各自归一），以及可微的稀疏选择机制。
**在修复并过 P6 之前，论文只能写「提出的架构」，不能写「已实现的贡献」。**

**已测得的正面结果（可如实报告）**：级联头的**显存**确实显著更低（L=498，fp32，fwd+bwd）——
flat 头在 B≥4 时 OOM，级联在 B=4 时 1471 MiB、B=8（bf16）2097 MiB。
即「层级参数化的复杂度优势」在**显存**上已被证实，缺的是精度与可训练性证据。

### 发现 B（严重）：编码器是随机初始化，且当前训练量 ≈ 0.56 epoch

- `encoder.build_encoder("35M")` 在 `backbone="custom"` 下直接 `RNAEncoder(n_layer=6, d_model=512, n_head=8, d_ff=2048)`，
  **无任何 checkpoint 加载路径**（grep 确认无 `load_state_dict` / `pretrained`）。
- 集群上**没有**任何可用的 RNA 基础模型权重：`/mnt/cunyuliu/rna-jepa/weights/` 只有 mRNABERT（mRNA 线遗留）；
  `rna_ss_data/rinalmo/` 只有 CSV 基准数据，**无权重**。
- 运行 1/2/3 为 3000 steps × batch 2 = 6,000 条 / 10,682 条 ≈ **0.56 epoch** → 必然欠训练。

**判定**：运行 1/2/3 只能验证通路，其 F1/ECE **不得**与 UFold / SPOT-RNA / MXfold2 横向比较，**不得**进论文结果表。

### 发现 C（中等）：目标函数按序列**求和**、未按长度归一

- `harness.negative_log_likelihood` 返回 `log Z(x) − Σ_(i,j)∈gt s_ij`，**不除以配对数或长度**。
- 语料长度实测：n=10,682，min=33，**p50=104**，p90=225，p99=414，max=498（≤128 占 63.9%，≤256 占 91.3%）。
  配合按长度分桶的 `iter_batches` → 批间损失天然相差数倍（日志中相邻步 99.09 / 59.67 / 72.32）。
- `grad_clip=1.0` 而**裁剪前**范数实测 40–673 → **每步都被裁死**，实际执行「定步长归一化梯度下降」，有效步长随批次长度波动。
- **不构成梯度错误**：逐对梯度 `p̂_ij − y_ij` 有界且正确。影响是（a）损失值不可跨步/跨臂比较，（b）有效步长波动。

### 修复：决策头显存（已实现并验证）

问题：`PairRepresentation.cross` 物化 `(B,L,L,3d)`、`PairTypeHead.cross` 物化 `(B,L,L,2d)`。
L=498、B=2 时约 3.0 GB + 2.0 GB，**在 2.2 GiB 的 MIG 切片上直接 OOM**（语料中 2.2% 的序列长于 384）。

两次迭代（第二次才真正解决问题）：

| 步骤 | 做法 | 实测结果 |
|---|---|---|
| 1 | 沿 `j` 分块（`chunk_size=64`） | 只压低了**单次**最大分配；autograd 仍保留**每个** chunk 的激活 → 峰值几乎不变（L=498/B=2 bf16 仍 4479 MiB） |
| 2 | 每个 chunk 加 `torch.utils.checkpoint`（`grad_checkpoint=True`） | **真正降低峰值 2–3×** |

修复后实测（fp32 / bf16，MIG 1g.5gb）：

| L | B | 修复前 | 修复后（fp32） |
|---|---|---|---|
| 256 | 8 | OOM | 2406 MiB |
| 384 | 4 | OOM | 1834 MiB |
| 498 | 4 | OOM | **2366 MiB** |

**正确性验证**（`tests/test_head_chunking.py`，11 项，全部通过）：
- 前向输出与整矩阵路径 **逐位相等**（scores / pair_types / mask 全部 `torch.equal`）
- 梯度在 **float64 下精确到 1e-15**（决定性证据：若有漏块/错索引会在这里暴露为 O(1) 差异）
- fp32 下梯度差异 ≈ 6e-8（相对 1e-7，即 float32 机器精度）——因为分块改变了矩阵乘的**累加顺序**，属舍入而非逻辑差异
- 构造性对称 `s_ij == s_ji` 在分块路径下**逐位成立**

> **记录一次自己的失误**：第一版梯度测试报出 **O(1)** 的差异（`pair_repr.proj.weight` 差 23.0 / 量级 27.4），
> 看起来像真 bug。实际原因是**测试脚本本身错了**——`torch.manual_seed` 写在构造函数**外面**，
> 而 `FlatDecisionHead.__init__` 会消耗全局 RNG（3 个 `nn.Linear` 初始化），
> 于是两次调用拿到**不同的参数**（`param_fingerprint` 112.3 vs 116.5）。
> 修正为每次调用内部播种 + `load_state_dict` 后才得到 6e-8。
> **教训**：比较两条代码路径前，必须先断言「起点完全相同」，否则测的是参数差异而不是路径差异。
> 该断言已固化为 `test_whole_and_chunked_paths_have_identical_parameters`。

### 修复：一条失效的测试（不是代码缺陷）

`test_teacher_throughput_refuses_real_tools_instead_of_using_the_mock` 断言 ViennaRNA **不可用**——
该前提在 ViennaRNA 2.7.2 装到 `/mnt/cunyuliu/pylibs` 之后失效（现在 `import RNA` 成功，解析器正确地返回真教师）。
改为断言真正的契约「**真工具永不静默退化为 mock**」，并**双向验证**：
可导入时返回真教师并报告锁定版本；monkeypatch 掉导入时抛 `TeacherUnavailableError` 且 CLI 退出码 3。
**全量测试：238 passed, 0 failed。**

---


## 运行 7：**C1-b 参考测量 —— ViennaRNA 精确配分函数概率**（TS0，1,288 条）

> 这条**不依赖任何训练**，因此现在就能测，而且它本身就是论文里的一个数。
> 工具：`eval/ss/reference_calibration.py`；指标与 `evaluate_decision.py` **共用同一份实现**
> （`ss.metrics.pooled_pair_calibration`，本次重构为单一实现，避免两处口径漂移）。

| 概率来源 | ECE | Brier | NLL | 平均预测概率 | 实际配对频率 | 边际校准误差 |
|---|---|---|---|---|---|---|
| **ViennaRNA 2.7.2 精确 BPP** | **0.0048** | 0.0051 | 0.0259 | 0.0087 | 0.0058 | 0.0029 |
| 我们的 CRF 精确边际（未训练头，前 200 条） | 0.0057 | 0.0071 | 0.0356 | 0.0090 | 0.0068 | 0.0022 |
| **我们的 System-1 头（20 步未训练）** | **0.5547** | 0.3431 | 1.1444 | 0.5605 | 0.0058 | 0.5547 |

**结论（三条，都是可写进论文的）**：

1. **精确配分函数的概率是良好校准的**（ECE 0.0048）。这**否定**了我原先的一个猜想
   （"连热力学精确模型都校准不好"）——**实测不支持，如实记录**。物理模型与 bpRNA 标注之间
   并不存在系统性失配；ViennaRNA 只是**轻微高估**配对概率（0.0087 vs 0.0058，边际误差 0.0029）。
2. 因此 **C1-c 的目标（≤0.02）是可达到的**：两个精确参照都在 0.005 量级，说明"免 DP 的校准"
   并非要求模型超越物理，而是要求它**逼近**精确边际的校准水平。**这加强了 C1 的立论**：
   目标已被证明存在且可达。
3. **"输出一个概率矩阵"与"输出一个校准的概率矩阵"确实是两件事**：同一个测试集上，
   未训练头 0.5547 vs 精确边际 0.0048，相差 **两个数量级**。这正是 C1 要闭合的缺口。

**待补**：C1-a 需要 SPOT-RNA / UFold 的权重或预测（尚未取得）；LinearPartition 未安装（C1-b 的另一参照）。

---

## 运行 8：RiNALMo-giga 冻结嵌入提取（进行中）

- 目的：补上 spec §0.9.5 发现 B（编码器随机初始化）。详见 **`records/RINALMO_BACKBONE.md`**。
- 权重：`multimolecule/rinalmo-giga`，650.9 M，33 层，hidden 1280，经 `hf-mirror.com` 下载（2.60 GB）。
- **加载陷阱（已实测并写进 preflight）**：`AutoModel.from_pretrained` **静默加载 0 个权重**
  （只 warning、不报错，输出形状正确）→ 会产出 10,682 条**随机**嵌入且训练照跑。
  已改为显式 `load_state_dict`，并**精确断言**缺失张量恰好只有未被使用的 `pooler.dense.*`
  （实测 498/498 编码器张量来自官方文件）。
- 提取：`tools/extract_rinalmo_embeddings.py --split all`，输出 `(ΣL, 1280)` fp16 拼接数组 + offsets，
  按**序列名**索引；`archiveii` / `pdb669` 各有 16 条超过 1024，**显式排除并记录**，不截断。
- 实测吞吐 **≈48 seq/s**，全部 10 个 split 约 37,000 条约 13 分钟。
- **许可：AGPL-3.0，论文使用前必须确认合规性。**

---


## 运行 9：RiNALMo 冻结嵌入 → 纯决策头训练（进行中）

- **嵌入提取完成**：10 个 split 全部落盘（`/mnt/cunyuliu/rna-jepa/embeddings/rinalmo-giga/`，
  `<split>.shard0of1.npz` + `manifest.json`，10 条 shard 记录）。
  实测吞吐 48–57 seq/s；`archiveii` 与 `pdb669` 各 16 条超过 1024，**显式排除并记录**。
- **训练臂已启动**：`rinalmo_full_b2_s0`，**head-only**（650 M 编码器完全不进训练进程），
  batch 2、`--head-chunk-size 16`、40,000 步、四项损失全开。
  首个日志：`step 50/40000 loss=92.1445 terms={'nll': 90.302, 'distill': 1.0028, 'rlcd': -0.1845, 'cal': 1.0242}`。
  → 冻结嵌入 → 决策头的通路**端到端打通**。
- **为什么用冻结嵌入**：650 M + AdamW 状态放不进任何空闲切片（MIG 1g.5gb，2.19–4.63 GiB）；
  冻结后编码器离开训练进程，head-only 只需极少量显存，**连最小的 2.19 GiB 切片也能跑**。
  代价是编码器不随任务适配，如实记录。
- **接入方式**：`--embedding-dir` + `--embedding-d-model`，新增 `HeadOnlyModel`；
  批内**不允许混用**冻结嵌入与自编码器激活（`objective_terms` 硬性拒绝），
  缺失序列**报错而非回退**（否则一半来自 650 M 冻结编码器、一半来自 35 M 可训练编码器，损失毫无意义）。

### 本次接线中修掉的 3 个真 bug（都由新测试暴露，非事后检查）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `'list' object has no attribute 'shape'` | `objective_terms` 把 `collate` 产出的**列表**直接喂给 head，未拼接为 `(B,L,d)` | 新增 `_stack_embeddings`，按 batch 最大长度零填充（安全：`objective_terms` 之后按 `[:length,:length]` 切片） |
| 2 | CPU 路径仍报 `got ndarray` | `_batch_to_device` 在非 CUDA 时**原样返回** batch（冻结损失模块接受 numpy），嵌入因此未转张量 | `_stack_embeddings` 直接接受 numpy 并转张量，避免再加一条设备搬运路径 |
| 3 | `no sequences matching split='bprna_tr0'` | 提取器只在**全部** split 结束后才写 `manifest.json`，运行中无 manifest → 按文件名推断 split | `from_dir` 在无 manifest 时从 `<split>.shardKofN.npz` 推断 split |

> 注：bug 1 与 2 说明——**单元测试全绿 ≠ 通路可用**。store 与 `build_decision_model` 的单测都过了，
> 而真正的训练路径是坏的。故新增 `test_head_only_run_trains_end_to_end_on_cpu` 直接跑完整驱动。

### 测试

`tests/test_embedding_head.py` 8 项通过；**全量 246 passed, 0 failed**（含既有 238 项）。

### 另修的工程问题

- `np.savez_compressed` 在慢挂载 + CPU 争抢下**极慢**（tr0 的 3.6 GB 数组压缩超过 20 分钟）→ 改为
  `np.savez`（不压缩）。同时它曾导致 `bprna_ts0` 在写入中途被杀而产生**坏 zip**，
  该文件已删除重做；已用脚本校验现存 shard 的 `offsets[-1] == h.shape[0]` 与逐条长度一致性（tr0: 0 处不符）。

---


## 运行 10：解码与速度审计（第三轮自检）—— **两条速度门作废**

详见 **`records/DECODE_AND_SPEED_AUDIT.md`** 与 `spec/spec.md` §0.9.6。要点：

1. **带状化几乎不加速**（400–600 桶 1662.5 → 1603.2 ms，仅 3.6%）。本实现的「以 j 的配对者为轴」
   递推下，带状化只把每区间候选限到 ≤ band → 总量 `O(L²·band/2)`，精确是 `O(L³/6)`，
   L=500/band=128 时运算量只差 1.3×；而真正的成本是 Python 对 `O(L²)` 个区间的循环开销。
2. **`_max_dp` 内层对 `i` 向量化**：Python 迭代 `O(L²)→O(L)`，实测 **3.0–4.1×**
   （L=100: 64.3→16.2 ms；L=300: 1009.6→245.3；L=500: 1952.1→1055.5），
   `N`/`dec`/`kp` **三张表逐位等价**（`tests/test_max_dp_fast.py` 25 项）。
3. **§8.3 S1「≥50×」与 S3「L=512 ≤ 10 ms」作废**。决策头省掉的是**配分函数**而非 DP；
   L≈500 时 System-1 ≈ 1.2 s 解码 + 0.37 s 前向，而 ViennaRNA BPP 是 C 代码、几十 ms ——
   **墙钟上我们更慢，差距来自实现语言**。论文只允许写「**不必求配分函数即可给出校准概率**」。
4. **修正我自己在发现 C 里的错误推理**：`clip_grad_norm_` 是全局等比缩放，而 AdamW 的 `m/√v`
   与缩放因子无关 → **等比裁剪对 AdamW 近似是空操作**，不是「每步被裁死」。
   **T-A4 因此降级为可选项**；仍成立的是损失按配对数求和、不可跨臂比较。

测试：**271 passed, 0 failed**（246 → 271，新增 25 项逐位等价）。

---


## 运行 11：物理基线实测（`eval/ss/run_baselines.py`，新增）

> 完整表与解读见 **`records/BASELINE_RESULTS.md`**。指标与 `evaluate_decision.py` 共用同一实现，可直接比较。

| split | 基线 | micro F1 | macro F1 | INF |
|---|---|---|---|---|
| bprna_ts0 | `vienna_mfe` | 0.5055 | 0.5086 | 0.5222 |
| bprna_ts0 | **`vienna_centroid`** | **0.5393** | 0.5288 | 0.5405 |
| bprna_ts0 | `vienna_mea` | 0.5241 | 0.5218 | 0.5346 |
| bprna_ts0 | `nussinov_turner`（我们 MLP_T=0） | 0.2124 | 0.2303 | 0.2362 |
| archiveii | `vienna_mfe` | 0.5764 | 0.6219 | 0.6264 |
| bprna_new（OOD） | `vienna_mfe` | 0.6379 | 0.6581 | 0.6638 |
| bprna_new（OOD） | **`vienna_centroid`** | **0.6770** | 0.6821 | 0.6871 |

**三条直接改变论文的结论**：

1. **centroid > MEA > MFE 在三个 split 上一致**，而 centroid/MEA 都建立在配分函数之上。
   → **C1 的正确标尺是「逼近 centroid 的 F1（TS0 0.5393）而一次配分函数都不算」**，比"快 50×"有力且可实测。
2. **`nussinov_turner`（0.2124）≈ 未训练头（0.2167）**，差 0.004。
   这**验证了"MLP_T=0 时起点精确等于 Nussinov+Turner"这一实现声明**，并证明 0.56 epoch 的头
   **确实还没学到东西**，而不是架构不行。
3. **ViennaRNA 在 OOD 上更好**（bpRNA-new 0.638 > TS0 0.506）。领域常引的"学习方法在 bpRNA-new 上全崩"
   **是学习方法的问题，不是数据问题**；这为我们的物理先验设计提供了直接实证支持。
   （E2Efold F≈0.0361 这个数字**引用前必须回原文核对**，本次未独立复现。）

**仍未取得**：UFold / SPOT-RNA（C1-a 头号对照，需下载权重）、BPfold（权重在集群但包未装）、
MXfold2 / CONTRAfold / LinearPartition、E2Efold。

---


## 运行 12：CPU 争抢事故与处置（吞吐减半 → 恢复）

### 现象

09:27 实测训练速率从 **18.6 步/分降到 9.5 步/分**（减半），`load average` 从 114 升到 **166**。

### 根因（实测，非推测）

`ps --sort=-pcpu` 显示一个**不属于本线**的进程占用 **1465% CPU**：

```
eval/factor_probe.py --ckpt /mnt/cunyuliu/rna-jepa/runs/...
```

`factor_probe` 是**早期 JEPA 线的产物**——而 spec §0.6 已因**否定性证据**明确移除 JEPA 线
（区域均值目标 60 步内饱和、因子探针对照是空操作）。进一步查 cron：

```
*/10 * * * * cd /mnt/cunyuliu/rna-jepa/code && ... scripts/monitor.py >> ... # RNAJEPA_MONITOR
```

**mRNA/JEPA 线的监控 cron 每 10 分钟从 `queue/pending/` 自动派发任务**
（日志实证：`dispatched 003_mrnabert_official_cds_ecoli_s3 (pid 1568909, ...)`），
其中包含已废弃的 JEPA 探针，持续抢走十几个核。

### 处置（最小、可逆、已记录）

1. **杀掉正在运行的 `factor_probe.py`**（废弃线的产物，无损失）。
2. **给该 cron 加 `--no-dispatch`**：**保留监控**（便宜、无害），**停止自动派发**（昂贵）。
   `monitor.py` 原生支持该开关（第 1331 行），**没有改动任何代码，也没有停用监控**。
   → 只停止**新**任务的派发；已在跑的任务不受影响。

### 结果

速率 **9.5 → 14.2 步/分**（1.5×），load 166 → 155。仍未回到 18.6，因为其他用户的
GROMACS 任务（`gmx mdrun`，合计约 3400% CPU）仍在占用宿主机。

> **需要用户确认**：这只是**临时**措施。若 mRNA 线需要继续派发任务，请告知，我立即恢复该 cron。
> 反之，如果 mRNA 线已收尾，建议直接停用该 cron，把整机 CPU 让给本项目的一周冲刺。

---

## 待办（按 Gate）

| Gate | 内容 | 状态 |
|---|---|---|
| 运行 5 完成 | 六臂各 40,000 steps 跑完并落 checkpoint（`--save-every 4000`） | **进行中** |
| **C2 分层目标** | 为级联设计 L0/L1/L2 分层目标 + 可微稀疏选择；过 P6（L0 召回 ≥ 0.98） | **未开始（阻塞 C2 的贡献主张）** |
| **预训练骨干** | 接入公开 RNA 基础模型权重（RiNALMo/RNA-FM，集群上没有，需下载）；不可得则用无标签语料自监督预训练 | **未开始（阻塞「优秀结果」）** |
| 损失归一 | 按 GT 配对数归一 + 重新标定 `grad_clip`；记录裁剪前范数分布 | 未开始 |
| Gate I | 在 **TS0(1,305) / ArchiveII(3,966) / bpRNA-new(5,401)** 上评测：F1(micro+macro) / INF / ECE / Brier / NLL / 非法结构率 / 延迟分桶 | 待做 |
| C1-a/b/c | 免 DP 校准判据（对 ViennaRNA 精确边际、SPOT-RNA/UFold 概率同口径 ECE） | 待做 |
| Gate J | §8 硬门 G1–G5 / 性能门 P1–P7 / 速度门 S1–S9 逐项核验 | 待做 |
| 基线 | BPfold（权重已在集群）实际复现 + ViennaRNA + Nussinov/Turner | 待做 |
| 带状/快速解码 | §5.0.2 的带状解码路径，**分别报告**精确解码 F1 与快速解码 F1/延迟 | 待做（在此之前不得引用任何加速比） |
