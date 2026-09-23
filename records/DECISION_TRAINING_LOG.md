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

## 待办（按 Gate）

| Gate | 内容 | 状态 |
|---|---|---|
| 训练完成 | 三臂各 3000 steps 跑完并落 checkpoint | 进行中 |
| Gate I | 在 **TS0(1,305) / ArchiveII(3,966) / bpRNA-new(5,401)** 上评测：F1(micro+macro) / INF / ECE / Brier / NLL / 非法结构率 / 延迟分桶 | 待做 |
| C1-a/b/c | 免 DP 校准判据（对 ViennaRNA 精确边际、SPOT-RNA/UFold 概率同口径 ECE） | 待做 |
| Gate J | §8 硬门 G1–G5 / 性能门 P1–P7 / 速度门 S1–S9 逐项核验 | 待做 |
| 基线 | BPfold（权重已在集群）实际复现 + ViennaRNA + Nussinov/Turner | 待做 |
