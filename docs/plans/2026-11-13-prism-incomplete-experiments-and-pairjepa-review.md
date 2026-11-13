# PRISM 未完成实验交接与 Pair-JEPA/SIGReg 复盘

> 日期：2026-11-13。本文覆盖此前的“已关闭”表述：关闭只代表当时阶段门不满足，不能删除待补证据或把工程失败写成科学无效。新指令为：D1、E2a、E2b 等待补；CL2–CL5 各自直接对照统一 B0，不再与 CL1 比较；Pair-JEPA/SIGReg 作为核心创新方向优先复盘。

## 一、统一基线与状态定义

B0 保持单 RiNALMo-giga、单 ResNet2D scorer、Turner、calibration、canonical-nested DP、原四目标、原优化器/解冻协议、单 checkpoint、无集成。除用户明确批准的新目标语义实验外，不在其他获胜臂上默认叠加改动。测试集不参与训练、调参、早停、校准或选臂。

状态必须分开：

- `TRAINED_NEGATIVE_VL0`：训练完成，但 VL0 不支持保留，不等于未做实验。
- `ENGINEERING_INCOMPLETE`：参数、CUDA、磁盘或实现问题导致未闭合，不能作科学否定。
- `TRAINING_INCOMPLETE`：已有部分训练曲线，但缺种子、对照或自然早停。
- `NOT_EVALUATED`：已有训练 artifact，但缺主八+Rfam 或指定验证。
- `NOT_STARTED`：尚未启动，不得写成负结果。

## 二、所有未完成或未得到完整测评的实验

| 实验 | 已有证据与缺口 | 当前状态 | 后续补跑入口 |
|---|---|---|---|
| D1 数据扩充 | 当前 804/202 数据及 804 条教师已生成，manifest 来源 SHA 已修；同一 step1917 两次真实 CUDA OOM，无三种子和结构测评 | `ENGINEERING_INCOMPLETE + NOT_EVALUATED` | 保留 OOM；先核对可用磁盘/显存和完整 ResNet2D CUDA 门，只从统一 B0 初始化补 seed42 工程门。通过后再跑三种子，不重算教师，不把旧 805/201 混入 |
| CL2 结构感知对比 | 三种子自然早停，mean best VL0=15.511690012885689；已有最终 projection 限制说明，但没有主八+Rfam 冻结结果 | `TRAINED + NOT_EVALUATED` | 新指令下直接与 B0=15.58693583577398 比，不再与 CL1 比；补主八+Rfam 及统计复核。seed42/2026 最终 resume projection 缺失不用于部署，不借此掩盖训练限制 |
| CL3 困难/结构化负样本 | 未形成正式三种子及结构测评 | `NOT_STARTED` | 从同一 B0 初始化只改对应负样本模块，先 seed42 工程门，再三种子；直接与 B0 比，不以 CL1/CL2 晋级为前提 |
| CL4 负样本加权 | 未形成正式三种子及结构测评 | `NOT_STARTED` | 同 CL3；不得叠加 CL2/CL3 获胜权重，不与 CL1 比 |
| CL5 扩展负池 | 未形成正式三种子及结构测评 | `NOT_STARTED` | 同 CL3；只在各自独立证据齐全后评审组合，不默认串联 |
| E2a motif 输入 | 参数门失败，motif 和 shuffled 均未训练；失败 QA SHA=`cbf8d5fe565ac689976ab77d5d7d2cfe9d8a0a4f098736ca1beb754adb471af7` | `ENGINEERING_INCOMPLETE + NOT_STARTED` | 修正参数生成/加载的单一根因后仅重做一次参数门；通过才运行 seed42 motif 与 shuffled，再决定三种子。不得称 motif 科学无效 |
| E2b 统计热力学 | TR1c 与 VL0 缓存完成；seed42 best VL0=19.728811457864218@6000，step9000 后段严重退化；seed2026 best=22.359044330264734@3500 后反复中断；seed3407、shuffled 和结构测评缺失 | `TRAINING_INCOMPLETE + NOT_EVALUATED` | 保留 PARTIAL，不用 seed42 单 seed 宣称收益。先解决共享盘空间及进程恢复，再从合法 resume 续 seed2026，之后补 seed3407 和同预算 shuffled；只有 online 同时优于 shuffled 与 B0 才补冻结评测 |
| Pair-JEPA/J1 | block-EMA 出现持续低秩、跨位置和跨序列近单位余弦坍塌；未形成可用固定/随机目标对照 | `TRAINED_NEGATIVE_MECHANISM + INCOMPLETE_CONTROLS` | 不直接重复 J1；先完成目标语义和抗坍塌复盘，见下文。部署仍只允许 online scorer |
| J2-online-target-SIGReg | 已按 online full-grid latent 方向尝试，但未取得同时满足抗坍塌与 VL0 的证据，未进入三种子、fixed/random 或冻结评测 | `MECHANISM_NOT_SUPPORTED + INCOMPLETE_CONTROLS` | 不把一次失败外推为 SIGReg 总体无效；先审查 target 语义、统计作用对象、梯度路径和优化冲突，再决定最小新臂 |
| J3 长程 context | 没有形成经过有效 J1/J2 目标门的结果 | `NOT_STARTED / BLOCKED_BY_TARGET_SEMANTICS` | 不通过改 context 掩盖目标坍塌；待目标表征有效性成立后独立评测 |
| fixed/random target | 未完成，无法区分学习到的结构语义和随机但多样的表征 | `NOT_STARTED` | 与候选目标同源、同预算比较，不能在 J1 失败后省略 |

明确不重跑：R-replay、L-length、O-LR、RIPE、B0、D0、CL1、E2c。E2c 已完成 VL0 负结果，不属于“未测评待补”。LinearDesign 和 raw-sequence 测速仍按此前指令暂停，不借补清单自行启动。

## 三、Pair-JEPA/SIGReg 核心复盘

### 3.1 当前真正的问题

1. **恒定解是优化捷径，不只是数值不稳定。** context predictor 与 EMA target 同构时，所有 pair 输出相同可同时降低 latent MSE 和适应下游参数。EMA 的 stop-gradient 只阻断梯度，并不保证 target 有语义；teacher 也会随 online 参数漂移。
2. **SIGReg 的各向同性不等于结构正确。** 随机或噪声表征也可以接近标准高斯。它最多约束分布形状，不能保证碱基身份、距离、局部结构或长程接触被编码。
3. **统计对象必须和坍塌对象一致。** 全 batch 合并 pair 可能得到漂亮的全局分布，却掩盖每条序列内部恒定。应保留每条序列合法 pair 的方差、有效秩、跨位置和跨序列相关，不能只看总 loss。
4. **直接任务与辅助目标可能优化方向不同。** 只报告总 VL0 或总 loss 会掩盖 direct NLL、latent MSE 和 SIGReg 各自的作用。必须保留分项、梯度范数/必要时的梯度 cosine，以及固定评估窗口。
5. **原对照链不完整。** fixed/random 未补齐时，无法区分“结构语义学习”与“随机但多样的输出”。因此当前既不能宣称 Pair-JEPA 失败，也不能宣称 SIGReg 已证明有效。

### 3.2 三种后续路线

**A. 优先：目标语义门（frozen target vs shuffled target）。** 用冻结的、具有结构约束的 pair target，避免 EMA target 与 student 同步坍塌。候选只需回答：结构 target 是否优于同源打乱 target，并且 direct NLL/VL0 不受损。优点是直接检验语义来源；风险是更接近结构蒸馏，论文中不能包装成完全无监督发现。

**B. no-EMA 共享 online target + SIGReg。** 去除 EMA 漂移，保留 online full-grid target 和 stop-gradient 分支。优点是实现较小、更接近 LeJEPA；风险是恒定解仍存在，SIGReg 可能只让噪声合法化。只有 A 证明 target 语义不是主要瓶颈时才值得先试 B。

**C. 长程 context。** 改进 block/axial/全局 context，检验是否能提供结构归纳偏置。它解决的是信息可见性，不解决恒定解，不能在 A/B 未成立时用来反复抢救。

### 3.3 推荐的最小可证伪顺序

1. 先做 seed=42 的 A：同一 B0 官方初始化与训练协议，唯一改变 target 来源与必要 predictor 接口；同预算 shuffled target 是唯一关键对照。保留 direct NLL、latent MSE、有效秩、每序列跨位置/跨序列相关和 VL0。
2. 只有结构 target 同时优于 shuffled 且不损害 VL0，才进入三种子；再做 fixed/random，确认不是随机表征收益。
3. 若目标语义成立但仍坍塌，再独立比较 no-EMA+SIGReg，不与 A/B/C 混改。
4. 只有前两层成立后才做 J3。任何阶段只改善分布而不改善结构目标，均记为无效，不再追加正则系数网格。

### 3.4 核心创新的表述边界

若 A 有效，应表述为“结构表征引导的 pair-context 学习/蒸馏”，不是完全无监督 JEPA。若 B 有效，才可重点讨论 JEPA/SIGReg 的抗坍塌自监督机制。论文创新必须落在实际被证明的模块，不能用随机高斯分布代替结构语义，也不能用 ResNet2D 的直接监督收益冒充 JEPA 收益。

## 四、CL2–CL5 的新比较合同

- 每臂均从同一 B0 官方初始化出发，只改各自一个模块；不与 CL1 比，也不在 CL1/CL2 上串联。
- CL2 已有三种子，直接补冻结评测；CL3、CL4、CL5 分别做 seed42 工程门后再跑三种子，不因 CL2 较好或较差而跳过。
- seed=42/2026/3407、steps=0、每500步 VL0、patience=6、best VL0 和自然早停不变。
- 每个臂独立与 B0=15.58693583577398 比，保留负结果；通过 VL0 证据门后再补主八+Rfam，不使用测试集选臂。
- 任一组合实验需单独评审，不能由 CL2–CL5 的名次自动推断组合有效。
