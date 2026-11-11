# RiNALMo 全量 D1 数据管线 Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** 在当前 RiNALMo 单模型上建立804条复合物训练链全保留的 D1 数据、教师、嵌入、长序列训练与独立验收管线，旧 xens2 仅作历史诊断。

**Architecture:** 以当前独立审计的804 train/202 holdout为唯一数据版本，D0复用已验收的L-length三种子，D1从同一RiNALMo官方初始化。数据先通过身份和canonical-nested守恒门，教师与嵌入按序列键恢复；PairMLP+R6采用列分块、activation checkpoint和禁用type-head的部署路径，禁止用长度过滤或截断冒充全量支持。

**Tech Stack:** Python、PyTorch CUDA、RiNALMo-giga、ViennaRNA 2.7.2、NumPy/NumPy NPZ、JSONL/JSON manifest、现有`train_plan_a.py`与独立QA。

---

### Batch 1: 数据版本与完整训练清单

**Files:**
- Create: `tools/prepare_rinalmo_full_d1_round53.py`
- Create: `tools/test_prepare_rinalmo_full_d1_round53.py`

**Risk:** High — 旧文件为805/201，当前有效版本为804/202；混用会破坏科学身份。

**Implementation:** 读取`data/rna_complex_round49/decontamination/{verification,decontamination,complex_train,complex_holdout}.json`，核验当前SHA、804/202数量、序列唯一性、canonical-nested端点/无共享端点/无交叉。生成TR1c 42564+804=43368的D1 JSONL和holdout清单，1-based源配对转为0-based训练配对并保留原始索引。清单记录长度分布、>1024/>3819样本、全部源SHA和旧版本隔离说明，不按长度或效果删除任何链。

**Minimum verification:** 合成边界与失败注入三轮；真实804/202独立复算，序列集合、标签、行数和SHA全部一致。

**Independent review:** Yes — 数据身份和标签守恒是后续训练的前置科学门。

### Batch 2: 全量教师与RiNALMo嵌入

**Files:**
- Create: `tools/build_rinalmo_full_d1_teacher_round53.py`
- Create: `tools/build_rinalmo_full_d1_embeddings_round53.py`
- Create: `tools/test_rinalmo_full_d1_assets_round53.py`

**Risk:** High — 现有教师仅189条、嵌入仅150条，且旧manifest不可混合。

**Implementation:** 以序列为唯一键核验现有教师/嵌入能否复用；为缺失序列生成ViennaRNA 2.7.2教师及RiNALMo全残基表征。分片原子写入并绑定run、数据、代码、长度和tensor shape SHA；拒绝重复序列、空教师、错误长度、NaN、静默CPU降级和未核验覆盖。嵌入生成只要求backbone全长前向，不将head/DP耗时误写为嵌入失败。

**Minimum verification:** 数量门与失败注入三轮；真实CUDA峰值、804条教师/嵌入序列集合、形状和独立SHA复核，所有旧分片身份单独列出。

**Independent review:** Yes — 必须证明804条均有合法训练输入，而非只生成manifest。

### Batch 3: 长序列训练/推理内存工程门

**Files:**
- Modify: `tools/train_plan_a.py`（仅内存等价路径）
- Modify: `tools/eval_rinalmo_full_round52.py`（显式长序列路径与失败记录）
- Create: `tools/probe_rinalmo_full_d1_long_sequence_round53.py`
- Create: `tools/test_rinalmo_full_d1_long_sequence_round53.py`

**Risk:** High — train最长7224 nt、holdout最长19000 nt；旧ResNet2D在7224 nt已真实OOM，PairMLP的z拼接和CRF也须实测。

**Implementation:** PairMLP按列计算/反向，避免完整latent grid拼接；保留R6、Turner、CRF/distill/RLCD/cal和canonical-nested DP，禁用未入损type-head。记录每个张量/峰值，先短序列等价，再7224 nt训练前向+反向及19000 nt推理。禁止改变科学batch、损失或标签；若精确CRF无法满足24GB，明确阻止正式训练并保留PARTIAL，不 secretly crop。

**Minimum verification:** 短序列前向/梯度独立复算；真实CUDA长序列峰值与输出守恒；OOM日志、CPU回退拒绝和失败注入三轮。

**Independent review:** Yes — 只有7224 nt训练门通过才能放行正式D1；19000 nt holdout若未通过必须单列，不能写成完整holdout覆盖。

### Batch 4: 正式D0/D1与独立验收

**Files:**
- Create: `tools/run_rinalmo_full_d1_round53.sh`
- Create: `tools/qa_rinalmo_full_d1_round53.py`

**Risk:** High — 需保持单变量数据扩充与自然收敛，不得从D0续训冒充同初始化。

**Implementation:** D0直接复用L-length三种子。D1在Batch 1–3全部通过后，以RiNALMo官方初始化、NLL-length、batch=2、head/backbone LR=1e-4/1e-5、steps=0、500步VL0、patience=6、best VL0训练42/2026/3407。只新增804条审计链，保持其他配置。按显存实测串行或安全并行，不强行填满显存。

**Minimum verification:** 三种子自然早停、best/resume逐张量一致、完整后段曲线；主八+Rfam与complex holdout逐seed三次预测，默认DP/greedy/Turner=0分表，B改善、A/C非劣、holdout同向与完整QA双跑。未通过前不得写最终结论或PPT。

**Independent review:** Yes — 最终科学归因、长序列覆盖和负结果复核。

## Execution handoff

按Batch 1→2→3→4执行，同一实现上下文完成低风险文件；仅在数据守恒、长序列CUDA门和最终科学裁决处进行独立复核。每批验收后提交明确相关文件并push，不提交大权重、教师、嵌入或秘密文件。
