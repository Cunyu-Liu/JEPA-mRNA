# RiNALMo-giga 骨干：加载陷阱、验证与冻结嵌入（2026-09-24）

> 目的：为决策头提供**预训练**序列表示，以补上 spec §0.9.5 发现 B 的缺口
> （`build_encoder("35M")` 是随机初始化）。本文件记录**实测**的坑与验证方式，供后续复用。

---

## 1. 为什么需要它

- spec §0.9.5 发现 B：`encoder.build_encoder("35M")` 直接构造 `RNAEncoder`，**不加载任何权重**；
  集群上原本也没有任何 RNA 基础模型权重（`weights/` 只有 mRNABERT；`rna_ss_data/rinalmo/` 只有 CSV）。
- Jev 自身的「快且可信」建立在**预训练骨干**之上（OpenJev 冻结开源 LLM，只读 option logits）。
  把决策头接到预训练 RNA 语言模型上，是恢复这一性质的正直做法。
- 已通过 `hf-mirror.com` 下载 `multimolecule/rinalmo-giga`：
  33 层 / hidden 1280 / 20 heads / rotary / max_position_embeddings 1024 / **650.9 M 参数** / 2.60 GB safetensors。
  路径：`/mnt/cunyuliu/rna-jepa/weights/rinalmo-giga/`。

## 2. 三个必须记住的坑（全部实测）

| # | 坑 | 现象 | 后果 |
|---|---|---|---|
| 1 | **`AutoModel.from_pretrained` 静默加载 0 个权重** | 只打印一行 warning（"Some weights ... were newly initialized"，列出几乎整个模型），**不报错**；输出形状完全正确 | 会产出 10,682 条**随机**嵌入，训练照跑、指标照出，**极难发现** |
| 2 | **`from_pretrained` 在 transformers 4.57.6 下直接失败** | `ValueError: The state dictionary ... is corrupted`（`_get_key_renaming_mapping`） | 无法用常规路径加载 |
| 3 | **`AutoTokenizer` 与 multimolecule 0.0.8 不兼容** | `AttributeError: 'list' object has no attribute 'keys'`（`SPECIAL_TOKENS_ATTRIBUTES`） | 分词器不可用 |

**根因（坑 1 与 2）**：checkpoint 的键带 `model.` 前缀（保存它的类把编码器属性命名为 `model`），
而 multimolecule 0.0.8 把编码器命名为 `rinalmo`（`RiNALMoForMaskedLM`）或**无前缀**（`RiNALMoModel`）。

**处置**：
- 不用 `from_pretrained`，改为**显式 `load_state_dict`**（见 `tools/rinalmo_preflight.py`）。
- 不用分词器：词表只有 28 个 token，手写 id（`<cls>`=1、`A/C/G/U/N`=6–10、`<eos>`=2）。

## 3. 键映射与验证（决定性证据）

对 `RiNALMoModel`（编码器）：
- 剥离 `model.` 前缀 → **498 个张量全部匹配**；
- 模型额外想要 `pooler.dense.{weight,bias}` —— **checkpoint 里没有**，因为 pooler 只用于 `[cls]` 池化输出，
  而我们要的是**逐残基**表示，**从不使用 pooler**。

因此 preflight 的判据是**精确断言**：

```
missing_keys 必须恰好等于 {pooler.dense.weight, pooler.dense.bias}
unexpected_keys 必须为空
```

任何其它缺失都会**硬失败**——因为「部分加载」的编码器依然能训练、能评测、能出图，只是结果是错的。

实测（`tools/rinalmo_preflight.py --device cuda`）：

```json
{"n_tensors_in_checkpoint": 504, "n_encoder_tensors_loaded": 498,
 "loaded_fraction_of_encoder": 0.996, "d_model": 1280, "n_layers": 33,
 "n_params": 650878731}
```

### 未能完成的独立交叉验证（如实记录）

原计划用模型卡上的 fill-mask 例子（`UAGCUUAUCAG<mask>CUGAUGUUGA` → `G 0.219446`）做端到端校验。
**该验证未通过，且不能作为证据**：`RiNALMoForMaskedLM` 需要 7 个 `lm_head` 键，而 checkpoint 只有 6 个
（缺 `lm_head.decoder.bias`），映射后 `lm_head.bias` 无对应值、只能随机初始化，输出的 top-5 全是 `<pad>`。
即**是版本间 head 结构差异导致该检查不可用**，不是权重有问题。
故本项目的加载证据只依赖上表的**张量级精确断言**（498/498 来自官方文件）+ 文件来源（官方仓库 id）。

### 批量不变性（验证 mask 正确性）

- 同批内换位置：`maxdiff = 0.0`（逐位相同）。
- 单独 vs 与更长序列同批（真实 padding）：fp32 下 `maxdiff = 8e-6`；
  **开/关 `attention_mask` 结果相同**（`8e-6` vs `8e-6`）→ RiNALMo 的注意力实现自行处理 padding，
  显式 mask 是冗余但无害的（仍保留，以消除 transformers 的告警并避免依赖该实现细节）。
- bf16 下同批不同组成有 `4.7e-2` 差异，**是 bf16 的批形状舍入噪声，不是 mask 缺陷**（fp32 下为 8e-6）。
  由于嵌入按**长度分桶的确定性顺序**生成，可复现性不受影响。

## 4. 提取产物

`tools/extract_rinalmo_embeddings.py`：

- 输出 `{split}.shard{k}of{n}.npz`，含 `h`（拼接后的 fp16 `(ΣL, 1280)`）、`offsets`、`lengths`、`seqs`；
- **按序列名查找**，不按索引 —— 因此被排除的序列不会造成错位；
- `max_position_embeddings = 1024`：`archiveii` 有 16 条、`pdb669` 有 16 条超过该长度，
  **显式排除并在 manifest 中记录**（`n_excluded_over_max_len`、`excluded_lengths`），**绝不静默截断**；
- 每个 shard 的 manifest 条目内嵌 `encoder_load` 报告（来源、加载比例、d_model、许可）。

实测吞吐：**≈48 seq/s**（MIG 3g.20gb 切片，bf16，batch 8），全部分片约 37,000 条约 13 分钟。

> **许可**：`multimolecule/rinalmo-giga` 标注 **AGPL-3.0**。论文使用前必须确认合规性，或改用许可更宽松的骨干。
> 该提示同时写入 manifest。

## 5. 环境

| 项 | 值 |
|---|---|
| 依赖安装位置 | **`/var/tmp/rnalmo_pkgs`**（`pip install --target`，**本地盘**；`/mnt` 上安装慢到不可用——实测 45 分钟仍未完成，本地盘约 3 分钟） |
| 关键版本 | multimolecule 0.0.8 · transformers 4.57.6 · torch 2.8.0 · tokenizers 0.22.2 · protobuf 6.33.6 |
| 运行方式 | `PYTHONPATH=/var/tmp/rnalmo_pkgs:tools HF_HUB_OFFLINE=1`（**不要**与训练环境 `lucaone` 混用：它需要 torch 2.5.1 与 transformers 4.26） |
| 隔离理由 | 训练环境的 transformers 4.26 不能升级（会破坏正在跑的训练与既有测试） |
