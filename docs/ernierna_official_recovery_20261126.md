# ERNIE-RNA 官方 bpRNA-new 资产恢复合同

## 范围与结论

本恢复固定 FoldNAR 基线提交 `906f554cd59454b09101cea447583836215be3e2` 和 ERNIE-RNA 官方提交 `43bc06de1088ed03ffd7de918ad4b2c2a3346a43`。只进行源码/权重下载、SHA、AST/依赖静态检查和 CPU 单元测试；没有执行模型构建、checkpoint 加载、张量计算、GPU 推理、训练、校准或外部 Test 调参。

**完整 bpRNA-new 预测尚未生成，不能登记为 full prediction 或 SOTA 数值。** 单序列 demo、论文值和 summary 均不能替代 5,388 条逐序列预测。精确资产状态和阻塞见 `manifests/foldnar_v1/ernierna_official_asset_provenance_20261126.json` 及远端 `readiness_audit.json`。

## 官方来源与许可

- 源码：https://github.com/Bruce-ywj/ERNIE-RNA
- 固定提交：https://github.com/Bruce-ywj/ERNIE-RNA/commit/43bc06de1088ed03ffd7de918ad4b2c2a3346a43
- 快照：https://codeload.github.com/Bruce-ywj/ERNIE-RNA/tar.gz/43bc06de1088ed03ffd7de918ad4b2c2a3346a43
- 代码许可：官方 `LICENSE` 的 MIT。官方 README 只明确声明 source code license，没有为 checkpoint 单独声明许可；不能把代码 MIT 自动扩张成权重许可。
- 预训练权重官方链接：https://drive.google.com/file/d/1CmNxJxgjDhRoBdlDODFFjDNNzJMHVzmO
- bpRNA-new 协议头官方链接：https://drive.google.com/file/d/1ii6XDkAsfVC--ZN2o3uiOcF5j9TTcvm7
- 链接出处：固定提交的 `checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_checkpoint.txt` 和 `checkpoint/ERNIE-RNA_ss_prediction_checkpoint/ERNIE-RNA_ss_prediction_checkpoint.txt`。
- 官方 release 没有权重附件；未使用第三方镜像。

所有官方源码、快照和权重目标均位于 `/teams/HoulabData_1788502323/liucunyu/foldnar/ernierna_official_recovery_20261126/`。Git 中只保存合同、测试和小型 provenance，不保存 checkpoint、预测大文件或源码快照。

### 2026-11-26 macOS 官方下载复核

本机未配置 HTTP/HTTPS/SOCKS 代理，当前 subagent 无内置浏览器控制权限。对两个官方文件使用 curl 和本机已有 `gdown` 直接访问，未登录、未使用第三方镜像：`drive.usercontent.google.com` 和 `drive.google.com` 均在约 15 秒连接阶段返回 curl 28；`www.googleapis.com` 同样连接超时。两个 `gdown` 请求分别运行 144.628/144.631 秒后仍无 HTTP 响应且本地零字节，因而中止。没有文件可供 magic、大小、scp 或端到端 SHA 检查，state-dict 静态检查保持 `NOT_RUN_NO_CHECKPOINT_BYTES`。逐端点原文错误记录在 provenance 的 `macos_official_download_retry_20261126`，full-ready 状态不变。

## 模型身份、head 和训练暴露

目标是官方 `--dataset_name bpRNA-new` 选择的 `ERNIE-RNA_attn-map_frozen_ss_prediction_bpRNA-1m_checkpoint.pt`，不是 bpRNA-new 上训练的模型。官方 README 明确：下游结构训练集为 bpRNA-1m，bpRNA-new 是该协议评估对象，不作为训练集。预训练语料的完整去污染和 bpRNA-new 暴露没有由该提交证明，因此不声称 pretraining-clean 或 homology-clean。

官方 `src/utils.py::ChooseModel` 使用最后一层第 5 号 attention map（Python 索引 5），经过 `Conv2d(1,8)`、`Conv2d(8,63)`，与原 attention 通道拼接为 64 通道，再经过 8 个 residual block 和最终 `Conv2d(64,1)`；输出对称化。`predict_ss_rna.py` 使用 constrained attention-map post-processing（100 次迭代）和 CT 转换（阈值 0.5）。

这是**一个预训练 checkpoint 加一个下游 checkpoint 的单模型**，不是集成。`load_model_ensemble_and_task` 是 fairseq 加载 API，但只传入一个路径并取 `rna_models[0]`，不能据函数名误报 ensemble。每条序列还输出 zero-shot CT；它来自未做结构微调的同一预训练编码器，是独立辅助输出，不能与 fine-tuned 输出平均或称为二模型集成。FoldNAR 目标预测仅为 `*_finetune_prediction.ct`。

## 可复现批合同

默认输入：`/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl`，SHA256 `271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9`，5,388 个唯一原始 `name`。准备阶段只使用 `name/seq`，不把 truth pairs 写入 FASTA 或批合同。

```bash
cd /home/rna_jev_ernierna_recovery
python tools/ernierna_bprna_new_batch.py prepare
python tools/check_ernierna_official_runtime.py \
  --output /teams/HoulabData_1788502323/liucunyu/foldnar/ernierna_official_recovery_20261126/bprna_new_batch/runtime_static_check.json
```

`prepare` 生成确定性的 `bprna_new_000000` 形式输入 ID，保留原始 ID、长度和序列 SHA 映射，并记录一个**未执行的未来命令**。`eligible_to_launch=false` 时禁止启动。

获得全部官方 CT 后，且仅在新的推理授权和 GPU 空闲时，执行：

```bash
python tools/ernierna_bprna_new_batch.py export
python tools/ernierna_bprna_new_batch.py audit
```

`export` 要求每个输入恰有 fine-tuned 和 zero-shot 两个 CT，拒绝缺失、额外 demo、重名、错序、序列/碱基不一致、非法索引和非对称配对。输出为 `finetune_predictions.jsonl` 与 `zeroshot_predictions.jsonl`，pair 是零基、闭区间、排序后的 `[i,j]`，保留 crossing 和短跨度，不做指标调参。只有 5,388/5,388 且 SHA 复核通过，`full_prediction_ready` 才能为 true。

## 验证边界

- `CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_ernierna_bprna_new_batch.py`
- 对所有官方 Python 文件仅做 AST parse；依赖通过 package metadata 查询。
- 现有 `lucaone_cu128` 环境缺 fairseq/hydra 等，torch/Python 版本也不等于官方 `environment.yml`，所以静态环境检查通过不等于官方运行环境 ready。
- 没有生成或登记 ERNIE-RNA bpRNA-new 指标；不得以本合同的存在替代预测文件。
