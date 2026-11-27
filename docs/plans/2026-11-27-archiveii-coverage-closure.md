# ArchiveII 覆盖闭合：Vienna clean 与 xens2 显存门控恢复

## 输入与边界

- Vienna clean 输入：`official_reference_clean_2544.jsonl.gz`，固定 2544 个官方 ID、原始 sequence 和 SHA。
- xens2 输入：`archiveii_full3966_cpu_recovery_20261127/recovery_plan.json`，固定 clean 外 1422 个 ID。
- xens2 冻结协议为两个 Plan-A checkpoint + R2D，`w_plana=0.7`、`plana_s1_weight=0.5`。
- 原 `xens2_eval` 跳过 `len>1024` 和缺失 embedding 的记录。1422 条中有 16 条 16s 超长记录且无对应 embedding，必须标为 `BLOCKED_PROTOCOL_LENGTH_GT1024` 或 `BLOCKED_PROTOCOL_EMBEDDING_MISSING`，不得改写序列、替换模型或伪造预测。合规上限为 1406 条，因此 xens2 的 full3966 仍可能只能达到 3950 条部分覆盖。

## Vienna CPU

工作根：`/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_vienna_clean_cpu_20261127`。

```bash
cd /home/rna_jev_archive_label_lineage
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
$PY tools/recover_archiveii_vienna_clean_cpu_20261127.py --baseline vienna_mfe
$PY tools/recover_archiveii_vienna_clean_cpu_20261127.py --baseline vienna_centroid
$PY tools/recover_archiveii_vienna_clean_cpu_20261127.py --baseline vienna_mea
```

每个进程仅一个 CPU 线程，`CUDA_VISIBLE_DEVICES=""`。逐 ID checkpoint、coverage 和预测 SHA 与前一轮 1422 恢复一致。合并时使用 clean 2544 的新 CPU 预测与此前完成的官方 1422 预测，不能把历史 summary-only 数值展开成记录级预测。

## xens2 GPU 队列

生产队列：

```bash
cd /home/rna_jev_archive_label_lineage
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
$PY tools/queue_archiveii_xens2_gpu_recovery_20261127.py --poll-seconds 60
```

默认队列工作根：`/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_xens2_gpu_queue_20261127`。

门控顺序：

1. 只读记录 `nvidia-smi` 设备与 free/used/total。
2. 探针前保守门槛：`free >= checkpoint_static_MiB + 2048 MiB`。不满足时状态为 `WAITING_PREPROBE`，不加载模型。
3. 满足后才运行最长合规记录（当前最长 491 nt）的冻结 xens2 探针，同时记录 PyTorch `max_memory_allocated` 和外部 `nvidia-smi` 峰值差。
4. 生产门槛：`free >= ceil(1.5 * max(torch_peak, external_peak_delta) + 2048 MiB)`。不满足时状态为 `WAITING_PRODUCTION`，不启动生产推理。
5. 门限满足后才运行 1422 条可恢复任务。任何显存失败均不触碰训练进程；已完成 ID 可从 checkpoint 恢复，其他错误不得静默重试。

`queue_status.json` 保存排队 PID、状态、设备、输入 checkpoint SHA、实测 peak 和门限；`probe.log`、`production.log` 保存实际命令输出。probe 与 production 使用不同目录，探针记录不混入生产预测。

## 完成与排名

只有真实记录级预测和 SHA 验证通过后才能计分。Vienna 可形成 full3966；xens2 若只完成 1406 条合规预测，则 full3966 仍为 3950/3966 部分覆盖、不排名，16 条保持显式 blocked。任何排队或部分状态都不能声称 xens2 full3966 已完成。
