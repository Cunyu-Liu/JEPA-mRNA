# ArchiveII full 3966：1422 条 CPU baseline 恢复协议

## 范围与输入

本任务只处理官方 3966 条参考中不在项目 clean 2544 内的 1422 条。任务清单为 `source_data/archiveii_full3966_cpu_recovery_20261127/recovery_plan.json`，每条记录固定官方 ID、原始 sequence、sequence SHA、参考 pair-set SHA 和 BPSEQ member SHA。1422 条当前全部为 ACGU；runner 仍对任何非 ACGU 输入直接写 `REJECTED_NON_ACGU`，不改写序列、不送预测、不把拒绝变成空预测。

只做 CPU baseline 推理，不使用 GPU，不修改 matched 训练、共享 registry、旧预测或旧指标。每个 worker 设置 `CUDA_VISIBLE_DEVICES=""`，OMP/OpenBLAS/MKL/NumExpr 均为 1 线程；不同 baseline 可作为独立单线程进程并行。

## 官方方法

- `contrafold`：官方串行 binary、native CONTRAfold 参数、MEA decode。
- `eternafold`：同一官方 binary，`EternaFoldParams.v1`、MEA decode。
- `mxfold2`：官方 CLI `python -m mxfold2 predict input.fa @TrainSetAB.conf`，CPU 默认 `--gpu=-1`，MixC/TrainSetAB，8 条一个 checkpoint shard。
- `vienna_mfe` / `vienna_centroid` / `vienna_mea`：ViennaRNA 2.7.2，全长序列。
- `rnastructure_fold` / `rnastructure_maxexpect`：官方 Fold `--MFE` 与 MaxExpect `--sequence`，解析首个 CT structure；多 CT 记录只保留数量信息，不擅自合并结构。

所有输出保持 zero-based inclusive pairs，不追加 min-loop、crossing 或 canonical 过滤。新输出不会覆盖已有 clean 2544 预测；后续合并身份必须显式标为 clean-frozen + official-only-CPU-recovery composite。

## 断点与恢复

默认工作根：`/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_full3966_cpu_20261127`。

每个 baseline 下有：

- `RUNNING.lock`：运行期间的 PID 和实际命令；防止同 baseline 重复写入。
- `records/<id-sha>.json`：逐 ID 结果，含名称、sequence SHA、状态、pairs/pair SHA。
- `coverage.json`：每次 shard/记录后的完成数、剩余 ID、coverage 和预测 SHA。
- `predictions.jsonl.gz`：仅从已验证 checkpoint 合并；部分完成明确为 `PARTIAL`。
- `predictor_metadata.json`：binary/config/weights 路径和 SHA。

运行与恢复均使用同一命令；已存在且名称、sequence、pairs、baseline 不匹配的 checkpoint 会 fail closed，而不是覆盖：

```bash
cd /home/rna_jev_archive_label_lineage
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
$PY tools/recover_archiveii_full3966_cpu_baselines_20261127.py --baseline contrafold
$PY tools/recover_archiveii_full3966_cpu_baselines_20261127.py --baseline eternafold
$PY tools/recover_archiveii_full3966_cpu_baselines_20261127.py --baseline mxfold2 --shard-size 8
$PY tools/recover_archiveii_full3966_cpu_baselines_20261127.py --baseline vienna_mfe
$PY tools/recover_archiveii_full3966_cpu_baselines_20261127.py --baseline vienna_centroid
$PY tools/recover_archiveii_full3966_cpu_baselines_20261127.py --baseline vienna_mea
$PY tools/recover_archiveii_full3966_cpu_baselines_20261127.py --baseline rnastructure_fold
$PY tools/recover_archiveii_full3966_cpu_baselines_20261127.py --baseline rnastructure_maxexpect
```

只读查看断点：

```bash
$PY tools/recover_archiveii_full3966_cpu_baselines_20261127.py \
  --baseline mxfold2 --status-only
```

## 完成判定

只有 `coverage.json` 的 `n_completed=1422`、`n_pass=1422`、`remaining_count=0`、`status=COMPLETE` 才能进入 full3966 全覆盖面板。任何其他状态只保留逐 ID 部分覆盖，不补 0、不排名。完成后仍需独立第二实现验证原始 checkpoint 与合并预测逐 ID 一致，并重算 full3966 和 clean2544 双面板。
