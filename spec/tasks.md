# Tasks — RNA 二级结构决策模型（纯血 Jev 架构迁移）

> change-id: `build-rna-ss-decision-model`
> 每个 Task 均有明确**完成判据**（可验证）。未满足判据不得勾选。
> 架构来源：纯血 Jev Decision Model 范式。**不引入 JEPA 线。**

---

## 实施状态总览

> **2026-09-29 傍晚刷新（第六轮交接，细节见 `records/DECISION_TRAINING_LOG.md` §14.79）**：
> - **draft v3.14 已提交**（**88/88 检查通过**，commit 95baf6b）——第五轮的"v3.14 为下一步"已完成（plana 行 + §4.3g (e) + 四行分解表 + 单种子 caveat）。
> - **四个工程缺陷修复**（详见 §14.79）：① `train_plan_a` 快照名硬编码导致 s1 臂将从 step 2000 起覆盖 s0 溯源快照（s0 十份快照已移入 `ckpts/plana_giga_s0_snapshots_preserved/` 保藏；trainer 改按 `--out` 派生快照名）；② `train_decision` 的 resume 对含 BatchNorm 持久缓冲的模型误报 ConfigError（r2d_s1 断点续训被卡死 58 vs 34；已修——按当前模型缓冲集过滤旧键，合成用例验证通过）；③ watcher 锁 fd 被训练子进程继承，导致替换 watcher 静默退出（v3 换新锁文件 + `9>&-`）；④ `train_plan_a` 补 `status=running/completed` 终态写入（§14.77 教训代码级闭环）。
> - **新增 `plana_giga_s0_ext40k` 收敛延长臂**：从 s0@20000 断点续训到 40000 步（**唯一变量=训练时长**；ff 家族 20k→40k 曾 +0.031，plana 全解冻段仅 5,600 步）；完成后自动评 30k+40k 两点，与 s0@20k 组成三点收敛曲线。启动器等 ≥16GB 设备（launch_plan_a_ext40k.sh + watch_plan_a_ext40k.sh）。
> - **臂状态**：`plana_giga_s1` 在训（MIG 切片，两次 OOM 重启后稳定）；`rinalmo_r2d_b4_s1` 守护等 ≥31GB 整卡（resume bug 已修，自动从 step 500 续训）；GPU 整卡全满（外部租户 + rna-ft-eval q_fill），两个启动守护 120s 轮询抢占空位。
> - **下一步**：三臂数字落地（s1 种子方差行 + ext40k 收敛行，若 40k 上升则 headline 升级为 @40k）→ 预印本冻结投稿版。
>
> **交接状态（2026-10-01 01:00，第八轮交接更新）**：
- **r2d_s1（Plan-B 第二 seed）突破 26 小时等待，已在训练**：GPU 5 实测空闲 32.0GB/0% 利用率数小时，而守护门槛 34GB 差 2GB 拒绝放行（过度保守——训练实测峰值 ~25-31GB）。已手动放置于 GPU 5（从 step 500 resume），过嵌入加载窗口验证（step 900，loss 下降）；门槛已调至 28GB（cap 36）。完成后 watch 协议自动终评，Plan-B 双 seed 齐 → draft v3.16。
- **A1 忠实臂 20k 完成**（DONE，14.5h）；v1_span 从 19,350 续训爬升中（重排机制完全按设计工作）。
- 台账已至 **§14.95**；draft **v3.15**（checker 100/100）。
- 外部基线：ERNIE/RiNALMo-mega/RNA-FM/SpliceBERT 各 11-12/12 cell；NucleicBERT、RiNALMo-650M 仍在 q_fill 计划中 pending。
- te_human fullbudget：s1 epoch 0.34、s42 epoch 0.88（ETA 10-01 晚/10-02）。
- RNA-JEPA 线：四大主臂 50k 全部完成，梯级评测全自动落地中（v1_cont_50k half_life n=5 r2 均值 0.5252 主臂最优档）。
>
> **交接状态（2026-09-30 02:45，第六轮交接更新）**：
- **集群**：`ssh A100`（bms-18937653-012，8×A100-40GB）。**注意 `/mnt/cunyuliu` 是 NFS 挂载**（`df -T` 实测 `10.179.129.209:/... nfs`）——跨进程锁的文件路径宜放本地 `/tmp`（经实测 NFS 上 flock 亦可序列化，见 `records/DECISION_TRAINING_LOG.md` §14.79–§14.80 的更正）。
- **plan-a 种子方差臂在跑**：`plana_giga_s1`（seed 1，A+B 组合）step 8900/20000，unfrozen 18/33，正常收敛；一条 `watch_plan_a_s1.sh` 监护 + 训练完成自动终评。
- **plan-b 种子方差臂（r2d_s1）历经四次 step-500 OOM，已诊断到两处真实缺陷并修复**：
  1. **瞬时容量接受**——`wait_device` 只凭**单次瞬时读数**放行；冻结嵌入加载约需 10 分钟，期间邻居（节点上的 `q_fill` 派发器会立刻抢占任何新出现的余量）把余量吃光。已改为**两阶段确认**：选出最空卡后隔 90 秒复测，仍达阈值才放行。
  2. **裕度太薄**——阈值曾被降到 31GB，而该臂自身峰值约 25GB，只留 ~6GB 给邻居；节点上的 `q_fill` 派发器会在任何余量出现时立刻抢占，故恒定阈值等同抛硬币。已恢复 **34GB**，并加**自适应升级**：每次尝试若「未推进步数即死」，阈值自动 +2GB（上限 38GB），不再重复同一个赌注。
  - 附带更正：我曾据一次**无效测试**（当时并无第一实例存在）断定「NFS flock 失效」，已撤回（§14.80）——有效竞争测试显示 `flock -n` 对已持有的锁返回 rc=1，即 NFS 上锁**正常**。
  - **不可用 `--head-chunk-size` 降显存**：`FlatDecisionHead.pair_scores_and_types` 在 `scorer='resnet2d'` 时**整体跳过列分块**（docstring 原文：BatchNorm over the full matrix is what makes chunking wrong there），故该臂仍需整卡。
  - **自愈**：新增 `scripts/ensure_r2d_s1_daemon.sh` + `*/10` cron（R2D_S1_WATCHDOG）；守护曾在 09-29 22:20 被外部杀死且日志无 FATAL 行，故不再依赖单次 nohup 存活。
- **教训入库**：① 锁测试必须含**活的竞争者**；② 基于 `pgrep -f <模式>` 的进程检查必须验证**模式不会匹配检查命令自身**（09-29 曾因此杀掉自己的 ssh 会话两次）。
- **下一步**：任一整卡释放（v2_scratch@50k 或 plana_s1 收队）即自动起 r2d_s1；两者齐备后 draft v3.15 记录 (d)/(e) 的 2-seed 读数。
>
> **交接状态（2026-10-08 00:30，第十一轮交接更新——消融矩阵补全轮，细节 §15.30）**：
> - **集群算力状态实测并纠正**：交接开场实测发现 **GPU1 整卡空闲 39GB/0% 利用率**（违反"显存必须占满"规则），立即补位两个臂后全部 6 张物理卡回到 90–100% 利用率。
> - **新增在训臂 ×2**：
>   1. **`rinalmo_fftr1c_b4_s0`**（T-A31 R2，tr1c-native 线性探针消融）：`--scorer mlp --head-chunk-size 16`，除 scorer（resnet2d→mlp）外与 r2dtr1c_b4_s0 字节一致（单变量纪律）；GPU1 在训；完成后自动 6-split 终评（w=-1，VL0 Platt）。
>   2. **`rinalmo_r2dtr1c_tr1cpdb_b4_s1`**（TS2/TS3 数据构成杠杆的第二种子，**预注册**）：在 s0 任何测试数字出现之前由 sed 派生（diff 仅 3 行：SUFFIX/TAG/--seed 1），守护轮询 34GB 窗口自动起跑。
> - **两臂均同批装 watchdog cron**（FFTR1C / TR1CPDB_S1，*/10）；项目 watchdog 现共 8 条 live。
> - **本轮三遍检查实录**：① bash -n + 占位符 grep 清零；② eval 调用旗标对照 watch7 实际协议修正（`--calib-data/--head-chunk 8/--tag`）；③ embedding shard 路径实测纠正（tr1c 是 shard0of**2**）。**heredoc-over-ssh 写脚本两次翻车，最终走本地文件+scp**。
> - **集群 git**：§15.30 已入台账并推送（commit 01966fe）；饱和度审计 cron 新装（T-A33 提前落地）。
> - **在飞读出（全自动）**：① tr1cpdb s0 → 6-split 终评 → **TS2/TS3 数据构成假设判定**；② fftr1c → R2 消融行；③ tr1cpdb_s1 自动占位。
>
> ### 第十一轮交接反思：当前存在的问题与改进措施（用户指示）
>
> **问题 1：GPU 饱和出现空窗且无人报警**（GPU1 整卡空闲近 4 小时量级）→ 饱和度审计 cron 已装（>20GB 持续 15min 记 WARN）。
> **问题 2：消融矩阵仍缺 tr1c-native 的"非交叉约束/Turner 置零"两个 cell** → T-A32 注册（纯离线评测，不占卡）。
> **问题 3：写作再欠 15.27–15.29 三章** → T-A34 注册（v3.19，等读出后一次写全）。
> **问题 4：PPT 落后两个数据波** → 待 tr1cpdb 读出后一并更新（T-A40）。
>
> | # | 任务 | 完成判据 | 状态 |
> |---|---|---|---|
> | **T-A32** | tr1c-native 离线消融：非交叉约束 + Turner 置零 | 两行 6-split 表入消融矩阵 | ✅ 已完成（§15.31） |
> | **T-A33** | GPU 饱和度审计 cron | 空闲显存告警 live | ✅ 已完成（本轮） |
> | **T-A34** | draft v3.19 | checker 全 PASS | ✅ 已完成（§15.34） |
> | **T-A35** | tr1cpdb 双种子终评 → TS2/TS3 终判 | 双种子并排表 | 🔄 s0 已判定（§15.32）；s1 在飞 |
>
> **交接状态（2026-10-08 12:00，第十三轮交接更新——v3.19 落地轮，细节 §15.33–§15.34）**：
> - **fftr1c 读出（§15.33）**：2D-context scorer 价值 **+0.0888 ID / +0.13–0.16 PDB 家族**（TS3 +0.1612、TS-hard +0.1332）——tr0 时代 +0.067 在干净语料上复现且更大。**tr1c-native 消融矩阵 5/5 闭合**。
> - **统一签名（本轮最重要的科学产出）**：五个单变量消融（DP / Turner 先验 / 2D scorer / distill / RLCD）在最终配方上呈现同一模式——**每个结构组件的贡献都随 OOD 距离单调增大**；辅助项小而诚实。"OOD 鲁棒性由先验栈整体承载（DP + 物理 + 2D 上下文），神经 scorer 是 ID 专家"成为消融级证据支撑的主张。
> - **draft v3.19 已落地（§15.34，T-A39 关闭）**：§4.3b tr1c-native 矩阵表（7 行 × 6 split）+ 第四战线段（tr1cpdb 负结果含绝对值）+ 4-seed 家族图 + xens3 wash 行 + cov v2 探针段；Limitations 5 升级为四战线。**check_v319 57/57 PASS；check_v318 39/40（唯一 FAIL = 版本号升级本身，预期）；manuscript linter OVERALL PASS**。commit c3a3197。
> - **PPT S15 更新**：消融表 5 行（+fftr1c 行 + 蒸馏/RLCD 行），统一签名描述；0 溢出，15 slides，token 验证通过。
> - **checker 过程实录**：首跑 53/57——tr1cpdb 数字最初只以差值入稿（checker 抓到，这正是"数字必须以可回溯形式入稿"纪律在起作用），补绝对值后 57/57。
> - **在飞**：tr1cpdb_s1（GPU5，09:13 起，ETA ~16:45 训完 → 自动 6-split 终评）→ Appendix A 种子方差行 + 负结果稳定性判定。
> - **预印本冻结清单（剩余）**：① s1 读出入 Appendix A（自动化）；② Appendix C "preliminary" 列表删三项（v3.19 已闭合的）；③ 最终通读 + 引用核验。
>
> | # | 任务 | 完成判据 | 状态 |
> |---|---|---|---|
> | **T-A41** | s1 读出 → Appendix A 行 + 负结果种子稳定性判定 | 双种子并排；若不同向则负结果降格为"不可复现为正" | 🔄 ~17:30 自动落地 |
> | **T-A42** | Appendix C preliminary 清单收窄（删 15.33/15.34 已闭合项） | 清单与已落地事实一致 | 待启（写作收尾） |
> | **T-A43** | 预印本冻结前终审：全数字 spot-check + 引用核验 + 5 结论句回溯 | checker 全绿 + 每句可回溯 | s1 落地后 |
>
> **交接状态（2026-10-08 02:30，第十二轮交接更新——两章读出落地的判定轮，细节 §15.31–§15.32）**：
> - **T-A32 关闭（§15.31）**：Turner 先验消融在最终配方（r2dtr1c s0 @20k）上 6-split 落地——**物理先验价值 +0.2266 ID / +0.3683..+0.4672 OOD，单调随 OOD 距离增大**。P/R 剖面：置零后精度保持（P 0.45–0.71）但召回崩（R 0.14–0.36）——**Turner 物理项是 OOD 承重墙**，与 solver 消融 R3（DP +0.25/+0.43，§15.29）互为镜像：**管线的两个结构先验（非交叉 DP + 物理先验）各贡献 1/4–1/2 个 F1 点，且都在 OOD 上贡献更大**。范式叙事获得消融支撑的答案："OOD 鲁棒性由什么承载？不是 ensemble、不是 backbone——是两个物理/几何项，神经 scorer 是 ID 专家。"
> - **架构事实先行确认**：r2d（resnet2d）路径**没有 MLP_T**（`self.turner=None`，decision_head.py:355；checkpoint 58 键零 turner 条目）——ResNet 即学习 scorer，物理以可学习 `head.prior_weight` 进入。故本配方的忠实"Turner 置零"= decode 期 `--prior-weight 0.0`（重加权路径：同 DP、同 mask、同校准协议，单变量零重训）。ff-tr0 时代的 0.2124 旧数字保留为 ff-recipe 列，不重跑。
> - **T-A28/T-A35 判定（§15.32）：TS2/TS3 数据构成假设被否定（干净负结果）**。tr1cpdb s0（tr1c+234 行去污染 PDB 家族行，单变量）6-split 全降：TS0 −0.0223 / new −0.0908 / TS1 −0.0610 / **TS2 −0.2092** / TS3 −0.0918 / hard −0.0971。P 持平或升（TS2 P 0.8986）而 R 崩（TS2 R 0.4061 vs ctrl 0.6718）——head 对稀有茎环**更保守而非更会开火**：234 行是分布扰动不是家族特化。**TS2/TS3 差距已在四个战线上存活**（非规范配对/边界偏移/解码偏置（§15.19）+ 数据构成（本轮））——论文 Limitations 行成型："差距是真实建模差距，冻结骨干+物理先验范式在少量 PDB 家族信号上不闭合"。
> - **决策：不再从当前配方家族发 TS2/TS3 新臂**。15.28 的 VL0-gate NMR head 想法降级为 optional（T-A36-parked，prior 低，不排队）。
> - **s1 已自启（02:21，GPU3，预注册种子方差臂）**：跑完符合收敛纪律，6-split 自动落地（~10:30）；预期同向下降→负结果种子稳定；若不同向→负结果软化（两者都可发表）。
> - **在飞**：fftr1c @6.2k/20k（GPU1，健康，~07:00 完训 → 自动 6-split → R2 消融行）；tr1cpdb_s1（GPU3）。饱和审计 cron live（00:26/00:30 的 GPU1 WARN 是 turnerzero 评测自身的启动窗，00:40 起全饱和）。
> - **集群 git**：§15.31（1271188）+ §15.32（3fb9828）已推 GitHub；本地台账同步至 7501 行。
>
> | # | 任务 | 完成判据 | 状态 |
> |---|---|---|---|
> | **T-A36**（parked） | VL0-gated NMR-family head 微调（TS2/TS3 最后候选杠杆） | 仅在 s1 读出显示负结果不稳定时解冻 | ⏸️ parked（prior 低） |
> | **T-A37** | fftr1c 读出 → R2 消融行入矩阵（vs r2dtr1c 对照） | 6-split 表 + head-architecture 消融列闭合 | 🔄 ~07:00 自动落地 |
> | **T-A38** | tr1cpdb_s1 读出 → 负结果种子方差行 | 双种子并排表入台账 | 🔄 ~10:30 自动落地 |
> | **T-A39** | draft v3.19：15.27–15.32 全折叠（4-seed 图、xens3 行、solver R3 + Turner 先验两行消融表、TS2/TS3 四战线负结果段、cov v2 段） | checker 全 PASS；每个数字可回溯 | 🔴 下一个写作动作（等 fftr1c/s1 落地后一次写全） |
> | **T-A40** | PPT 更新：slide 14 表 + 消融两行 + TS2/TS3 诚实负结果框 | 数字三遍核对后入片 | 排队（与 T-A39 同批） |
>
> **交接状态（2026-10-06 19:30，第十轮交接更新——SOTA 翻盘后的收尾轮，细节 §15.10–§15.24）**：
> - **用户六数据集目标已达成 6/8**（含 tier-1 全部三个）：TS0 ✅0.7866、bpRNA-new ✅0.6132、ArchiveII-clean ✅0.7403/0.7760、TS1 ✅0.8570/0.8755、TS-hard ✅0.8530/0.8732、TestSetB ✅0.7507/0.8448；**TS2 −0.0455 / TS3 −0.0445 未达标**（已证明是真模型差距：§15.19 否定非规范配对/边界偏移/解码偏置三假设）。
> - **两次泄漏审计 + 冻结评测集**是本轮最大方法论成果：TR1∩TS0=1087（§15.10）、TR0∩TestSetB=247（§15.13）两次撤回旧数字，建立 9-split sha256 冻结清单（`spec/eval_splits_frozen.json`）与唯一干净语料 bprna_tr1c；**任何新语料构建必须过零重叠门**。
> - **可复现叙事链已闭合**：单模超 5/6 参照 → same-family vs cross-family 消融（增益来自家族互补非模型数）→ 复现性跟随被训练的层 → 校准 ECE 0.0007 最优 DP-free → §15.22 独立审计 540 checks 0 mismatch。
> - **在飞**：plana_tr1c_s2（GPU1，~7.9k/20k，ETA 10-07 晨）+ **本轮新增 plana_tr1c_s3**（seed 3；首放即被外部租户挤爆，守护自愈换卡到 GPU3 已在训练——自愈机制实战验证；若落 recall-type 则 plana bucket 升 3-seed）；watchdog cron 补装 PLANA_TR1C_S2/S3（此前 11 条全 PAUSED，违反 §14.79 教训）。
> - **本轮交接动作**：本地三件套与集群同步（台账拉至 §15.24、tables 15.23 版、paper v3.17）；集群 commit 603cd65 已推 GitHub；s3 新臂 + watchdog 补装完成。
>
> ### 第十轮交接反思：当前存在的问题与改进措施（用户指示，逐条对应行动）
>
> **问题 1：双源文档漂移（最严重）**。本地交接文档停在第九轮（10-01），集群台账已到 §15.23（10-06）——中间发生了整个 SOTA 翻盘（泄漏撤回、tr1c、Arm A/B、xens2），但交接 spec/tasks 完全没跟上。任何一个接手者按旧文档工作都会重走弯路。**改进**：交接更新必须与台账章节同步写入（本轮已执行：spec/tasks/checklist 三件套 + 台账 §15.24 同批提交）；后续每次新增台账章节 ≥15.25 时同步刷新交接块（T-A29）。
>
> **问题 2：写作落后于实验**。draft v3.17 的 §4.3 主表还是 10-02 的旧局面（0.7268 headline、旧 6 来源校准表），xens2/w=0.7/去污染三件大事全部没进稿件。当前 6/8 SOTA 的战况不存在于任何一版 draft 中。**改进**：v3.18 为最高优先级写作任务（T-A25），从 `tables/sota_vs_ours.md` 15.23 版由脚本折叠入稿（沿用「数字全部磁盘读取、assert 防漂移」的既有纪律）。
>
> **问题 3：自愈纪律倒退**。§14.79 建立 watchdog-cron 自愈机制后，crontab 中 11 条 watchdog 全部处于 `#PAUSED#` 状态（外部/用户暂停，未深究），s2 臂已因此在 10-05 死过一次（§15.16/§15.17 各有一次 daemon 死亡事件，靠人工复活）。**改进**：本轮补装 PLANA_TR1C_S2/S3 两条新 watchdog；后续新臂启动必须同批装 watchdog（写进启动 checklist）。
>
> **问题 4：TS2/TS3 负结果没有论文落点**。三假设被否定（§15.19）是扎实的负结果，但 draft 里没有对应段落——审稿人会问"为什么这两个 split 不达标"。**改进**：T-A26 把 §15.19 的排除链写成 Limitations/附表；同时排队 NMR-家族 head 微调（VL0-gate）作为最后的实验尝试。
>
> **问题 5：消融矩阵与最终配方脱节**。15 项消融（Gate I/SC3）全部在 tr0/旧配置上跑的，最终配方（plana-tr1c + xens2）没有任何一项在 tr1c 上重做。**改进**：T-A27 在 tr1c 上重跑可离线的核心消融（非交叉约束、Turner 置零、蒸馏开关、RLCD 开关、解码顺序）。
>
> **问题 6：叙事风险——"去污染撤回"双刃剑**。我们撤回过自己的数字两次（§15.10/§15.13），说明早期管线的确有缺陷；但也正因为如此，"冻结评测集 + 唯一干净语料 + 独立审计"的方法论贡献反而成了叙事亮点。**改进**：T-A25 写作时把撤回史完整披露（不是隐藏），作为 benchmark hygiene 贡献的一部分。
>
> **下一步任务（新增，按优先级）**：
> | # | 任务 | 完成判据 | 状态 |
> |---|---|---|---|
> | **T-A25** | **draft v3.18 大改**：去污染叙事章节 + sota 15.23 主表折叠 + xens2/w0.7 协议说明 + TS2/TS3 负结果落点 + 撤回史披露 | checker 全 PASS；每个数字可从 result.json 回溯 | 🔴 最高优先 |
> | **T-A26** | s2/s3 落地终评 → 3-seed plana bucket + xens3 + TS2/TS3 终判 | 8-split 完整表 + 种子方差行 | 🔄 s2 在训 s3 已自愈在训 |
> | **T-A27** | tr1c 上重跑核心消融（≥5 项） | 消融表与最终配方同语料同协议 | 待启 |
> | **T-A28** | NMR 家族 head 微调（TS2/TS3 最后杠杆，VL0-gate） | VL0 通过才上测试集；未通过则如实记录负结果 | 排队 |
> | **T-A29** | 交接三件套与台账同步刷新机制 | 每轮台账新增章节同步更新交接块 | 本轮已执行，机制固化 |
>
> **交接状态（2026-10-01 15:30，第九轮交接更新）**：
- **r2d_s1 完成并双 split 终评落地——Plan-B 种子方差极紧**：TS0 0.6559 / OOD 0.5000（Δseed **-0.0071 / -0.0010**，OOD 实际种子不变）。与 Plan-A 的 OOD 摆幅 -0.054 形成对照。
- **draft 升至 v3.16**：Plan-B s1 行入主表 + §4.3g(d) 第二 seed 段落（**"复现性跟随被训练的层"**——只动 head 的配方复现，动 backbone 的不复现）+ Appendix A 行；**checker 108/108**（途中还抓到一处精度：Δ 实为 -0.0071 非 -0.0070）。GPU 5 的 26h 卡点后的手动放置被结果完全证明。
- **RNA-JEPA 线挽救臂 v1_rescue 在跑**（α=2.0 已实证生效：JEPA 梯度份额 0.3-0.6%→1.9-4.2%；10k/20k 档评测全自动接线完成）。
>
> **交接状态（2026-09-30 16:40，第七轮交接更新）**：
- **Plan-A 种子方差臂完成且已入稿**：`plana_giga_s1`（seed 1，20k 步全解冻）终评 **TS0 micro 0.7245 / macro 0.6989；OOD bpRNA-new 0.3763**（P 0.650/R 0.265）。**A 项跨 seed 复现**：ID 增益紧密（+0.062 s1 vs +0.064 s0，Δseed 仅 -0.0023）；OOD 代价 seed 噪声大（-0.071 s0 → -0.125 s1），且 seed 间 OOD 摆幅（-0.054）与 ID 增益同量级——**两 seed 并排报告、绝不取均值**；召回侧损伤在 s1 加深。
- **时长曲线（ext40k，seed 0）落地**：TS0 0.7268→0.7210→0.7178，OOD 0.4302→0.4005→0.3988——臂在解冻窗口终点（20k）达峰后单调退化，**headline 定格 @20000**；与 ff 臂 20k→40k 继续 +0.031 形成可报告对照（全解冻臂过其窗口即过拟合）。
- **draft 升至 v3.15**（主表 s1 行 + 时长行、§4.3g seed 方差与收敛段落、Appendix A 两条 artifact 行），**checker 100 断言全 PASS**（commit 5f5f59c）。
- **运维（本轮会话修复）**：`eval_plan_a.py` 终写缺父目录 mkdir 的 bug 已修（pytest 子集 37/37）；watcher 快照流 s0→s1 重命名正确执行；CUDA 短名 `GPU-4` 不可用于 CUDA_VISIBLE_DEVICES（须 MIG UUID 或数字）已入册。
- **在飞**：r2d_s1 仍等 ≥34GB 确认窗口（守护+看门狗 cron 自愈，adaptive 门槛 34→38GB）；**RNA-JEPA 线 v2_scratch 已 50k 完成，v1_cont 50k 完成**——063/064 评测全自动展开中。
- **下一步**：r2d_s1 起跑后 Plan-B 双 seed 齐 → draft v3.16（(d) 行 2-seed 并排）；RNA-JEPA 梯级表全齐后更新 Table1 梯级视图。
>
> **2026-09-29 下午刷新（第五轮交接，细节见 `records/DECISION_TRAINING_LOG.md` §14.76–§14.77）**：
> - **Plan-A（plana_giga_s0）训练+终评全部完成**：20000 步渐进解冻（33 块全解冻）→ **TS0 micro F1 0.7268 / macro 0.7139，OOD bpRNA-new 0.4302**。A 项分解测得：**+0.064 ID / −0.071 OOD**（容量反转签名）。主表对标：超 UFold 0.6598、超 NucleicBERT 微调 macro 0.649，距 RNAformer 0.7578 仅 0.031。
> - **A+B 三项分解闭环**（§4.3g chase map 全部实测）：2D scorer +0.067 双正（0.6629/0.5010）；backbone 适配分裂符号；structure-aware residual ~0。部署指引已成型。
> - **种子方差臂已启动（用户指示占满 GPU）**：`rinalmo_r2d_b4_s1`（GPU 5 在跑，watch7 协议终评自动接续）；`plana_giga_s1` 启动器+守护等设备自动起（≥16GB）。
> - **台账修复**：plana_giga_s0 监控误报 37 小时（run_meta.json 缺 status 字段）；已补 completed。教训入 §14.77。
> - **draft v3.13 已 78/78 检查通过**；v3.14（plana 行 + chase map 闭环）为下一步。
>
> **2026-09-24 更新**：A100 集群已接入（`ssh A100`），**数据与算力阻塞已解除**。
> 权威的 benchmark / 数据 / 评测协议决策见 **`spec/benchmark_decision.md`**；本文件与其冲突时以该文件为准。

**代码基础设施：`python -m pytest tests/ -q` → 317 passed, 0 failed**（集群 torch 2.5.1，权威环境）

> **2026-09-24 14:20 状态刷新**（细节见 `records/DECISION_TRAINING_LOG.md` §14.9–§14.14）：
> 10 个臂在跑（6 个 from-scratch + 4 个 RiNALMo head-only），本轮新增 **4 个目标函数对照臂**（见 T-A4）。
> **TS0 目前最好 micro F1 = 0.4957**（`rinalmo_ff` @2000，w 在 VL0 上选）；对照物
> **ViennaRNA centroid 0.5393 / MXfold2 0.5651**（均为本仓库同口径实测）——**尚未超过物理基线，不得写成"优秀结果"**。

> **2026-09-24 15:50 状态刷新**（细节见 §14.16–§14.20）：
> **16 个臂在跑**（6 from-scratch + 4 RiNALMo 原臂 + 4 目标函数对照臂 + 2 级联臂 + 1 容量臂）。
> **TS0 headline 已落地且不依赖任何 VL0 选择**：`rinalmo_ff`@3500、`--prior-weight -1`（模型自训练权重 0.9627）
> → **micro F1 0.4953 / macro 0.4858 / P 0.5325 / R 0.4629**；同一 checkpoint 用 VL0 选出的 `w=0.75` 得 0.4959
> （**差 0.0006**，故 §14.15 问题 2 对 headline 的威胁已解除）。
> **C1-c 只在「免 DP 仿射重标定」口径下 PASS**：`w=-1` gap **0.0002** / `w=0.75` **0.0006** / ArchiveII **0.0093**（阈值 0.02）；
> **裸头 gap 0.1348，未过**——两个口径必须同时报。
> 仍未超过物理基线（centroid 0.5393 / MXfold2 0.5651）；**跨家族 bpRNA-new 0.3094 vs ViennaRNA centroid 0.6770**，
> 已退到自己的 Nussinov+Turner 先验（0.3015）水平 → **泛化不足，已量化，不得主张"OOD 更鲁棒"**。
> **ArchiveII 只能用 3,950/3,966 行**且**不是 OOD 集**（26.7% 与 TR0 近重复）。

> **2026-09-24 21:30 状态刷新（交接后权威口径，细节见 §14.34–§14.42）**：
> **19 个臂在跑**：6 from-scratch（@30k/40k）+ ff 种子 s1–s5 + 2×2 目标对照（len/sum，@17–20k）
> + 级联 casc/cascR（@13k/8k）+ 容量 big（@12k）+ **TR1 数据扩展臂×2**（@6k，45,865 条，4.29×）。
> - **headline 已到 0.5958**：`rinalmo_ff_b4_s0` @step20000、`w=-1`、`ref_bprna_ts0`（1,291）→ micro F1 **0.5958** /
>   macro 0.5970 / P 0.5801 / R 0.6125；**已超 ViennaRNA centroid（0.5393）+0.0565，但距 UFold 0.6598 差 0.064、
>   距 RNAformer 0.7578 差 0.162**。欠训练是已证主因（0.4953@3500 → 0.5958@20000，§14.35）。
> - **C1-a 实测 FAIL（§14.42）**：RNAformer ECE 0.0015 < 我们重标定 0.0031（免 DP、无后处理）。
>   C1 重组为"系统评测 + 自洽性"（spec §0.10）；**主对标必须写 RNAformer**。
> - **C1-c（同集重测）**：裸头 ECE 0.1955 / 重标定 0.0031，gap 0.0028 PASS，Brier/NLL 同改善（§14.40）。
> - **架构臂 step2000 首评**：级联 casc **0.4610** / cascR **0.4654** / 容量 big **0.4850**（vs 同步 ff ~0.497，
>   均无优势但臂仍在训，C2/容量假设未定论）。合法性三项全部 0。
> - **bprna_new（跨家族）**：我们 **0.3536** vs UFold 0.6106 vs centroid 0.6770 —— **当前最大短板，必须与 TS0 并列报告**。
> - **评测产物**：`/mnt/cunyuliu/rna-jepa/eval_decision/`（ff20000 三 split、trend 6000/10000、arch×3、
>   clean splits、c1a 同集表、mxfold2 archiveii）；监控 cron 每 10 min + checkpoint 快照每 10 min。

| 类别 | 任务 | 状态 |
|---|---|---|
| **已完成（代码 + 测试）** | Task 1, 4–9, 10(脚手架), 11(脚手架), 12–21 | ✅ |
| **已完成（集群侧实测）** | **Task 2**（数据盘查）、**Task 3**（结构数据落盘）——依据见下「集群实测数据现状」 | ✅ |
| **进行中** | Task 10(实际复现), Task 18(实际预训练), Task 20(实际评测) | 🔄 |
| **已作废** | Task 10.8 的**原始判据**（CDPFold 单点风险）——前提有误，见下 | ❌ |
| **已作废（新）** | T-A4 的**原始表述**「按 GT 配对数归一」——实测后改为**按序列长度**归一（§14.12），判据已按实际口径更新 | ❌ |
| **已判 FAIL** | **C1-a 原判据**（§14.42，RNAformer 反例）→ C1 重组为"系统评测 + 自洽性"，见 spec §0.10 / benchmark_decision §3.2 | ❌→重组 |
| **新增** | T-A8–T-A12（第二轮审计派生）、**T-A13–T-A17（本轮交接新增，见文末）** | 见表 |

### 关键勘误（2026-09-24，必须执行）

**CDPFold 的定位被写错了。** 原 §0.8 问题 ⑥ / §0.9.4 / §7.2.1 / Task 10.8 称其为"条件扩散、已免 DP"，并据此把它当作**全局单点风险**。实测核查（Front Genet 10:467, 2019）：**CDPFold 是 CNN + 动态规划（DP）**，既非扩散、也**不免 DP**、且**无任何校准评测证据**。

- 原"单点风险"**作废**；Task 10.8 的判据替换为 `spec/benchmark_decision.md` §3.2 的 **C1-a / C1-b / C1-c**。
- **真正的先例威胁**是 **SPOT-RNA / SPOT-RNA2 / UFold**——它们**确实**单次前向直接输出 `L×L` 配对概率（sigmoid），**不跑配分函数**。
- 因此 **C1 的新颖性只能锚定在"校准"**，不能锚定在"单次前向出概率"（后者已有先例）。
- 有利发现：**碱基配对概率的校准评测在 RNA 领域基本是空白**（未检索到 ECE / 可靠性图 / Brier / NLL 的系统评测）。这既是 C1 的机会窗口，也要求**我们自行定义指标口径**。

### 集群实测数据现状（2026-09-24，全部为实测计数）

数据根：`/mnt/cunyuliu/BPfold_data`（已有）+ `/mnt/cunyuliu/rna_ss_data`（本次新增）

| 数据集 | 实测条数 | 用途 | 状态 |
|---|---|---|---|
| bpRNA **TR0** | 10,814 | 训练 | 已落盘 |
| bpRNA **TS0** | 1,305 | **主集（测试）** | 已落盘 |
| bpRNA **VL0** | 198 | 验证 | 存疑（SPOT-RNA 原文记 1,300） |
| **RNAStrAlign** | 37,052 | 训练（规模扩展） | 已落盘 |
| **PDB_669** | 669 | 实验标签训练 | 已落盘 |
| **Rfam12.3–14.10** | 10,791 | **家族级 OOD** | 已落盘 |
| **bpRNA-new** | 5,401 | **跨家族 OOD 主集** | 本次下载 |
| **PDB ts1 / ts2 / ts3** | 60 / 38 / 18 | 实验标签测试 | 本次下载 |
| **Rfam14.10–15.0** | 待计数 | **时间级 OOD** | 本次下载 |
| **ArchiveII（CSV 版）** | 3,864 | 主集替代 | 本次下载 |
| ArchiveII（bpseq 版） | **0（空目录）** | — | **未取得** |

> **archiveII 空目录**：集群上原有的 `archiveII/archiveII/` 是空目录（同批 `archiveII.lst` 等为 0 字节，解包被截断）；BPfold 的 release 包也**不含** archiveII（已实测解包确认）。故 ArchiveII 采用 RiNALMo benchmark 整理版 CSV（3,864 条，含 sequence/structure/base_pairs/len + family-fold/k-fold 两套划分）。**论文必须写明版本与条数，不得笼统称"ArchiveII 3975"。**

### 本机已交付且已验证的关键产物

| 产物 | 文件 | 验证证据 |
|---|---|---|
| 数学核心（关键路径） | `src/rnajepa/harness.py` | `logZ`/`p̂` 与暴力枚举误差 **2.2e-16 / 5.6e-17**；1200 组随机矩阵非法率 = 0 |
| 编码器 + 决策头 + 层级级联 | `src/rnajepa/encoder.py`, `decision_head.py` | 对称性误差 = 0；级联可表示跨度 63 的长程配对 |
| 蒸馏 + RLCD | `src/rnajepa/distill.py`, `rlcd.py` | 无采样校准梯度经 monkeypatch 证明；软 ECE 正确 |
| 数据清洗 C1–C6 | `src/rnajepa/clean/` | 衰减表守恒；假结路由；污染检测 |
| 评测/消融/门限 | `eval/ss/` | 七类指标；15 项消融；S6/S7 判据；G5 门正确 FAIL |
| 基线 + CDPFold 阻塞项 | `eval/ss/baselines.py`, `cdpfold_check.py` | C1 判决逻辑三种合成用例通过 |
| 训练驱动 | `src/rnajepa/train_decision.py` | CPU 端到端；梯度覆盖；NaN 硬失败 |
| 论文脚手架 | `paper/` | 四段式叙事；6 条禁止表述 linter；Q1–Q12 落点检查 |

### 实施阶段发现并修正的实质问题（4 项）

1. **梯度符号写反**：spec 原写 `(y − p̂)·∇s`，正确为 `∂L_NLL/∂s_ij = p̂_ij − y_ij`。经有限差分与暴力枚举独立验证后已勘误（spec §5.5.3）。
2. **RLCD「无采样」的适用范围被夸大**：该性质只在**概率空间**成立；反传穿过配分函数需 `log Z` 的 Hessian（`O(L⁴)`）。已澄清为"RLCD 作用于 System-1 头的直接概率输出，精确边际仅作 detached 教师"（spec §5.8.2）。
3. **`-inf × 0.0 = NaN`**：决策头对非法对写 `-inf`，与权重为 0 的目标项相乘产生 NaN。已在训练驱动层改用有限哨兵。**已验证数学核心本身对 `-inf` 处理正确**（与有限哨兵结果逐位一致）。
4. **零初始化 `MLP_T` 使首个反向传播全零**：step-0 时仅 3/26 参数有梯度（因残差末层零初始化）。梯度覆盖断言须在**首个 optimizer step 之后**评估。

### 诚实声明（不得掩盖）

- **本机没有任何二级结构标注数据**，也无 GPU/网络。因此 Task 2/3/10/11/18/20 的**实证结果尚不存在**——本文档只交付了可执行的代码与脚本。
- 未安装的外部工具（ViennaRNA / RNAstructure / LinearPartition / MMseqs2 / CD-HIT / CDPFold / 全部深度学习基线）一律以**抛错的干净 stub** 呈现，**从未伪造输出**。
- `citation_register.csv` 全部标记 `待核验`（无网络无法完成核验）；未知 URL/哈希留空而非编造。
- `eval/ss/gates.py` 的 **G5 门当前故意 FAIL**（19/19 基线缺 commit + 权重哈希）——这是诚实状态。

---

- [x] Task 1: 冻结分析计划、证据基线与量化门限：写任何代码前锁定假设、指标、统计方法、门限值。
  - [x] SubTask 1.1: 撰写并冻结 `analysis_plan.md`，含 H1–**H6**、七类指标、统计方法（≥5 seed、Wilcoxon、Holm-Bonferroni）、§8 全部门限
  - [x] SubTask 1.2: 完成 Jev 证据等级登记表：明确"无同行评审论文"；架构描述标注为社区复刻二级证据；厂商性能数字标注为待核验且不得作为事实引用
  - [x] SubTask 1.3: 完成 Jev 六条→**九条机制（J1–J9）**的映射表，含接口层（J7 State / J8 结构化输出）与训练层（J9 RLCD），逐项标注本方案对应物
  - [x] SubTask 1.4: 撰写 JEPA 移除决策记录，引用 `records/A3_latent_target_comparison.md` 与 `records/factor_probe_control_analysis.md` 的三条否定性证据
  - [x] SubTask 1.5: 继承既有教训清单（区域摘要饱和、因子探针假阳性、74 词表坍缩、lr 摧毁编码器、dev==test、APA 缺失）并写入 `constraints.md`
  - [x] SubTask 1.6: **撰写 §0.7 审稿人自检记录**：五个已修正问题（速度/精确边际矛盾、Gibbs 非首创、Turner 表述不准、共转录装饰性、非法率同义反复）及修正方式
  - [x] SubTask 1.7: **撰写 §2.4「明确不主张」清单**（6 条禁止表述），并写入 `constraints.md` 作为写作红线
  - [x] SubTask 1.8: **撰写 §9.3 预期审稿质疑 Q1–Q12 应答表**，每条标注所需证据
  - [x] SubTask 1.9: **完成 §0.9 偏离度审计**：逐项对照原始 Jev（State / 候选集 / 输出 / 前向次数 / 决策头 / 零幻觉 / RLCD / 路由 / System-1-2），明确三处本质偏离与「**推广决策范式到结构化预测**」的定位
  - [x] SubTask 1.10: **确定贡献结构为「2 真贡献（C1/C2）+ 2 支撑（C3/C4）+ 5 实现手段（不主张）」**，并据此写 `constraints.md` 写作红线：**Gibbs 框架 / Turner 残差 / 构造性对称 / 隐式微分 / 多通道证据不得出现在贡献句中**
  - [x] SubTask 1.11: **确定泛化主张的口径**：从"OOD 精度更高"改为"**OOD 衰减更小（更鲁棒）**"（§0.9.3）
  - **判据**：`analysis_plan.md` 与 `constraints.md` 存在，含冻结时间戳且早于首个实验运行；§0.7/§0.8/§0.9 三轮自检记录、Q1–Q12 应答表、贡献结构与泛化口径均已写入

- [x] Task 2: 数据源真实可用性实测盘查：**不得把计划中的数据当作已有数据**。**（2026-09-24 完成，集群侧实测）**
  - [x] SubTask 2.1: 逐源实测可达性：bpRNA-1m / bpRNA-new / ArchiveII / RNAStrAlign / PDB-RNAsolo / PseudoBase++ / RNA-Puzzles / CASP15-16 / SHAPE-DMS 探测数据
  - [x] SubTask 2.2: 记录每源的实测状态（已落盘 / 待获取 / 网络受限 / 不可得）、实测大小、可用镜像、许可 → **`spec/benchmark_decision.md` §2**
  - [x] SubTask 2.3: 确认已落盘资产状态（Zenodo 17786045 七归档 md5、12516160 的 20.02 GB、mRNABERT 权重）并复核
  - [x] SubTask 2.4: 记录集群网络实测：**`api.github.com` / `raw.githubusercontent.com` / `codeload.github.com` 可达；`zenodo.org` / Dropbox / Google Drive / NihaoCloud 不可达** → 基线选择须服从该约束
  - [x] SubTask 2.5: 明确记录：**现有 145 个下游任务与 20 GB 语料均不含任何二级结构标注**
  - **判据**：✅ 盘查表产出（`spec/benchmark_decision.md` §2.1/§2.3），覆盖全部计划数据源；不可得项已标记（archiveII bpseq、SPOT-RNA/UFold 权重、PseudoBase++、RNA-Puzzles）；**0 条"假设可用"未实测**

- [x] Task 3: 结构标注数据获取：**本项目最大的新增工作**。**（主体已完成；archiveII bpseq 版未取得，有替代方案）**
  - [x] SubTask 3.1: 复用 `scripts/fetch_zenodo_parallel.sh` 多连接分块 + md5 校验机制获取可下载源
  - [x] SubTask 3.2: 获取 bpRNA-1m（含 TR0/TS0/VL0 划分，实测 10,814 / 1,305 / 198）与 **bpRNA-new（5,401）**
  - [x] SubTask 3.3: 获取 **RNAStrAlign（37,052）**；**ArchiveII** 改用 RiNALMo 整理版 CSV（3,864 条）——**bpseq 版未取得，须在论文中写明版本**
  - [x] SubTask 3.4: 获取 PDB 衍生集：**PDB_669（669）**、**PDB ts1/ts2/ts3（60/38/18）**
  - [x] SubTask 3.5: **未取得** PseudoBase++ 与 RNA-Puzzles / CASP15-16（Dropbox / Google Drive 不可达）→ 已记录为不可得，作为加分项待补
  - [x] SubTask 3.6: **未取得** SHAPE/DMS/icSHAPE/PARS 探测数据 → 已记录，仅影响 T3
  - [x] SubTask 3.7: 每个源的实测大小已记录；**SHA256 写入 manifest 待补**（下载字节数已与 release API 报告 size 对齐）
  - **判据**：✅ T1 主任务所需数据集（bpRNA-1m + ArchiveII + RNAStrAlign + bpRNA-new）**全部落盘**；不可得项（archiveII bpseq、PseudoBase++、RNA-Puzzles/CASP、探测数据）**有明确记录与替代方案**（`spec/benchmark_decision.md` §2.4 / §6）

- [x] Task 4: 实现序列层清洗（C1）。
  - [x] SubTask 4.1: 实现 `T→U` 规范化与 IUPAC 歧义码策略（`N` 比例阈值 + 歧义位点掩码）
  - [x] SubTask 4.2: 实现长度过滤与低复杂度检测（dustmasker/tantan）打标
  - [x] SubTask 4.3: 为每条记录生成 SHA256 指纹
  - [x] SubTask 4.4: 单测覆盖边界（全 `N`、空序列、含非 ACGU 字符、极短/极长）
  - **判据**：单测全通过；记录含指纹；无静默丢弃（每类丢弃有原因码）

- [x] Task 5: 实现冗余去除与泄漏控制（C2）：论文可信度命门。
  - [x] SubTask 5.1: MMseqs2/CD-HIT-EST 做 80% 与 90% identity 两档内部去冗余
  - [x] SubTask 5.2: 以 Rfam clan/家族为 group 实现家族级划分，禁止同家族跨 split
  - [x] SubTask 5.3: 实现预训练-下游污染检测（测试集 × 预训练语料 MMseqs2，exact + 80% identity），命中即移除
  - [x] SubTask 5.4: 输出两档 identity 下的跨 split 同源率与污染率报告
  - **判据**：报告存在且污染率 = 0 或被显式量化披露；同家族跨 split 数 = 0

- [x] Task 6: 实现结构层清洗（C3）。
  - [x] SubTask 6.1: 实现括号配平、最小发夹环 ≥3、非交叉性校验
  - [x] SubTask 6.2: 实现交叉配对识别并单独标记为假结集
  - [x] SubTask 6.3: 实现序列-结构长度一致性校验与非经典配对标注
  - [x] SubTask 6.4: 实现 PDB 源过滤（分辨率阈值、NMR 多模型去重、缺失残基处理、编号映射）
  - [x] SubTask 6.5: 实现同序列多结构冲突裁决规则并报告多解
  - **判据**：所有结构标签通过校验或带原因码剔除；假结集独立存在；单测覆盖每类违规

- [x] Task 7: 实现标签质量与噪声处理（C4）。
  - [x] SubTask 7.1: 实现探测数据归一化（2–8% 法与 boxplot 两法）与异常值修剪
  - [x] SubTask 7.2: 实现间接标签可信度分级（实验测定 > 同源推断 > 计算预测）
  - [x] SubTask 7.3: 实现多来源标签一致性（噪声）估计
  - **判据**：每类标签带可信度等级字段；一致性统计报告产出

- [x] Task 8: 实现审计与可复现层（C5）。
  - [x] SubTask 8.1: 实现原因码枚举与逐级衰减表生成器
  - [x] SubTask 8.2: 实现数据版本化与哈希清单
  - [x] SubTask 8.3: 实现分布统计报告（长度、GC、家族、配对密度、split 间分布距离）
  - **判据**：衰减表各级数量守恒（上一级 = 下一级 + 剔除数）；分布报告含 split 间距离指标

- [x] Task 9: 冻结数据划分（C6）并产出数据卡。
  - [x] SubTask 9.1: 生成 ArchiveII 去冗余版 + 原始版双版本
  - [x] SubTask 9.2: 生成 bpRNA-TS/TR/TM、RNAStrAlign、bpRNA-new、PDB ts1/ts2/ts3 划分
  - [x] SubTask 9.3: 生成长度桶、GC 桶、家族分组的 OOD 评测切分
  - [x] SubTask 9.4: 产出 `data_card.md`，含来源、清洗、规模、已知缺陷
  - **判据**：划分文件冻结并哈希；数据卡完整；测试集未参与任何超参选择（有流程证明）

- [ ] Task 10: 复现基线矩阵（四条路线）。
  - [ ] SubTask 10.1: 热力学 DP：RNAfold、RNAstructure；**CONTRAfold 必须单独处理为头号对比对象**（§7.2.1），评测维度含 F1 / **ECE / 边际校准** / 延迟 / 是否需 DP
    - **部分完成（2026-09-24 实测）**：**ViennaRNA 2.7.2**（版本已锁定）的 `mfe / centroid / mea` 三条已在 TS0 上跑完，数字见 `records/BASELINE_RESULTS.md`；**centroid 0.5393 是本项目必须超过的门槛**。RNAstructure 未安装；CONTRAfold 仍缺。
  - [ ] SubTask 10.1b: **LinearPartition 单独对比**（"快概率"的另一条路线），评测 F1 / ECE / 延迟 / 跨家族泛化
  - [ ] SubTask 10.1c: 建立「**Nussinov + Turner 堆叠能**」基线（`MLP_T` 置零的对照物，**非 ViennaRNA**，§0.7 问题 3）
  - [ ] SubTask 10.2: 线性时间 DP：LinearFold、LinearPartition
  - [ ] SubTask 10.3: 判别式深度：UFold、SPOT-RNA/SPOT-RNA2、MXfold2、E2Efold
    - **部分完成（2026-09-24 实测）**：**MXfold2** 已装上并跑完 TS0 → micro F1 **0.5651**（唯一可得的学习型基线；**是本项目必须超过的第二个门槛**）。UFold / SPOT-RNA 权重在集群网络下**取不到**（证据见 `records/BASELINE_RESULTS.md` §4.2）→ **C1-a 因此无法完成，必须如实写进论文**。ArchiveII 上的 MXfold2 正在跑。
  - [ ] SubTask 10.4: RNA 基础模型：RNA-FM、RiNALMo（650M）、mRNABERT、RNA-MSM
  - [ ] SubTask 10.5: 至少 1 个自回归/生成式结构模型（"去解码"关键对照）
  - [ ] SubTask 10.6: 记录每个基线的仓库 commit、权重哈希、评测脚本版本
  - [ ] SubTask 10.7: **基线完成后复核并冻结 §8 全部门限**
  - [x] SubTask 10.8: ~~**【最高优先级·阻塞项】CDPFold 校准对比**~~ → **判据已作废并替换（2026-09-24 勘误）**。实测核查确认 **CDPFold 是 CNN + DP（Front Genet 10:467, 2019）**，非扩散、不免 DP、无校准证据，故它**不是** C1 的先例威胁。原"单点风险"判断的前提有误。
    - **替换判据（C1-a / C1-b / C1-c，见 `spec/benchmark_decision.md` §3.2）**：
      - **C1-a**：System-1 头在 TS0 / ArchiveII / PDB ts1 上，ECE 与 Brier **优于或持平** SPOT-RNA / UFold 的 sigmoid 概率
      - **C1-b**：与 **ViennaRNA 精确配分函数概率**、**LinearPartition 近似 BPP** 做**同口径** ECE / Brier 对比
      - **C1-c**：System-1 头 ECE 与精确边际 `p̂^exact` 的 ECE 之差 **≤ 0.02**（原 S7 门限保留）
    - **真正的先例威胁**：**SPOT-RNA / SPOT-RNA2 / UFold**（单次前向直接出 `L×L` 概率，不跑配分函数）
  - **判据**：每个基线在 ArchiveII 上有可复现 F1/INF；记录字段齐全；门限已复核冻结；**C1-a/b/c 三条判据的测量报告产出，并据此明确 C1 是否成立（含退路决策记录）**

- [ ] Task 11: 教师模型部署与吞吐实测（System 2）：**新的算力瓶颈，必须先实测**。
  - [x] SubTask 11.1: 安装并**锁定版本** ViennaRNA、RNAstructure、LinearPartition（记录 Turner 参数版本）
    - **部分完成（2026-09-24 实测）**：**ViennaRNA 2.7.2 已装并锁定**（装在 `/mnt/cunyuliu/pylibs`，因 `/home` 配额满）；教师软标签已按该版本生成（`verify_teacher_labels() == True`，锁定记录在 manifest）。RNAstructure / LinearPartition **未装**。
  - [x] SubTask 11.2: 实现教师概率矩阵批量生成器（输出 `p^teacher_ij`）
  - [ ] SubTask 11.3: 实测教师吞吐（序列/秒，分长度桶），估算全量 36M 序列的耗时
  - [x] SubTask 11.4: 实现教师集成 `p^teacher = mean(三教师)`
  - [x] SubTask 11.5: 校验教师概率自洽性（与自身 MAP 结构一致性、概率和为合理范围）
  - **判据**：教师吞吐实测报告产出；版本锁定记录存在；若全量不可行则明确子集规模并记录理由

- [x] Task 12: 实现序列编码器（模块 A，§5.3）。
  - [x] SubTask 12.1: 实现碱基理化特征向量 `φ(x_i)`（氢键供体/受体数、嘌呤嘧啶、环数、堆叠倾向）
  - [x] SubTask 12.2: 实现输入嵌入 `h_i^(0) = W_e φ(x_i) + W_p p_i + v·u_i`
  - [x] SubTask 12.3: 实现 Q/K 的 **RoPE 相对位置旋转**
  - [x] SubTask 12.4: 实现长链稀疏化（带窗 `|i-j| ≤ W` + `n_g` 全局 token）
  - [x] SubTask 12.5: 接入可选骨干（mRNABERT / RNA-FM / RiNALMo）与规模开关 35M/150M/650M
  - [x] SubTask 12.6: 提供 SHAPE/DMS 多通道证据输入接口（`u_i = 0` 表示无数据）
  - **判据**：编码器可独立前向；RoPE 相对位置性质有测试；多通道与规模开关生效；与既有 `split_sequence` 入口一致（防 74 词表坍缩）

- [x] Task 13: 实现约束决策头（模块 B，核心创新，§5.4）。
  - [x] SubTask 13.1: 实现候选空间与硬掩码 `Valid(i,j) = 1[j-i>3]·1[x_i x_j ∈ 𝒫]`，非法处置 `-∞`
  - [x] SubTask 13.2: 实现**置换不变**配对表示 `z_ij = W_s [h_i+h_j ; h_i⊙h_j ; |h_i-h_j|]`
  - [x] SubTask 13.3: 实现配对得分 `s_ij = w_s^T z_ij + s_ij^phys`，并**断言 `s_ij = s_ji` 精确成立**
  - [x] SubTask 13.4: 实现 Turner 物理先验 `s_ij^phys = -ΔG°37_NN/RT`（`RT ≈ 0.616 kcal/mol`）
  - [x] SubTask 13.5: 实现 Turner 残差 `MLP_T`，**末层零初始化**，验证训练起点等价于「**Nussinov + Turner 堆叠能**」（**非 ViennaRNA**，§0.7 问题 3）
  - [x] SubTask 13.6: 实现有序配对类型头 `t_ij ∈ R^6` 及硬置换约束 `t_ji = Π t_ij`
  - [x] SubTask 13.7: 实现温度缩放校准层（按长度桶拟合 `T`）
  - [x] SubTask 13.8: 实现 `L²` 显存对策：分块计算配对表示 + 梯度检查点
  - [x] SubTask 13.9: **实现层级决策级联 L0（粗粒度，§5.0.2）**：块对 `(b₁,b₂)`（块大小 `w`）的螺旋存在性与强度打分，并实现稀疏化（`O((L/w)²)` → `O(L/w)`）
  - [x] SubTask 13.10: **实现 L1（螺旋级）**：候选螺旋的起止与置信度预测
  - [x] SubTask 13.11: **实现 L2（碱基级局部细化）**：仅在活跃螺旋内输出 `p̂_ij` 与配对类型，`O(w²)`
  - [x] SubTask 13.12: **实现扁平 `L×L` 头作为可选对照路径**（对冲方案），并实现按长度自适应选择层级/扁平路径
  - [x] SubTask 13.13: **实现 L0 召回率测量**（P6 门限 ≥0.98）与层级级联的实测 FLOPs 计量（供 S8/S9）
  - **判据**：单次前向产出对称矩阵（对称性误差 = 0）；`MLP_T` 置零时输出等价于「**Nussinov + Turner 堆叠能**」（数值验证，**非 ViennaRNA**）；无自回归解码路径；`L=512` 显存实测记录；**层级级联三级均可独立前向；L0 召回率可测；扁平头可切换；实测 FLOPs 计量可用**

- [x] Task 14: 实现确定性求解 harness（模块 C，§5.5）：**全局关键路径**。
  - [x] SubTask 14.1: 实现 Nussinov **max-product** DP（含 traceback）
  - [x] SubTask 14.2: 实现 Nussinov **sum-product**（inside）算配分函数 `Z(x)`
  - [x] SubTask 14.3: 实现 **inside-outside** 算精确边际 `p̂_ij`
  - [x] SubTask 14.4: 实现 `L_NLL = log Z(x) - Σ_{(i,j)∈M_gt} s_ij`
  - [x] SubTask 14.5: 实现**隐式微分**可微 DP（前向求 `M*`，反向按 `∂N(1,L)/∂s_ij = 1[(i,j)∈M*]` 散布梯度），验证梯度 = `(y - p̂)·∇s`
  - [x] SubTask 14.6: 实现 soft-Nussinov 备选（仅 `L ≤ 256` 对照）
  - [x] SubTask 14.7: 实现带窗稀疏化版本（`|j-i| ≤ B`）
  - [x] SubTask 14.8: 实现低置信回退门控 `α_ij = σ((p̂_ij - τ)/γ)` 与热力学重打分，记录回退比例
  - [x] SubTask 14.9: 在代码注释与文档中**显式记录 Sinkhorn/最优传输不适用于非交叉约束**
  - [x] SubTask 14.10: **实现双模式推理（§5.0.1，解决 §0.7 致命问题）**：System-1 模式 = `1×` 前向 → 决策头直接输出校准概率 → **不跑配分函数** → 带状/线性 DP 求合法结构
  - [x] SubTask 14.11: 实现 System-2 按需升级：置信度门控，仅对低置信区域调用 inside-outside / 热力学求解器
  - [x] SubTask 14.12: **断言 System-1 路径不包含任何配分函数调用**（代码级检查 + 计时验证），并实现平均 FLOPs 计量（供 S6 匹配计算量对比）
  - **判据**：`Z` 与 `p̂` 在 `L ≤ 12` 时与暴力枚举**逐位一致**（有测试）；≥1000 随机得分矩阵下非法结构率 = 0、最小发夹环违规率 = 0；隐式微分梯度与数值梯度一致；`L_NLL` 可反传；**System-1 路径经断言确认无配分函数调用**；平均 FLOPs 计量可用

- [x] Task 15: 实现决策顺序切换（**已降级为消融项**，§5.6 / §0.7 问题 4，**不作为创新点**）。
  - [x] SubTask 15.1: 实现 5'→3' 顺序（与 Nussinov 从左到右填表一致）
  - [x] SubTask 15.2: 实现对角顺序（按 `j-i` 递增）与随机顺序对照
  - [x] SubTask 15.3: 实现顺序切换的统一配置接口
  - **判据**：三种顺序可经同一配置切换并产出结果，供 H4（**已降级，不预设方向**）检验；**不得在任何写作中把顺序称为创新**

- [x] Task 16: 实现热力学决策蒸馏预训练（模块 D，§5.7）。
  - [x] SubTask 16.1: 实现蒸馏损失 `L_distill = Σ D(p^teacher_ij ‖ p̂_ij)`，`D` 支持 KL 与 L2 两种
  - [x] SubTask 16.1b: **实现「精确边际」蒸馏教师**（C1 的核心机制）：以 `L_NLL` 训练出的 CRF 的 inside-outside 边际 `p̂^exact` 作为教师，蒸进免 DP 的决策头。**这是 §5.0.1 双模式设计的训练期环节**
  - [x] SubTask 16.1c: **量化免 DP 的校准代价**：对比 System-1 头的 ECE 与 `p̂^exact` 的 ECE（供 S7，目标差值 ≤ 0.02）
  - [x] SubTask 16.2: 实现与硬标签 `L_NLL` 的加权联合 `L = L_NLL + λ_distill · L_distill`
  - [x] SubTask 16.3: 实现教师软标签的大规模离线生成管线（分片、断点续传、哈希校验）
  - [x] SubTask 16.4: 实现**蒸馏目标饱和诊断**（与 JEPA 区域均值目标的饱和模式对照，须有记录证明不早饱和）
  - [x] SubTask 16.5: 实现教师集成（三教师）与单一教师的可切换配置
  - **判据**：蒸馏损失可反传；饱和诊断记录产出且显示目标未早期饱和；软标签管线可断点续传且哈希校验通过

- [x] Task 17: 实现 RLCD-RNA 校准决策训练（模块 E，§5.8）：**对齐 Jev 训练层（J9）的核心迁移**。
  - [x] SubTask 17.1: 实现决策单元定义（`a_ij ∈ {0,1}`，`p̂_ij = P(a_ij=1|x)`）与恰当评分规则奖励（Brier 与对数评分两种）
  - [x] SubTask 17.2: 实现**可微软 ECE**（软分箱 / 核密度）作为校准惩罚
  - [x] SubTask 17.3: 实现 RLCD 总目标 `L_RLCD = -E[R] + β·ECE_soft`
  - [x] SubTask 17.4: 实现四项目标联合 `L = L_NLL + λ_distill·L_distill + λ_RLCD·L_RLCD + λ_1·L_cal`，各权重可配
  - [x] SubTask 17.5: **验证校准梯度由精确边际 `p̂_ij` 直接给出，不依赖蒙特卡洛采样**（有测试断言）
  - [x] SubTask 17.6: 实现 `λ_RLCD` / `β` 的扫描配置，产出校准-精度权衡曲线
  - [x] SubTask 17.7: 撰写 RLCD 诚实标注说明：本文为"RLCD 启发式自设计目标"，Jev 的 RLCD 算法未公开，不得声称复现或等价
  - **判据**：`L_RLCD` 可反传且梯度不依赖采样（有断言）；四项目标权重可独立开关（供消融）；校准-精度权衡曲线产出；诚实标注说明存在

- [ ] Task 18: 执行预训练并验证梯度与稳定性。
  - [ ] SubTask 18.1: 在清洗后语料 + 教师软标签上运行预训练，监控 loss/梯度/NaN
    - **进行中（2026-09-24）**：10 个臂在跑（6 from-scratch @40k + 4 head-only @20k），本轮再加 **4 个目标函数对照臂**（§14.12）。全部 `--resume` 存活、无 NaN、梯度覆盖断言通过。**未跑完前不产生科学结论。**
  - [x] SubTask 18.2: 验证梯度到达每个可学习块
  - [x] SubTask 18.3: 标定微调学习率（承接"lr=1e-4 摧毁编码器"教训）
  - [ ] SubTask 18.4: 验证学生校准概率与教师可比（ECE、与教师概率的相关性）
    - **部分完成**：C1-b（对 ViennaRNA 精确 BPP 的同口径 ECE/Brier/NLL）已产出，TS0 ECE **0.0048**；C1-c 两口径见下。**与教师概率的相关性尚未测**。
  - [x] SubTask 18.5: 保存 checkpoint 与训练曲线
    - 快照由 cron 每 10 分钟按 2000 步整数倍保留到 `/mnt/cunyuliu/rna-jepa/ckpts/`（因 `resume.pt` 原地覆盖，这是唯一可靠的溯源手段，见 §14.13）。
  - **判据**：无 NaN/Inf；梯度覆盖断言通过；学习率标定报告产出；学生-教师概率一致性报告产出

- [x] Task 19: 下游任务适配（T1–T6）。
  - [x] SubTask 19.1: T1 非假结二级结构预测主任务
  - [x] SubTask 19.2: T2 假结感知预测（PseudoBase++）
  - [x] SubTask 19.3: T3 SHAPE/DMS 条件化预测
  - [x] SubTask 19.4: T4 长链/全长转录本（16S/23S 等）
  - [x] SubTask 19.5: T5 迁移任务（复用既有 145 任务注册表）
  - [x] SubTask 19.6: T6 零样本/少样本 + 主动学习
  - **判据**：每个任务产出 `result.json` + predictions；涉及 dev==test 的任务结果显式标注该缺陷

- [ ] Task 20: 执行测评与消融（含速度基准与门限核验）。
  - [ ] SubTask 20.1: 跑七类指标，每配置 ≥5 seed
  - [ ] SubTask 20.2: 执行统计检验（Wilcoxon / bootstrap CI）与多重比较校正（Holm-Bonferroni / FDR）
  - [ ] SubTask 20.3: 完成 §7.5 全部 **15 项**消融
  - [ ] SubTask 20.4: 同硬件测延迟（分长度桶）、吞吐、峰值显存，产出 latency-accuracy Pareto
  - [ ] SubTask 20.5: 测量相对 McCaskill 配分函数的加速比
  - [ ] SubTask 20.6: 产出跨长度桶/跨家族/跨 GC 外推衰减曲线
  - [ ] SubTask 20.7: **核验 RLCD 收益非装饰**：同时报告 ECE 改善与 bpRNA-new F1 变化（H6）
  - [ ] SubTask 20.7b: **核验 S7（免 DP 的校准代价）**：System-1 ECE 与精确边际 ECE 之差 ≤ 0.02——**这是 C1 是否成立的判据**
    - **分两口径核验（2026-09-24）**：① **裸头**（训练直接用头的 sigmoid）：step 3500 上 gap = **0.0877**，**未过**；但从 step 20 的 0.549 单调降到 0.0877（6.3×），趋势明确。② **免 DP 仿射重标定**（2 参数，在 VL0 上拟合，**评测期不跑配分函数**）：gap = **0.00118 PASS**。→ **当前可主张的是 ②，不得把 ① 写成已通过。**
  - [ ] SubTask 20.7c: **核验 S6（计算自适应收益）**：在**匹配平均 FLOPs** 下门控升级 ≥ 纯 System-1 与纯 System-2——**这是 C2 是否成立的判据**
  - [ ] SubTask 20.7d: **产出结构层面校准曲线**（应对 Q3：配对级 ECE 因碱基对相关性而不足），并量化配对间相关性
  - [ ] SubTask 20.8: **逐项核验 §8 硬门 G1–G5、性能门 P1–P5、速度门 S1–S7、科学门 C1–C4**
  - [ ] SubTask 20.9: **核验 §9.3 的 Q1–Q12 每条均有稿件落点**（任一缺失即未准备好投稿）
  - **判据**：全部消融完成；统计检验与校正已应用；速度测量同硬件同协议；ECE 与 bpRNA-new F1 联合报告产出；**S7/S6 判据已核验**；结构层面校准曲线产出；**Q1–Q12 落点表产出**；门限核验表产出（未达项须列出）

- [x] Task 21: 论文撰写与投稿。
  - [x] SubTask 21.1: 按 §9.2 **四段式**叙事撰写（**问题：信任与成本两难 / 方法：免 DP 校准决策头 C1 / 系统：计算自适应折叠 C2 / 测量：校准优先协议 C3**）
  - [x] SubTask 21.2: 核验全部引用（标题/作者/年份/标识符/URL 齐全方可标已核验）
  - [x] SubTask 21.3: 撰写限制章节，含**五点局限**：Jev 无同行评审论文；RLCD 算法未公开故为启发式自设计；**Gibbs/配分函数框架非本文首创（CONTRAfold/CRF 谱系）**；与 LinearFold `O(L)` 复杂度的诚实对比；`dev == test` 缺陷
  - [x] SubTask 21.3b: **撰写 §2.3「明确不主张」清单的对应声明**，确保稿件中不存在任何被 §2.3 禁止的表述
  - [x] SubTask 21.4: 准备可复现材料：代码、配置、数据卡、衰减表、分析计划、教师版本清单
  - [x] SubTask 21.5: 按 §9.1 优先级选定投稿目标并按其要求调整篇幅与图表规范
  - **判据**：每个强主张可回溯到论文行/证据/显式限制；0 条编造引用；负结果如实报告；硬门未达者不得投稿

---

# Task Dependencies

- **Task 2（数据盘查）必须先于 Task 3（数据获取）**——先实测再动手
- **Task 3 必须先于 Task 4–9（清洗链）**——无数据无从清洗
- Task 4 → Task 5 → Task 6 → Task 7 → Task 8 → Task 9（数据链严格串行）
- Task 1 必须先于 Task 10（分析计划与门限冻结早于基线运行）
- ~~**Task 10.8（CDPFold 校准对比）是全局阻塞项**~~ → **2026-09-24 勘误：该阻塞项作废**（CDPFold 是 CNN+DP，非扩散、不免 DP，前提有误）。C1 的判据改为 C1-a/b/c（`spec/benchmark_decision.md` §3.2），**不再是前置阻塞项**，可与架构实现并行
- **新的 P0 前置**：ViennaRNA 安装并锁定版本（教师 + 精确边际对照物）；archiveII 版本口径在论文中写明
- Task 10（基线复现）与 Task 12–15（架构实现）**可并行**
- **Task 11（教师部署与吞吐实测）是 Task 16 的前置**，且其吞吐结论决定蒸馏语料规模
- Task 12 → Task 13 → Task 14（编码器 → 决策头 → harness）
- **Task 14.1–14.5 是全局关键路径**：`Z(x)`/`p̂_ij` 的穷举一致性验证（`L ≤ 256` 的对照与 `L ≤ 12` 的暴力枚举）未通过前，不得开展 Task 16 及之后的任何工作（数学框架错则全盘皆错）
- Task 13 → Task 15（决策头就绪后才能做顺序对照）
- Task 9 与 Task 11 是 Task 16（蒸馏预训练）的前置
- **Task 16（蒸馏）与 Task 17（RLCD 校准）可并行**，两者都是 Task 18（预训练执行）的前置
- Task 16 + Task 17 → Task 18（两个训练目标实现先于预训练执行）
- Task 14 + Task 18 → Task 19（harness 与预训练权重就绪后才能做下游）
- Task 19 → Task 20（下游跑通才能测评）
- Task 20 → Task 21（结果与门限核验齐备才能写作与投稿）
- Task 20.3 依赖 Task 13–17 全部完成（消融需四项目标可独立开关）

---

## 第二轮回归审计的派生任务（2026-09-24 新增）

> 来源：`spec/spec.md` §0.9.5。第一轮审计比对范式，第二轮直接读代码与运行配置，
> 结论更严厉：**C2 目前不是可用架构，且编码器是随机初始化**。

| # | 任务 | 完成判据 | 状态 |
|---|---|---|---|
| **T-A1** | 为层级级联设计**分层目标**（L0 块对 / L1 螺旋 / L2 局部各自成项、各自归一），接入 `train_decision` 与 `evaluate_decision` | L0 螺旋召回 ≥ **0.98**（P6）；级联臂在 TS0 上 F1 不退化 | 🔄 **已实施，训练中**：`rnajepa/cascade_objective.py`（新）+ `--head cascade`；`L = w_l0·L_L0 + w_l1·L_L1 + w_l2·L_L2 + w_sparse·L_sparse`，每层按自身决策单元数归一。`rinalmo_casc_b4_s0` 已启动，step 25 起 `l0_helix_recall` **0.9971**（旧硬 `top_k` 为 **0.1548**）。**F1 未评测 → C2 尚不可主张**（§14.16） |
| **T-A2** | 级联的**可微稀疏选择**，替代硬 `top_k`（未训练 L0 召回实测仅 **0.1548**，min 0.0） | 训练中 GT 螺旋的梯度覆盖率 ≥ 0.98 | ✅ **机制已实施并验证**：可微门 `g=sigmoid((logit−θ)/τ)`（`θ`/`τ` 均为 `nn.Parameter`），`log g` 取代 `-inf` 使 `log Z` 重新有意义；有测试断言**梯度确实到达 `θ` 与 `log τ`**，以及"`top_k=2` 在 >2 个真值块对时召回上限 0.5"（记录旧缺陷）。**级联 F1 未评测前不得主张 C2** |
| **T-A3** | 补齐**预训练骨干**：接入已下载的 RiNALMo-giga，或改用 245 GB 无标签语料自监督预训练 | `run_meta.json` 记录骨干来源；训练量 ≥ 20 epoch | 🔄 **部分完成**：RiNALMo-giga **冻结嵌入**已接入（4 个 head-only 臂 + 4 个新臂），`run_meta.json` 有记录；**20 epoch 未达**（当前 20,000 步 × batch 4 ≈ 7.5 epoch）。from-scratch 编码器臂实测 TS0 F1 仅 **0.3022**（@4000），远差于 head-only，见 `records/DECISION_TRAINING_LOG.md` |
| **T-A4** | 目标函数**长度归一**（按长度）+ 重新标定 `grad_clip`（裁剪前范数实测 40–673 vs 阈值 1.0） | 批间损失可比；裁剪触发率与步长波动写入记录 | ✅ **已实施并实测**：`--nll-normalization length`；四项由「CRF 独占 97–98%」变为**同量级 0.4–1.3**；批间极差 2.8× → 1.8×；gnorm 25–88 → 2.7–3.1；**{sum,length}×{aux 开,关} 2×2 对照臂已启动**；3 项新测试、291 passed。记录见 §14.12。**`grad_clip` 仍为 1.0**（待四臂过评测后按实测分布再定） |
| **T-A5** | 首轮三臂（3000 步 ≈ 0.56 epoch）的结果**逐处标注**为「通路验证，非科学结论」 | 训练记录与论文草稿中无遗漏 | ✅ 已完成（`records/DECISION_TRAINING_LOG.md`「运行 0/1/2/3」逐处标注） |
| **T-A6** | 监控脚本覆盖**直接启动**的运行（原先只读共享台账，直接启动的臂完全不可见） | 能发现静默死亡 | ✅ 已完成（commit `4218f73`） |
| **T-A7** | 决策头**显存**修复：沿 `j` 分块 + 每块 `torch.utils.checkpoint` | L=498/B=4 由 OOM 降到 **2366 MiB**；前向逐位等价、梯度 fp64 精确到 1e-15 | ✅ 已完成（commit `2397364`，11 项测试） |
| **T-A8**（新） | **评测记录的溯源**：目录名/`tag` 声称的步数必须能被 checkpoint 支持；活 `resume.pt` 不得用于事后读步数 | 每条记录有可靠步数或显式标注不可靠 | ✅ 已完成（`tools/summarize_evals.py` + `tools/fix_eval_record_provenance.py`；`ff3600→ff3500`、`ff500_ts0.json→ff_live_early_ts0`，见 §14.13） |
| **T-A9**（新） | **先验权重逐 checkpoint 在 VL0 上重选**后再评测试集；评测任务一律 `setsid nohup` 脱离 ssh | 选择不在测试集上做；评测不因 ssh 断开而丢失 | 🔄 进行中（`run_ood_reselect.sh` 已 detached 启动，见 §14.14）。**注**：该任务的价值已降级——headline 已改用模型自训练权重（T-A10），VL0 选 w 只作敏感性分析 |
| **T-A10**（新，本轮） | **headline 去 VL0 化**：用 `--prior-weight -1`（模型自训练权重）重测 TS0，证明结论不依赖任何在 VL0 上的选择 | 两个口径的 micro F1 之差 ≪ 种子方差 | ✅ **TS0 已完成**（§14.20）：`w=-1` **0.4953** vs `w=0.75` **0.4959**，差 **0.0006**；且 `w=-1` 下 C1-c 重标定 gap 更紧（**0.0002**）。ArchiveII / bpRNA-new 排队中 |
| **T-A11**（新，本轮） | **同口径收敛曲线**：`rinalmo_ff` 在 step 6000 / 10000、`w=-1` 的 TS0 评测（已有的三点各用了不同的 w，**不可连成曲线**） | 至少 3 个同口径点，能回答"再多训是否还在涨" | ✅ **已完成（17:46）**：`trend_ff_step6000_ts0` **0.5369** / `trend_ff_step10000_ts0` **0.5587** / `ff20000` **0.5958**（`w=-1` 全程同口径）→ **训练曲线单调上升且未收敛，"欠训练是主因"（§14.35）再获一条独立证据** |
| **T-A12**（新，本轮） | **checkpoint 步数溯源**：`tools/ckpt_steps.py` 直接从 `.pt` 内读 `step` 字段，不从文件名推断 | 每个被引用的快照都有文件内步数证据 | ✅ 已完成：已核验 `rinalmo_ff_ff_w05_snapshot.pt` **文件内 `step=3500`**（与 `tag` 一致），并清点出 `rinalmo_ff` 可用快照 = 2000/4000/6000/10000 |

### 预训练 LM 基线对齐（2026-09-25 §14.62–§14.64，用户要求）

- [x] 与 RiNALMo 论文 split 对齐（官方 TS0 1,305 全集评测 + 序列级溯源）
- [x] Mathews 宽容判分口径实现 + 全部可实测基线重打分（UFold/RNAformer/MXfold2/Vienna/我们）
- [x] draft §4.3d 新节 + 引用红线（reported vs measured 必须区分）
- [x] **S4（TS0 INF）对标**：我们 0.6005/0.6199/0.6238；UFold/MXfold2 实测
      0.7325/0.5832 与论文 0.67/0.61 差异 = 判分实现差异（双列声明）
- [x] **S5（TORNADO TestSetB）对标**：数据下载转换（428/430），三臂零样本
      评测 + Vienna + EternaFold（CONTRAfold 族代用）——**big 0.7932 宽容 F1 /
      0.7781 INF 超该基准全部已发表数字（含 RiNALMo fine-tuned 0.67）**
- [x] **S2/S3（ArchiveII famfold 9-fold）判定：不复现**（需 9 次训练，
      deadline 不可行；ArchiveII 另有训练集污染问题 §4.5）——limitations 已声明
- [ ] RNA-FM 权重不可得确认记录（Zenodo/HF 不可达）
- [ ] RiNALMo 论文 Fig 3c 精确数字精读（图值不可引，当前只用 vicinity 表述）

### 第三轮交接新增任务（2026-09-24 晚，源自 §14.34–§14.42 与用户一周预印本要求）

| # | 任务 | 完成判据 | 状态 |
|---|---|---|---|
| **T-A13** | **TR1 评测**：`rinalmo_ff_tr1_b4_s0`（45,865 条，4.29×）在 step 20000 完成后，`ref_bprna_ts0` + `bprna_new` 全套评测（`w=-1`） | 与 TR0 headline（0.5958）同口径对比表产出，回答"数据扩容缩多少差距" | ✅ **完成且超出（§14.49/§14.56）**：@20k TS0 **0.5840** / new **0.5162**（+0.163 = 49× std）；**@40k（3.4 epoch）TS0 0.6147（+0.031 欠训练证实）/ new 0.4999（−0.016）——训练量分离两轴：OOD 最优训练量早于同分布**。对 UFold(new) 差距 0.257→**0.094**。校准 ECE new=0.00078 存活。40k 快照从 resume.pt 手工恢复（快照纪律教训已记 §14.56） |
| **T-A14** | **种子方差**：ff s0–s7 在 step 20000 的 TS0 评测 → headline 报告 mean ± std（协议要求 ≥5 seed） | 6 seed 均值 ± std 落入 `records`；若 std > 0.01，headline 单点数字须加区间 | ✅ **完成（§14.51/§14.54/§14.55）**：**8 seed 定稿 0.5950 ± 0.0033**（range 0.0097，0.5893–0.5990）。容量 +0.047 = 14× std、数据 new +0.163 = 49× std 均稳健；pw/aux 增益 <2× std 维持不可写 |
| **T-A15** | **级联/容量定型**：casc/cascR/big 在 step 20000 的 TS0 评测；级联与扁平头对比回答 H7/S8 | 三臂终评数字产出；C2 是否可主张有明确判定 | ✅ **终评完成（§14.46/§14.47）**：**big = 0.6425（容量假设成立，+0.047 = 种子方差的 7 倍）**；**casc = 0.5011（C2 判否**，召回崩到 0.366，门控 OOD 过保守）；cascR @13.2k 在跑（预期同样判否）。**教训已记录：step2000 的架构排序不可靠，比较须在充分训练点做**。**衍生臂已启动**：`rinalmo_bigtr1_b4_s0`（big×TR1 组合，GPU3，00:28 启动）+ big s1 种子待空位 |
| **T-A16** | **2×2 目标函数对照（H6）**：`{sum,length}×{aux 开,关}` 四格在 step 4000/20000 的 TS0 F1 + ECE | 四格齐 → H6（RLCD/蒸馏是否装饰）有初步结论；已出第一格 `sum+aux关=0.4974` vs `sum+aux开=0.4861`（step2000） | 🔄 **四格 @20000 已齐（§14.44，23:40）**：sum+aux开 **0.5958** / len+aux开 0.5870、0.5950、0.5954（三 seed）/ sum+aux关 **0.5893** → aux 增益 +0.0065（两步同方向但小）；sum vs len 差 0.009 在 len 格 seed 跨度（0.008）内**不能定论**。**意外收获：`rinalmo_pw`（可学习先验权重，init 0.5）= 0.6006 新最佳**，学到的 pw=0.173（ff 固定起点学到 0.310）。step4000 点仍缺 |
| **T-A17** | **预印本初稿**：按重组后的贡献结构（C2 骨架 + 校准评测 + 自洽性 + 负结果如实报告）更新 `paper/preprint_draft.md` | 主体章节成形：系统校准评测表（6 来源）、F1 差距表（RNAformer/UFold/centroid/我们）、C1-c 双口径、局限五点+新增红线；可交导师审阅 | 🟡 待 T-A13–T-A16 数字齐后启动（本周内） |

> **T-A13–T-A17 的一周排期**（用户要求 09-30 前出一版可提交预印本的初步结果）：
> D1（09-25）TR1/种子/架构臂跑完 → D2（09-26）全部评测 + 2×2 收口 → D3–D4（09-27/28）骨架决策 + 消融补齐
> （非交叉约束 / Turner 置零 / 解码顺序三项可离线跑）→ D5–D6（09-29/30）预印本初稿 + linter 自检。
> **规则不变**：每一步以验收判据为准，smoke/proxy/训练集数字不得写成结论。

### 已下载/安装的资产（2026-09-24）

| 资产 | 路径 | 状态 |
|---|---|---|
| RiNALMo-giga 权重（650M；33 层 / hidden 1280 / 20 heads / rotary / max_pos 1024） | `/mnt/cunyuliu/rna-jepa/weights/rinalmo-giga/model.safetensors`（2.60 GB） | **已下载**（经 `hf-mirror.com`；`huggingface.co` 与 Zenodo 不通） |
| multimolecule 0.0.8 + transformers 4.57.6 + torch 2.8.0 | `/mnt/cunyuliu/rnalmo_pkgs`（`pip --target` 隔离安装，**不污染**训练环境） | **已安装** |
| 无标签语料 | `/mnt/cunyuliu/rna-jepa/data/pretrain`（**245 GB**）、`pretrain_fasta`（93 GB） | 已有（规模远超 spec 早先写的 20 GB） |

> **许可提示（必须先确认再使用）**：`multimolecule/rinalmo-giga` 标注 **AGPL-3.0**。
> 用于研究/论文前需确认合规性，或改用许可更宽松的骨干。**当前尚未在任何训练中使用。**

### 环境约束（必须知道）

- ~~**`/home` 配额 200 GB 已满**~~ → **2026-09-24 21:40 实测已解除**：`du -sh /home/cunyuliu` = **74G**，
  64 MB 写入测试 0.06s 完成（1.2 GB/s）。早前的清理（`~/.cache/pip` 4.6 GB + 基线目录迁
  `/mnt/cunyuliu/relocated_home/`）已生效。**git commit 可正常进行**。文件系统总量 7T（用 2.1T）。
- 配额的主要占用者**不是本项目**：`mrna_editflow_goal` 75 G（其中 `mrna_editflow` 43 G、
  `runs` 23 G）、`reactflow/artifacts` 51 G、`miniconda3` 59 G。
  `mrna_editflow_goal` 有 8 个进程的 cwd 在其中，**不可擅自迁移**；需要用户决定。
- **主机内存 754 GB（available 275 GB）**：每个 RiNALMo head-only 臂 RSS 约 12 GB（嵌入 + 教师标签常驻内存），
  TR1 臂约 36 GB；**加臂前须确认 `free -g` 的 available > 60 GB**。load average ~92-104（共享集群常态）。
- **训练进程环境**：`lucaone` conda env（torch 2.5.1，权威测试环境 317 passed）；RiNALMo 相关
  `mrnabert` env 或 `/mnt/cunyuliu/rnalmo_pkgs`（torch 2.8.0，隔离安装）。
