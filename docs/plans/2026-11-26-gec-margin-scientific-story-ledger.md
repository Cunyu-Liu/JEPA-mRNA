# GEC / structured-margin 科学叙事与训练留痕

日期：2026-11-26
分支：`scientific-story-ledger-20261126`
整合基线：`780044cc2c4c2fa3ae355edf1fb6e1d68eb44019`
范围：bpRNA model-dev；既有 VL0 为 196 条，GEC 为 TR1c cluster80 dev 9,143 条。均非外部测试、严格 OOD 或 SOTA 证据。

## 当前一句话

已完成证据支持的是“容量、SRD、stacking closure 按预注册门停止；GEC seed0 科学 FAIL；残差主要定位于非正 score-margin FN；crossover 证明上下文输入有因果作用，但真实 ensemble 并非一致有益；NLL 四臂恢复为结构参照”；项目旧 bpRNA-new 标签已证实为 `DERIVED_FROM_BPFOLD_PREDICTIONS`，structRFM/BPfold/CONTRAfold 旧 raw 结果仅作 circular-audit-only；新的同官方参考面板只支持描述性冻结协议比较；LinearPartition 19,000 nt 只有 baseline 运行回答而 GEC unsupported；**structured margin 是正在检验的候选目标，不是已成立结论，也没有任何 margin 四臂结构结果可供摘要或标题使用。**

## 时间、提交与运行快照

- 科学结果及 NLL 资格链整合到 `780044c`；capacity/SRD/stacking 的结果提交分别为 `c289384afcd801cb0509941c40e973f7c320f230`、`03f9afe430c445fffd5d0e386ba3c16970747ab8`、`dde6641db437d1d1c35ee876843d30d0a5cf6f68`。
- GEC 四臂训练与独立复评为 `bd01b4a120a8ed3b7c77690f593a8caa03fb533d`、`342a714b24998ed0c739908613f21e50866d6b56`；残差冻结审计为 `0589b2f489aa456f7a885b6d5dfe625b5de1b76b`；frozen context crossover 为 `810441e7480ea06a717097c57a49130c820c9636`，对应 PPT 第 33 页记录提交 `03f7f4e14ef906359cbd9d1b519089b6e61116f4`。FoldNAR 整合 HEAD 为 `fd497e9819e39a29887ec26698a46c91284d230b`，真实 deck 为 14 张、记录至第 33 页。第 34 页的 round71 builder 引入提交为 `1782cd64a7c6253beefe77c93634a2652f6fab5e`，render/OCR 留痕提交为 `5a7ccbd418f520eb44f02f0fdf6fbdf9f6ff238e`，仍须等待最终 hash-verified 评估。
- 本轮 FoldNAR 整合 HEAD 为 `79a2ed45879f1f4e81d6719ed54600bf7f8da807`；record-level label quality 来源提交为 `3e64df111cffaeaf21add48094434123234ea40f`。该静态 CPU 审计不访问外部 Test、不启动 GPU 任务、不修改 model-dev 或训练 IDs。
- RiNALMo-giga pretrain exposure 来源提交为 `4247d1e4a971ceb8c63e3f8338e026a23c18941b`，使用既有合法资产做静态 CPU 审计；157,227 条 population-row 独立复核通过，不训练、不推理、不下载、不新跑 homology search，也不用于模型/臂/checkpoint/阈值/超参选择。
- NLL 资格修复为 `c4bfbb8734af3535393ef424f4c1b0b39af7b081`、`e388b504d7888b8a355d33fe80150ebd650a0adc`、`ecc40d3a8ec545bf332199a935e6380b718d415b`。
- margin 数值核心、独立合同复核、显存探针与正式调度分别为 `92f0bc2e6440bcbfc682d2f8ca3e502ec86a6b48`、`55be915f6e7eda26254d87ab133624fa99cd5b8c`、`cd61c57f48cc0fe543b869b3319e4bd8b7f820f5`、`a846d90519e575a79b7db76f63692afb755e8ac6`；`a846d90` 已核对与 GitHub 分支远端一致。
- BPfold 标签协议与 official-reference 面板来源提交为 `e7ff4e76fe078178ab41d2052349627317b5a3e3`。旧项目 bpRNA-new 标签已证实 `DERIVED_FROM_BPFOLD_PREDICTIONS`，所有旧 raw/clean 派生面板仅为 circular-audit-only；新面板和冻结预测只做 CPU 复算，不训练、不推理、不选模。
- Model-dev v3 登记提交为 `302b70a228d77b0f1847489c93ead402e33510b5`，registry SHA256=`4a047b8ed6a520e00a52543a3aa2e3f595fa0a29c565753d39e9ccbaf8f4d904`；它只做 additive exposure sensitivity/final reporting，不读 margin 结果，不启动 GPU，不做新训练、推理、homology search 或 route selection。
- current independent margin evaluator 固定为 `/home/rna_jev_margin_evaluator`，HEAD `536b79e450c1ad18b67e06cb39ecb3730546221b`；本 ledger 更新只读核对该提交，不修改评估器、候选训练或调度。
- **登记排队快照**：`structured_margin_full_v1/generation_01/startup_report_v1.json` 为 `QUEUED_PROBE_MEMORY`，scheduler PID `3976280`，完成臂 0，`scientific_conclusion=null`。**本次更新实时状态**：`scheduler_state.json` 为 `QUEUED_ARM_MEMORY`，`matched_control` 与 `ensemble_context` 两臂运行，`destroy_interval_shuffle` 与 `independent_context` 两臂等待资源，完成臂仍为 0，`scientific_conclusion=null`。这里区分初始快照与后续调度；排队、探针、启动、运行或等待资源的进程都不是科学结果。本 ledger 不修改调度器、资源门或任何运行任务。
- ArchiveII/PDB/Rfam 标签谱系整合基线固定为 `/home/rna_jev_foldnar_integration`@`8bd87223edb9b5a21c1e13676b4192de54505d9f`；ArchiveII official-reference 与 full3966 CPU 恢复/三轮 clean-clone delivery 固定为 `/home/rna_jev_archive_label_lineage`@`1558cf007b8a114349b73ed3b7becd77d76ac251`；TS0/TestSetB 官方/project 双面板与三轮 clean-clone delivery 固定为 `/home/rna_jev_archive_label_lineage`@`8a2008a4cda3c43b3000f25151b80408b0bdb110`；TS0 full1305 覆盖恢复固定为 `/home/rna_jev_ts0_coverage`@`a8b1089a67a84665c49f0c228064693a53548590`；TestSetB full430 覆盖恢复固定为 `/home/rna_jev_testsetb_coverage`@`7342f26303d3acc8d61f395c040534242d0f7895`，PDB/Rfam 原始来源审计、RCSB/mmCIF/DSSR 双协议重建与 Rfam family/sequence release 谱系固定为 `/home/rna_jev_pdb_rfam_label_lineage`@`105bb55d2e959e587c85d3de7607fdc3c14c5c73`。这些是外部预测/标签的 CPU 审计与复算，不是 margin 结果，也不用于模型选择。
- 磁盘配额事故恢复代码固定为 `/home/rna_jev_margin_incident_recovery`@`442636dd660f93f1d39156ca9bbdbd107ecdbf42`，preflight SHA256=`b72b144915c9e07ecdfc701b55d9b5266df27875afe27f6b28e9fc30e26ce4dd`。恢复调度只读快照为完成臂 0、`scientific_conclusion=null`；本 ledger 不启动、停止或修改任何训练/恢复进程。
- 第二次 scheduler Errno 122 后的最终修复固定为 `/home/rna_jev_margin_incident_recovery`@`aa9f1c4a92e92d70841e33cec455734f160075c4`，本地 HEAD 与 GitHub 分支 HEAD 一致；接管证据固定为 `/home/foldnar_incident_evidence/recovery_takeover_status.json`。历史 teams scheduler state 与当前本地接管 state 分开记录，不把旧快照覆盖成新状态。
- matched 自然收敛与 ensemble 恢复只读审计固定为 `/home/rna_jev_margin_incident_recovery`@`9cc035be714ce0487593ce042aaee259d5a6cd7d`，审计 artifact 为 `/home/foldnar_incident_evidence/matched_converged_19000_audit.json`。自然收敛仅解除 matched 训练终态资格，不提前产生四臂结构评估或 margin 科学结论。

Artifact 根以下统一缩写：`F=/teams/HoulabData_1788502323/liucunyu/foldnar`，`A=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts`。

## 1. 容量扩张：单 seed/raw 信号没有跨 seed 转化为结构收益

- **Claim（仅限当前形式）**：简单增加表示/结构头容量不是稳健生产路线。
- **Phenomenon**：seed0 的 AUPRC、r@P0.8 与部分 pair/stem 曾改善，诱发了“容量不足”的解释。
- **Intervention**：`head_deep`、`repr_head` 对同 seed matched control 做 seeds 0/1/2、六条确认运行；fresh head、共享初始化、9,600 steps、VL0 196 条，并从 raw scores 独立重算。
- **Expected observation**：若容量是稳健因素，pair/stem 结构终点应在逐 seed 与三 seed 均值上按冻结门一致改善，而非仅 raw 排序改善。
- **Actual**：三 seed 均值（候选−control）中，`head_deep` micro/macro pair-F1 为 **−0.0039389391/−0.0099052807**，`repr_head` 为 **−0.0345147191/−0.0519265689**；`repr_head` micro/macro stem-F1 为 **−0.0333460085/−0.0564139810**。即使 `repr_head` micro AUPRC=**+0.1219403702**，也不能抵消结构失败。裁决 `STOP_REPRESENTATION_HEAD_CAPACITY_THREE_SEED`，不选冠军 seed/checkpoint。
- **Artifact**：`A/representation_head_mechanism_4arm_v1/capacity_freeze_audit.json`；[封存记录](2026-11-24-repr-head-capacity-freeze-audit.md)。
- **Limitation**：只否定当前容量形式与 model-dev 路线；表示、目标、优化因果未分离，不证明所有容量或新形式无效。
- **Next**：关闭该容量扩张分支，保留 raw 与结构终点不一致作为后续干预必须同时考核的约束。

## 2. SRD：求解器/stacking 包装未超过当前 MEA

- **Claim（冻结设置）**：nested SRD stacking 未优于当前 MEA，单来源分支停止。
- **Phenomenon**：raw score 与最终结构不一致，提出是否由解码器或 stacking 候选表达不足造成的问题。
- **Intervention**：冻结 `control_matched_seed0_step9600`，在同一 VL0 196 条上比较 Nussinov MAP、当前 MEA、nested canonical matching、nested SRD stacking、best-of-N；固定约束并独立重算。
- **Expected observation**：SRD 必须严格提高 micro/macro pair-F1，同时保持 precision 和 micro stem-F1；约束满足率与 fallback 不能靠失败换取收益。
- **Actual**：SRD micro/macro pair-F1=**0.800210/0.804848**，当前 MEA=**0.825713/0.805772**；SRD−MEA 分别为 **−0.025502/−0.000924**，micro stem-F1 **−0.031502**。195/196 条 SRD 与 MAP 相同；裁决 `STOP_SRD_NOT_BETTER_THAN_CURRENT_MEA_OR_CONSTRAINT_FAILURE`。
- **Artifact**：`A/structured_redecoding_v1/seed0_control_matched/srd_summary.json` 与 `independent_verification.json`；[结果记录](2026-11-24-structured-redecoding-v1-result.md)。
- **Limitation**：单来源、单 checkpoint、单设置；不证明所有结构化解码无效。多模型重组为 `NOT_RUN_SINGLE_SOURCE_STOP`，不能写成“多模型也失败”。
- **Next**：不搜索 stacking bonus、temperature、gamma 或阈值，不以 QUBO 包装替代科学问题。

## 3. Stacking closure：局部边界信号未通过 macro 与 destroy 对照

- **Claim（seed0）**：显式 stacking closure 未形成可继续扩张的稳健机制。
- **Phenomenon**：FN 审计提示 helix-boundary 是重要错误环境，但错误环境本身不证明 stacking 是根因。
- **Intervention**：VL0 196 条、9,600 steps、同参数 `local_control/stacking_closure/destroy_offset/destroy_channel_shuffle` 四臂；固定 local_control 的 1,700 个目标边界 truth pairs，并设结构与 destroy 门。
- **Expected observation**：候选不仅改善目标边界，还须保持 macro pair/stem，并在 micro/macro pair/stem F1 上全部优于两个 destroy 对照，才允许后续 seeds 1/2。
- **Actual**：候选相对 control 的 micro pair-F1=**+0.0039898209**、目标边界 recall=**+0.0064705882**，但 macro pair-F1=**−0.0056405920**、macro stem-F1=**−0.0194305860**。`destroy_offset` micro pair-F1=**0.8479153165** 高于候选 **0.8468992248**，四项 destroy 门均未全部通过；裁决 `STOP_STACKING_CLOSURE_FORM_SEED0`。
- **Artifact**：`A/stacking_closure_form_v1/pilot/evaluation/pilot_verdict.json` 与 `independent_recheck.json`；[seed0 结果](2026-11-24-stacking-closure-seed0-result.md)。
- **Limitation**：单 seed 的 destroy 较好只说明候选未证明机制特异性，**不能解释成随机扰动有益，也不能改选 destroy 为冠军**；局部/raw 改善不能抵消 macro 失败。
- **Next**：不启动 seeds 1/2，不继续堆模块；将“必须优于破坏机制特异性的对照”保留给 GEC。

## 4. GEC seed0：ensemble context 未通过 matched/destroy/independent 对照

- **Claim（当前 NLL seed0）**：GEC 科学假设失败，不能宣称 ensemble context 改善结构。
- **Phenomenon**：需要检验多模型 ensemble context 是否提供超出随机/独立上下文的结构信息。
- **Intervention**：`matched_control/ensemble_context/destroy_interval_shuffle/independent_context` 四臂共享初始化、batch/schedule，固定 MEA；TR1c dev 9,143 条独立复评，包含 paired CI、家族敏感性和目标子群。
- **Expected observation**：ensemble 对 destroy 与 independent 的 micro/macro pair/stem F1 严格占优，对 matched control 满足 precision/F1 门，并改善隔离/长跨距目标；seed0 只决定是否提出多种子，而非稳健性结论。
- **Actual**：ensemble micro/macro pair-F1=**0.6812310470/0.6746040482**，destroy=**0.7077029737/0.7360553533**；ensemble−destroy 为 **−0.0264719267/−0.0614513051**。ensemble micro/macro stem-F1=**0.3220558228/0.6280752650**，destroy=**0.3590030804/0.6614507239**。独立复评 `overall_pass=false`、`metric_gates.pass=false`，状态 `FAIL_GEC_SEED0_INDEPENDENT_REEVAL`。
- **Artifact**：`F/gec_seed0_v3_independent_reeval_v1/independent_reeval.json`、`gate_summary.json`；四臂 prediction/raw 输入由其 manifest 绑定。
- **Limitation**：单 seed、model-dev；不能推广为“所有 context 无益”，也不能把 destroy 的相对优势写成随机有益。seed0 失败不授权 seeds 1/2、外部测试或 SOTA。
- **Next**：保留失败结论，转向 frozen-forward 残差定位，而不是挑指标或改门。

## 5. 残差定位：98.31% 是 FN 分类描述，不是 margin 有效性证据

- **Claim（描述性）**：ensemble 冻结残差中的未选 truth pairs 主要落在非正 score-margin 类，提供候选干预靶点。
- **Phenomenon**：GEC 失败后仍需区分“score/weight 未支持 truth pair”与“正权重下的 MEA 竞争”。
- **Intervention**：对 frozen checkpoint 做 stop-gradient forward、合法 pair 分类、forced-pair/MEA 反事实和 destroy→ensemble 转移审计；不重训、不修改历史预测。
- **Expected observation**：获得可守恒核对的 FN 分解与子群定位；该诊断只负责提出可证伪干预，不预设干预有效。
- **Actual**：ensemble 在 247,013 个 truth pairs 中 selected=144,110，FN=**102,903**；其中 `nonpositive_score_margin`=**101,164**、`positive_weight_mea_competition`=**1,739**、illegal=0。因此 **101,164/(101,164+1,739)=98.31005899%** 的 FN 为非正 score-margin。总体 recall=**0.5834105897**，isolated recall=**0.1090534979**。
- **Artifact**：`F/gec_residual_error_audit_v1/ensemble_context_summary.json`、`comparison.json`、`verification.json`；独立复核 `status=PASS` 指审计完整性通过。
- **Limitation**：这是 frozen-forward 关联与反事实解码分类，**不识别训练因果，不证明 pair-Hamming structured margin 会改善这些 FN，更不保证最终 pair/stem F1 改善**；98.31% 也不是 margin 的预期提升幅度。
- **Next**：将该定位转化为独立目标函数的预注册四臂检验，而非直接把诊断比例写进疗效叙事。

## 6. Frozen context crossover：输入有因果作用，不等于真实 ensemble 一致有益

- **Claim（两个必须分开的命题）**：① 在 frozen seed0 模型上，替换上下文输入会因果性改变结构预测；② 真实 ensemble 上下文一致优于 matched/destroy/independent 输入。crossover 直接支持①，但不支持②，GEC seed0 科学 FAIL 不变。
- **Phenomenon**：第 4 节比较的是不同训练臂，无法单独分离“输入上下文”与“训练后参数”的作用；需要同一模型下只替换输入的干预。
- **Intervention**：四个 frozen 模型 × 四种输入上下文共 16 组合，固定同一批 9,143 条 dev IDs、模型参数与 MEA，独立从 raw pairs/truth 重算；四个 native 预测均逐条吻合历史输出。PPT 第 33 页由 `03f7f4e14ef906359cbd9d1b519089b6e61116f4` 留痕，本 ledger 只引用该记录。
- **Expected observation**：若输入上下文有因果作用，同模型仅换输入应改变结构指标；若真实 ensemble 机制一致有益，则 ensemble 模型使用真实 ensemble 输入应优于使用 matched/destroy 等替代输入，而不只是“换输入会变化”。
- **Actual**：同一 `ensemble_context` 模型的 micro pair-F1 在 native ensemble 输入为 **0.6812310470**，matched 输入为 **0.6993017226**，destroy 输入为 **0.6871289606**；真实 native 反而低于两种替代输入。independent 输入−matched 输入在 `matched_control/ensemble_context/destroy_interval_shuffle` 三个模型上分别为 **−0.1041019496/−0.1932951914/−0.2897765982**，均明显退化，直接证明上下文输入不是无关装饰；但“有作用”的方向不恒为真实 ensemble 有益。
- **Artifact**：`F/gec_frozen_context_crossover_v1/full/summary.json`（SHA256 `5d1984e9271e190cbc83c79d3c87f6571580fefc3ae84e0bbec5cca7db3fefd7`）与 `independent_verification.json`（SHA256 `06322b53c93d6e5428b347176443e48a0b4ce0416c0ebc0b0c28557c58808054`，`status=PASS`）；执行 commit `810441e7480ea06a717097c57a49130c820c9636`，PPT 第 33 页记录 commit `03f7f4e14ef906359cbd9d1b519089b6e61116f4`。
- **Limitation**：single seed0、cluster dev、frozen causal diagnostic；证明输入干预有效，不证明真实 ensemble 表示已经解决问题，也不推翻第 4 节对照失败。`gec_pass_claim=false`，不授权 seeds 1/2、外部测试或 SOTA。
- **Next**：把“真实 ensemble 并非一致最优”作为 margin 检验必须面对的反证，继续第 8 节四臂目标检验；crossover 本身不是 margin 效果证据。

## 7. NLL 资格修复：恢复结构参照，不重写科学 FAIL

- **Claim**：历史 NLL 四臂可作固定结构评估参照，但不是 margin  checkpoint/目标，也不是 GEC 阳性证据。
- **Phenomenon**：旧独立复评中四臂 `checkpoint_and_restored_convergence=false`，源于 best restore 后 `bad_evaluations` 由 stopped=5 变为 best=0 的元数据口径缺陷。
- **Intervention**：从 raw validation ledger、stopped/best checkpoints、预测绑定与历史代码/数据身份重新资格审查，并由第二实现独立复核；不重算或改变科学门。
- **Expected observation**：只有当缺陷确为唯一资格阻塞、best/restore tensor 一致且预测绑定 best 时，才恢复 `QUALIFIED_STRUCTURE_REFERENCE`；原科学 FAIL 必须保留。
- **Actual**：四臂均 stop step=**13,000**、best step=**10,500**、raw stopped `bad_evaluations=5`，资格为 `QUALIFIED_STRUCTURE_REFERENCE`；`reference_use.original_scientific_fail_unchanged=true`、`scientific_gate_recomputed_or_changed=false`、`margin_superior_to_nll_claim_allowed=false`。
- **Artifact**：`F/gec_seed0_v3_requalified_v1/qualification_summary.json`、`requalification.json`、`independent_recheck.json`；最终资格提交 `ecc40d3` 已包含于整合基线 `780044c`。
- **Limitation**：资格 PASS 仅修复参照身份，不推翻第 4 节科学结果；NLL 与 margin validation loss 不是共同标量，禁止同步数/同 loss 选冠军。
- **Next**：若比较目标函数，双方须各自收敛、恢复 best，再用同一冻结 MEA 与结构指标独立复评。

## 8. Structured margin 四臂：正式检验已授权，科学结果仍为空

- **Claim（待检验假设，不是结论）**：exact pair-Hamming cost-augmented structured max-margin 是否能针对残差定位改善 GEC，并证明 ensemble 的机制特异性。
- **Phenomenon**：第 5 节非正 score-margin FN 提供干预动机；第 4 节说明仅有动机不足以支持 GEC；第 6 节进一步证明“上下文有因果作用”与“真实 ensemble 一致有益”是不同命题，后者仍未支持。
- **Intervention**：在 `a846d90` 使用独立 margin 目标/artifact namespace，seed0 四臂 `matched_control/ensemble_context/destroy_interval_shuffle/independent_context` 各自训练至收敛；`lambda_nll=0`、length normalization、terminal 20/33 后每 500 step 验证、patience=5、restore best；固定 MEA（temperature=1、gamma=1），禁止 NLL checkpoint 复用、decoder 搜索和 seeds 1/2。
- **Expected observation**：四臂全部收敛并通过 artifact 身份审查后，ensemble 才可在同一固定 MEA 下接受第 4 节全部对照、结构终点、paired CI、目标子群和家族集中性检验。若声称 margin>NLL，还须有 NLL 合格参照的同协议直接结构比较；validation loss 不可跨目标比较。
- **Actual（截至本次更新）**：数值穷举/两步 smoke/合同复核属于工程通过；resource probe 仅是 B=4/L=500/terminal-20 的显存与运行时测量。实时调度为 matched/ensemble 两臂运行、destroy/independent 两臂等待资源；**完成臂 0、无四臂收敛结构评估、无 margin>NLL 结果、`scientific_conclusion=null`**。PPT 第 34 页仅已有最终四臂 verdict 的生成与独立验收工具链（`1782cd6` 等），没有最终 hash-verified 四臂评估可填入，不得把工具或预留页码写成结果页。
- **Artifact**：`F/structured_margin_smoke_v1/generation_01/smoke_verification_v1.json`；`F/structured_margin_memory_probe_v1/resource_probe_v1.json`；`F/structured_margin_full_v1/full_authorization_v1.json`；`F/structured_margin_full_v1/generation_01/startup_report_v1.json` 与实时 `scheduler_state.json`。配置来源为 `a846d90:configs/gec_structured_margin_convergence_v1.json`。
- **Limitation**：两步 smoke、授权、排队或启动均不证明收敛，更不证明科学有效；98.31% 只说明干预动机。seed0 即使通过也只能提议后续多种子，不能单独建立稳健性、外部泛化或 SOTA。
- **Next**：不干扰现有资源门控调度；等待四臂各自收敛和 best restore，独立检查身份/日志/artifact，再做固定 MEA 结构复评。通过或失败都追加真实结果，并由最终评估驱动第 34 页；此前摘要与标题保持“待检验”。

## 9. structRFM 官方 checkpoint 的旧派生标签审计：仅 circular-audit-only

- **Claim（仅官方 raw 口径）**：官方 structRFM v0.0.8 单 checkpoint 已在项目 raw bpRNA-new 5,388 条上完成新推理与独立评分；该结果描述外部 baseline，**不是我们的候选正向性能，也不是 clean SOTA**。
- **Phenomenon**：历史 structRFM 预测与缺失的历史权重字节无法回溯绑定，不能用旧 replay 代替官方 checkpoint 结果。
- **Intervention**：固定官方 v0.0.8 checkpoint、官方 constrained DP 与项目 raw 标签，完成 5,388 条 GPU 推理；独立核对 prediction identity、coverage 与 exact-pair 指标，历史预测保持不变。
- **Expected observation**：只有新官方 checkpoint 的完整预测、身份和 raw 指标全部通过，才可进入同口径 baseline 算术；不得借 raw 排名推断去污染泛化或 SOTA。
- **Actual**：coverage=**5,388/5,388=1.0**，TP/FP/FN=**73,864/62,616/61,320**，micro/macro pair-F1=**0.5437893868896873/0.5338105702966888**。状态为 `PASS_INDEPENDENT_OFFICIAL_5388_IDENTITY_AND_METRICS`，执行提交 `b53a4f8245776eda43f1bb2ee651a654df68425b`，并由 FoldNAR HEAD `fd497e9819e39a29887ec26698a46c91284d230b` 整合。
- **Artifact**：`F/structrfm_official_v0.0.8/reports/structrfm_official_bprna_new_metrics_20261126.json`、`structrfm_official_provenance_20261126.json` 与 `full_delivery_verification_20261126.json`；原始预测为 `F/structrfm_official_v0.0.8/predictions/structrfm_official_v0.0.8_bprna_new_5388.jsonl`。
- **Limitation**：标签范围为 `PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED`，`raw_not_clean=true`；协议审计进一步证明旧标签 `DERIVED_FROM_BPFOLD_PREDICTIONS`，因此该分数只保留为 circular-audit-only，不再具备独立排名或 SOTA 资格。它是官方 checkpoint 对旧派生标签的历史审计，不是 FoldNar/GEC/margin 候选结果，不授权 clean、OOD、外部候选性能或 SOTA 宣称。
- **Next**：只允许与同一 raw 5,388 标签、同一 strict micro-F1 口径的 baseline 做绝对差距描述；候选正向性能仍须等待其自身合格实验。

## 10. Official-reference 5,388：xens2 参照下的新差距

- **Claim（同官方参考的绝对 micro-F1 差距）**：在 SHA256=`4924e3512e89fd965233a1a14b327c3d87e6c583764b7682c2d293b1454159e4` 的 official-reference 5,388 面板、相同 strict zero-based inclusive exact-pair 口径下，xens2 micro-F1=**0.48901660903024696**；BPfold 项目 replay=**0.7745179151021447**，Δ=**0.2855013060718978**；CONTRAfold=**0.629675588219163**，Δ=**0.14065897918891607**。这不是相对百分比，也不是我们的候选提升。
- **Phenomenon**：旧 structRFM 差距混用了派生标签与不同参考身份，不能迁移到本面板。BPfold 项目六 checkpoint replay 与 BPfold 官方 v0.2 release 是不同预测资产，必须与 xens2 在同一官方参考、相同 5,388 ID 上重新计分。
- **Intervention**：固定官方参考面板、全覆盖预测与 pooled TP/FP/F1，直接计算各系统−xens2 的 micro-F1 绝对差；官方 release 独立使用其 author-release prediction identity，不用项目 replay 替代。所有结果只由冻结预测 CPU 复算。
- **Expected observation**：三者均覆盖 5,388/5,388；registry 与 system metrics 逐字段一致，差距可由冻结 micro 值独立相减。部分覆盖、blocked/summary-only 与 0.9617 循环值不得进入描述排名。
- **Actual**：xens2 为 `three_checkpoint_ensemble`，micro=**0.48901660903024696**。BPfold replay 为 `six_checkpoint_ensemble`，micro=**0.7745179151021447**，故 **0.7745179151021447−0.48901660903024696=0.2855013060718978**。CONTRAfold 为 `deterministic_single_predictor`，micro=**0.629675588219163**，故差值 **0.14065897918891607**。BPfold official release 独立 micro=**0.6339521964768657**，相对 xens2 Δ=**0.14493558744661877**；其 SHA256=`bc41c0bc...`，不同于 replay SHA256=`9832e97b...`，**official release 与项目 replay 是不同 prediction identity**。
- **Artifact / links**：[official-reference registry](/home/rna_jev_bpfold_protocol/manifests/foldnar_v1/bprna_new_official_reference_5388_prediction_registry.json) 的 `xens2_reference`、`strongest_overall`、`strongest_single_predictor` 与 `rows`；[system metrics](/home/rna_jev_bpfold_protocol/source_data/bprna_official_reference_5388/system_metrics.json) 与其逐字段一致；[独立验证](/home/rna_jev_bpfold_protocol/source_data/bprna_official_reference_5388/independent_verification.json) 为 `PASS_INDEPENDENT_SECOND_IMPLEMENTATION`。
- **Limitation**：这些是 `DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION` 的同面板描述差值，不是 clean-OOD、pretraining-unseen 或无限定 SOTA，不证明集成普遍优于单模型，也不是 FoldNar/GEC/margin 候选正向性能。模型形式与预训练暴露必须同时报告，外部 Test 不得用于 checkpoint、阈值、臂或模型选择。
- **Next**：废弃旧 structRFM raw 差距的当前比较用途；后续只可复用本官方面板及同一协议重算。0.9617 永远不得进入本面板或排行榜，official release 与 replay 不得合并成同一系统身份。

## 11. LinearPartition 19,000 nt holdout：只有 baseline 计算回答，GEC unsupported

- **Claim（单条 holdout 描述）**：官方 LinearPartition-C 已在 `7AM2_1` 完整 19,000 nt holdout 上完成一次 serial-CPU `APPROXIMATE_SPARSE` 运行，提供真实 coverage、资源与 candidate/MEA recall 边界；**长序列只有 baseline 回答，GEC unsupported**。
- **Phenomenon**：短序列结构结果不能外推为 19,000 nt 可计算性或准确性；candidate coverage、最终 MEA 输出与资源消耗必须分别核对。
- **Intervention**：固定官方 LinearPartition-C binary（上游 commit `b450fb3e63189073b68d385589035f992080aa3a`）、beam=100、BPP cutoff=0.0、单 CPU 线程和 600 秒 timeout，不截断输入；独立重读 raw BPP/MEA，复算 coverage、span recall 和 GNU time 资源。
- **Expected observation**：必须完整输出 19,000 nt，先报告 candidate recall，再报告 MEA 与资源；空 span 桶保持 null，FLOPs 无法测量时不得估算替代。baseline 通过不等于 GEC 通过。
- **Actual**：coverage=**19,000/19,000=1.0**、`truncated=false`；wall time=**362.79 s**，peak CPU RSS=**8,481,693,696 bytes**。BPP candidate overall recall=**53/73=0.726027397260274**，MEA overall recall=**8/73=0.1095890410958904**；两者在 span 200–999 均为 **0/13**，span ≥1000 因 reference=0 而为 **null**。FLOPs=`NOT_MEASURED`，不作估算。执行/独立验证提交为 `024ef5aadeed1d560802f1a13387b5079d6a56b5`，结果记录提交为 `bbc4366bae1e4643bccb2e960470abcee77a83c2`，整合 HEAD 为 `fd497e9819e39a29887ec26698a46c91284d230b`。
- **Artifact**：`F/longseq_linearpartition_contract_20261126/runs/linearpartition_c_19000_holdout_cpu_20261126/7AM2_1/run_report.json`、`metrics.json`、`independent_verification.json` 及上级 `holdout_delivery_verification.json`；raw BPP/MEA 路径与 SHA256 均由这些文件绑定。
- **Limitation**：范围为 `HOLDOUT_DESCRIPTIVE_ONLY_NOT_FOR_MODEL_SELECTION`，只有一条 holdout，不用于模型选择、校准、early stopping 或训练。BPP candidate recall 不是最终结构准确率，MEA recall 也不是我们的候选结果；这不是我们的候选正向性能或 clean SOTA。GEC 没有此 19,000 nt 全长预测、candidate recall、结构准确性或资源对照，故长序列 GEC 仍为 unsupported。
- **Next**：只可声称 LinearPartition baseline 在该单条 holdout 上完成了全长计算并报告上述边界；任何 GEC 长序列性能、scaling 或优于 baseline 的句子都必须等待其自身完整、独立的直接证据。

## 12. Label quality：软复核不是删除授权，来源类别不是实验真值

- **Claim（记录级质量边界）**：TR1c `DATABASE` 与 CHANRG `COMPARATIVE_INFERRED` 均不能称为实验真值；noncanonical 或 short-span 只触发软复核，**不自动删除**。
- **Phenomenon**：数据库来源、比较推断标签与实验测定是不同 provenance；格式合法的 noncanonical/短跨距配对也不等于无效标签。无 issue flag 同样只能记为 `NOT_DETECTED_UNKNOWN`，不能升级为 clean。
- **Intervention**：在 FoldNAR HEAD `79a2ed45879f1f4e81d6719ed54600bf7f8da807` 上，对 TR1c cluster train/dev 与 CHANRG train/validation 共 **148,285** 条记录做独立 raw 重算，核对 stable ID、0-based inclusive pairs、对称性、最小 loop、partner conflict、canonical/crossing 与删除守恒；外部 Test 未访问，训练 IDs 未修改。
- **Expected observation**：来源类别逐数据集保留，per-record experimental lineage 无证据时保持 `UNCHECKED`；硬冲突只可列为 proposed exclusion candidate，软复核不得直接变成删除，实际删除必须有单独授权与留痕。
- **Actual**：记录级 issue 计数为 **95,052 条 noncanonical**、**10,250 条 short-span**；二者可重叠，不能相加成唯一记录数。合计 review candidate=**95,783** 且全部为 soft-review-only；**硬冲突记录=0**、proposed exclusion=0、**实际删除=0**，`training_ids_modified=false`、`model_dev_v2_modified=false`。TR1c 42,564 条均为 `DATABASE`，CHANRG 105,721 条均为 `COMPARATIVE_INFERRED`；全部 148,285 条 per-record experimental lineage 仍 `UNCHECKED`。独立验证 `verification_passed=true`、`independent_raw_recomputation=true`、`builder_imported=false`。
- **Artifact**：`F/label_quality_v1/summary.json`、`deletion_conservation.json`、`independent_verification.json` 与 `artifact_manifest.json`；来源代码提交 `3e64df111cffaeaf21add48094434123234ea40f`，artifact manifest 的三轮验证均通过。
- **Limitation**：这是质量审计，不是 cleaned dataset、路线选择结果或删除授权。TR1c 的数据库成员身份不证明逐条实验 lineage；CHANRG 的 Rfam consensus/comparative inference 不得改写为实验真值。PDB/mmCIF、modified-parent 与逐条实验 lineage 仍未核实；无 flag 不代表 clean。
- **Next**：保留原始记录与 IDs；任何后续清洗必须针对具体硬冲突另行授权并独立守恒复核，不能把本节的软复核计数直接转换为删除清单。

## 13. Pretrain exposure：已知来源高度暴露，完整官方语料仍未封闭

- **Claim（暴露分层而非 clean 认证）**：当前 RiNALMo-giga backbone 的已知公开来源审计显示，cluster80 dev 9,143 条为 **EXACT=7,978 / HOMOLOGY=539 / NOT_DETECTED_UNKNOWN=626**；**bpRNA-new EXACT=5,381/5,388**，剩余 7 条均为 HOMOLOGY。**完整官方预训练语料仍 `UNCHECKED`，dev 不能外推 clean-OOD**。
- **Phenomenon**：下游 cluster/family 去泄露不能移除 backbone 预训练接触；已知来源未检出与完整历史语料未覆盖也不是同一命题。把一个 dev 总分或 raw 榜单直接写成 pretraining-unseen/uncontaminated SOTA，会隐藏上述暴露层。
- **Intervention**：固定官方声明的 RNAcentral、Rfam、nt、Ensembl 109 与 Ensembl Bacteria 56 来源边界，扫描 hash-bound RNAcentral 22.0、Rfam 14.9、Ensembl 快照并复用既有 80% identity/bidirectional-coverage 搜索；规范化 full-sequence exact 与 homology 分层，由独立 JSONL/FASTA parser 全量重扫。历史 nt 无固定训练日期 release/hash，不拿当前 nt 冒充历史训练快照。
- **Expected observation**：`EXACT` 只表示已知来源全长序列命中，`HOMOLOGY` 只表示无 exact 但有已登记同源命中；`NOT_DETECTED_UNKNOWN` 与 `UNCHECKED` 不得改写为 clean。外部 Test 只能做最终暴露/质量审计，不能用于选型。
- **Actual**：dev 的 exact-only 层为 7,978 EXACT + 1,165 exact-undetected；加入已知来源 homology 后为 7,978/539/626，三者守恒至 9,143。bpRNA-new 为 5,381 EXACT + 7 exact-undetected，而这 7 条均进入 HOMOLOGY，因此已知来源 homology-undetected=0，不能构造 clean 七条对照或作暴露—性能因果比较。完整官方 coverage=`UNCHECKED`；独立验证 `status=PASS`、registry rows=**157,227**，只证明已声明已知来源证据可复算，不封闭历史 nt。
- **Artifact**：`/home/rna_jev_pretrain_exposure/manifests/pretrain_exposure_v1/summary.json`、`asset_registry.json`、`sequence_registry.jsonl.gz` 与 `independent_verification.json`；来源提交 `4247d1e4a971ceb8c63e3f8338e026a23c18941b`。
- **Limitation**：支持的措辞只有 **exposure-aware descriptive performance on the registered benchmarks**。已知 EXACT/HOMOLOGY 暴露使相关总体不得声称 clean-OOD、pretraining-unseen 或 uncontaminated SOTA；`NOT_DETECTED_UNKNOWN` 不是 clean，`UNCHECKED` 不是阴性暴露。历史 nt 未封闭，不能估计完整暴露率、认证无污染余量或宣称对未见 RNA 家族优越。RiNALMo-ft 的 bpRNA TR0 监督微调暴露是额外层级，不得与当前 backbone 自监督预训练暴露混用。
- **Next**：任何 benchmark 叙述必须保留对应 EXACT/HOMOLOGY/NOT_DETECTED_UNKNOWN/UNCHECKED 层；dev 结论只限 development/model-dev，不外推 clean-OOD。外部 Test 暴露结果不得反向选择模型、臂、checkpoint、阈值或超参。

## 14. 下一页准入：暴露分层必须显式出现，margin 仍无结论

- **Claim（后续内容门禁，不是新结果）**：**下一页必须暴露分层**；至少展示 dev 的 EXACT=7,978 / HOMOLOGY=539 / NOT_DETECTED_UNKNOWN=626，以及 bpRNA-new 的 EXACT=5,381/5,388、其余 7 条 HOMOLOGY，并同场保留完整官方预训练语料 `UNCHECKED`。本节只约束后续内容，不修改 PPT。
- **Phenomenon**：只给合并排行榜或总分会把已知暴露、已知来源未检出与完整语料未核实压成一个误导性的 clean/OOD 叙事；工程工具链或双训练运行状态也容易被误写为科学阳性结果。
- **Intervention**：后续页面直接引用第 13 节冻结 registry 的分层数字与语义，不用 UNKNOWN/UNCHECKED 创建 clean 组，不把 bpRNA-new 的 7 条 HOMOLOGY 当作未暴露对照；dev 结论不外推 clean-OOD，SOTA 仅限 exposure-aware descriptive wording。
- **Expected observation**：页面必须能逐层回溯 EXACT/HOMOLOGY/NOT_DETECTED_UNKNOWN/UNCHECKED，并保留历史 nt 未封闭的 limitation。margin 只有在双训练完成、各自收敛、best restore、独立身份检查与固定 MEA 结构复评全部完成后才可进入结果叙事。
- **Actual（冻结状态）**：**margin 双训练未完成，完成臂 0，`scientific_conclusion=null`，margin 仍无结论**。current independent margin evaluator 为 `/home/rna_jev_margin_evaluator`、commit `536b79e450c1ad18b67e06cb39ecb3730546221b`，只作为后续独立复评身份；当前没有 hash-verified 四臂结构结果。本轮**不得干扰 GPU 或改原 16 门**，也未执行训练、推理、资源调度或科学门修改。
- **Artifact**：暴露分层来自 `pretrain_exposure_v1/summary.json` 与 `independent_verification.json`；margin 无结论状态仍由 `structured_margin_full_v1/generation_01/scheduler_state.json` 和 `full_authorization_v1.json` 绑定；独立评估器身份由上述路径与 commit 绑定。
- **Limitation**：暴露审计不创造新的模型性能结果，也不授权用外部 Test 选型。评估器 commit、页面工具或待生成页面均不等于实验完成；margin>NLL、机制特异性、clean-OOD 与 SOTA 均保持 unsupported。
- **Next**：后续制作页面时先逐项通过本门禁；等待 margin 双训练及剩余对照真实完成后，再由固定独立评估器复评并追加结果。无论通过或失败，都不得回填或改写当前“无结论”快照。

## 15. Model-dev v3：12 个历史 run 的可回溯暴露登记，不产生新性能结果

- **Claim（registry 登记）**：model-dev v3 已在提交 `302b70a228d77b0f1847489c93ead402e33510b5` 冻结，给既有 12 个 GEC seed0 run 增加可回溯的暴露分层与最终报告字段；不增加训练、推理或 route-selection 结果。
- **Phenomenon**：第 13 节暴露 population 需要与既有 run、train/dev IDs、配对和家族环境精确连接，否则分层数字无法用于后续 sensitivity；但 registry 连接本身也不能冒充新模型评估。
- **Intervention**：以 FoldNAR base `79a2ed45879f1f4e81d6719ed54600bf7f8da807`、label-quality `3e64df1` 和 pretrain-exposure `4247d1e` 为冻结来源，非 import 独立重读 gzip/registry，重算 sequence/pair/family/concentration 与资格；v1/v2 保持 immutable。
- **Expected observation**：三轮独立验证应逐轮通过 257 项检查和 31 个单元测试，deterministic rebuild byte-identical；dev 分层序列与 pair 守恒，完整官方语料继续 `UNCHECKED`，UNKNOWN 不升级为 clean。
- **Actual**：验证状态 `PASS_THREE_ROUNDS`，三轮均为 **31 passed / 257 checks**，registry SHA256=`4a047b8ed6a520e00a52543a3aa2e3f595fa0a29c565753d39e9ccbaf8f4d904`。Dev 序列仍为 7,978/539/626；对应 pair 数为 **206,588 / 19,528 / 20,897**，合计 247,013。12 个 run 的 sensitivity eligibility 为 EXACT=true、HOMOLOGY=false、NOT_DETECTED_UNKNOWN=false；因此 EXACT 仅 sensitivity-only，HOMOLOGY/UNKNOWN 仅 diagnostic-only。`metric_values_status=NOT_COMPUTED_BY_REGISTRY_BUILD`，没有新 sensitivity 性能数值。
- **Artifact**：`/home/rna_jev_model_dev_v3/manifests/foldnar_v3/model_dev_registry.json` 与 `verification_summary.json`；`F/model_dev_v3/artifact_manifest.json`、`verification_round3.json`。v3 manifest 记录 `v1_v2_modified=false`、`margin_results_read=false`、`gpu_task_launched=false`。
- **Limitation**：该 registry 是 additive reporting layer，不得 rescue/veto primary full-panel 结论，不得影响 route selection，也不得把外部 Test 用于选型。EXACT sensitivity-only 不等于 clean-OOD 或 primary gate；HOMOLOGY/UNKNOWN diagnostic-only 不得改写成 sensitivity 证据或 clean 组。
- **Next**：后续独立 margin evaluator 只能在训练终端资格完成后读取同一冻结暴露 registry；分层用途与原 16 门不变必须由独立门禁再次核对。

## 16. Evaluator × model-dev v3 联合门禁：EXACT sensitivity-only，其余 diagnostic-only

- **Claim（不可降级的用途边界）**：**EXACT sensitivity-only，HOMOLOGY/UNKNOWN diagnostic-only，原16门不变**。当前 evaluator `536b79e450c1ad18b67e06cb39ecb3730546221b` 只增加 `PREREGISTERED_SENSITIVITY_ONLY` 分析；它不重写第 4 节科学 FAIL，也不产生 margin 结论。
- **Phenomenon**：EXACT 层满足样本量、pair 数与家族集中度门槛，因此可作为附加敏感性层；HOMOLOGY 与 UNKNOWN 虽有序列/pair 数量，却没有 mapped families，family/HHI/unmapped 检查失败，只能保留诊断。若把这些层用于 rescue/veto primary gate 或 checkpoint selection，会改变预注册科学问题。
- **Intervention**：逐 run 检查 v3 的 12 个 `exposure_stratified_eligibility` 块，并对照 evaluator 源码中的 `PREREGISTERED_SENSITIVITY_ONLY`、`gate_use=FORBIDDEN`、`checkpoint_selection_use=FORBIDDEN` 与 exactly-16-gate fail-closed assertion；同时核对 v3 artifact manifest 的 immutability flags。
- **Expected observation**：每个 run 均为 EXACT `SENSITIVITY_ONLY` 且 `sensitivity_eligible=true`，HOMOLOGY/UNKNOWN 均为 `DIAGNOSTIC_ONLY` 且 `sensitivity_eligible=false`；三层 `route_selection_allowed=false`。原 16 门名称、定义、通过状态和 route eligibility 不变。
- **Actual**：12/12 run 逐层符合上述用途。v3 记录 `original_16_gates_modified=false`、`current_seed0_main_gate_changed=false`、`gate_count_changed=false`、`gate_definitions_changed=false`；当前 seed0 仍为 `FAIL_GEC_SEED0_INDEPENDENT_REEVAL`，**n_gates=16 / n_failed_gates=12**，pass=false。Evaluator 的 exposure sensitivity 明确禁止 gate/checkpoint/arm selection；registry 本身仍无新性能数值。
- **Artifact**：`/home/rna_jev_margin_evaluator/tools/structured_margin_exposure_sensitivity_v1.py`、`structured_margin_independent_eval_v1.py`；`/home/rna_jev_model_dev_v3/manifests/foldnar_v3/model_dev_registry.json`；`F/model_dev_v3/artifact_manifest.json`。
- **Limitation**：EXACT sensitivity 是已暴露子层的附加描述，不是 clean-OOD、pretraining-unseen 或 primary SOTA 证据；HOMOLOGY/UNKNOWN diagnostic 不得转成 sensitivity 或 clean 对照。任何分层结果都不能替代 full-panel paired CI 与原 16 门。Margin 双训练仍未完成，无 margin>NLL 或机制特异性结论。
- **Next**：保持原 16 门与当前 FAIL 不变；待 margin 终端资格真实完成后，current evaluator 才可在同一 raw pairs、相同 IDs/coverage 上追加 sensitivity，且必须继续禁止分层 rescue/veto、checkpoint/arm selection 和外部 Test 选型。

## 17. BPfold 标签协议：旧 raw 为循环审计，official-reference 才是独立参考面板

- **Claim（标签谱系与面板边界）**：协议提交 `e7ff4e76fe078178ab41d2052349627317b5a3e3` 证明项目旧 bpRNA-new 标签为 **`DERIVED_FROM_BPFOLD_PREDICTIONS`**；旧 raw/clean 派生结果一律仅为 **circular-audit-only**。新 official-reference 5,388 面板 SHA256=`4924e3512e89fd965233a1a14b327c3d87e6c583764b7682c2d293b1454159e4`。**0.9617 仅循环审计不得排名**。
- **Phenomenon**：项目标签并非与 BPfold 官方 `BPfold_data/test_data/bpRNAnew` 参考标签相同，而是由 BPfold v0.2 release 预测 BPSEQ 经项目 min-loop/crossing 投影生成。用 release 预测再评这些派生标签会产生 FN=0 的循环一致性高分。
- **Intervention**：独立第二实现解析官方 release、官方参考、项目标签和项目 replay 四种身份，逐条复算 5,401 条名称/序列/pair、5,388 条投影与 13 条 bad-alphabet 排除；另从官方参考归档重建冻结 5,388 ID 顺序的官方面板，不运行训练、推理或模型选择。
- **Expected observation**：项目 pair sets 必须逐条等于同名 release 的历史投影；官方参考与项目标签差异不得全部误归因于过滤。0.9616708875167174 必须被识别为 release→派生标签的循环审计，不得进入 official-reference 结果或排名。
- **Actual**：5,388 条项目标签全部精确匹配 release 投影；只有 **106 条**与官方参考集合相同，**5,282 条**不同。官方 release 对旧项目标签 micro-F1=**0.9616708875167174**、FN=0，只支持循环一致性。官方 release 是独立 prediction identity，与项目六 checkpoint replay 不得混用；同一 release 对官方参考 5,388 交集为 **0.6339521964768657**，官方 5,401 全量则为 0.633624758107719，两个面板不得互换。
- **Artifact / links**：[标签协议摘要](/home/rna_jev_bpfold_protocol/source_data/bpfold_label_protocol_audit_20261127/summary.json)（独立状态 `PASS_INDEPENDENT_PROTOCOL_AUDIT`）；[official-reference registry](/home/rna_jev_bpfold_protocol/manifests/foldnar_v1/bprna_new_official_reference_5388_prediction_registry.json)；[system metrics](/home/rna_jev_bpfold_protocol/source_data/bprna_official_reference_5388/system_metrics.json)；[官方 5,388 面板](/home/rna_jev_bpfold_protocol/source_data/bprna_official_reference_5388/official_reference_5388.jsonl.gz)；[第二实现验证](/home/rna_jev_bpfold_protocol/source_data/bprna_official_reference_5388/independent_verification.json)（5,388 条、28 个全覆盖系统通过）。
- **Limitation**：旧 raw 的 0.737007、0.696764、0.543789 及其差值只保留历史循环审计身份，不能排名或声称独立 SOTA。新 official-reference 面板也只是 `DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION`，仍需保留预训练暴露、模型形式和覆盖率限制，不授权 clean-OOD 或选模。
- **Next**：当前可比叙述只使用同一官方参考 5,388 面板、相同 strict pair 协议与全覆盖身份；任何旧派生标签数字必须带 circular-audit-only，0.9617 永不进入排行榜、checkpoint 选择或标题。

## 18. SOTA 统一门禁：19 个派生资产全部 circular-only，官方面板不含 0.9617

- **Claim（跨 registry 最终限制）**：旧 recovered registry 中 **19/19 派生标签资产**（18 raw + 1 TR1c-clean）均为 `circular-audit-only; historical value retained; not independent SOTA`，且 `independent_sota_eligible=false`；**28/28 official-reference scored rows 均不含 0.9617**。旧差距禁止回写。
- **Phenomenon**：只修改单个 BPfold 行不足以消除循环审计，因为同一派生标签还用于其他 recovered 模型、CONTRAfold TR1c-clean 与 xens2 历史参照；只排除一个高分也不足以把新面板升级为 clean-OOD 或无限定 SOTA。
- **Intervention**：逐行扫描 recovered registry 的 `label_lineage.label_source`、`disposition`、`sota_use` 和 `independent_sota_eligible`；扫描 official registry/system metrics 的全部 rows 是否出现 0.9616708875167174，并核对 ranking scope、独立验证的排除检查和 model-selection 禁止标记。
- **Expected observation**：所有派生资产保持历史数值但丧失独立排名资格；official release、replay、xens2 和其他模型只能在同一官方参考上描述比较。旧 structRFM 差距值不得返回当前 ledger。
- **Actual**：跨 registry 门禁验证 19 个派生资产全部 circular-only，xens2 旧 raw reference 同样 `independent_sota_eligible=false`；official registry 与 metrics 的 28 行均无 0.9617。新结果仅为 `DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION`，并继续受第 13 节预训练暴露限制。
- **Artifact / links**：[修复后的 recovered registry](/home/rna_jev_bpfold_protocol/manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json)、[official-reference registry](/home/rna_jev_bpfold_protocol/manifests/foldnar_v1/bprna_new_official_reference_5388_prediction_registry.json)、[system metrics](/home/rna_jev_bpfold_protocol/source_data/bprna_official_reference_5388/system_metrics.json) 和 [独立验证](/home/rna_jev_bpfold_protocol/source_data/bprna_official_reference_5388/independent_verification.json)。
- **Limitation**：官方参考修复解决标签独立性与同协议可比性，不解决所有模型的预训练暴露、集成/单模型形式差异或外部泛化；不得把描述性最高值写成我们的候选提升或 clean-OOD SOTA。外部 Test 审计不参与任何选型。
- **Next**：后续 ledger 只接受新 official-reference 差值及完整身份/覆盖率；任何旧差距、raw 排名或 0.9617 回写均触发验证失败。本轮不修改 PPT。

## 19. ArchiveII official-reference：clean 2544 描述比较与 3966 覆盖禁排名门

- **Claim（面板与标签身份）**：旧 ArchiveII `DERIVED_FROM_BPFOLD_PREDICTIONS` 标签及旧分数只保留 **circular-audit-only**。新面板在未过滤官方参考上做 strict zero-based inclusive pair CPU 复算；clean 2544 共同全覆盖结果仅 `DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION`，不用于选模、checkpoint 选择或 margin 结论。
- **Phenomenon**：旧标签会把预测来源与评价参考混在一起；仅替换参考还不够，因为 release、项目 replay、集成和单预测器的覆盖范围及模型形式不同，缺失预测也不能补 0 后排名。
- **Intervention**：逐条核对 BPfold_data 与 MXfold2 的 3,966 条官方 BPSEQ 身份，使用冻结预测独立复算 clean 2,544 交集和官方 3,966 全集；共同覆盖面内以 10,000 次序列级 paired percentile bootstrap（seed=20261127）计算**相对 xens2 的 macro-F1 差值** 95% CI。
- **Expected observation**：clean 2544 内可报告真实 micro/macro-F1 及配对 macro 差值 CI；全集只有全覆盖身份可做全集描述，部分覆盖必须 `PARTIAL_COVERAGE_NOT_RANKED`，没有全覆盖单预测器时不得给出单预测器排名。
- **Actual（clean 2544）**：

  | Prediction identity / form | micro-F1 | macro-F1 | Δmicro vs xens2 | Δmacro vs xens2 [paired 95% CI] |
  |---|---:|---:|---:|---:|
  | xens2 / three-checkpoint ensemble | **0.8156522070450173** | 0.8596587124620273 | 0 | 0 [0, 0] |
  | BPfold replay / six-checkpoint ensemble | **0.8999331354520079** | 0.9161611813979388 | +0.0842809284069906 | +0.0565024689359115 **[+0.0511175356573565, +0.0619467954524887]** |
  | RiNALMo-ft / single model，clean 内最强单预测器 | **0.8204522394549935** | 0.8630290508840547 | +0.0048000324099762 | +0.0033703384220274 **[−0.0012121956293115, +0.0078699689970171]** |

  BPfold replay 的配对 macro 差值 CI 不跨 0；RiNALMo-ft 的对应 CI **跨 0**，所以其相对 xens2 的微小描述性差值不是已解析的统计优势。CI 只属于 macro-F1 差值，不得写成 micro-F1 CI。
- **Actual（official 3966 base 历史快照，commit `4daccb0eb1c78db565532634ed0d77cf1e6ddaae`）**：base 36 个 inventory identities 中，clean 2544 全覆盖 20 个；当时全集只有 `bpfold_official_release/archiveii` 为 3,966/3,966 全覆盖，micro/macro-F1=**0.7974584107660678 / 0.8200813622940291**，其余 19 个有部分预测的身份均为 2,544/3,966、coverage=0.6414523449319214、`PARTIAL_COVERAGE_NOT_RANKED`，16 个 blocked/protocol-mismatch。base 最强单预测器为 **`UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR`**。这些只描述 CPU 恢复前状态；当前 full3966 的 5 个全覆盖和 MXfold2 最强单见第 27 节，BPfold official release 与项目 replay 仍是不同 prediction identity。
- **Artifact / links**：[冻结复算 system metrics](/home/rna_jev_archive_label_lineage/artifacts/archiveii_official_reference_20261127/system_metrics.json)、[registry recommendation](/home/rna_jev_archive_label_lineage/artifacts/archiveii_official_reference_20261127/registry_recommendation.json)、[clean 2544 官方参考](/home/rna_jev_archive_label_lineage/source_data/archiveii_official_reference_20261127/official_reference_clean_2544.jsonl.gz)（SHA256=`cde658b5d61813ff9530509bd25631d6f33dcb08021bae107636f0a212c11afb`）和 [round3 clean-clone delivery](/home/rna_jev_archive_label_lineage/artifacts/archiveii_official_reference_20261127/verification_round3.json)（`PASS_CLEAN_CLONE_AND_DELIVERY`）。
- **Limitation**：这是外部 Test 的 audit-only 描述复算，不授权外部 Test 选型，不消除预训练暴露、集成/单模型形式或家族独立性限制，也不构成我们的 margin 候选提升或无限定 SOTA。全集 release 结果不能与 replay 的 clean 2544 结果混排。
- **Next**：保留同一面板、同一协议、全覆盖和 paired CI 的完整身份；任何部分覆盖单预测器名次、xens2 full 差距、缺预测补 0 或旧派生标签分数回写均触发验证失败。当前 full 恢复见第 27 节，本轮不修改 PPT。

## 20. PDB/Rfam 标签谱系：当前 PDB62/91，旧 58/95 仅为历史快照

- **Claim（fail-closed 标签边界）**：PDB/Rfam 六个面板均为 filtered canonical/nested projections，目录名、BPfold 结果位置或 registry 的 `EXPERIMENTAL` 标签本身不能证明实验标签谱系。当前 RCSB/mmCIF/DSSR/metadata 重建为 **62 条参考、91 条 UNCHECKED**，其中 4 条来自 official sequence-scheme DSSR 修复；**PDB62 仍不得排名**。旧 **58/95 仅是 round3 历史快照**，不得与当前值混用。
- **Phenomenon**：冻结 JSONL 与官方 RNAformer PLK 没有 PDB/entity/chain 坐标映射；未观测残基不能当负例。跨面板 exact overlap 与 Rfam release membership 也分别破坏“独立面板”和“时间外新记录”的简化解释。
- **Intervention**：对 153 条 PDB source candidates 重放官方 PLK，按唯一 exact full-polymer/observed-only `(PDB, entity, auth_chain)` 匹配并运行固定 DSSR；再用官方 sequence schemes 修复 4 条 DSSR endpoint 映射，结合 metadata 冻结 round6 双协议资产。对 Rfam 14.9/15.0 官方 FASTA、family/full_region 独立复算 exact/region 身份，不读取或使用任何模型结果。
- **Expected observation**：当前 62 条参考与 91 条 UNCHECKED 必须守恒为 153；4 条 scheme 修复只能解决指定 endpoint，不得产生新 metadata 参考或排名资格。RNAformer 与 DSSR 原始配对差异必须保留并重算指标，TS-hard 与 Rfam temporal 的限制不变。
- **Actual（当前 round6 PDB rebuild）**：153 条中 **62 条重建、91 条 UNCHECKED**；剩余原因为 `NO_METADATA_ELIGIBLE_CHAIN_UNCHECKED`=32、`METADATA_AMBIGUOUS_UNCHECKED`=55、`SOURCE_BAD_ALPHABET_UNCHECKED`=4。分面板重建为 TS1/TS2/TS3/TS-hard=**16/27/9/10**。official seq-scheme DSSR 修复为 **4 条**，`new_metadata_references=0`；被排除 endpoint 数为 `pdb_ts1#633034`=89、`pdb_ts2#633066`=2、`pdb_ts3#633080`=2、`pdb_ts_hard#633122`=89。当前 RNAformer-vs-DSSR raw pair exact-match 为 **58 true / 4 false**，4 条差异仍须完整指标重算，不得沿用旧指标或排名。未观测位置继续 `MASKED_NEVER_NEGATIVE`，DSSR pairs 不做 canonical/nested 投影。
- **Actual（历史 round3，仅快照）**：旧 153=58+95、四类原因 26/61/4/4、TS1/TS2/TS3/TS-hard=15/26/8/9、raw pair **56 true / 2 false**，均只描述 round3 当时的重建状态。**旧 58/95 与 56/2 不再是当前值**，不得覆盖 round6 的 62/91 与 58/4。
- **Actual（TS-hard overlap）**：`ref_pdb_ts_hard` 28/28 均有 cross-panel exact-sequence overlap，**TS-hard 与 TS1/TS3 不序列不相交**；TS1/TS2/TS3 的对应 cross-panel overlap 为 22/1/11。即使 TR1c exact overlap 为 0，也不能省略跨面板重叠或将四者当相互独立的 PDB 证据。
- **Actual（Rfam temporal）**：436 条当前 novelty 为 **269 NOVEL_SEQUENCE_RELEASE_SUPPORTED、43 旧序列、12 TR1c 排除、2 cross-panel、110 UNCHECKED**，详见第 26 节；原始 TR1c exact=33、cross-panel exact=2 是独立暴露列。Rfam 14.9 exact=**43** 的旧序列不能称为后一时间区间的新记录，110 条仍 UNCHECKED。`TEMPORAL_FIRST_RELEASE_UNCHECKED` 只保留在未解决记录，pair provenance 与独立实验标签资格仍未闭合，Rfam 标签仍是 `COMPARATIVE_INFERRED`，六个面板均 `NOT_QUALIFIED`，不能称全 temporal 独立。
- **Artifact / links**：[PDB/Rfam 谱系报告](/home/rna_jev_pdb_rfam_label_lineage/source_data/pdb_rfam_label_lineage_audit_20261127/report.md)、[当前 round6 报告](/home/rna_jev_pdb_rfam_label_lineage/source_data/pdb_rcsb_rebuild_20261127/round6_combined/report.md)、[当前 rebuild summary](/home/rna_jev_pdb_rfam_label_lineage/source_data/pdb_rcsb_rebuild_20261127/round6_combined/rebuild_summary.json)、[当前双协议验证](/home/rna_jev_pdb_rfam_label_lineage/source_data/pdb_rcsb_rebuild_20261127/round6_combined_verification.json)（`PASS_RNAFORMER_DOUBLE_PROTOCOL_DSSR_INDEPENDENT_VERIFIER`）、[历史 round3 summary](/home/rna_jev_pdb_rfam_label_lineage/source_data/pdb_rcsb_rebuild_20261127/round3_build/rebuild_summary.json) 和 [历史验证](/home/rna_jev_pdb_rfam_label_lineage/source_data/pdb_rcsb_rebuild_20261127/round3_build_verification.json)。
- **Limitation**：62 条参考仍不是完整 PDB 面板，4 条差异不是可忽略噪声，91 条 UNCHECKED 不是阴性或失败预测。所有内容均 `AUDIT_ONLY_NO_MODEL_SELECTION`，没有 prediction/metric/checkpoint/model-selection outcome，不支持 PDB/Rfam 排名、实验真值、独立泛化或 temporal-OOD 结论。
- **Next**：任何 PDB 性能比较必须先处理 91 条缺失身份、4 条 DSSR 差异和跨面板重叠并完整重算；Rfam temporal 必须补齐 pair provenance 与逐记录 first-release dates。当前不启动推理、不修改 PPT。

## 21. 磁盘配额事故与 9500 断点恢复：工程连续性不是科学收敛

- **Claim（当前运行边界）**：本次中断有真实 `OSError: [Errno 122] Disk quota exceeded`，发生在 `/home` staged checkpoint 向 teams 目标复制时，不能归因为模型、CUDA 或科学门失败。修复 commit `442636dd660f93f1d39156ca9bbdbd107ecdbf42` 保持原训练合同不变；**四臂未收敛，当前无 margin 科学结论**。
- **Phenomenon**：大 checkpoint 在团队配额压力下会使训练/调度中断；仅重新启动进程不足以证明连续性，必须核对 checkpoint、optimizer/RNG/stream、下一 batch 和逐 step ledgers，同时避免把活跃大文件继续写向配额事故路径。
- **Intervention**：双臂 preflight 对 9,500 step checkpoint、optimizer step、tensor/RNG 身份、下一批记录和 train/batch/compute/validation 连续性做只读审计；训练使用 frozen source `a846d90519e575a79b7db76f63692afb755e8ac6`、`steps=0`、batch=4、500-step validation/save、最多解冻 20/33，不以固定步数替代自然收敛。活跃 run 与大 checkpoint 暂存分别位于 `/home/foldnar_incident_run` 和 `/home/foldnar_incident_checkpoints`，最终独立评估只在合法终态后由既有 publisher 镜像到 teams。
- **Expected observation**：恢复必须从 9,500 的下一批继续，不重复/跳批，不篡改原 16 门；10000 训练点或 checkpoint staging 只证明恢复运行。只有四臂均满足自然收敛和终态资格后，才允许独立评估及最终 teams 镜像；在此之前没有 margin>NLL、机制特异性或 SOTA 结论。
- **Actual（只读快照）**：`recovery_preflight_local.json` 为 `PASS_INTERRUPTED_RESUME_AUDIT`，matched/ensemble 双臂 checkpoint 与 optimizer 均为 **step=9500**、`converged=false`、下一 step=9501，三条训练 ledger 均守恒到 9500，preflight `scientific_conclusion=null`。matched 的 launch identity 为 `launch_kind=resume`、recovery commit=`442636d…`、preflight SHA256=`b72b1449…ce4dd`。随后 matched train log 已到达 step=10000（optimizer=10000、20/33 unfrozen），并完成 `/home/foldnar_incident_checkpoints/matched_control/resume.pt` staging。**step=10000 validation 已完成**：validation_loss=**0.4183340173506429**、`eligible=true`、`improved=true`、`save_best=true`、`stop=false`。继续只读跟踪至最新快照时，train log 已到 step=15000，validation 已完成到 step=14500；当前 best 为 step=12500、validation_loss=0.4035744189001614、bad_evaluations=4，仍 `stop=false`。这些只是中途训练/validation 状态，不是自然收敛或结构评估结果。
- **Actual（调度与存储）**：`recovery_scheduler_state.json` 的 `recovery_code_commit=442636dd660f93f1d39156ca9bbdbd107ecdbf42`，matched `run_meta` 为 running、`converged=false`，调度进程为 running/resume，`completed_arms=[]`、`evaluation=null`、`scientific_conclusion=null`；destroy/independent 尚未形成终态。代码留在 `/home`，常规 artifacts 与最终 publication mirror 留在 teams；只有本次活跃 run/大 checkpoint 按事故记录暂放 `/home`，不得把该例外推广为新存储默认。
- **Artifact / links**：[9500 preflight](/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_full_v1/incident_recovery_20261127/recovery_preflight_local.json)、[恢复调度状态](/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_full_v1/incident_recovery_20261127/recovery_scheduler_state.json)、[matched launch identity](/home/foldnar_incident_run/matched_control/recovery_launch_identity_20261127.json)、[staged-checkpoint trainer](/home/rna_jev_margin_incident_recovery/tools/run_frozen_margin_trainer_with_staged_checkpoint_20261127.py) 和 [最终评估镜像器](/home/rna_jev_margin_incident_recovery/tools/mirror_structured_margin_evaluation_for_publication_20261127.py)。事故原始日志为 `/home/foldnar_incident_logs/matched_control.log`，其中保留 Errno 122 与后续 9500 resume/10000 staging 证据。
- **Limitation**：9500 preflight PASS、10000 step、进程 running、checkpoint 可写或镜像器等待都不是科学 PASS；当前四臂未收敛，独立评估尚未开始，不能比较 margin/NLL validation loss，也不能选择臂或 checkpoint。活跃 `/home` 文件仍需在合法终态后完成 hash-verified teams 镜像，当前不得宣称最终镜像完成。
- **Next**：让既有恢复调度自然运行，不干扰 GPU/进程；后续仅读取终端 `run_meta`、独立评估和镜像 hash。若任何臂未达到预注册自然收敛，维持无结论，不用 step cap 或中途 validation 代替。此轮不修改 PPT。

## 22. 第二次 Errno 122 与最终接管修复：scheduler 退出不等于训练退出

- **Claim（第二次事故事实）**：第一次恢复后，旧 scheduler PID `4077848` 在 teams 上执行 `atomic_json` 写 `recovery_scheduler_state.json` 时再次发生 **`OSError: [Errno 122] Disk quota exceeded`** 并退出；这是第二次独立的 scheduler 状态写盘事故，不是 matched 训练、checkpoint 或科学门失败。
- **Phenomenon**：旧 teams state 最后停在 `MONITORING`、matched `resume/running`，同时留下零字节 `recovery_scheduler_state.json.tmp.4077848`、旧 lock/PID 文件。它们只是中断现场，不能继续当作当前 scheduler 身份，也不能据此重启 matched。
- **Intervention**：将第二次 failure log、旧 teams state/events 与 `/home` takeover status 分开冻结，核对进程生命周期与最终代码 `aa9f1c4a92e92d70841e33cec455734f160075c4`；不修改旧证据、不杀进程、不重启训练。
- **Expected observation**：旧 scheduler 已退出而 matched 原 PID 继续存活；最终修复的本地/远端 commit 一致，接管证据 `scientific_conclusion=null`。只有核清身份后才允许记录接管，不能把进程监控事件改写成科学结果。
- **Actual（事故与身份）**：`team_quota_scheduler_failure.log` 明确显示第二次 Errno 122 位于 `save("MONITORING") → atomic_json → recovery_scheduler_state.json`；只读 `/proc` 核对旧 scheduler `4077848` 已退出，matched `4077864` 仍存活。`recovery_takeover_status.json` 的 `head` 与 `remote_head` 均为 `aa9f1c4a92e92d70841e33cec455734f160075c4`，`scientific_conclusion=null`。
- **Actual（scheduler 接管）**：当前 `/home/foldnar_incident_scheduler/recovery_scheduler_state.json` 为 `MONITORING`、scheduler PID **4111890**，matched 记录为 `launch_kind=adopted_running`、PID **4077864**、`process_start_ticks=1329819713`；`/proc` 中该 PID/start_ticks 仍一致，且只有这一份 matched trainer。state 的 `recovery_code_commit=a61a6ea8e52f72fa129387b74cb28b3af8bc64ff` 是接管进程启动身份；最终 HEAD `aa9f1c4…` 增加 `atomic_pid` 持久化，二者不得混写。`recovery_scheduler.pid` 内容为 `4111890\n`，PID 4111890 的 fd 3 持有 `/home/foldnar_incident_scheduler/recovery_scheduler.lock`。
- **Actual（events 与旧文件）**：本地 events 只有 `RECOVERY_SCHEDULER_STARTED pid=4111890` 和随后的 `ARM_ADOPTED matched_control pid=4077864`，没有新的 matched `ARM_STARTED`，因此是 **adopt 原进程而非重复训练**。旧 teams `recovery_scheduler.pid=4070244`、lock/state/events 仅保留历史事故身份，不被当作当前 PID。
- **Actual（存储边界）**：训练输出 `/home/foldnar_incident_run`、大 checkpoint `/home/foldnar_incident_checkpoints`、scheduler state/events/lock/PID `/home/foldnar_incident_scheduler`、接管证据 `/home/foldnar_incident_evidence` 均在本地；评估根配置为 **`/home/foldnar_incident_eval`**，当前评估未开始、目录尚未创建。teams 的共享初始化/GEC 初始化只作只读输入；**teams 写入仅限合法终态后的最终小证据镜像**，不再承接活跃训练、checkpoint、评估工作或 scheduler 状态。
- **Actual（最新已知 best 与科学状态）**：takeover 快照为 train step=15500、20/33 unfrozen、15500 validation pending；最新已完成 best 为 **step=15000、validation_loss=0.3896151208942566、`best_step=15000`、`bad_evaluations=0`、`improved=true`、`save_best=true`、`stop=false`**，`best.pt` 存在。`run_meta` 仍 running、`converged=false`，本地 scheduler `completed_arms=[]`、`evaluation=null`；后续臂为 ensemble/destroy/independent，ensemble 仍从 /home 的 9500 complete checkpoint 等待恢复。**四臂未收敛，当前无 margin 科学结论**。
- **Artifact / links**：[第二次 scheduler failure log](/home/foldnar_incident_evidence/team_quota_scheduler_failure.log)、[旧 teams state](/home/foldnar_incident_evidence/team_quota_scheduler_state.json)、[旧 teams events](/home/foldnar_incident_evidence/team_quota_scheduler_events.jsonl)、[takeover status](/home/foldnar_incident_evidence/recovery_takeover_status.json)、[当前本地 scheduler state](/home/foldnar_incident_scheduler/recovery_scheduler_state.json) 和 [当前本地 events](/home/foldnar_incident_scheduler/recovery_scheduler_events.jsonl) 和 [takeover 时 matched validation](/home/foldnar_incident_evidence/matched_validation_at_takeover.jsonl)。旧 teams lock/PID/state/events 仅作事故证据，当前身份另见本地接管 state。
- **Limitation**：第二次 Errno 122 与最终修复 HEAD 不构成收敛、独立评估或 margin>NLL 证据；四臂仍未收敛，当前无 margin 科学结论。最终镜像尚未完成。
- **Next**：继续核对本地 scheduler state/events/lock/PID、adopt 唯一性和存储边界；不干扰运行，不修改 PPT。

## 23. TS0：保留 projected 1288，full1305 恢复后现有 7 个全覆盖身份

- **Claim（协议边界）**：TS0 项目子集是 `DERIVED_FROM_OFFICIAL_REFERENCE_PROJECTION`，只能在 projected-reference 1288 内做描述性复算；不得与未过滤 official 1305 混排。xens2 为三 checkpoint 集成，RNAformer-bprna 为单模型，模型形式和训练暴露必须同场报告。
- **Phenomenon**：官方参考比项目子集多 17 条，恢复前所有可用预测都缺这 17 条；把子集高分写成官方全集名次会掩盖覆盖缺口。当前只为最终冻结 baseline 恢复官方缺口，xens2 的 TR1c homology-only 暴露与外部模型训练 UNCHECKED 仍不消除。
- **Intervention**：保留 projected 1288 的冻结复算与 paired macro CI；BPfold 采用作者 official release full1305，Vienna/Nussinov、CONTRAfold、EternaFold 和 RNAstructure 仅对官方多出的 17 条做 CPU 补充。面板重算不训练、不选模，缺预测不补 0，RNAstructure 不改写 R/Y 碱基。
- **Expected observation**：projected 原值必须保持；full1305 现有 7 个 `PASS_FULL_COVERAGE` 身份可描述，RNAstructure Fold/MaxExpect 只能 1302/1305、`PARTIAL_COVERAGE_NOT_RANKED`。恢复前零全覆盖只能作为历史快照。
- **Actual（projected 1288）**：xens2 集成 micro-F1=**0.8032405714870655**、macro-F1=0.7999193349056970，为描述性最强集成/总体；最强单预测器 RNAformer-bprna micro-F1=**0.7568974968024850**、macro-F1=0.7444430454311176。RNAformer 相对 xens2 Δmicro=−0.0463430746845805、Δmacro=−0.0554762894745794，paired macro 95% CI=**[−0.0674523877056411, −0.0435507990639508]**。xens2 路线为 TR1c exact=0、homology-only=575、clean n=713，状态 `HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN`；RNAformer 暴露为 `UNCHECKED`。
- **Actual（当前 full1305）**：现有 **7 个全覆盖身份**：BPfold official release、Vienna MFE/centroid/MEA、Nussinov-Turner、CONTRAfold、EternaFold。BPfold official release 为作者六 checkpoint 集成，1305/1305，micro-F1=**0.6812363564199246**、macro-F1=0.6574981350034410，仅 `DESCRIPTIVE_FROZEN_OFFICIAL_FULL_AUDIT_ONLY_NOT_MODEL_SELECTION`，训练暴露仍 `UNCHECKED`。RNAstructure Fold/MaxExpect 因官方工具拒绝 3 条含 R/Y 序列，均为 **1302/1305**、missing=3、`PARTIAL_COVERAGE_NOT_RANKED`，不得补 0 或排名。其他未恢复神经模型继续保持部分覆盖或阻塞，不用 GPU 补全。
- **Actual（恢复前历史）**：旧 21 个 scored identities 的 1288/1305、coverage=0.9869731800766284、official full coverage=0 只保留为恢复前快照；不得再用旧零全覆盖否定当前 7 个全覆盖，也不得把新增 full 值回写到 projected 原表。
- **Artifact / links**：[TS0 system metrics](/home/rna_jev_ts0_coverage/artifacts/ts0_testsetb_official_reference_20261127/bprna_ts0_system_metrics.json)、[full1305 recovery audit](/home/rna_jev_ts0_coverage/artifacts/ts0_official_coverage_recovery_20261127/recovery_audit.json)、[official coverage gaps](/home/rna_jev_ts0_coverage/artifacts/ts0_testsetb_official_reference_20261127/official_coverage_gaps.json) 和 [round3 delivery](/home/rna_jev_ts0_coverage/artifacts/ts0_testsetb_official_reference_20261127/verification_round3.json)（`PASS_CLEAN_CLONE_AND_DELIVERY`、registry diff EMPTY）。
- **Limitation**：这些值只适用于 projected 1288 的冻结协议描述，不是 official 1305 排名、独立 SOTA、clean-OOD 或模型选择证据；标签来源独立不等于训练独立。
- **Next**：full1305 只能在 7 个全覆盖身份内继续做暴露受限的描述，RNAstructure 1302/1305 与其他部分覆盖不得排名；任何 projected 原值回写、训练暴露省略或恢复前零覆盖冒充当前状态均触发验证失败。本轮不修改 PPT。

## 24. TestSetB：保留 projected 428 与暴露，full430 恢复后现有 7 个全覆盖身份

- **Claim（暴露优先于名次）**：TestSetB 项目子集仅支持 projected-reference 428 描述性复算。RiNALMo-ft 虽为面板内单模型最高值，但 **TR0 精确序列暴露 247/428**，状态 `EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT`，因此 **不可独立排名**；xens2 集成也有 TR1c homology-only 暴露。
- **Phenomenon**：仅看 micro-F1 会把 RiNALMo-ft 的 0.8670 写成独立优势；仅看标签来源又会漏掉训练集精确重叠。恢复前官方全集多 2 条 bad-alphabet 记录，所有可用预测均缺这两条；覆盖恢复也不能消除训练暴露。
- **Intervention**：保留 projected 428 的数值、CI 和逐行暴露，仅对官方多出的 2 条以 9 个 CPU decoder 补充最终冻结 baseline；MXfold2 明确 `--gpu -1`，RNAstructure 不改写原始 S 字符。缺预测不补 0，外部 Test 不选模。
- **Expected observation**：原 20 个 projected 身份数值不变；full430 现有 7 个全覆盖身份可描述，MXfold2 为全覆盖中最高数值但仅描述。RNAstructure Fold/MaxExpect 只能 429/430、`PARTIAL_COVERAGE_NOT_RANKED`。
- **Actual（projected 428）**：xens2 三 checkpoint 集成 micro-F1=**0.8403202122116432**、macro-F1=0.8322478063381703，为描述性最强集成；RiNALMo-ft 单模型 micro-F1=**0.8670430033503144**、macro-F1=0.8628944070360038，为描述性最强单/总体。RiNALMo-ft 相对 xens2 Δmicro=+0.0267227911386712、Δmacro=+0.0306466006978335，paired macro 95% CI=**[+0.0169091444489763, +0.0452470510286115]**；该 CI 不消除暴露，不能升级为独立优势。
- **Actual（训练暴露）**：RiNALMo-ft 路线为 TR0 exact=**247/428**、homology-only=65、clean n=116，`EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT`；xens2 路线为 TR1c exact=0、homology-only=140、clean n=288，`HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN`。两者都不能写成完整未见或 clean-OOD。
- **Actual（当前 full430）**：现有 **7 个全覆盖身份**：CONTRAfold、EternaFold、MXfold2、Nussinov-Turner、Vienna centroid/MEA/MFE。全覆盖中最高数值为 **MXfold2 micro-F1=0.6468595682161368**、macro-F1=0.6692063276068385，430/430，**仅描述、不用于模型选择或独立 SOTA**；其训练暴露仍为 `AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED`。RNAstructure Fold/MaxExpect 因原始 S 字符被拒绝，均为 **429/430**、missing `testsetb_104.bpseq`、`PARTIAL_COVERAGE_NOT_RANKED`，不得补 0 或排名。RiNALMo-ft 和 xens2 等未恢复身份仍保持 428/430 部分覆盖及原暴露，不用 full 面板重排它们。
- **Actual（恢复前历史）**：旧 20 个 scored identities 的 428/430、coverage=0.9953488372093023、official full coverage=0 仅为恢复前快照；原 20 个 projected 身份指标经独立验证保持不变。旧零全覆盖不得覆盖当前 7 个全覆盖，新增 full 值也不得回写 projected 原表。
- **Artifact / links**：[TestSetB full/project system metrics](/home/rna_jev_testsetb_coverage/artifacts/testsetb_official_coverage_recovery_20261127/system_metrics.json)、[official coverage gaps](/home/rna_jev_archive_label_lineage/artifacts/ts0_testsetb_official_reference_20261127/official_coverage_gaps.json) 和 [round3 独立验证](/home/rna_jev_testsetb_coverage/artifacts/testsetb_official_coverage_recovery_20261127/verification_round3.json)。
- **Limitation**：projected 数值、paired CI 和“最强单”标签都只是暴露受限的描述；full430 的 MXfold2 最高数值同样不解决完整外部暴露。均不支持独立排名、SOTA、clean-OOD 或选模。
- **Next**：需要独立结论时必须使用与实际模型路线一致的去暴露评估；full430 仅在 7 个全覆盖身份内描述，RNAstructure 429/430 和其他部分覆盖不得排名，任何利用 247/428 精确暴露记录进行独立排名的表述均触发验证失败。本轮不修改 PPT。

## 25. Full 覆盖恢复联合不变量：projected 原值/CI/暴露不变，旧零覆盖仅历史

- **Claim（跨 artifact 一致性）**：full1305/full430 覆盖恢复只增加官方缺口预测与 full 面板描述，不改变 TS0/TestSetB projected 原结果。恢复前零全覆盖是历史状态，当前两面板各有 7 个全覆盖身份，二者不得互相覆盖。
- **Intervention**：分别读取恢复前 `/home/rna_jev_archive_label_lineage` 与当前 `/home/rna_jev_ts0_coverage`、`/home/rna_jev_testsetb_coverage` 的 system metrics，逐字段比较 projected 四个关键身份的 metrics、相对 xens2 差值和训练暴露，并核对恢复前 paired macro CI；当前若保留 CI 字段也必须一致。
- **Actual（不变量）**：TS0 xens2=0.8032405714870655、RNAformer=0.7568974968024850；TestSetB xens2=0.8403202122116432、RiNALMo-ft=0.8670430033503144。对应 CI 与暴露均保持原值，尤其 TestSetB RiNALMo-ft TR0 exact=247/428 仍不可独立排名。当前 full 的 BPfold release=0.6812363564199246、MXfold2=0.6468595682161368 只新增 full 描述，不回写 projected。
- **Actual（部分覆盖不变）**：RNAstructure 在 full1305 仍为 1302、full430 仍为 429，均 `PARTIAL_COVERAGE_NOT_RANKED`；其他未恢复身份保持部分覆盖或阻塞。旧 1288/1305、428/430 与零全覆盖仅是恢复前快照。
- **Artifact / links**：[恢复前 TS0 metrics](/home/rna_jev_archive_label_lineage/artifacts/ts0_testsetb_official_reference_20261127/bprna_ts0_system_metrics.json)、[恢复前 TestSetB metrics](/home/rna_jev_archive_label_lineage/artifacts/ts0_testsetb_official_reference_20261127/testsetb_system_metrics.json)、[当前 TS0 metrics](/home/rna_jev_ts0_coverage/artifacts/ts0_testsetb_official_reference_20261127/bprna_ts0_system_metrics.json) 和 [当前 TestSetB metrics](/home/rna_jev_testsetb_coverage/artifacts/testsetb_official_coverage_recovery_20261127/system_metrics.json)。
- **Limitation**：跨 artifact 一致只证明协议与数值未回归，不证明训练独立、full 覆盖模型可无限制排名或任何 margin 结论。任何 projected CI/暴露变化、旧零覆盖冒充当前或 RNAstructure 部分覆盖排名均触发联合验证失败。本轮不修改 PPT。

## 26. Rfam family/sequence release：family 首次出现不等于序列首次出现

- **Claim（family 层证据）**：Rfam family/clan 只能由 exact sequence/region/seed 证据指定；当前 **11,186 条 `UNIQUE_FAMILY_EXACT_SEQUENCE_REGION_SEED`**，另有 `AMBIGUOUS_FAMILY_UNCHECKED`=22、`FAMILY_UNCHECKED`=18。`family_first_release` 来自官方 14.9/14.10/15.0 family tables 的 accession presence 并以 seed 交叉核对，不用 created/updated timestamp 代替。
- **Phenomenon**：family accession 在某一 release 首次出现，不证明某条具体序列也在该 release 首次出现；同一 family 的旧序列和新序列可共存。目录名、family membership 和 timestamp 都不能直接证明 temporal independence。
- **Intervention**：独立读取官方 family table、seed、FASTA 与 per-family alignment，分别冻结 `family_first_release` 和 `sequence_first_release`；ambiguous/absent evidence 保持 UNCHECKED，不使用配对、模型结果或外部 Test 选型。
- **Expected observation**：11,226 条 family-lineage 记录应满足 11,186+22+18；family first-release 计数应完整守恒。sequence 层另按 exact FASTA/alignment 分类，不能由 family release 直接推出。
- **Actual（family release）**：family first-release **14.10=166、15.0=99**，另有 **14.9_OR_EARLIER=10,921、UNCHECKED=40**，合计 **11,226**。Rfam family 面板中 10,790 条有 10,754 条唯一 family、22 条 ambiguous、14 条 unchecked；temporal 面板 436 条有 432 条唯一 family、4 条 unchecked。registry 内容仅为 `PROPOSED_ONLY_NOT_REGISTERED`，shared registry 未修改。
- **Actual（sequence temporal novelty）**：436 条 temporal 记录互斥分为 **269 条 `NOVEL_SEQUENCE_RELEASE_SUPPORTED`、43 条 `NOT_NOVEL_PRE14_9_SEQUENCE`、12 条 `NOT_INDEPENDENT_TR1C_EXACT`、2 条 `NOT_INDEPENDENT_CROSS_PANEL_EXACT`、110 条 `UNCHECKED`**，合计 436。269 条只表示 sequence release 有支持，不包括其余 167 条，也**不能称全 temporal 独立**。
- **Actual（独立暴露列与证据缺口）**：原始 TR1c exact overlap 为 **33**、cross-panel exact=2，在 novelty assignment 前独立记录；其中 12 条进入 `NOT_INDEPENDENT_TR1C_EXACT`，所以 33 与 12 是不同字段，不能相互替换。官方 15.0 原始状态为 direct exact=320、other-region exact=6、not exact=110，也不能直接等同最终 novelty。14.10 alignments 失败的 RF04301、RF04302、RF04303、RF04305、RF04306、RF04307、RF04308、RF04309、RF04310 共 9 个 family，相关记录保持 UNCHECKED，不猜测补全。
- **Actual（sequence release 与 novelty 不可混用）**：`sequence_first_release` 为 **14.9_OR_EARLIER=43、15.0=278、UNCHECKED=115**，合计 436；它与最终 novelty 的 269/43/12/2/110 是不同分类，因为 TR1c/cross-panel 排除和 alignment 证据缺口会改变独立资格。ID 集合验证显示 33 个 raw TR1c exact 扣除 pre14.9 后为 12 个独立排除 ID；五类互斥并守恒 436，family+temporal 记录合计 11,226。
- **Actual（历史边界）**：早期报告的“所有 first-release UNCHECKED”和 `TEMPORAL_FIRST_RELEASE_UNCHECKED` 全局限制仅是 release-lineage 尚未闭合时的历史快照；当前只能对 269 条使用 `NOVEL_SEQUENCE_RELEASE_SUPPORTED`，其余记录继续保持各自旧序列、暴露或 UNCHECKED 状态，不能称全 temporal 独立。
- **Artifact / links**：[family release summary](/home/rna_jev_pdb_rfam_label_lineage/source_data/rfam_release_lineage_20261127/round1_map_v2/release_summary.json)、[release 独立验证](/home/rna_jev_pdb_rfam_label_lineage/source_data/rfam_release_lineage_20261127/final_release_verification.json)、[registry recommendation](/home/rna_jev_pdb_rfam_label_lineage/source_data/rfam_release_lineage_20261127/final_registry_proposal/registry_recommendation.json) 和 [provenance](/home/rna_jev_pdb_rfam_label_lineage/source_data/rfam_release_lineage_20261127/final_registry_proposal/provenance.json)、[novelty strata](/home/rna_jev_pdb_rfam_label_lineage/source_data/rfam_release_lineage_20261127/final_registry_proposal/novelty_strata.json) 和 [temporal 独立验证](/home/rna_jev_pdb_rfam_label_lineage/source_data/rfam_release_lineage_20261127/final_registry_verification.json)。
- **Limitation**：family release≠sequence release。上述 family 数字不支持全 temporal 独立、clean-OOD、SOTA 或模型选择； temporal 资格必须由逐序列证据另行判定。
- **Next**：继续冻结逐序列 novelty、TR1c/cross-panel 排除和 UNCHECKED 守恒，不修改 PPT。

## 27. ArchiveII full3966 CPU 恢复：BPfold release 总体、MXfold2 composite 最强单

- **Claim（full 当前面板）**：ArchiveII full3966 现有 **5 个全覆盖身份**。`bpfold_official_release/archiveii` 为描述性最强总体/集成，`mxfold2/archiveii_official_reference_3966/cpu_recovery_v1` 为描述性最强单预测器；均仅 `DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION`，不授权独立 SOTA 或选模。
- **Phenomenon**：原 clean2544 之外有 1,422 条官方记录，baseline 覆盖恢复会新增 clean-frozen+CPU-recovery composite 身份。它们与 author official release、项目 replay 和 xens2 是不同 prediction identities，不能把 full 指标回写到 clean 原表。
- **Intervention**：只对官方 clean2544 之外的 1,422 条做 CPU baseline 恢复，原始 sequence/SHA 不改写；合并后独立重算 full3966 和 clean2544 双面板。缺预测不补 0，外部 Test 不选模。
- **Expected observation**：BPfold release、MXfold2、CONTRAfold、EternaFold 和 RNAstructure Fold 为 5 个 full 身份；Vienna 没有旧 clean 记录级预测时不得构造 full，MaxExpect 未补齐时保持部分覆盖，xens2 不足 full 时不得给 full 差距。
- **Actual（full3966）**：BPfold official release 为六 checkpoint 集成，3966/3966，micro-F1=**0.7974584107660678**、macro-F1=0.8200813622940291，为描述性最强总体/集成。MXfold2 CPU composite 为 deterministic single predictor，3966/3966，micro-F1=**0.7023050285738565**、macro-F1=0.7457279552838411，为描述性最强单预测器。其余 full 身份为 CONTRAfold、EternaFold、RNAstructure Fold。
- **Actual（clean 原值不变）**：clean2544 的最强集成仍为 BPfold replay micro-F1=0.8999331354520079，最强单仍为 RiNALMo-ft=0.8204522394549935；full 新增结果不得替换这些 clean 身份或其 CI。
- **Actual（full 禁差距与部分覆盖）**：xens2 仍只有 **2544/3966**，coverage=0.6414523449319214、`PARTIAL_COVERAGE_NOT_RANKED`，full 差值保持 **`UNCHECKED_XENS2_NOT_FULL_COVERAGE`**，不得计算 full Δmicro/Δmacro 或 CI。Vienna MFE/centroid/MEA 虽各完成 1,422 条 CPU 恢复，但 clean2544 的 `n_with_prediction=0`、`BLOCKED_NO_PREDICTIONS_IN_PANEL`，所以 full 也只有 1,422/3,966；**无 clean 记录预测不构造 full**，不以空结构或补 0 合并。MaxExpect 仅恢复 12/1422，composite 为 **2556/3966**、missing=1410，继续 `PARTIAL_COVERAGE_NOT_RANKED`。
- **Artifact / links**：[full3966 system metrics](/home/rna_jev_archive_label_lineage/artifacts/archiveii_full3966_cpu_recovery_20261127/system_metrics.json)、[coverage gaps](/home/rna_jev_archive_label_lineage/artifacts/archiveii_full3966_cpu_recovery_20261127/official_coverage_gaps.json)、[registry recommendation](/home/rna_jev_archive_label_lineage/artifacts/archiveii_full3966_cpu_recovery_20261127/registry_recommendation.json) 和 [round3 clean-clone delivery](/home/rna_jev_archive_label_lineage/artifacts/archiveii_full3966_cpu_recovery_20261127/verification_round3.json)（`PASS_CLEAN_CLONE_AND_DELIVERY`、shared registry diff EMPTY）。
- **Limitation**：覆盖恢复不解决训练暴露，也不把 CPU composite 变成官方 release 或独立模型证据；full 最强值仅是当前 5 个全覆盖身份内的描述。
- **Next**：继续锁定 xens2 禁 full 差距、Vienna 不构造 full 和 MaxExpect 部分覆盖门，不修改 PPT。

## 28. ArchiveII clean/full 联合不变量：full 新增不得回写 clean 或 base 历史

- **Claim（跨 artifact 一致性）**：full3966 CPU 恢复只新增 full 面板与 composite 身份，不改变 clean2544 的 xens2、BPfold replay、RiNALMo-ft 指标、delta、paired macro CI 或最强身份。base 的“仅 official release 全覆盖、无 full 单预测器”仅为恢复前历史。
- **Intervention**：逐字段比较 base `archiveii_official_reference_20261127/system_metrics.json` 与当前 `archiveii_full3966_cpu_recovery_20261127/system_metrics.json` 的 clean 三个关键身份，并分别核对 base full=1 与当前 full=5。
- **Actual（clean 不变量）**：xens2=0.8156522070450173、BPfold replay=0.8999331354520079、RiNALMo-ft=0.8204522394549935；三者的 metrics、相对 xens2 delta 和 paired macro CI 均与 base 一致。clean 最强集成仍是 BPfold replay，最强单仍是 RiNALMo-ft。
- **Actual（full 与部分覆盖不变量）**：当前 full 为 BPfold release=0.7974584107660678 总体、MXfold2 composite=0.7023050285738565 最强单；xens2 full 差距仍 UNCHECKED，Vienna 无 clean 记录不构造 full，MaxExpect 2556/3966 不排名。任何 full 数值回写 clean 或 base 历史均触发联合验证失败。
- **Artifact / links**：[base system metrics](/home/rna_jev_archive_label_lineage/artifacts/archiveii_official_reference_20261127/system_metrics.json) 与 [当前 full-recovery system metrics](/home/rna_jev_archive_label_lineage/artifacts/archiveii_full3966_cpu_recovery_20261127/system_metrics.json)。
- **Limitation**：跨 artifact 一致只证明 clean 数值未回归，不证明 full 覆盖模型训练独立、官方 release 与 composite 身份相同，或任何 margin 结论。本轮不修改 PPT。

## 29. Matched 自然收敛：训练终态成立，不等于四臂结构结论

- **Claim（matched 训练终态）**：matched_control 已按预注册 patience 自然收敛，停止于 **step=19000**，restore best 为 **step=16500、validation_loss=0.35262170191251785**。`stop_reason=CONVERGED_VALIDATION_PAIR_HAMMING_STRUCTURED_MARGIN_AFTER_TERMINAL_UNFREEZE`，不是固定 step cap、人工停止或配额失败。
- **Phenomenon**：停止状态和恢复后 best 状态的 `bad_evaluations` 不同，若混用会误判合同：stopped convergence 连续 5 次未改善，restored-best convergence 在 best 时为 0。
- **Intervention**：以 `9cc035be714ce0487593ce042aaee259d5a6cd7d` 的独立只读审计核对 run_meta、19000 validation、best/resume checkpoint 和三条训练 ledger，不启动额外结构评估或模型选择。
- **Expected observation**：19000 stop=true、best=16500、bad=5；best_convergence 除 bad=0 外与 stopped convergence 相同。best/resume 的模型张量及 run identity 一致，训练/验证链无缺失。
- **Actual（convergence 状态）**：`convergence.best_step=16500`、`best_validation_loss=0.35262170191251785`、`bad_evaluations=5`、`eligible=true`；`best_convergence` 为同一 best/loss、**bad_evaluations=0**、eligible=true。step19000 validation_loss=0.40534604989096473，`improved=false`、`save_best=false`、`stop=true`，与 patience=5 的停止序列一致。
- **Actual（ledger 完整性）**：train/batch/compute 各 **19,000** 行，validation **38** 行（500–19000、步长500），`duplicates=0`、`missing=0`、`record_order_consistent=true`、`losses_finite=true`。batch 与 compute 的记录顺序逐步一致，没有以终态元数据代替原始 ledger。
- **Actual（best/resume 张量一致）**：resume 为 step16500、`restored_best_eligible`，SHA256=`5a9167ff1d26327c38900ec6e248fd752b8520eac7dc91d1a253afcb3735f1a6`；best 为 step16500、`best_eligible`，SHA256=`acec9e6f3bc025597fef673422ab2b1d4081587ba7a5172b3494ce9c05e42ad0`。best 的嵌套导出与 checkpoint 张量一致，best 与 resume 的 encoder/head/residual 三组 SHA256 全部相同，`best_resume_tensor_identity_match=true`。
- **Actual（ensemble 审计时状态）**：ensemble 与 matched 的共享科学身份一致，首个 eligible best 为 **step=10000、validation_loss=0.40810874768920063**，`improved=true`、`save_best=true`、`best_step=10000`、`bad_evaluations=0`、`stop=false`；current best SHA256=`7407f2dcc27204b8ca78d359af0ac8fd6bf7e66d99bc7c2ca0c69dc0fdfd397d`。审计时 resume 已到 **10500**、20/33 unfrozen，下一批为 10501 且匹配 saved stream，run_meta 仍 running、converged=false。
- **Actual（四臂结论门）**：当前 scheduler 仅 `completed_arms=[matched_control]`，ensemble 为 running/resume，`evaluation=null`、`scientific_conclusion=null`。matched 一臂自然收敛不得外推为四臂完成，ensemble 10000 best 不得与 matched 16500 best 直接比较为结构优劣；独立结构评估仍未开始。
- **Artifact / links**：[matched/ensemble 只读审计](/home/foldnar_incident_evidence/matched_converged_19000_audit.json)（`PASS_MATCHED_19000_AND_ENSEMBLE_RECOVERY_READONLY_AUDIT`）和 [审计源码](/home/rna_jev_margin_incident_recovery/tools/audit_structured_margin_converged_19000_v1.py)。`structural_model_selection_performed=false`、`additional_evaluation_started=false`、`scientific_conclusion=null`。
- **Limitation**：matched 自然收敛只证明该臂训练终止和 best 恢复资格，不能证明 margin>NLL、机制特异性、pair/stem 改善或 SOTA；其余三臂未完成，四臂结构结论仍不存在。
- **Next**：继续核对 19000 步 ledger、38 次 validation 和 best/resume 张量一致性；ensemble 只在审计时记录恢复进度，不修改 PPT。

## Claim → evidence 索引（审稿时逐项回溯）

| Claim | Phenomenon | Intervention | Expected observation | Artifact | Limitation |
|---|---|---|---|---|---|
| 当前容量扩张不稳健 | seed0/raw 曾正向 | 三 seed、两臂同 control | pair/stem 跨 seed 一致改善 | `capacity_freeze_audit.json` | 当前形式/model-dev；因果未分离 |
| 当前 SRD 不优于 MEA | raw 与结构脱节 | 冻结 checkpoint 五解码臂 | pair-F1 严格升且 precision/stem 保持 | `srd_summary.json` | 单来源单设置；多模型未运行 |
| stacking closure 停止 | 边界 FN 环境 | 同参数四臂+两 destroy | macro 保持且全面优于 destroy | `pilot_verdict.json` | seed0；destroy 不等于随机有益 |
| GEC seed0 FAIL | ensemble 机制待证 | NLL 四臂+独立复评 | ensemble 胜过全部相关对照 | `independent_reeval.json` | 单 seed/model-dev；不推广到所有 context |
| FN 主要为非正 score-margin | GEC 残差需定位 | frozen-forward 分类/反事实 | 可守恒分解并形成候选靶点 | `ensemble_context_summary.json` | 描述性；不证明 margin 有效 |
| 上下文有因果作用；真实 ensemble 非一致最优 | 不同训练臂无法分离输入作用 | 4 模型 × 4 输入 frozen crossover | 换输入可改变预测，但 native ensemble 应胜过替代输入才支持一致有益 | `gec_frozen_context_crossover_v1/full/summary.json` | 支持前一命题，不支持后一命题；seed0/model-dev |
| NLL 仅恢复结构参照 | 收敛元数据缺陷 | raw ledger/checkpoint 双实现复核 | 唯一资格缺陷解除且科学门不动 | `qualification_summary.json` | 不是科学 PASS 或 margin 参照 checkpoint |
| margin 待检验 | 98.31% FN 定位 | seed0 margin 四臂至各自收敛 | 固定 MEA 下通过全部科学与对照门 | authorization/startup/scheduler；结果尚缺 | 当前无收敛结构结果，不得阳性宣称 |
| structRFM 旧派生标签审计 | 历史预测无法绑定官方权重 | 官方 v0.0.8 全量新推理+独立评分 | 5,388 条身份/coverage/raw 指标完整 | `structrfm_official_bprna_new_metrics_20261126.json` | 标签来自 BPfold 预测；仅 circular-audit-only；不是独立 SOTA |
| Official-reference 下 BPfold/CONTRAfold 相对 xens2 的差值 | 旧 raw 标签循环且预测身份不同 | 同 5,388 官方参考 strict micro-F1 相减 | 身份/coverage 一致且差值可复算 | `bprna_new_official_reference_5388_prediction_registry.json` | 描述性协议审计；不是候选提升、clean-OOD 或选模 |
| BPfold 官方参考 5,388 面板 | 项目旧标签与官方参考不同 | 官方参考重建+第二实现复算 | 同面板/同协议/全覆盖描述比较 | `bprna_new_official_reference_5388_prediction_registry.json` | 描述性协议审计；0.9617 禁排名；不选模 |
| LinearPartition 19,000 nt 仅 baseline 可计算 | 短序列不能外推全长 | 官方 CPU 全长运行+raw 独立复算 | 不截断；coverage/recall/资源完整 | `7AM2_1/run_report.json`、`metrics.json` | 单条 holdout 描述；GEC unsupported；不用于选型 |
| 标签软复核与零删除 | 来源/配对质量不能混为实验真值 | 148,285 条 raw 独立重算与守恒 | 软复核不删除，硬冲突另行授权 | `label_quality_v1/summary.json` | DATABASE/COMPARATIVE_INFERRED 非实验真值；无 flag 非 clean |
| 预训练暴露分层 | 下游去泄露不等于 backbone 未接触 | 已知来源 exact+homology 独立重扫 | 分层守恒且不升级 UNKNOWN/UNCHECKED | `pretrain_exposure_v1/summary.json` | 完整官方语料 UNCHECKED；dev 非 clean-OOD；仅描述性表现 |
| 下一页暴露分层与 margin 准入 | 总分掩盖暴露，运行状态冒充结果 | 冻结分层+独立评估器身份门禁 | 分层显式出现；双训练完成后再复评 | exposure registry；margin evaluator `536b79e` | 当前双训练未完成、margin 无结论；不改 PPT/GPU/原 16 门 |
| Model-dev v3 暴露登记 | 分层需连接 12 个历史 run | 非 import 独立重算与三轮验证 | 序列/pair 守恒；additive-only | `foldnar_v3/verification_summary.json` | 无新性能；EXACT sensitivity-only，其余 diagnostic-only；不选路线 |
| Evaluator×v3 用途与16门不变 | 分层不能改写主科学门 | 12 run 逐层+evaluator 源码联合门禁 | EXACT sensitivity-only；其余 diagnostic-only；16门原样 | evaluator tools；v3 registry/manifest | 不 rescue/veto，不选 checkpoint/arm；margin 无结论 |
| ArchiveII official-reference | 旧标签循环且预测覆盖不齐 | 未过滤官方参考+冻结预测独立复算 | clean 2544 同协议值与 paired macro CI；3966 仅全覆盖可描述 | `archiveii_official_reference_20261127/system_metrics.json` | base 快照无全覆盖单预测器；当前见 full 恢复；audit-only，不选模、不是 margin 结果 |
| ArchiveII full3966 CPU 恢复 | clean 外 1422 条可补预测但身份不同 | CPU composite+full/clean 双面板独立复算 | 5 个 full 可描述；BPfold 总体、MXfold2 最强单 | `archiveii_full3966_cpu_recovery_20261127/system_metrics.json` | xens2 无 full 差距；Vienna 不构造 full；MaxExpect 不排名 |
| ArchiveII clean/full 联合不变量 | full 新增不得回写 clean 或 base | base/current clean 指标、delta、CI 逐字段比较 | clean 三个身份不变；base full=1、当前 full=5 | base 与 full-recovery system metrics | 不证明训练独立；composite≠official release；无 margin 结论 |
| TS0 projected/official 双面板 | 子集原值与 full 恢复身份不同 | projected 1288 冻结复算+full1305 CPU 覆盖恢复 | 保留子集值/CI；7 个 full 可描述，部分覆盖不排名 | `bprna_ts0_system_metrics.json`；`ts0_official_coverage_recovery_20261127` | RNAstructure 1302 不排名；homology/UNCHECKED 暴露不省略 |
| TestSetB projected/official 双面板 | projected 高值暴露与 full 覆盖身份不同 | projected 428 复算+full430 CPU 覆盖恢复+暴露审计 | 原 20 个 projected 值不变；7 个 full 可描述 | `testsetb_official_coverage_recovery_20261127/system_metrics.json` | 247/428 精确暴露不独立排名；RNAstructure 429 不排名 |
| Full 覆盖恢复联合不变量 | full 新增不得回写 projected 或暴露 | 恢复前后 artifact 逐字段比较+CI 冻结 | 四个 projected 身份不变；full 各 7 个；旧零覆盖仅历史 | archive pre-recovery 与 ts0/testsetb current metrics | 不证明训练独立；RNAstructure 部分覆盖不排名 |
| Rfam family/sequence release | family accession 首次出现不等于序列首次出现 | 官方 family table/seed/FASTA/alignment 独立谱系 | family 计数守恒；sequence novelty 另行分类 | `rfam_release_lineage_20261127` | 不支持全 temporal 独立；proposal-only，不选模 |
| PDB/Rfam 标签谱系与 PDB62 重建 | 原面板缺坐标/时间/配对谱系且互相重叠 | 官方 PLK、mmCIF/DSSR scheme、metadata、Rfam release 独立重算 | 当前 62/91、58 一致/4 差异；旧 58/95 仅历史 | `pdb_rfam_label_lineage_audit_20261127`；`pdb_rcsb_rebuild_20261127/round6_combined` | 不排名；TS-hard 非独立；Rfam temporal 非自动 novel；不选模 |
| 配额事故与 9500 恢复 | Errno 122 中断且大 checkpoint 压迫 teams 配额 | 双臂 preflight+原合同恢复+/home 暂存+终态 teams 镜像 | 下一批连续；四臂自然收敛后才评估 | `recovery_preflight_local.json`；`recovery_scheduler_state.json` | matched 10000 validation 已完成但未收敛；四臂未完成；无 margin 结论 |
| 第二次 scheduler 配额事故与接管 | 旧 scheduler 写 teams state 再遇 Errno 122 | 本地 state/events/lock/PID+原 PID adopt+最终小证据镜像 | 不重复 matched；所有活跃工作留 /home | `recovery_takeover_status.json`；本地 scheduler state/events | 四臂未收敛；评估 pending；无 margin 结论 |
| matched 自然收敛 | 早停与恢复 best 是不同状态 | 19000 停止序列+16500 best/resume+19000 步审计 | stopped bad5、restored-best bad0；张量一致 | `matched_converged_19000_audit.json` | 仅 matched 训练终态；ensemble 继续；四臂无结构结论 |

## 摘要 / 标题准入审稿清单

只有对应**直接、已完成、可回溯**的证据全部存在，相关表述才可进入摘要或标题；预期、计划、诊断关联、工程 PASS 均不够。

- [ ] **状态与范围**：写明 seed、数据面板、model-dev/external、单设置/多种子；scheduler、smoke、authorization 不得改写成“实验成功”。
- [ ] **训练完整性**：每个被比较臂均达到预注册收敛并 restore best；日志、代码/配置哈希、checkpoint、预测和独立 artifact 审查一致。启动或固定 step cap 不等于收敛。
- [ ] **结构终点**：使用同一冻结 MEA 的 micro/macro pair/stem F1 与 precision 门；不得以 AUPRC、r@P0.8、局部 recall 或训练/validation loss 替代结构失败门。
- [ ] **机制特异性**：ensemble/候选必须直接优于 matched 及 destroy/independent 对照；destroy 单次相对较好不得写成“随机有益”，也不得事后选择冠军。
- [ ] **因果作用与一致有益分离**：crossover 可支持“替换上下文会改变预测”，但真实 ensemble 一致有益仍需 native ensemble 直接优于替代输入及训练对照；不得用前一命题替代后一命题。
- [ ] **目标子群与不确定性**：隔离/长跨距等 claim 必须有计数、paired sequence CI、家族集中性/leave-one-family-out 等预注册检查，不能只报一个子群均值。
- [ ] **跨目标比较**：margin>NLL 必须来自各自独立收敛后的同协议结构复评；禁止比较 NLL 与 margin validation loss 或用 same-step 诊断作优胜结论。
- [ ] **多种子与外部性**：稳健机制需后续获授权并完成多种子直接证据；严格 OOD、外部测试、SOTA 只能在相应面板与 provenance 实际完成后宣称。
- [ ] **残差措辞**：98.31% 只能表述为“冻结 ensemble FN 中非正 score-margin 的分类占比”；不得写成准确率、提升幅度、因果比例或 margin 有效性。
- [ ] **NLL 资格措辞**：`QUALIFIED_STRUCTURE_REFERENCE` 只能支持“可作结构参照”；不得省略为“NLL/GEC 科学通过”，且原 seed0 FAIL 必须同场保留。
- [ ] **负面结果完整性**：容量、SRD、stacking 与 GEC 的停止/失败不得被后续候选叙事抹去；任何新阳性 claim 必须同时说明其不同干预与直接证据。
- [ ] **官方 checkpoint 旧派生标签措辞**：structRFM 历史结果必须同时保留旧标签 `DERIVED_FROM_BPFOLD_PREDICTIONS` 与 circular-audit-only；不得写成我们的正向性能、独立排名、clean/OOD 或 SOTA。
- [ ] **跨 registry 禁排名门禁**：19/19 派生标签 raw/clean 资产仅 circular-audit-only，旧差距禁止回写；28/28 official-reference rows 不含 0.9617。新面板仍只是描述性冻结协议审计，不是 clean-OOD 或选模证据。
- [ ] **BPfold 协议与 SOTA 限制**：旧派生标签及全部旧 raw/clean 差值只能 circular-audit-only，不得排名。当前描述比较只能使用 SHA256=`4924e351...` 的 official-reference 5,388 面板；0.9617 仅循环审计不得排名，official release 与项目 replay 身份不得混用，外部 Test 不得选模。
- [ ] **Model-dev v3 / 原16门**：EXACT 只能 sensitivity-only，HOMOLOGY/UNKNOWN 只能 diagnostic-only；不得 rescue/veto primary 或选择 checkpoint/arm。原16门定义与 seed0 FAIL（16门、12失败）不变，current evaluator 只能是 `536b79e450c1ad18b67e06cb39ecb3730546221b`，旧 evaluator 不得回写。
- [ ] **下一页与 margin 准入**：下一页必须显式展示暴露分层及完整官方语料 UNCHECKED；margin 双训练未完成、仍无结论。独立评估器固定 `/home/rna_jev_margin_evaluator`@`536b79e450c1ad18b67e06cb39ecb3730546221b`，不得干扰 GPU 或改原 16 门；本轮不修改 PPT。
- [ ] **Pretrain exposure / SOTA 边界**：dev 必须保留 7,978 EXACT、539 HOMOLOGY、626 NOT_DETECTED_UNKNOWN；bpRNA-new 为 5,381/5,388 EXACT，其余 7 条 HOMOLOGY。完整官方语料 UNCHECKED，dev 不能外推 clean-OOD；只能写 exposure-aware descriptive performance，UNKNOWN/UNCHECKED 不得写成 clean。
- [ ] **Label quality 边界**：TR1c `DATABASE` 与 CHANRG `COMPARATIVE_INFERRED` 不得称为实验真值；95,052 noncanonical 与 10,250 short-span 仅为可重叠的软复核记录数，硬冲突 0、实际删除 0，不得自动删除或把无 flag 写成 clean。
- [ ] **长序列边界**：19,000 nt 只能陈述 LinearPartition 单条 holdout 的全长 coverage、真实资源与 BPP/MEA recall；candidate recall 不得冒充结构准确率，baseline coverage 不得冒充 GEC coverage，GEC unsupported 不得省略。
- [ ] **ArchiveII 覆盖与 CI**：旧派生标签只能 circular-audit-only；clean 2544 的 xens2/BPfold replay/RiNALMo-ft 必须使用 official-reference 真实值并标明 paired macro 差值 CI，RiNALMo-ft CI 跨 0。base 3966 仅 official release 全覆盖、单预测器 UNCHECKED 仅历史；当前 full3966 有 5 个全覆盖，BPfold release=0.7974584107660678 总体、MXfold2=0.7023050285738565 最强单，仅描述。xens2 2544/3966 不得给 full 差距，Vienna 无 clean 记录不构造 full，MaxExpect 2556/3966 不排名。
- [ ] **TS0 projected 与 full 覆盖**：保留 projected xens2=0.8032405714870655、RNAformer=0.7568974968024850 与暴露；full1305 现有 7 个全覆盖，BPfold release=0.6812363564199246 仅描述，RNAstructure 1302/1305 不排名。旧零全覆盖仅恢复前快照。
- [ ] **TestSetB 暴露与 full 覆盖**：保留 projected xens2=0.8403202122116432、RiNALMo-ft=0.8670430033503144 和 TR0 exact=247/428，CI 为正也不可独立排名。full430 现有 7 个全覆盖，MXfold2=0.6468595682161368 仅描述；RNAstructure 429/430 不排名，旧零全覆盖仅历史。
- [ ] **PDB/Rfam 禁排名与 temporal 限制**：当前 PDB 为 62 条重建、91 条 UNCHECKED，4 条 DSSR scheme 修复，58 条 raw pair 一致而 4 条差异须重算；不得排名。旧 58/95、56/2 仅 round3 历史快照。TS-hard 28/28 跨面板重叠且与 TS1/TS3 不相交性不成立。Rfam temporal 43 条已在 14.9，first-release/pair provenance UNCHECKED，其他记录也不得自动称 novel。
- [ ] **Rfam family/sequence release**：11,186 条唯一 family，first-release 14.10=166、15.0=99，14.9或更早10,921、UNCHECKED40；family release≠sequence release；temporal 仅 269 条 NOVEL 支持，43 旧序列、12 TR1c、2 cross-panel、110 UNCHECKED，原始 TR1c exact=33 与 12 排除为不同字段；sequence first-release=43/278/115 与最终 novelty 不可混用，五类互斥守恒 436，不得称全 temporal 独立。registry 仅 proposal，shared registry 未修改。
- [ ] **配额事故与恢复状态**：Errno 122 不得写成模型/CUDA 失败；9500 preflight 只证明可恢复，matched 10000 validation 已完成但 stop=false，不等于四臂收敛。完成臂 0、evaluation=null、scientific_conclusion=null；活跃大 checkpoint 暂 /home 是事故例外，最终 teams 镜像尚未完成，当前无任何 margin 科学结论。
- [ ] **matched 自然收敛**：stop19000/best16500、loss=0.35262170191251785；stopped bad5 与 restored-best bad0 不得混用。best/resume 张量和 19000 步/38 次 validation 必须一致；ensemble 审计时继续 10500，四臂未完成仍无结构结论。
- [ ] **第二次 scheduler 事故与接管**：第二次 Errno 122 是旧 scheduler 写 teams state 失败，不是训练失败；4111890 只能 adopt 唯一 matched PID4077864，不得重复启动。训练/checkpoint/scheduler/评估根均在 /home，teams 只写最终小证据镜像；评估 pending、四臂未收敛。best15000 loss=0.3896151208942566、bad0、stop=false 只是中途 validation，不是科学结论。

## 校验

`tools/verify_scientific_story_ledger.py` 检查本文相对 Markdown 链接、上述核心 JSON 数值/裁决、98.31% 的分母与计算、crossover 输入指标与 independent 扰动差值、NLL 科学 FAIL 不变、structRFM 旧派生标签身份与指标、BPfold 标签谱系/circular-only 边界、official-reference 面板 SHA/独立验证、xens2/BPfold/CONTRAfold 新差距算术及19派生资产/28官方行跨registry禁排名门禁、LinearPartition 全长 coverage/资源/recall 与 GEC unsupported 边界、label quality 来源分类/软复核/零删除守恒、pretrain exposure 分层/完整语料 UNCHECKED/SOTA 限制、下一页准入门及 FoldNAR/label-quality/pretrain-exposure/独立 margin 评估器/model-dev-v3 源提交、v3 三轮验证及 evaluator×v3 分层用途/原16门联合门禁、BPfold-protocol 源提交、ArchiveII official clean 2544 真实值/paired macro CI、3966 全覆盖与单预测器禁排名门、full3966 五全覆盖/BPfold release/MXfold2、xens2 禁 full 差距、Vienna 不构造 full 与 MaxExpect 2556 部分覆盖门、clean/base 指标与 CI 联合不变量、TS0 projected 1288 原值/CI、full1305 七全覆盖/BPfold release 与 RNAstructure 部分覆盖门、TestSetB projected 428 原值/CI、TR0 247/428 精确暴露、full430 七全覆盖/MXfold2 与 RNAstructure 部分覆盖门、恢复前后 projected 指标/CI/暴露联合不变量、当前 PDB62/91UNCHECKED/4 条 DSSR scheme 修复与 58/4 配对一致性、旧 58/95 和 56/2 历史身份、TS-hard 重叠、Rfam temporal 限制、Rfam family 11186/first-release 166/99、family≠sequence release、temporal 269/43/12/2/110 互斥分类、sequence first-release 43/278/115、ID 集合守恒与 33/12 暴露门、磁盘配额 Errno 122、9500 断点连续性、matched 10000 validation 真实状态、第二次 scheduler Errno 122、aa9f1c4 最终修复/takeover 证据、本地 scheduler state/events/lock/PID 与 matched adopt 唯一性、训练/checkpoint/评估 /home 存储及 teams 小证据镜像门、best15000 validation/bad0/stop=false 与四臂未收敛状态、matched stop19000/best16500 自然收敛、张量/ledger 完整性、ensemble10000 best/审计时10500 与四臂结构结论门、PPT 第 33 页来源及 margin 最新调度/无结论状态。它只读源码与 artifact，不查询或控制 GPU，不修改科学门。
