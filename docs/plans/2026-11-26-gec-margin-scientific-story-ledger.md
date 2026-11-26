# GEC / structured-margin 科学叙事与训练留痕

日期：2026-11-26
分支：`scientific-story-ledger-20261126`
整合基线：`780044cc2c4c2fa3ae355edf1fb6e1d68eb44019`
范围：bpRNA model-dev；既有 VL0 为 196 条，GEC 为 TR1c cluster80 dev 9,143 条。均非外部测试、严格 OOD 或 SOTA 证据。

## 当前一句话

已完成证据支持的是“容量、SRD、stacking closure 按预注册门停止；GEC seed0 科学 FAIL；残差主要定位于非正 score-margin FN；crossover 证明上下文输入有因果作用，但真实 ensemble 并非一致有益；NLL 四臂恢复为结构参照”；structRFM 官方 raw 结果及其与 BPfold/CONTRAfold 的差距只是外部基线算术；**structured margin 是正在检验的候选目标，不是已成立结论，也没有任何 margin 四臂结构结果可供摘要或标题使用。**

## 时间、提交与运行快照

- 科学结果及 NLL 资格链整合到 `780044c`；capacity/SRD/stacking 的结果提交分别为 `c289384afcd801cb0509941c40e973f7c320f230`、`03f9afe430c445fffd5d0e386ba3c16970747ab8`、`dde6641db437d1d1c35ee876843d30d0a5cf6f68`。
- GEC 四臂训练与独立复评为 `bd01b4a120a8ed3b7c77690f593a8caa03fb533d`、`342a714b24998ed0c739908613f21e50866d6b56`；残差冻结审计为 `0589b2f489aa456f7a885b6d5dfe625b5de1b76b`；frozen context crossover 为 `810441e7480ea06a717097c57a49130c820c9636`，对应 PPT 第 33 页记录提交 `03f7f4e14ef906359cbd9d1b519089b6e61116f4`。FoldNAR 整合 HEAD 为 `fd497e9819e39a29887ec26698a46c91284d230b`，真实 deck 为 14 张、记录至第 33 页。第 34 页的 round71 builder 引入提交为 `1782cd64a7c6253beefe77c93634a2652f6fab5e`，render/OCR 留痕提交为 `5a7ccbd418f520eb44f02f0fdf6fbdf9f6ff238e`，仍须等待最终 hash-verified 评估。
- NLL 资格修复为 `c4bfbb8734af3535393ef424f4c1b0b39af7b081`、`e388b504d7888b8a355d33fe80150ebd650a0adc`、`ecc40d3a8ec545bf332199a935e6380b718d415b`。
- margin 数值核心、独立合同复核、显存探针与正式调度分别为 `92f0bc2e6440bcbfc682d2f8ca3e502ec86a6b48`、`55be915f6e7eda26254d87ab133624fa99cd5b8c`、`cd61c57f48cc0fe543b869b3319e4bd8b7f820f5`、`a846d90519e575a79b7db76f63692afb755e8ac6`；`a846d90` 已核对与 GitHub 分支远端一致。
- **登记排队快照**：`structured_margin_full_v1/generation_01/startup_report_v1.json` 为 `QUEUED_PROBE_MEMORY`，scheduler PID `3976280`，完成臂 0，`scientific_conclusion=null`。**本次更新实时状态**：`scheduler_state.json` 为 `QUEUED_ARM_MEMORY`，`matched_control` 与 `ensemble_context` 两臂运行，`destroy_interval_shuffle` 与 `independent_context` 两臂等待资源，完成臂仍为 0，`scientific_conclusion=null`。这里区分初始快照与后续调度；排队、探针、启动、运行或等待资源的进程都不是科学结果。本 ledger 不修改调度器、资源门或任何运行任务。

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

## 9. structRFM 官方 raw：补齐外部基线，不是候选正向结果

- **Claim（仅官方 raw 口径）**：官方 structRFM v0.0.8 单 checkpoint 已在项目 raw bpRNA-new 5,388 条上完成新推理与独立评分；该结果描述外部 baseline，**不是我们的候选正向性能，也不是 clean SOTA**。
- **Phenomenon**：历史 structRFM 预测与缺失的历史权重字节无法回溯绑定，不能用旧 replay 代替官方 checkpoint 结果。
- **Intervention**：固定官方 v0.0.8 checkpoint、官方 constrained DP 与项目 raw 标签，完成 5,388 条 GPU 推理；独立核对 prediction identity、coverage 与 exact-pair 指标，历史预测保持不变。
- **Expected observation**：只有新官方 checkpoint 的完整预测、身份和 raw 指标全部通过，才可进入同口径 baseline 算术；不得借 raw 排名推断去污染泛化或 SOTA。
- **Actual**：coverage=**5,388/5,388=1.0**，TP/FP/FN=**73,864/62,616/61,320**，micro/macro pair-F1=**0.5437893868896873/0.5338105702966888**。状态为 `PASS_INDEPENDENT_OFFICIAL_5388_IDENTITY_AND_METRICS`，执行提交 `b53a4f8245776eda43f1bb2ee651a654df68425b`，并由 FoldNAR HEAD `fd497e9819e39a29887ec26698a46c91284d230b` 整合。
- **Artifact**：`F/structrfm_official_v0.0.8/reports/structrfm_official_bprna_new_metrics_20261126.json`、`structrfm_official_provenance_20261126.json` 与 `full_delivery_verification_20261126.json`；原始预测为 `F/structrfm_official_v0.0.8/predictions/structrfm_official_v0.0.8_bprna_new_5388.jsonl`。
- **Limitation**：标签范围为 `PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED`，`raw_not_clean=true`；未作精确/同源去污染，checkpoint 训练暴露未解决。它是官方外部单模型 baseline，不是 FoldNar/GEC/margin 候选结果，不授权 clean、OOD、外部候选性能或 SOTA 宣称。
- **Next**：只允许与同一 raw 5,388 标签、同一 strict micro-F1 口径的 baseline 做绝对差距描述；候选正向性能仍须等待其自身合格实验。

## 10. BPfold / CONTRAfold 绝对差距：同 raw 口径的 baseline 算术

- **Claim（绝对 micro-F1 差距）**：在项目 raw bpRNA-new 5,388 条、相同 strict micro-F1 口径下，BPfold 六 checkpoint 集成与 CONTRAfold 确定性单模型分别高于 structRFM 官方单模型 **0.19321803153859785** 和 **0.15297454451260917**；这不是相对百分比，也不是我们的候选提升。
- **Phenomenon**：raw baseline 排名同时混有不同模型形式与未解决训练暴露；只报一个较高分数会误导读成集成/单模型等同或 clean SOTA。
- **Intervention**：固定项目 `bprna_new.jsonl`（SHA256 `271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9`）、zero-based inclusive exact pairs、5,388/5,388 coverage，直接计算 comparator−structRFM 的 pooled strict micro-F1 绝对差。
+- **Expected observation**：三者身份、模型形式、coverage 与 raw 标签口径一致可比时才允许相减；BPfold official-reference 标签结果和 CONTRAfold TR1c-clean 子集均不得混入。
+- **Actual**：BPfold `six_checkpoint_ensemble` micro-F1=**0.7370074184282851**，绝对差=**0.7370074184282851−0.5437893868896873=0.19321803153859785**；CONTRAfold `deterministic_single_predictor` micro-F1=**0.6967639314022964**，绝对差=**0.6967639314022964−0.5437893868896873=0.15297454451260917**。数值来自 FoldNAR HEAD `fd497e9819e39a29887ec26698a46c91284d230b` 的冻结 registry，structRFM 执行提交仍为 `b53a4f8245776eda43f1bb2ee651a654df68425b`。
+- **Artifact**：`/home/rna_jev_foldnar_integration/manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json` 的 `strongest_project_raw` 与 `strongest_project_raw_single_predictor`；structRFM 来源仍为 `F/structrfm_official_v0.0.8/reports/structrfm_official_bprna_new_metrics_20261126.json`。
+- **Limitation**：三项都是 `raw ranking only; not clean SOTA` 口径；raw 未去污染，BPfold 训练 overlap 未解决，CONTRAfold 的 TR1c-clean 结果属于另一面板。两个差距都不是 FoldNar/GEC/margin 候选正向性能，不证明集成优于所有单模型，也不授权 clean/OOD/SOTA。
+- **Next**：保留该差距作为 raw 基线差距事实；若未来评估候选，必须独立取得同协议候选预测并重新计算，不能从 baseline 差值反推候选得分。
+
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
| structRFM 官方 raw baseline | 历史预测无法绑定官方权重 | 官方 v0.0.8 全量新推理+独立评分 | 5,388 条身份/coverage/raw 指标完整 | `structrfm_official_bprna_new_metrics_20261126.json` | 外部 baseline；raw 非 clean；不是候选正向性能或 SOTA |
| BPfold/CONTRAfold 相对 structRFM 的 raw 绝对差 | baseline 模型形式与暴露不同 | 同 5,388 raw 标签 strict micro-F1 相减 | 身份/coverage 一致且差距可复算 | `bprna_new_recovered_prediction_registry.json` | 外部 baseline 算术；不是候选提升或 clean SOTA |

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
- [ ] **官方 raw baseline 措辞**：structRFM 结果必须同时保留官方外部 baseline、5,388 条 raw 非去污染与候选隔离边界；不得写成我们的正向性能、clean/OOD 或 SOTA。
- [ ] **raw 绝对差距**：BPfold 集成、CONTRAfold 单模型与 structRFM 的差值只能是同面板 micro-F1 绝对算术；不得混用 official-reference/TR1c-clean 标签、改写成相对百分比或候选提升。

## 校验

`tools/verify_scientific_story_ledger.py` 检查本文相对 Markdown 链接、上述核心 JSON 数值/裁决、98.31% 的分母与计算、crossover 输入指标与 independent 扰动差值、NLL 科学 FAIL 不变、structRFM 官方 raw 身份与指标、BPfold/CONTRAfold 绝对差距算术、PPT 第 33 页来源及 margin 最新调度/无结论状态。它只读源码与 artifact，不查询或控制 GPU，不修改科学门。
