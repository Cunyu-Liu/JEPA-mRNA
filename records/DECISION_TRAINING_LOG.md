

## §15.925（2026-11-14）Lattice parsing beam-CKY解码实验

- 原文精确Lattice parsing仍是O(L^3)，不能直接替代成更快的精确DP；实现的是每个span保留前B状态的beam-CKY，复用相同score、canonical/min-loop mask和非交叉约束。
- 大beam与精确Nussinov完全一致，小beam通过无交叉、无共享端点和min-loop检查，相关测试累计31项通过。
- VL0网格B=4/16/64/256的micro-F1均0.8755383766，B=4最快，因此冻结B=4。TS0外部验证micro-F1=0.7791047470，与精确DP逐值一致，三次预测一致且无unsupported。
- 19,000nt beam4长链已启动，比较其与精确DP约7小时的成本及预测一致性；未形成长链精度或速度结论。


## §15.926（2026-11-16）双视图 P1 离线候选与远端连接阻断

- 当前唯一执行依据为 `docs/plans/2026-11-16-dual-view-jepa-accuracy-contract.md`；D1、RIPE、旧局部 Pair-JEPA、19,000 nt beam 和三 R2D 投票均未恢复或启动。
- P0 已通过，原 xens2 bpRNA-new 重放与历史预测逐条一致；P1 旧身份的 19 项 CPU、H1–H6 CUDA 和部署证据在并发源码变化后不得直接沿用。
- 本地候选补 SIGReg state-dict resume 回退与旧 H0 CLI/metadata 默认值；基于 `adc1b48` 的隔离测试共 27/27 PASS。首次 resume 回归 1/7 失败为旧配置属性缺失，修复后全过；首次临时 import 和 HTTP2 clone 失败亦保留在候选 manifest。
- `px2-jcy.matpool.com:22` 多轮持续 `Connection refused`。候选未上传，真实 CUDA、宿主、部署与最终 P1 gate 均未按新哈希执行；H0–H6 pilot 未启动，无精度或收敛结论。
- 已准备 `.round56_work/run_p1_candidate_gate.sh` 和 `validate_p1_candidate_gate.py`：上传前后校验哈希、拒绝覆盖 artifact，只有全部工程证据通过才输出 `PASS_P1_ENGINEERING_ONLY`。合成聚合测试可接受完整证据并拒绝伪造 converged=true；该合成测试不是 P1 或科学结果。


## §15.927（2026-11-17）新集群 P1 候选上传与 Blackwell 环境门

- 新集群 `px1-jcy.matpool.com:27693` 使用密钥认证，未在脚本、文档或 Git 中保存密码。原 px2 连接失败历史保留，不改写。
- 远端新增 SIGReg buffer 测试已与本地 resume 候选合并，合并后本地和远端均为 28/28 PASS（21 双视图＋7 resume），候选 train/dual/test SHA256 分别为 `67cc71d0…8392a4`、`908b3a52…a6266`、`66331fd6…b208`。
- 多个 scp 连接曾被间歇关闭，均在源码替换前停止；单文件上传哈希复核通过，上传器加入每文件有限重试，最终远端 28 项测试完成。
- 真实 CUDA 随后失败并停止：RTX 5090 sm_120 不受旧 torch 2.5.1+cu121（最高 sm_90）支持，报 `no kernel image is available for execution on the device`。没有 CPU 降级，没有执行宿主训练或 H0–H6 pilot。
- 旧 `lucaone` 保持不变，正在准备克隆隔离环境 `lucaone_cu128`（torch 2.7.1+cu128、triton 3.3.1）。首次安装命令因前台超时下截断，未声称安装完成；新一轮 artifact 使用 `candidate_cu128_v1`，旧失败目录保留。


## §15.928（2026-11-17）P1 通过并启动受限 model-dev pilot

- `lucaone_cu128` 在 RTX 5090 的真实张量计算通过，torch=2.7.1+cu128、capability=(12,0)。H1–H6 合成 CUDA forward/backward 全过，FLOPs=29,626,006,352（仅 dual-view 模块 profiler 口径），无 torch-geometric。
- 28 项双视图＋resume 测试、H0/H2 真实宿主一步、旧 resume 与序列-only 部署全过。部署 checkpoint 仅 encoder/head，DualViewJEPA/StructureTokenEncoder/build_structure_example/make_view_masks 调用均为 0，确定性两次一致。
- 旧 resume 夹具第一次误假定 SIGReg step=1；实际序列与结构两个 target 正则使显式值和 buffer 均为 2。夹具改为二者相等且 >=1，失败目录保留，独立目录重跑通过。
- P1 gate：`/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/manifests/p1/candidate_cu128_v1/gate_decision.json`；commit=`0a44150` 已推送。
- H0–H6 pilot 已冻结并并行启动：Plan-A s0 step20000 初始化、TR1c/VL0 和原四主损失、seed42、B=4、500 步受限机制预算、每100步 VL0、patience=3。只使用 model-dev 早停和比较，外部测试集不参与。
- 自动评测仅对 best checkpoint 做序列-only 精确 DP 和独立计数。H2 必须超过 H0/H3/H4/H5/H6 才允许 P3；否则停止并保留负结果。当前尚无 pilot、收敛或精度结论。


## §15.929（2026-11-17）pilot 初始并发 OOM 与受门恢复

- 七臂同时启动发生真实显存超订：H3/H5 在反向阶段 CUDA OOM，H4/H6 被系统终止。失败 stdout 全部保留，不写成科学负结果或收敛。
- 原监控器误调用系统 python3（集群不存在），提前退出；已改为 `lucaone_cu128` 环境 Python。监控错误与训练失败分开记录。
- 当前先等待 H0/H1/H2 完成，再以 H3/H4、H5/H6 两臂一组从原 checkpoint 恢复；无 checkpoint 的臂从相同初始化开始，配置/数据/seed/主损失不变。禁止再次扩大并发，失败不自动重试。
- 恢复脚本 commit=`928d3f6` 已推送。全部七臂完成后才运行 model-dev 序列-only 评测和独立裁决，当前仍无 pilot 结果。


## §15.930（2026-11-17）七臂早停完成与评测 OOM 恢复

- H0–H6 均在 step400 按 patience=3 自然早停，best 均为 step100；训练阶段恢复未改配置、数据或 seed。
- H5/H6 的 model-dev 序列-only 精确 DP 评测已完成。首轮七路统一评测再次发生 CUDA OOM，H0–H4 缺 `model_dev_eval.json`，聚合器按设计拒绝输出不完整裁决。
- 现仅补缺失评测，H0/H1、H2/H3、H4 分组，不重训、不换 checkpoint、不访问外部测试。恢复脚本 commit=`d305abb` 已推送。
- 全部七份 model-dev 结果和独立计数齐套后才生成 pilot 裁决，当前仍无最终科学结论。


## §15.931（2026-11-17）H0–H6 pilot 负结果与主线停止

- 七份 model-dev 结果已齐套，独立 TP/FP/FN 与 micro/macro-F1 复算通过。decision=`/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/manifests/p2/pilot_v1_decision.json`，SHA256=`b6b47fb4e779d2510d1e2a571a9d6458ff183bee198ae3e7481f41271a211c41`。
- macro-F1：H0=0.8514596293、H1=0.8511860896、H2=0.8442637614、H3=0.8394108513、H4=0.8480104682、H5=0.8471288978、H6=0.8472334148。
- H2 相对最强对照 H0 的 Δ=-0.0071958679，且未超过 shuffle/random/MLP/global-mean。H2 micro-F1=0.8658792361 虽略高，但预注册判定指标为 macro-F1，不得换指标掩盖失败。
- 裁决=`STOP_PILOT_NEGATIVE_DEV_ONLY`，p3_started=false。三种子正式训练、主八＋Rfam 冻结评测、motif 和功能下游均未启动。该单 seed、VL0、step100 best 的机制负结果不是外部未见家族精度结论，也不支持结构模态有效的主张。
- 全部配置、best/resume、逐序列预测、显存、停止原因和失败日志保留；不调整 SIGReg 权重、训练预算或对照身份挽救结果。


## §15.932（2026-11-17）负结果 PPT 与最终复核

- 指定副本 `交接文档/PRISM_组会汇报_20261107_完整版_副本.pptx` 仅新增第20页，含七臂 macro/micro-F1、H0/H2 差值、停止原因与限制。SHA256=`1f9f68bb389d681ce0fcf5e3aae1d8d9a2b8d709a46b77d58f218abd6f05a6ac`。
- LibreOffice 多次中文缺字仅作转换器失败保留，未当成交付验收；PowerPoint 原生第20页中文显示正常、无遮挡。ZIP、20页、边界、微软雅黑、1.5行距、数值和 manifest 一致。
- 新页补齐 a:ea/a:cs 字体设置，绘图脚本 commit=`93c1ffb` 已推送。P1、pipeline、恢复代码此前分别推送 `0a44150`、`d5dc3ef`、`928d3f6`、`d305abb`。
- 最终科学状态不变：`STOP_PILOT_NEGATIVE_DEV_ONLY`；未启动三种子、正式冻结评测、motif 或功能下游，无新架构精度突破。


## §15.933（2026-11-17）v2 数据与暴露审计门

- v2 合同及全部 spec/tasks 镜像已更新，v1 负结果不恢复、不改写。
- 三遍生成＋三遍复读＋独立验证通过，audit=`manifests/v2/data_audit/data_exposure_audit.json`，SHA256=`ee1b7df5ac3e53f9a043e2531469b9883a8fa3effd93c4c158a0eef0b69c7e82`；代码 commit=`2cfe676` 已推送。
- TR1c=42564 唯一序列、15521 个 80% 簇；pair=1157273，删除 crossing=4539、illegal=4598。train 的 PDB resolution/NMR/missing residue 元数据不存在，标 blocked。
- Rfam 与 TR1c exact overlap=284、cluster overlap=1038；TS0 cluster overlap=418；ArchiveII=104；TS1/2/3/hard=27/7/12/12；TestSetB=16；bpRNA-new=1。exact=0 不得写成无同源污染。
- ncRNA 去除 TR1c exact 后 train/val/test=102060/17054/25038，删除行=3804/270/304，唯一重叠=2000/270/304；原始文件保留，clean overlap=0。
- RiNALMo 全预训练语料未在本地冻结，暴露审计 blocked；因此本门是 `PASS_AUDIT_WITH_EXPLICIT_LIMITS`，不是 clean-OOD 证明。


## §15.934（2026-11-17）完整宿主机制测量与初始化对照启动

- 28 项回归与独立 profile 复核通过。完整宿主 forward/backward profiler=`199494385321` FLOPs，batch=2、length=41/45、峰值=`2906620416` bytes，scope 为 RiNALMo＋原 ResNet2D/Turner＋dual-view 的 profiler 支持运算；精确推理 DP 和 Python/C 解码不估计。
- 共享编码器梯度 cosine=`-0.0529394841`，nll/jepa norm=11.4621/4.1695，范围 last_four_unfrozen_encoder_tensors，不再是 v1 的 None。该单 batch 负值只记录梯度冲突，不作普遍机制结论。代码 commit=`09d19fe`。
- 原生 RiNALMo H0/H2 已以 seed42 在真实 CUDA 上启动，和 v1 Plan-A warm-start 使用相同数据、主损失与受限预算。原 resume 损坏，优化器延续 blocked；只比较新优化器。代码 commit=`017bd84`。


## §15.935（2026-11-17）原生初始化对照完成

- 原生 RiNALMo H0/H2 在 step500 均为 `step_cap`、`converged=false`，best@500，未触发早停；model-dev 序列-only 精确 DP 评测完成。
- native macro-F1：H0=0.6539801156、H2=0.6588592058，H2−H0=+0.0048790902；warm Plan-A 分别为0.8514596293/0.8442637614。原生欠训练和 warm-start 遗忘/初始化是明确混杂，小正值不得写成结构模态有效。
- 优化器延续因原 resume 损坏而 blocked，未伪造历史优化器状态。report=`manifests/v2/initialization_report.json`，SHA256=`325f79f1986e78d96d8467bd8fcc3dbea998c503ac0d24a71edfacbbf413bf29`。
- 监视器曾在另一臂评测未完成时退出，但随后读取两份完整 stdout 和 artifact 确认双臂均完成，未把监视器状态误记为训练失败。


## §15.936（2026-11-17）BPfold 实现与 RNA–蛋白界面 QA

- 已确认旧 `--motif` 是 Vienna/Turner 特征，不是 BPfold/BRIQ。新实现使用官方 `key.energy`（MIT，76086行，SHA256=`b657c553…ce8b`），按官方 outer/inner key 和 min-normalization 构造两通道，31项回归通过；commit=`ac8065e`。
- 旧 checkpoint 缺少新投影键，首轮四臂均 strict-load 失败。现仅允许 `motif_input.*` 新键缺失，其余 strict；新目录 `artifacts/v2/bpfold_v2` 重跑，旧失败保留。修复 commit=`4140cea`。
- RNA–蛋白界面未使用旧 RNA–RNA labels。1EHZ 无蛋白链，不可作界面样本。10BE/10FB 以最近 RNA–肽链重原子距离 <5Å 生成接触和同链负样本：10BE=35/767、10FB=19/66（界面/非界面），QA 通过但只是两 entry 资产门，不是训练结果。
- BPfold dev 队列正在运行；只有 P1 超过 P0/shuffle 才允许外部评测。


## §15.937（2026-11-17）BPfold 初始化问题闭环

- BPfold 初次 strict-load 缺新键后，连续修正了允许缺失集合的命名空间及 state_dict 合并方式，失败 v1/v2/v3 均保留，未用 strict=False 绕过。
- 当前做法：先验证缺失恰为 `motif_input.*`，再合并其零初始化 state_dict，最后 strict=True 加载。v4 两臂已确认真实进程存活，31项回归通过；commit=`db90e26` 已推送。
- 等待 v4 四臂完整 dev 裁决，不提前宣称 motif 有效。


## §15.938（2026-11-17）BPfold 前两臂评测与界面全量 QA

- inner/outer、shuffled 均在 step400 自然早停，best@100。`load_plana` 初评未重建 BPfold 模块而失败，已按 `bpfold_motif` 元数据修复并补评，commit=`9ef41b3`。
- inner/outer 与无 motif H0 完全一致：macro=0.8514596293、micro=0.8638651784；shuffled macro=0.8515088595，仅高0.0000492302，不显示有意义 motif 收益。Turner=0/energy-only 尚未完成，不提前下最终结论。
- 界面 holdout 真实标签独立 QA：202候选中83条实际 JSONL，79条同时有正负残基，119个 blocked；不能把202当可训练数。输出 SHA256=`35d15ede…e672`，验证代码 commit=`15f1e75`。
- train 标签仍在生成，界面模型尚未训练。


## §15.939（2026-11-17）RNA–蛋白界面旧标签与探索基线撤销、原始 mmCIF 重建

- 复核发现旧提取器把整条 gemmi auth chain 的枚举位置当 RNA 索引，未核对 polymer/label_seq/参考序列；同名链可包含蛋白。例：12DM_3 旧“166 个 RNA 残基”实际为蛋白，正确 RNA 参考序列长度1473。旧 QA 只检查距离阈值和计数，不能证明坐标—序列对应。
- 旧 train/holdout 标签、sequence/random 三个 seed 的500步未收敛探索输出及误标 macro-F1 的数值均撤销，不进入科学结论。目录保留为 `artifacts/v2/interface_unaligned_v1`；分片命名冲突的中间件另保留为 `interface_shard_collision_v1`，不得合并复用。
- 新提取直接读取原始 mmCIF entity/poly-seq-scheme/atom_site，按 label_asym/label_seq 对齐完整 RNA 参考序列；蛋白伙伴限 polypeptide，未观测残基不生成负标签，坐标/序列不一致显式 blocked。保存 partner、实验方法、分辨率、mmCIF 哈希和接触距离。
- 真实结构回归覆盖蛋白/RNA 区分、完整 RNA 映射、缺失残基、参考序列不一致和多链拒绝；分片文件与合并键名错误已修复并保留失败。独立验证器不复用提取器，重读原始 mmCIF 后逐残基复算最近肽链原子、阈值、伙伴、标签守恒和 PDB＋序列簇隔离；QA 未完成前没有启动新界面训练。
- 新 probe 只使用冻结 RiNALMo/H0/H2 表征，结构特征只从冻结模型预测 pair scores 计算；不把 H2 的下游 probe 当成主线精度救援。对照含组成保持的链内标签 shuffle、长度/GC 和手工特征。训练无固定步数上限，内部 PDB＋簇验证连续30 epoch 平台期才停止；正类 F1、二分类 macro-F1、逐链指标与 micro/macro-AUPRC 分开，阈值仅由内部验证选择。
- 既有 v1 双视图和 BPfold motif 负结果不变。界面重建、测试器和训练器完成不等于下游性能结论，RiNALMo 预训练对 PDB 的暴露仍 UNKNOWN。


## §15.940（2026-11-18）界面来源复核、全量亲本恢复与长链结构工程边界

- 原始 mmCIF 直接映射的未修饰子集 train=639/804、holdout=167/202 已完成独立逐残基 QA：接触距离、真实肽链伙伴、参考序列、缺失坐标、PDB/序列簇隔离全通过；train/holdout exact sequence overlap=0。该子集已保留为 `interface_unmodified_subset_v1`，不直接外推为全量通过。
- 其余 train165、holdout35 条全部经 hash-bound `parents.json` 和原始 `labels.json` 证明为 modified-parent 规范化差异，完整 polymer 精确一致，无近似对齐或静默删样本。真实11EL_A的PSU位于一基label_seq=516、输出零基索引515、亲本U；11项回归通过。
- 最终全量804/202标签正在按此规则重建并做独立复算。旧蛋白链混入、分片碰撞、合并键名、CIF引号和测试期望错误全部保留为工程失败，不写成模型负结果。
- H0原ResNet2D对最长训练链9MR4_5=7224nt的真实CUDA结构特征工程门OOM：进程约31.24GiB，额外452MiB分配失败。该链编码器/结构全长可运行不能声称长序列有效；不截断、不换近似结构模型。h0_structure/h2_structure未启动，保留 `long_feature_engineering.log`；序列-only冻结表征与结构辅助臂分开验收。
- 界面修复代码 commit=`8628d1f` 已推送，未夹带旧工作区其他未提交文件。RiNALMo/H0/H2的真实CUDA短链冻结表征与有限值检查通过，仅为工程预检。
- 指定PPT已增加至22页：第20页双视图负结果，第21页数据/初始化审计，第22页BPfold motif负结果。第20–22页中文备注、ZIP结构和PowerPoint原生视觉复核通过；PPT SHA256=`8d8a740de8e6e4cdc85aac7a78a8d0d8642a4232b7f81033e0852dce9ca71fd9`。界面新性能尚未入图。
- 预印本 `paper/rna_evidence_v2_claim_matrix.md` 明确当前不是精度突破或可投稿的完整算法结果；后续界面probe不抢救主线，结构臂工程blocked不得隐藏。


## §15.941（2026-11-18）全量界面 QA 与长链注意力工程门通过

- 最终全量标签 train=804/804、holdout=202/202，观测残基1,453,871/231,155，正负残基770730/683141与125642/105513；未观测位置不生成负样本。train/holdout PDB、80%簇和精确序列重叠均为0。QA=`artifacts/v2/interface/label_qa_summary.json`，SHA256=`1c996c9c9a412afdb8b30398f3bd33239914fccde4f71321ff132ad1404b32bb`；contacts SHA256=train `6abae275…4fe9`、holdout `e915f145…acdb`。
- 原全注意力在19,000nt编码器上前向OOM，结构pair representation在7,224nt亦OOM，失败保留。新增查询维512分块全序列注意力，不改Q/K/V、RoPE、softmax、模型权重或输入长度，不启动19,000nt beam。
- 首轮广播mask切分错误、数值差异、空pair一致率定义错误均保留并修正；没有放宽科学门。最终短/中链relative L2<0.01且cosine>0.9999，原/分块重复前向稳定，完整VL0 196/196 pair集合相同、mean pair agreement=1.0、macro-F1均为0.8514596292970453。不是bitwise等价声明。
- 7,224nt与19,000nt完整编码器前向均有限，峰值6,139,810,304与8,163,816,960 bytes；只证明计算可运行，不证明长序列有效。结构辅助h0_structure/h2_structure仍因完整ResNet2D pair矩阵OOM保持blocked，不能用编码器成功替代。
- 代码 commit=`4eb7e7d` 已推送。两个并行CUDA特征任务已恢复，七类sequence-only对照在三套缓存齐套后自动进行42/2026/3407三seed无固定步数上限训练；尚无界面性能结论。
- RiNALMo论文核对预训练源为RNAcentral22.0、Rfam14.9、Ensembl109/56与NCBI nt，36M唯一序列、17M个70%簇。Rfam与RNAcentral页面有CC0证据；公开源仅用于暴露审计，不进入训练。nt当前文件405.95GB且历史快照未确定，Ensembl精确清单与许可仍待核验，因此预训练暴露完整审计尚未关闭。


## §15.942（2026-11-18）界面数据页原生验收、全量表征与首批收敛 run

- 指定PPT第23页仅展示界面数据与工程门：804/202链、真实RNA–肽链接触、同链负样本、独立隔离、完整VL0 pair等价与长链边界，不包含界面性能宣称。PowerPoint原生中文/表格/布局复核通过，PPT SHA256=`0cabb3167adda74a7a222f845790e586ff794030115dfed618774fbc5d12d182`。构建脚本 commit=`aa577d7` 已推送，证据JSON已同步远端。
- RiNALMo/H0/H2的train/holdout全部冻结表征完成，使用经验证的512查询块，原checkpoint、QA与kernel哈希齐全。序列-only七类对照随后并行运行，峰值按run保存，不占满显存而超订。
- 已自然收敛的四个run：RiNALMo seed42/2026/3407与H0 seed42，均 `internal_validation_plateau_30_epochs`，best epoch分别为178/241/219/244。其余run继续执行，未把首批AUPRC或F1写成完整矩阵或功能提升结论。
- 后续只有在21个可运行run全部自然收敛、逐残基概率与独立计数复核、长度/实验方法分层及结构blocked证据齐套后，才生成性能汇总/PPT。原双视图与BPfold主线负结果不变，界面probe不作为抢救叙事。


## §15.943（2026-11-18）等价小批量加速与 H2 编码器身份纠正

- 独立残基的小批量 BCE 与逐链 BCE 在真实 CUDA 上具有相同损失和全部参数梯度，普通/带结构模式及批边界均通过；13项回归通过，commit=`332106f`。仅改变计算打包，不改数据、模型、初始化、主损失、验证或阈值。best/resume 保留，已收敛结果不重训。
- 随后发现 H2 seed42 与 H0 seed42 的 probe checkpoint 和逐残基预测完全相同。独立读取 pilot checkpoint 证明 H0/H2 step100 编码器逐张量完全一致，max_abs=0；warmup_head_steps=1600，两个 split 的特征缓存也一致。不是文件路径错误。
- 因当前 H2 只是 H0 共享冻结编码器的别名，停止其余重复 H2 run，seed42 与失败日志保留。身份文件 `artifacts/v2/interface/h2_encoder_identity.json`，代码 commit=`77018f6` 已推送。汇总不计算独立 H2−H0 效果，不把别名当成结构学习正结果或新负结果。
- 后续仅保留 RiNALMo、H0、手工特征、组成保持随机、长度和 GC 六类独立对照，自然收敛后再聚合。结构辅助臂仍工程blocked，完整F2不宣称通过；该 warmup 限制也不用于恢复或调优主线。


## §15.944（2026-11-18）最终界面指标、精确预训练暴露与汇报验收

- 六类独立界面probe共18 run均自然收敛，未以固定步数上限终止。H0 micro-AUPRC=0.7715063953±0.00526532、二分类macro-F1=0.6853715146±0.00413946；RiNALMo=0.7215875464、手工特征=0.5437771、链内标签shuffle=0.62916613、长度=0.55373391、GC=0.55341552。H0在100–499nt仅0.6232793522（n=23），≥1500nt为0.7725384619（n=88）；shuffle已有较高分数，不能把H0高分直接称为复合物理解。
- sklearn独立复核18/18 run通过，最大指标差2.220446049250313e-16。AUPRC使用distinct score thresholds消除同分顺序依赖，旧值保存在各run的`result_before_auprc_tie_fix.json`；仅重算指标，不重训、不改checkpoint、预测或阈值。最终汇总`artifacts/v2/interface/converged_baseline_summary.json` SHA256=`2b465cd3816407cb92c886d2b9fe9e3f00d48d81834f34113fab9ccda7f24870`；分层汇总`interface_stratified_summary.json` SHA256=`ca01a2e9caa51468ba1e90b91c8f539e6b740c0ab82de3b92371781c39e9f3e5`。
- H2与H0仍是共享冻结编码器的别名，不能计算独立JEPA功能收益；h0_structure/h2_structure因原ResNet2D全pair矩阵OOM保持blocked，完整F2未通过。序列-only界面probe不是二级结构主线或结构模态的抢救证据。
- RNAcentral22.0精确重叠：TR1c=33379/42564、VL0=184/196、bpRNA-new=5303/5388、TS0=1098/1288、Rfam=9989/10790、ncRNA test=21452/25038、界面holdout=86/202。Rfam14.9对应为28914/42564、112/196、5380/5388、1047/1288、10223/10790、23756/25038、28/202。证据`/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit/exact_overlap.json` SHA256=`6cd88fc5e3b6385ac15f179c225bf575f79c1cec308c1328ce1af31baf558877`。阳性匹配已排除clean-OOD叙事，但不量化为全部预训练污染率。
- 预训练官方源已核对为RNAcentral22.0、Rfam14.9、Ensembl109/56与NCBI nt。Ensembl已确定315＋30731=31046个ncRNA文件，断点下载中；验收manifest尚未生成。nt当前快照约405.95GB，历史快照未确定，未下载；完整语料、80%同源与家族审计仍未闭环，不能用部分来源反推无接触。
- PPT第24页仅报告污染感知界面probe，第25页披露精确暴露。首次更新脚本对25页 deck 的页数断言失败，未保存部分修改；已改为显式页定位并保留幂等修复。第25页初版TS0与底部文字重叠，已缩短表格并重新分配底部位置。PowerPoint原生第24/25页中文、表格、备注和无重叠复核通过，几何/边界QA同步通过；最终deck SHA256=`8786c22c61e7ae911cf49962803f67bd71282511ad3162eab2c3df67f4554bc1`。
- `spec.md`、`tasks.md`、`checklist.md`及`spec/`、`spec_sync/`镜像已同步为污染感知状态。文档完成不是实验完成；v1主线负结果、原生欠训练、H2别名、motif负结果和结构OOM均未删除，未启动三种子正式精度评测，无新单模型精度突破。
- RiNALMo原文再次核对长度过滤为16–8192nt，训练时对>1022nt输入随机裁剪；后续审计区分源序列精确成员关系与逐碱基训练暴露。新增Ensembl独立验收和精确重叠脚本：验收未完成前均显式拒绝执行，pending日志保留。PPT脚本与Ensembl下载器commit=`7e8d79f`已推送。

主要复核命令（不重复执行已通过的耗时全量距离QA）：
```bash
cd /home/rna_jev_xens2_dualview_round56
/root/miniconda3/envs/lucaone_cu128/bin/python tools/verify_interface_labels.py
/root/miniconda3/envs/lucaone_cu128/bin/python tools/validate_rinalmo_attention_chunk.py
/root/miniconda3/envs/lucaone_cu128/bin/python tools/aggregate_interface_results.py
/root/miniconda3/envs/lucaone_cu128/bin/python tools/analyze_interface_strata.py
/root/miniconda3/envs/lucaone_cu128/bin/python tools/verify_interface_metric_qa.py
/root/miniconda3/envs/lucaone_cu128/bin/python tools/audit_pretrain_exact_overlap.py
```


## §15.945（2026-11-18）预训练资产深检与同源审计准备，尚无检索结果

- 新增 `verify_ensembl_download_manifest.py`：检查源清单URL双射、31046文件、字节数、独立SHA256、gzip CRC/解压、非空FASTA和16–8192nt长度分布。新增 `audit_ensembl_exact_overlap.py`，只有独立资产验收通过才运行；两个pending拒绝路径均实测通过，commit=`34b1a63`已推送。下载尚未完成，没有扩展精确重叠数值。
- 同源查询冻结为165854条唯一序列，query manifest SHA256=`5d929e307d93c5ee09c220e5d3ed33a770cc1c700e856f9f9a708cdc86a3af21`，FASTA SHA256=`2ee5ed75c602a533dbd55f61ac855708670d250aff2564247fc08bc66f34a43c`。两次生成哈希一致，独立复读ID/长度/归属/文件哈希通过；commit=`4adfb82`。
- 固定拟用MMseqs2 17.b804f，逐查询easy-search、80% identity、`cov-mode 0`、`c=0.8`；不是以代表序列linclust替代逐查询搜索。最多保留300个target/query只用于命中识别，不声称完整同源计数；不检出不是无同源证明。
- MMseqs2首次安装因集群阿里云channel 404失败，改用显式官方conda-forge/bioconda URL并在`/home/rna_pretrain_audit_env`隔离安装，不改torch环境。版本未验收前不得启动检索。
- Ensembl可追踪target准备器和检索runner已加入，未验证资产/缺失target均拒绝执行；commit=`2c3b670`。这只是审计准备，不是预训练、模型实验或家族独立性结论。


## §15.946（2026-11-18）MMseqs2 工程通过与逐查询检索启动

- MMseqs2 17.b804f 已在隔离环境 `/home/rna_pretrain_audit_env` 安装并验证，官方channel安装成功；先前阿里云镜像404失败保留。现有CUDA训练环境未修改。
- 聚合器3项合成测试通过，覆盖80%阈值、确定性best、空命中、未知query、低于阈值与畸形行；commit=`9ea5478`已推送。
- 真实MMseqs2 smoke首次因`/teams`不支持临时symlink失败，第二次因临时目录父目录未创建失败，均无科学命中结果。改为`/home/rna_pretrain_homology_tmp`并显式创建父目录后，已知Rfam源序列在真实gzip target上以100% identity检出，格式/阈值/索引链路通过。`engineering_smoke.json`仅为工程证据；runner修复commit=`be9d91c`。
- 已并行启动RNAcentral22.0、Rfam14.9全量165854条查询的80% identity＋双向80%覆盖搜索，8线程/来源。Ensembl只设一个后续队列，等待既有下载manifest后依次执行独立验收、精确重叠、target构建、搜索与聚合，不重复下载。
- 当前没有全量同源命中汇总，也没有家族独立性结论。检索启动不等于完成，空命中不得写成无同源，完整历史nt快照仍未确定。


## §15.947（2026-11-18）Rfam14.9 同源搜索与独立计数通过

- Rfam14.9的165854条查询搜索与聚合已完成。原始结果`matches.tsv`为1,225,275,707 bytes，SHA256=`288bfab82dfbc4cb85cb9649dddb2569fa8c7e482bda8f70e1c456f93bbc91c4`；search/result/逐query文件哈希复读通过。
- 共155635条唯一查询检出80% identity＋双向80%覆盖匹配，10219条未检出仅表示本算法和阈值下未知。独立逐query复读ID、正负状态和best阈值一致，QA=`homology_round56/rfam_14_9/homology_independent_qa.json`。
- 关键集合：TR1c=37541/42564、VL0=184/196、bpRNA-new=5386/5388、TS0=1221/1288、Rfam=10530/10790、ncRNA test=24347/25038、界面holdout=93/198（长度过滤后的查询数，不等于原始202条全部评测）。
- 汇总`homology_round56/rfam_14_9/homology_summary.json` SHA256=`a486ccc03f832fbedb899fe69fe8c575b2256a15b7e66e2708aa4ab95f558bc8`。该结果只覆盖Rfam来源，未检出不证明无同源，不能当作完整家族独立性或干净泛化证明。
- RNAcentral搜索、Ensembl下载后独立验收/精确/同源队列仍在运行；联合汇总需三个来源全部通过，当前尚未生成。


## §15.948（2026-11-18）Ensembl 资产与精确暴露通过、RNAcentral 内存分块恢复

- Ensembl 31046文件全部下载并完成独立字节/SHA256、gzip CRC与非空FASTA验证，总字节701138004；原始记录4040439、16–8192nt过滤后4032030。verification SHA256=`1a9b2804e8cde6536506e8291abcd0cace34ff99b961581fe11957d90523c30b`。
- Ensembl精确重叠：TR1c=7333/42564、VL0=73/196、bpRNA-new=10/5388、TS0=195/1288、Rfam=391/10790、ncRNA test=2415/25038、界面holdout=5/202。`ensembl_exact_overlap.json` SHA256=`2f63c9631c83f4fbdceb2a9c13f7ba621a8acbf3fd1aae82f9e0c0a074ba2445`；计数已与verification独立对账，不与RNAcentral直接相加。
- RNAcentral首次prefilter估计185GiB并被系统终止，失败目录保留为`rnacentral_22_active_failed_prefilter_memory_v1`。固定identity、双向覆盖、查询与target不变，增加`--split-memory-limit 50G`恢复，已进入6分块处理，不将失败写成零命中。
- 聚合器改为流式保留best，6项测试通过；对Rfam原始15183453条比对得到与旧结果完全相同的155635条best，未改写Rfam历史汇总。修复commit=`33805b9`。
- 官方Rfam14.9 `full_region`与`family`元数据已下载并绑定SHA256；latin1按SQL声明解析。只按名称映射的`rfam14_9_family_mapping_summary.json`为部分方法：rfam_fam仅434条通过，不能解释为其余无家族。正在按精确序列→官方区域→家族映射，缺元数据与多家族歧义显式保留，完整家族独立性仍不可宣称。
- 公开bpRNA站点/Dropbox连接失败；一次猜测的PMC页面与DOI不匹配，未采用其任何内容。EuropePMC网页提取未成功，不伪造链接或补充材料身份。后续以可核验的Rfam精确序列映射为主，不从bpRNA内部数字猜家族。


## §15.949（2026-11-18）三源精确去重与第26页原生验收

- 三来源精确重叠按规范化完整序列去重，不把不同来源相加。TR1c=37870/42564、VL0=193/196、bpRNA-new=5381/5388、TS0=1253/1288、Rfam=10327/10790、ncRNA test=24912/25038、界面holdout=86/202。
- `plk`等名称对应不同序列，不能按名称直接合并。联合器对冲突候选直接回查源FASTA，并逐源复核原始unique/rows计数；16个集合均满足联合值不小于单源且不超过总和，独立不变量QA通过。`exact_union_summary.json` SHA256=`d441239311605a2572b3f561357531689b0ea24a19068e1ac417dca75148baa4`；commit=`8a63a5c`已推送。
- 该联合仅覆盖RNAcentral22.0、Rfam14.9、Ensembl109/56的精确匹配，nt、完整同源和家族审计仍未闭环；未命中仍是未知，不是clean或独立家族。
- PPT新增第26页，不改写第25页单源审计。表格同时列出各源和三源联合值，备注披露长序列裁剪及未闭环项。PowerPoint原生中文、五列数据、TS0与底部说明无重叠；deck SHA256=`d70e6d9f31f4cc512531b30dd4fd8cccfd7c25ee9e2a3f2a1f88e2d02b2c3b6e`，脚本commit=`115abb6`。旧deck在reporting目录保留`before_exact_union_sha8786`。
- Rfam家族前缀QA已完成两个集合，非全量结论：rfam_fam唯一官方家族8002、歧义110、缺元数据2111、无精确匹配567；bprna_new唯一4938、缺元数据442、无精确匹配8。TR1c继续运行，未计算完整训练—测试家族重叠。
- RNAcentral分块搜索、Ensembl可追踪target构建和家族映射均在继续；文档/PPT更新不等于科学门完成。


## §15.950（2026-11-18）Ensembl 同源、Rfam 家族映射与第27页验收

- Ensembl target/index 4,032,030条记录完成独立ID、长度、序列SHA和总数核验；小夹具先行通过。此前QA读取器前瞻头与文件句柄接口错误均保留并修正，未把检查器错误写成数据错误。Ensembl同源结果165854条查询中61092条检出，104762条未检出不等于无同源；summary SHA256=`6e1f4ee958ae8a45b0966816a3a287ea86654d2b97f00e18f770051a7e9a740c`，独立逐查询复核通过，QA脚本commit=`a01d14d`。
- Rfam14.9官方精确序列家族映射共58742行独立复核通过。rfam_fam唯一官方家族8002、歧义110、缺元数据2111、无精确匹配567；bprna_new唯一4938、缺元数据442、无精确匹配8；TR1c唯一24392、歧义35、缺元数据4487、无精确匹配13650。映射家族数分别1597、477、1732。
- 可映射家族交集中，Rfam与TR1c共享120，bpRNA-new为0。120说明下游split去重不能替代家族隔离；0只支持bpRNA-new的下游标签划分，不抵消RiNALMo预训练精确暴露。缺元数据、歧义和未匹配均不强行赋家族。summary SHA256=`bdef1812946c4c00c36801b1cbddfb4433a3d3c433b8ecf71a8a6256f79c8a99`，代码commit=`610ee8e`。
- RNAcentral第二次失败发生在第3/6分块，估计38GiB的prefilter与Ensembl搜索约22GiB叠加触发内存终止，前两块已完成，不能记零命中。Ensembl完成后以相同50GiB配置串行恢复；两次失败目录均保留，未降低identity、覆盖率或查询数量。
- PPT新增第27页，分别展示Rfam/Ensembl同源和家族映射，不把120或0包装成泛化或精度结论。原生视觉通过，deck SHA256=`582410ef9e10da4ccde1ce9443c88b670c7554a26b273703cdb5c4a4fd4fee83`，脚本commit=`0a0aa3d`。RNAcentral同源、跨来源联合、nt历史快照及预印本仍未闭环。


## §15.951（2026-11-18）三源同源联合完成与 nt 历史快照限制冻结

- RNAcentral 以相同80% identity＋双向80%覆盖、165854条查询和50GiB分块串行完成全部6个分块、alignment和聚合。检出151835，未检出14019不是无同源；summary SHA256=`3a5c1805823b12156bd51738ecb3cbce29bc51941135cf13d7442adccf623c21`。前两次内存失败目录保留，没有降低阈值或缩小查询集。
- 三源同源联合与独立逐query复核完成：至少一个来源检出162489/165854，未检出3365仍为未知。VL0=196/196、bpRNA-new=5388/5388、Rfam=10711/10790、TS0=1281/1288、ncRNA test=25012/25038、界面holdout=111/198。`homology_union_summary.json` SHA256=`69a0b13b2ef0cb061807e10f50efca3a6da7889ee006cd032e2e72bca2197131`。源内去重而非三列相加，未检出不等于无同源。
- 官方RiNALMo仓库树完整，main=`2c2c5c14a5ae609d8c560a5d9ca32e51e0288955`。`remote_data.json`仅给出实时nt.gz和current_release，没有历史nt日期或内容哈希；提交时间也不能证明训练快照。未下载当前nt冒充原始训练集。`nt_snapshot_assessment.json`状态`BLOCKED_HISTORICAL_NT_SNAPSHOT_UNPINNED`，SHA256=`5f4282bdcda743cc14de112a4b6e6c8d5436b9404dff3180553f27a04c336232`，commit=`96aa207`。这不影响已知来源的阳性暴露，但阻断完整历史污染分母。
- PPT新增第28页：联合检出、各源计数、未闭环nt与120个共享家族/下游0的边界齐全，PowerPoint原生视觉通过。deck SHA256=`abc4f31ecd043ec407bf8ad6e748eb93362e7f96371e63214717dbf35afd472e`，脚本commit=`9f586b6`。
- 当前阶段是**已知三源审计完成且nt不可追溯**，不是完整历史语料去污染、独立泛化或新模型精度突破。Rfam家族映射只覆盖精确可映射子集；v1主线、motif负结果及完整F2未通过均不变。


## §15.952（2026-11-18）污染感知预印本证据草稿与双重检查

- 新增 `paper/rna_evidence_v2_manuscript_draft.md`，3258词，定位“可靠数据、严格对照与机制边界”，不是新架构精度论文。正文含来源身份、精确/同源审计、家族映射、双视图负结果、初始化混杂、motif负结果、界面probe、局限、复现命令和官方参考来源。
- 首轮自定义复核发现长度分层浮点精度口径问题：正文四位小数正确，检查器改用四位比较，源数据和正文未改。随后7个模型臂、6类界面对照、7个查询集合及7份哈希全部通过。
- 原项目检查首轮Q1–Q12未填，已逐项补充：未评估的免DP校准、RLCD、50×速度和决策顺序明确不主张，污染、标签、OOM与nt限制给出具体落点；正文另增“Calibration and pair dependence”局限。复跑a/b/c/d全部PASS，Q1–Q12全部有落点。
- 本地与远端复核均通过，草稿SHA256=`f9b8469f8299a521168b87d47d0f0405b396157e0169dfb33c06463227c2cdc4`，代码和检查器commit=`ee75844`已推送。
- 文档状态仍为**非投稿就绪**：主精度门未通过、nt历史快照不可追溯、完整F2未通过，H2别名与结构臂OOM均保留。没有为迎合发表而补造引用或把负结果改写成突破。


## §15.953（2026-11-18）预印本 PDF 构建、表格修复与视觉验收

- 采用本机现有 Pandoc＋LibreOffice 构建 A4 单栏证据草稿，未新增依赖，未套用未经确认的IEEE模板或虚构作者信息。
- 首次PDF文本检查发现表格正文缺失和“堆叠能”不可提取；根因是Pandoc生成空`tblGrid`，不是科学数据或中文字体问题。已按页面宽度显式写入2–3列表格网格，禁止表格跨行断裂，并增加DOCX列数断言。
- 重建后DOCX列数为3/3/2/3/2，PDF包含关键数值、Q12、中文“Nussinov + Turner 堆叠能”及附录回答。首页、结果表和末页实际渲染通过：表格完整、无溢出或截断、中文正常、Q1–Q12与参考文献可读。
- 最终PDF共6页，SHA256=`7fb512bb33f91ed4a25fdf392e811722e91f4cea231f1a42993f7967ac46b7e1`；源Markdown保持`f9b8469f…cdc4`不变。构建脚本commit=`e0b6fd7`已推送，DOCX/PDF/manifest同步至`artifacts/v2/reporting`。
- 版式交付完成不等于投稿就绪或科学门通过。主精度门、nt历史快照、完整F2和H2独立效果仍未解决；不以排版替代实验。


## §15.954（2026-11-18）独立 PDF 重建与最终证据边界

- 在新的临时目录用相同源Markdown和构建脚本再次生成DOCX/PDF，未覆盖验收文件。归一化正文文本、全部表格文本、表格网格3/3/2/3/2和页数6均与已验收版本一致。
- 参考PDF SHA256=`7fb512bb…b7e1`，重建PDF SHA256=`3c5c3657…32c1`。元数据和容器时间戳可不同，因此只声明文本与版式可复现，不声明bitwise一致。临时中间件已清理，最终PDF哈希未变。
- 最终未决科学项仍为：历史nt快照不可追溯；完整F2未通过；H2无独立JEPA效果估计；原生初始化未收敛；正式三种子与外部精度评测未启动。预印本草稿和PDF是证据交付，不是上述科学门的替代。


## §15.955（2026-11-18）可发表性评估与后续证据门

- 当前3258词草稿及6页PDF适合作为内部技术报告，不足以支撑“新算法提高未见家族精度”的完整算法论文。数据审计和标签QA是贡献，但不能以排版、投票或下游proxy替代主科学问题。
- 建议优先走“污染感知的数据可靠性与评估协议”。必须进一步回答接触如何影响精度，而不只报告有接触：使用已冻结逐序列预测和预先固定的exact/homology/not-detected分层，not-detected仍为未知而非clean。
- 历史checkpoint、原始预测、集合和阈值不得改变，不能按效果选组。raw、own-clean、matched-clean、exposure-aware分别报告，只有身份/覆盖/独立计数齐全且支持样本数足够时才做配对区间和Holm校正；没有合格候选时保留无精度结论。
- 数据量、独立家族数、80%簇数和预训练来源数不可互换。单backbone、单seed model-dev、混来源标签不足以支持普适算法创新；完整nt缺失只限制完整污染分母，不抹去已知阳性暴露。
- 下一步先核对历史冻结预测身份和格式，再决定分层分析可行性。身份或覆盖不足则标blocked，不重新训练来填补叙事。


## §15.956（2026-11-18）冻结预测暴露分层：同源未检出对照不存在

- 原 xens2 的 P0 历史与重放逐序列完全一致，三个 checkpoint 的实际 SHA256 均与 P0 一致。bpRNA-new 共5388条、无跳过，独立复算 macro-F1=0.5422090871138705，micro-F1=0.553142073575729，TP=64542、FP=33639、FN=70642。
- 精确暴露组n=5381、macro-F1=0.5424236738973192；精确未检出仅n=7、macro-F1=0.3772531611514662。7条仍为未知而非clean，组间序列构成不同，不做显著性或因果结论。全部5388条至少在一个已知来源检出80%同源，同源未检出n=0；空组的零指标不是测得性能，因此现有bpRNA-new不能形成同源未检出精度对照。
- 只读分析不改模型、阈值或数据。旧替代预测有385条完整记录不同，旧macro-F1=0.542298358496208，不与P0身份混用。先前误计query注释导致的5388差异为审计器错误，已修正为385并保留错误目录；回归3/3通过。证据`xens2_bprna_new.json` SHA256=`c1c39477b84c3a139a52a091f1011548b1841c50bf6f5e572dc56b75318007ec`，代码commit=`e6dbffd`。
- 第29页PPT已收录该无对照结果，PowerPoint原生AX/渲染、源哈希/几何、ZIP和本地/远端deck一致性三类检查通过；deck SHA256=`5a6e87ab7594a25c535589a8111a9873b92238ec901f0ab001a344ca2c8ed19a`，脚本commit=`39700ea`。
- 更新后证据稿3346词，Q1–Q12、7臂/6界面对照/7查询集合/8哈希检查全通过，SHA256=`5ec633801ee951d392ac16d57f5401af9f2694ded8f9812fffcb193e667dd6a1`，commit=`8ccca43`。6页PDF首/中/末页视觉通过，SHA256=`09ba5a7ab2c923fa6fe8b30e6e55bda7525135344e218cde26a79098db0d3bab`；新目录独立重建归一化文本、表格和页数一致，重建PDF SHA256=`d4d1b778a769cc227a781a866bb69f0afb5aa3cc9a80d511c9904e1779b306f8`，不声称bitwise一致。初次同步时正文残留28页及旧deck哈希导致检查失败，已纠正并重新构建/复核，不覆盖失败事实。
- 当前结论只证明**无法在bpRNA-new上按已知来源构建同源未检出精度对照**，不是模型泛化或精度突破证据。下一数据门应优先评估有可靠时间戳和原始实验标签的时间外独立结构集；不得直接以未检出组、下游家族名或新数据条数替代独立性。
