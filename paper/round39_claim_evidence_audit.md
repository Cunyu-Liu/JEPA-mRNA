# Round 39 论文叙事：现象—对照—限制审计

| 推进叙事 | 已直接观察到的实验现象 | 可以写到什么程度 | 必须保留的限制与未完成证据 |
|---|---|---|---|
| 去污染是主结果的前提 | TR0 的 TestSetB 从旧 clean-181 变为 clean-116；tr1c 的 TS0/ArchiveII 经 80% identity+query coverage 过滤后分别为 713/1107 | 精确重复不足以覆盖同源泄露；逐模型、逐 split 的 clean 集合必须分别构建 | 600 单元和全模型训练语料审计未完成；不能把 raw 复测称为完整 clean，不能用旧 6/8 或 5/8 推断最终胜场 |
| xens2 是当前主力 | raw 全板敏感性 5/8、最佳单 checkpoint 4/8；bpRNA-new、TS2、ArchiveII 未领先 | 只能作为当前敏感性与短板陈述 | 预注册 clean 参照 6/8 仅是实验合同附注；最终 SOTA 必须由逐模型 clean 三次均值判定 |
| 新强对手改变攻关门槛 | BPfold 自身训练并集 clean 三次复算：ArchiveII=0.8212189843（n=731）、TS2=0.9494949495（n=3） | ArchiveII 必须面对显著强于旧 xens2 的对照；TS2 可记录直接观测值 | TS2 仅 3 条，禁止只报高 F1 而隐藏 n；其他模型 clean 未齐，暂不写最终排名 |
| xens2 推理尚未速度 SOTA | B=32 raw-sequence xens2=19.89 seq/s，r2d=53.91；三次编码器前向与三个头是直接耗时来源 | 当前未达速度目标；需要结构性的推理优化 | plana s0/s1 与 r2d 的骨干权重不同，不能无损共享编码；多模型批化和单学生蒸馏均需实测，不能把预期提速写成结果 |
| 蒸馏目标必须同时看教师与 GT | 纯 r2d KL 的教师 micro-F1=0.8029、VL0 ECE=0.0344，但 TS0=0.6247；旧四项目标 TS0=0.6694 | 纯 KL 提高教师一致性和校准，却不能保持结构精度，不投入其三种子 | 单 plana 可训练骨干是另一候选；其收敛前不得宣称继承 xens2 精度或达到速度 SOTA |
| 结构先验解释 OOD 收益 | 40/40 消融显示 Turner/2D 的 PDB-OOD 损失更大，DP 的 ID 损失更大；GenC/GenF 少预测且零正确配对率高 | 可以写组件作用依任务而变，Turner/2D 在 PDB-OOD 更重要 | 不能概括为所有组件都随 OOD 距离单调增强；GenC/GenF 失败机制不能被 OOD mean 掩盖 |
| 置信度可用于质量分层 | 大样本 split 的 VL0 仿射降低 structure-level ECE，八 split Spearman 全为正；TS2/TS3 过校正 | 可以写“部分 split 改善且排序信号存在” | 最低 ECE 未证明；缺少统一概率基线、risk–coverage/abstention 和跨 split 稳定性，平均置信度高不等于更好 |
| JEV 头可跨骨干迁移 | NB 主 benchmark 8/8 正迁移；原论文 acceptor/donor/CPEB3 不显著为正，其他下游也有负结果 | 只能说主 benchmark 迁移成立，原论文任务未显示普遍正迁移 | 不得写成 NucleicBERT 原论文任务全面提升；NB 二级结构五种子和独立 QA 未闭环，contact/distance blocked |
| 长序列部署边界明确 | 短中段表现较强，>400 nt 仍是弱项，精确 DP 为 O(L³) | 可以界定当前适用范围和开放问题 | 不得声称长序列优势或近似线性解码；需要长度×长程配对联合分层和显存/时间证据 |
| 最终目标 8/8、速度 SOTA、最低 ECE | 当前三个目标均未闭环 | 只能作为研究目标，不是结果标题 | 不能通过选择弱参照、减少 seed、缩短训练或隐藏负结果达标；最终结论必须由完整 clean、同机三次评测和独立复算支持 |

## 写作原则

1. 每个强主张后紧跟对应原始 artifact、对照实验与限制，不用机制图代替因果证据。
2. raw、clean-subset、预注册参照和最终 clean 主板分开命名；不同 clean n 不得混用。
3. 多模型融合/蒸馏是候选模型，不冒充原 xens2 的无损加速；任何精度、速度或 ECE 变化均完整报告。
4. 负结果 bpRNA-new、TS2、ArchiveII、GenC/GenF、长序列和原论文下游任务不得被平均值或新标题掩盖。
