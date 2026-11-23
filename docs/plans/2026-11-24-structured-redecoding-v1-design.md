# Structured Re-Decoding v1：受控独立分支

## 有限问题与边界

在相同冻结 pair scores 下，检验联合候选选择、helix stacking 与候选重组能否超过当前 MEA。该分支不训练、不修改 checkpoint，不解释或挽救容量路线失败；RAW_SCORE 仍占主要可归因错误。仅使用 bpRNA VL0 model-dev，外部评估、严格 OOD 与 SOTA 关闭。

## 数学对象与物理对应

合法结构是 canonical、min-loop=3、无 partner conflict 且 non-crossing 的 matching。对结构 \(S\) 定义未校准结构目标：

\[
J(S)=\sum_{(i,j)\in S}w_{ij}+\lambda\,|\{(i,j):(i,j)\in S,(i+1,j-1)\in S\}|,\quad \lambda=1.
\]

其中 \(w_{ij}\) 是冻结 `plana_off` 分数，不称为概率；第二项对应连续 Watson–Crick 配对形成的 helix stacking。当前 MEA 仍由 inside–outside marginals、固定 temperature=1、gamma=1 计算。

## 候选、求解器与必要对照

1. 当前 Nussinov MAP。
2. 当前 MEA。
3. nested canonical matching：与 MAP 使用同分数、同约束的精确 DP。
4. nested SRD stacking：从 MAP+MEA 候选结构抽取 maximal helices，允许其 outer-prefix，在候选支持集上做精确 helix-prefix DP。
5. best-of-N：在同一 \(J(S)\) 下只选择完整 MAP 或 MEA，不允许重组。
6. 多模型重组：成员 checkpoint/导出身份另行核验后才运行；固定 xens2 0.7/0.3、best-of-N 与 SRD 重组比较，记录成员来源。

**等价性对照**：长度不超过 12 的合成问题上，对相同候选支持集与相同 \(J\)，比较精确 DP 和穷举所有 nested structures 的目标值。若相同候选和相同目标下 DP 已全局最优，QUBO 只可能是另一种求解器，不记作新增科学价值。

**候选对照**：best-of-N 与 SRD 使用完全相同的候选和目标；差异只能来自 helix 重组。单来源与多来源 SRD 也必须使用同一求解器和目标。

## 停止门与成本

SRD 相对当前 MEA 必须同时严格提高 micro/macro pair-F1，micro precision 与 micro stem-F1 降幅均不超过 0.005，nested/canonical 满足率 100%，失败序列数为 0。只赢 MAP、independent pairing 或 native flow 不算通过；只提高 macro 或 raw 指标也停止。多模型重组若不能超过单来源 SRD，删除该复杂分支。逐序列记录求解时间、候选结构/pair/helix 数、内存、fallback 与失败，不能只报告成功样本。

通过仅允许提交单独冻结的外部敏感性申请，不自动开放外部。任何失败均保留负结果，不搜索阈值、temperature、gamma 或 stacking bonus 救援。
