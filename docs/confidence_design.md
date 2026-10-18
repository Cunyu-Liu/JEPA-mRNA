# Jev 式输出置信度 — 设计方案（2026-10-18，用户目标 3）

## 用户的诉求

Jev 原版对每个输出给一个置信度。本项目（RNA 二级结构决策模型）的输出是
**碱基对集合**（预测结构 = L×L 决策矩阵 → DP 解码），需要一个 Jev 式的
"这个预测有多可信"读出。结论先行：**项目已有三层基础，缺的是把它们组装
成产品化的置信度输出 API + 结构级聚合**。

## 现状盘点（三层已有基础）

| 层级 | 已有什么 | 出处 | 校准质量 |
|---|---|---|---|
| L1 配对级概率 | `p_ij = sigmoid(s_ij)`，candidate mask 上输出 | decision_head.py + evaluate_decision.py 全链路 | ECE 0.0007（重标定后，全测源最优 DP-free）——**这就是 Jev 式 option probability 的对应物，且已校准** |
| L2 边际 vs 精确 | C1-c 门：System-1 头 ECE vs inside-outside 精确边际 ECE 差 ≤0.02（实测 0.0002~0.0093） | matched_prefix_c1c | 已核验 PASS（免 DP 仿射重标定口径） |
| L3 helix 置信度 | L1 级联臂的 helix_logits（候选螺旋起止+置信度） | decision_head.py cascade 分支 | 已实现（级联臂未上主表） |

## 设计：三层置信度输出 API

**目标接口**（`eval/ss/predict_with_confidence.py` 新原型）：

```python
result = predict(seq, ckpt)
# result.structure        —— dot-bracket（现有 nussinov_map 解码）
# result.pair_conf       —— {pair: p_ij} 字典：每个配对位的校准概率（已有 sigmoid+affine）
# result.structure_conf  —— 结构级置信度（新，三种聚合见下）
# result.uncertain_pairs —— 低置信配对列表（p < τ），供 System-2 升级门控复用
```

**结构级置信度的三种候选聚合**（全部 DP-free，保持 C1 纪律）：

1. **配对概率几何均值**（主推）：`conf = exp(mean(log p_ij : (i,j) ∈ pred))`
   ——与 NLL 同尺度、对低置信配对敏感、可解释（"平均每次配对决策的把握"）。
   已有全部原料（probs 矩阵在 evaluate 决策路径里现成）。
2. **期望精度**（epsilon-margin）：`P(majority-correct) ≈ Π p_ij / (Π p_ij + Π (1-p_ij))`
   ——结构评分规则的边际化，理论最干净，但长结构数值下溢需 logsumexp。
3. **margin 置信度**：`conf = mean(|s_ij - s*_ij|)` 最优 vs 次优决策间隙
   ——Jev 的 "score gap" 直觉对应物，但需修改 DP 保留次优解（k-best DP），
   工程量大，列为 v2 候选。

**校准验证协议**（与既有 C1-c 同源，不新增发明）：
- 在 TS0/VL0 上把 `structure_conf` 分桶，测**结构级 reliability curve**：
  报告的置信 vs 实际 per-seq F1 的偏差（ECE_struct）；
- 阈值 τ 的选择在 **VL0** 上做（纪律：测试集不参与超参选择）；
- 与 ViennaRNA base-pair 概率的同类聚合对比（C1-b 同口径扩展）。

## 与 Jev 原版的对齐与差异（诚实定位）

- **对齐**：Jev 的 option probability = softmax over options；我们的 pair
  probability = sigmoid over candidate mask（选项空间是 L²/2 个配对位 +
  不配对）。校准是 Jev 信任故事的卖点（C1 主张），置信度是它的产品化输出。
- **差异**：Jev 单决策置信度；我们是结构级聚合（配对间有相关性——这正是
  Q3 审稿质疑，结构级校准曲线就是对它的回应，spec §9.3 已有落点检查）。
- **禁止表述**（沿用 §2.3 红线）：不写"首次置信度输出"（ViennaRNA BPP 早有
  配对概率）；我们的是**校准验证过的置信度 + 免 DP + 结构级聚合三合一**。

## 实施计划（T-A73 拆解）

1. **原型** `tools/predict_with_confidence.py`：输入 seq+ckpt，输出三层置信度
   （聚合 1 为主），~100 行，不触碰训练路径（纯推理期产品化）；
2. **校准验证** `tools/confidence_reliability.py`：TS0 全量 structure_conf
   分桶 reliability 曲线 + ECE_struct，与 ViennaRNA 聚合同口径对比；
3. **入板**：结果 JSON → 台账 → （数字三遍核对后）PPT 一页；
4. **验收**：ECE_struct ≤ 配对级 ECE 的 5×（相关性导致的退化上限）且
   reliability curve 单调（不单调则降级为 ordinal 置信度声明）。

## 现成可复用资产清单

- `all_probs_s1` 池（evaluate_decision.py 已存每条序列概率矩阵）
- 仿射重标定（sigmoid(a·s+b)，VL0 拟合，已入 C1-c 表）
- `PairLevelMetrics.from_pairs`（per-seq F1，作 reliability 的 actual 轴）
- ViennaRNA BPP teacher 标签（结构级聚合的对照物）
