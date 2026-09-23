# 泛化主张的措辞规则（Generalization Claim Wording Rule）

> change-id: `build-rna-ss-decision-model`
> 依据：spec §0.9.3（泛化性判断）、`spec/constraints.md` §5
> 本文档是**一条硬性措辞规则 + 其机制理由 + 自动守卫**。任何稿件若违反，即为不合格。

---

## 1. 规则（一句话）

> 跨家族 / OOD 的泛化主张**只能**写成「**OOD 衰减更小（更鲁棒）**」，
> **绝不能**写成「**OOD 精度更高**」或「跨家族精度超越基线」。

| ❌ 禁止 | ✅ 允许 |
|---|---|
| "higher OOD accuracy" | "**smaller OOD degradation**" |
| "higher out-of-distribution accuracy" | "**more robust** under distribution shift" |
| "better OOD accuracy than baselines" | "**degrades less** than baselines" |
| "cross-family accuracy surpasses the baselines" | "**smaller cross-family degradation**" |
| "OOD 精度更高" | "**OOD 衰减更小（更鲁棒）**" |
| "跨家族精度超越基线" | "**跨家族场景下退化更小**" |

**允许的加强表述**（仍须有证据）：`smaller OOD degradation`、`more robust`、
`degrades less`、`衰减更小`、`更鲁棒`、`退化更小`。

---

## 2. 理由（为什么"更鲁棒"才是可辩护的主张）

1. **泛化瓶颈是数据与问题本质，不是架构**（spec §0.9.3）。结构标注只有 ~102k 条且家族集中，
   这是硬约束，架构无法突破；领域已知**所有**方法在 bpRNA-new 上都差
   （MXFold2 论文报告 E2Efold `F ≈ 0.0361`）。因此"绝对值更高"缺乏机制支撑。
2. **物理先验跨家族不变**——Turner 参数是物理常数，不随家族漂移。热力学蒸馏 + 残差先验让模型
   **起点就是物理模型**（精确说：起点 = 「Nussinov + Turner 堆叠能」，**不是** ViennaRNA）。
   这直接支撑「退化更小」这一**相对**主张。
3. **螺旋级参数化更少、更可迁移**（H7 的机制依据）——螺旋是进化保守的功能单元，
   碱基级配对模式在家族间漂移更大。参数更少 ⇒ 更不易过拟合家族特异模式 ⇒ 衰减更小。
4. **"更小衰减"是可证伪且更稳健的命题**：它只要求相对量（同源 → 跨家族的落差），
   不要求绝对精度超越最强基线；后者即使侥幸成立也更可能是数据泄漏或调参产物。

> **例外**：只有当 **H7 意外成立且效应显著**时，才允许在**明确标注该前提**的情况下报告
> 绝对值提升。默认口径永远是"更小衰减"。

---

## 3. 自动守卫（Guard / Lint）

守卫实现在 `paper/check_manuscript.py`，两个层次：

### 3.1 全局稿件守卫

`check_manuscript.py` 的检查项 **(d)** 扫描稿件全文，命中下列任一模式即报错：

- `higher OOD accuracy` / `higher out-of-distribution accuracy`
- `better OOD accuracy` / `improve(d)? OOD accuracy`
- `higher cross-family accuracy` / `superior cross-family accuracy`
- `OOD 精度更高` / `跨家族精度超越` / `跨家族精度更高`

```bash
python paper/check_manuscript.py --manuscript <稿件>            # 全部四项检查
python paper/check_manuscript.py --manuscript <稿件> --checks d  # 只查泛化措辞
```

### 3.2 可编程守卫（供 CI / 写作脚本直接调用）

```python
from paper.check_manuscript import check_generalization_wording

check_generalization_wording(
    "Our model shows smaller OOD degradation than the baselines."
)["ok"]      # -> True

check_generalization_wording(
    "Our model achieves higher OOD accuracy than the baselines."
)["ok"]      # -> False  (findings 给出命中的模式与片段)
```

`check_generalization_wording` 的语义：

- 命中任一**禁止模式** → `ok = False`（无论是否同时出现允许措辞）；
- 否则 → `ok = True`（`allowed_hits` 列出命中的允许措辞，便于确认主张确实落在"更小衰减"口径上）。

### 3.3 使用边界（重要）

- 本文件、`limitations.md`、`contributions.md`、`reviewer_objections.md` 中出现的**禁止措辞是反例**，
  **不要**把这些守卫文件当作稿件去 lint（否则会"正确"地报错）。
- 稿件正文（含摘要、贡献句、图注、表格标题）才是 lint 对象。
- 投稿前把检查项 (d) 与 `limitations.md` 的措辞表一并核对：两处的口径必须完全一致。

---

## 4. 与其它约束的关系

| 关联 | 说明 |
|---|---|
| `limitations.md` | 其"泛化主张的措辞约束"表与本文件**逐字一致**；改动须两处同步 |
| `contributions.md` | 泛化**不是** C1–C4 之一（H7 是机制假设），因此泛化措辞不得写进贡献句 |
| `spec/analysis_plan.md` | H7 的判据是"层级 vs 扁平"的 bpRNA-new F1 比较；本规则约束的是**结论如何表述**，不改变判据 |
| spec §0.9.3 | 规则来源；"不要把泛化当作主打卖点" |
