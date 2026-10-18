# 泛化主张的措辞规则（Generalization Claim Wording Rule）

> change-id: `build-rna-ss-decision-model`
> 依据：spec §0.9.3（泛化性判断）、`spec/constraints.md` §5、
> **`records/DECISION_TRAINING_LOG.md` §14.18（实测推翻原规则）**
> 本文档是**一条硬性措辞规则 + 其机制理由 + 自动守卫**。任何稿件若违反，即为不合格。

---

## 0. 规则已被实测重写（2026-09-24）

**原规则**：跨家族主张**只能**写成「OOD 衰减更小（更鲁棒）」。

**为什么作废**：该规则的前提是"我们退化得更小"。§14.18 把它**测了**，结论相反：

| split | 我们（micro F1） | ViennaRNA centroid | 我们自己的 Nussinov+Turner 先验 |
|---|---|---|---|
| TS0（同分布） | 0.4953 | 0.5393 | 0.2124 |
| **bpRNA-new（跨家族）** | **0.3094** | **0.6770** | **0.3015** |

两件事同时发生：① 物理基线**上升**（0.5393 → 0.6770），不是衰减；
② 我们**掉到自己的物理先验水平**（0.3094 vs 0.3015），即判别力跨家族基本消失。

因此「OOD 衰减更小 / 更鲁棒」**同样是不可主张的**。一条会继续祝福这个说法的 lint 规则，
等于在强制一个已被数据否证的结论。

---

## 1. 规则（一句话，现行）

> 跨家族泛化**只能**写成**已量化的局限**：跨家族泛化不足，且我们**没有超过自己的物理先验**。
> **既不能**写「OOD 精度更高」，**也不能**写「OOD 衰减更小 / 更鲁棒」。

| ❌ 禁止（第一类，从来不可辩护） | ❌ 禁止（第二类，**实测后从"允许"降为"禁止"**） | ✅ 允许 |
|---|---|---|
| "higher OOD accuracy" | "smaller OOD degradation" | "cross-family generalization **is insufficient**" |
| "better OOD accuracy than baselines" | "more robust under distribution shift" | "cross-family performance **collapses to** the prior" |
| "higher cross-family accuracy" | "degrades less than baselines" | "does **not** beat its own physical prior" |
| "cross-family accuracy surpasses the baselines" | "smaller cross-family degradation" | "**跨家族泛化不足**（已量化）" |
| "OOD 精度更高" | "OOD 衰减更小（更鲁棒）" | "**退到自身物理先验水平**" |
| "跨家族精度超越基线" | "跨家族场景下退化更小" / "更鲁棒" | "跨家族表现已量化，不作正面主张" |

**允许的表述**（仍须有证据）：`cross-family generalization is insufficient`、
`collapses to the prior`、`does not beat its own physical prior`、
`跨家族泛化不足`、`退到自身先验水平`。

**唯一例外**：只有当**新的、干净的家族级划分实验**证明跨家族判别力确实高于自身先验时，
才允许在**明确标注该前提与证据**的情况下改写本节。默认口径永远是"已量化的局限"。

---

## 2. 理由（为什么只能这样写）

1. **原机制的四个论据全部被数据削弱**（逐条对照旧版）：
   - 旧论据「物理先验跨家族不变，所以我们起点就是物理模型，因此退化更小」——
     前半句成立，后半句**不成立**：起点是物理模型，**终点也退回了物理模型**（0.3094 ≈ 0.3015），
     所以"相对优势"不存在。
   - 旧论据「螺旋级参数更少、更可迁移」——**尚未验证**（家族级划分从未执行），
     且当前级联臂的 F1 未评测，不能用来支撑措辞。
   - 旧论据「更小衰减是相对量、更稳健」——前提是我们至少保持相对优势，实测没有。
   - 旧论据「绝对值更高缺乏机制支撑」——**仍然成立**，所以第一类禁止保持不变。
2. **物理基线上升是这条结论里最硬的事实**。它说明跨家族评测上"谁更鲁棒"的方向与直觉相反，
   必须如实呈现，而不是选择性地只报我们自己的绝对数字。
3. **可证伪性**：现行表述给出的是两个可被独立复现的数字（0.3094 / 0.6770）和一个可被检验的
   机制假设（头部学到的家族特异基序不可迁移）。旧表述给的是一个无法从我们的数据中推出的方向性断言。

---

## 3. 自动守卫（Guard / Lint）

守卫实现在 `paper/check_manuscript.py`，检查项 **(d)**，两个层次：

### 3.1 全局稿件守卫

命中下列**任一类**模式即报错（`ok = False`）：

**第一类**（从来不可辩护）：

- `higher OOD accuracy` / `higher out-of-distribution accuracy`
- `better OOD accuracy` / `improve(d)? OOD accuracy`
- `higher cross-family accuracy` / `superior cross-family accuracy`
- `OOD 精度更高` / `跨家族精度超越` / `跨家族精度更高`

**第二类**（§14.18 之后新加入）：

- `smaller OOD degradation` / `less OOD degradation`
- `more robust` / `degrades less`
- `衰减更小` / `更鲁棒` / `退化更小`

```bash
python paper/check_manuscript.py --manuscript <稿件>            # 全部四项检查
python paper/check_manuscript.py --manuscript <稿件> --checks d  # 只查泛化措辞
```

### 3.2 可编程守卫（供 CI / 写作脚本直接调用）

```python
from paper.check_manuscript import check_generalization_wording

check_generalization_wording(
    "Cross-family generalization is insufficient: on bpRNA-new our model does not "
    "beat its own physical prior."
)["ok"]      # -> True

check_generalization_wording(
    "Our model shows smaller OOD degradation than the baselines."
)["ok"]      # -> False  (第二类；曾经是允许措辞)
```

`check_generalization_wording` 的语义：

- 命中任一**禁止模式** → `ok = False`（无论是否同时出现允许措辞）；
- 否则 → `ok = True`（`allowed_hits` 列出命中的允许措辞，便于确认局限确实被写出来，
  而不是仅仅"没有写错"）。

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
| `records/DECISION_TRAINING_LOG.md` §14.18 | **实测依据**：bpRNA-new 0.3094 / centroid 0.6770 / 自身先验 0.3015；旧规则因此作废 |
| `records/DECISION_TRAINING_LOG.md` §14.15 问题 1 | ArchiveII 26.7% 与 TR0 近重复 → 它也**不能**用作泛化证据 |
