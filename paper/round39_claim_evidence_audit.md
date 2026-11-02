# Round 39 论文叙事证据审计（实验进行中）

## 当前可以写入的结论

| 叙事 | 实验现象与对照 | 必须保留的限制 |
|---|---|---|
| 主 benchmark 需要逐模型去泄露，而不是统一 clean-181 | EternaFold、BPfold、MXfold2 各八 split 已按自身训练并集去泄露，24 格数据、审计、三次预测和计数独立复算通过。EternaFold bpRNA-new=0.6853840073（n=5388，SD=0） | 各模型 clean n 不同；异子集 F1 不直接排 SOTA，最终补同名、同序列、同 GT 的配对敏感性 |
| RNAformer inter-family 的重复性和长序列问题已定位并修复 | 关闭 12 个 RotaryEmbedding 缓存后，TS1 原始 logits 和预测三次一致；L>512 的精确注意力分块通过逐元素等价和最长序列验证。raw 八格独立复算完成 | raw 8/8 不等于 raw+clean 完成；biophysical config 为 `rotary_emb:false`，不套用 inter-family 根因 |
| 单学生蒸馏并未解决全板泛化 | 三种子 clean 8/8 已闭环，但 bpRNA-new、TS2、ArchiveII 未达 SOTA；同交集上学生 bpRNA-new 比 BPfold 低约 0.227 F1 | 不能再用“完成三种子”替代“性能达标”；不再沿同一单 plana 路线加种子 |
| VL0 融合不能宣称 8/8 | 六个 clean split 后，TS2/TS3 未领先，8/8 数学上不可达，已停止剩余推理 | 拟合集上的融合增益不能外推为跨 split 泛化；不生成新教师或启动新蒸馏 |
| 混合 NLL 的局部涨点不是泛化改善 | seed0 仅 TS2/TS3/TSB 上升，其余五格下降；bpRNA-new 所有长度桶退化，ArchiveII 退化随长度增加 | 这是候选否决后的解释性诊断，不用于测试集调参；下一候选需先解决结构泛化，而非统一调阈值 |
| 置信度校准与结构质量排序是不同目标 | VL0 校准降低 ECE，但 Spearman 从 0.8510 降至 0.7415、AURC 变差；SPOT-RNA 的所选配对置信度全为 1.0，无法排序 | 当前只有 VL0 拟合集证据，不能称跨 split 最低 ECE 或最佳置信度；BPfold raw utility 不是概率，不直接计算 ECE |

## 当前不能升级为结论的主张

- **8/8 SOTA**：强基线全量复测尚未结束，主候选仍有三格负结果；主标题保持指定的 5/8、4/8 与 bpRNA-new、TS2、ArchiveII 未领先。
- **速度 SOTA**：xens2 的三编码器、三决策头根因明确，但正式比较必须等 eligible baseline 齐套后，在独占 4090 上以 raw-sequence 全 TS0 三次复测。并发耗时、head-only、cached embedding 均不能替代。
- **完整 600 单元**：E2Efold/ContextFold 资产、UFold/eFold/CONTRAfold provenance 等边界仍保留；blocked 不写成 0 分，也不从分母中偷偷删除后宣称全基线胜利。
- **机制结论**：长度分层只说明混合 NLL 的误差模式，不证明某种新结构特征必然有效。新机制必须先通过 VL0 和合法训练数据门，再做三种子及全板验证。

## 后续证据顺序

1. 完成 eligible baseline 的 raw+clean 八 split、三次重复和独立复算，保留所有 blocked 与空 split。
2. 在同一序列/GT 交集上复核关键强对手，特别标出 BPfold TS2 n=3 等小样本单元。
3. 新候选只在 VL0 证明有明确泛化收益后进入三种子；收敛后做全板、独占速度、ECE/Brier/NLL、Spearman 和 risk–coverage。
4. 只有完整原始 artifact 支持时才更新 PPT 和论文主结论；否则保留负结果与限制。
