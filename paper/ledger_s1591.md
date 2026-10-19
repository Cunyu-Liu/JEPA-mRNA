### 15.91 预印本 v3.28：评审第二轮反馈入稿（T-A94，2026-10-19 深夜）

- **用户反馈 ⑥⑧ 的"文章中"要求落地**（上轮只进了 PPT，本轮补齐正文）：
  1. **ECE 重要性解释段**入 §4.3i（ECE 表后）："Why the reader should care about ECE"——校准=置信度可按面值用于 triage/复算预算/升级决策，0.031 vs 0.160 的部署含义；
  2. **架构机制分析段**入 §3（Method 开头）："Why this architecture reaches the measured standings"——冻结骨干（36M 序列知识 vs 14k 标签、防泄漏）/ 平面头（决策局部性、收敛快、家族互补保留、124.8 seq/s 的架构根源）/ 解码级先验（OOD 归属 +0.37/+0.25/+0.16 的消融签名），三机制如何组合成测得战绩；
  3. **图嵌入更新**：fig1 拆 fig1a（单模 5/8）+ fig1b（集成 7/8）双图入 §4.3h；fig2/fig5 换 v2 重画版；fig6 换 HTML 架构图（同路径新渲染）；
  4. **Appendix D 图注全部同步**（1a/1b 拆分、fig2 ranked panel 口径、fig5 无碰撞标注+联合前沿包络、fig6 全 schematic 图注含三机制面板）。
- **口径审计发现并修复**：稿内 6/8（最强参照口径）与图内 7/8（inter-fam 零重叠口径）冲突 → §4.3h 新增 "Two reference conventions, both printed" 段：6/8 主表保持 headline（claim 更少），7/8 为图口径（TS3 +0.072/hard +0.102 翻案 + TSB clean-181 +0.027），两口径并列打印；TS2 两口径皆负（−0.044，零重叠参照）不受影响。机制段 7/8 表述同步改 6/8。
- **PDF 重建**：pandoc→Chrome headless 管道，26 页 7 图（p5 架构/p15 fig1a+1b/p17 fig2+3+4/p21 fig5）+ 8 项关键内容程序化检查全过。
- **交付**：本地 交接文档/paper/{preprint_draft.md, preprint_draft_v3.28.html, preprint_draft_v3.28.pdf}；集群与 GitHub 本轮同步。
