# RNA-JEV 当前 claim → evidence → artifact → limitation 矩阵

日期：2026-11-24。仅纳入已有直接证据；新形式 pilot 运行期间不得预写其结果。

| Claim | Required evidence | Artifact | Counterexample / limitation |
|---|---|---|---|
| 主要可观测错误在 score 形成阶段 | 三 seed 错误归因与 raw 导出 | 既有 RAW_SCORE=2,416/2,650 FN（91.17%） | 不能分离表示、目标、优化和头部形式因果 |
| 简单容量扩张不是稳健生产路线 | 三 seed 同 control delta、逐 seed 与均值门、六 run 完整性 | `representation_head_mechanism_4arm_v1/capacity_freeze_audit.json`，SHA256 `6f435ceaca…071d` | seed0 曾正向；结论仅限当前容量形式与 model-dev，不排除新形式 |
| raw 排序改善不等于结构改善 | AUPRC/r@P0.8 与 pair/stem 对照 | 同上；head_deep/repr_head 三 seed 均值 | AUPRC/r@P0.8 为正不能抵消 pair-F1、recall 或 stem 失败 |
| SRD 不能超过当前 MEA（本冻结设置） | 196 条 VL0 五臂、约束、fallback、成本、独立重算 | `structured_redecoding_v1/seed0_control_matched`，summary `d0147bcd…e5bb` | 单来源单设置；不证明所有结构化解码无效。多模型因单来源停止而未运行 |
| 同候选同目标下 QUBO 无新增价值 | 精确 DP 与独立穷举求解器等价 | `tests/test_structured_redecoding.py`，11 项 SRD 测试通过 | 只有候选、目标或外部规格变化时才可能产生新价值；未授权外部规格 |
| helix 边界是重要 FN 环境 | 三 control FN 分类与 raw 重算守恒 | `pair_form_hypothesis_v1/stacking_fn_audit.json`，边界 1,789/2,981 | 描述性错误环境，不证明 stacking 是根因；孤立 pair recall 更低 |
| stacking closure 的局部信号未形成稳健机制 | 数学命题、同参数 control、两个 destroy、目标子群和结构门 | `stacking_closure_form_v1/pilot`；候选 micro pair-F1 +0.00399、目标 recall +0.00647，但 macro pair/stem 与 destroy 门失败 | seed0 已停止；不启动 seeds 1/2、不追加模块，局部/raw 改善不能抵消 macro 结构失败 |
| 当前没有统一完整 SOTA | 同协议全面板、frozen predictions、baseline provenance | `manifests/foldnar_v1/*_registry.json` | 历史 6/8 为特定参照口径；62/63 baseline 仍有 provenance 阻塞，统一完整 SOTA=0/8 |

## 叙事边界

当前可成立的故事是“诊断 → 候选机制 → 跨种子反证 → 停止错误路线 → 新形式预注册证伪”，不是“容量或解码已经解决性能问题”。新形式只有在 seed0 全部门、目标边界和 destroy 对照通过后，才可进入三种子确认；外部、严格 OOD 和 SOTA 继续关闭。
