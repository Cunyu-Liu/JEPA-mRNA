# Stacking Closure Cell v1：新形式准入设计（未训练）

## 1. 科学对象与现有 FN 规模

目标仅为 **helix 边界 FN，尤其已有正确 stacking 邻居的边界 FN**。三个冻结 control 的 VL0 共有 2,981 FN：helix 边界 1,789（60.01%），helix 内部 873（29.29%），孤立 pair 319（10.70%）。边界 FN 中 500 个至少一个相邻真实 pair 已预测正确；三个 seed 的全部“有正确邻居支持 FN”分别为 167/228/215，占各 seed FN 的 18.05%/22.35%/20.75%。孤立 pair recall 虽仅 0.1425，但不在本假设的修复对象内。

这只是错误环境描述，不证明 stacking 是根因，也不分离表示、目标和优化因果。artifact：`pair_form_hypothesis_v1/stacking_fn_audit.json`。

## 2. 数学命题与可检验预测

设对称 pair representation 为 \(H_{ij}\)，定义内/外 stacking 邻居 \(H^{in}_{ij}=H_{i+1,j-1}\)、\(H^{out}_{ij}=H_{i-1,j+1}\)。候选 **Stacking Closure Cell** 为：

\[
C_{ij}=[H_{ij},H^{in}_{ij},H^{out}_{ij},H_{ij}\odot H^{in}_{ij},H_{ij}\odot H^{out}_{ij}],\qquad
R_{ij}=\mathrm{sym}(W_2\,\phi(W_1C_{ij})),
\]

其中 \(\mathrm{sym}(R)_{ij}=(R_{ij}+R_{ji})/2\)，canonical/min-loop 外及序列边界强制为零。用**替换一个既有模块**的 \(H+R\) 形式，不增加 `blocks/hidden/d_z`。与普通 ResNet/attention 的区别是显式二阶 closure 相互作用和固定的 \((i\pm1,j\mp1)\) 反方向拓扑，而非更多通用邻域计算。

可证伪预测：候选相对匹配 control 应优先提高“有正确邻居支持的边界 pair”recall 和 stem-F1；把 stacking 邻居换成 \((i\pm1,j\pm1)\) 或对邻居通道做固定置换后，该优势应消失。若只提高 AUPRC、只改善非目标子群或 destroy 对照同样改善，则命题被否定。

## 3. 物理对应

相邻 canonical pairs \((i,j)\) 与 \((i+1,j-1)\) 构成连续碱基 stacking；helix 末端对应 loop/junction 或链边界。显式邻居及乘积项表达“当前配对与闭合相邻配对是否共同支持同一 helix”，不是装饰性的物理命名。孤立长程 pair 没有此局部 stacking 支撑，因此不要求该算子修复。

## 4. 必要性、匹配对照与负对照

- **control_matched**：相同初始化、数据、NLL、schedule、decoder、总参数和近似 FLOPs；以同宽通用局部 cell 替换同一模块，但只使用非 stacking 邻域。
- **stacking_closure**：唯一科学变量为上述反方向 closure 边与二阶乘积。
- **destroy_offset**：相同参数和计算，将内/外邻居改为 \((i+1,j+1)\)、\((i-1,j-1)\)，破坏 helix 几何。
- **destroy_channel_shuffle**：相同参数和计算，对邻居通道使用冻结算法生成的固定置换，保留值分布但破坏配对对应。
- 不启动 seeds 1/2，除非 seed0 同时通过全部预注册结构门、目标子群门且优于两个 destroy 对照。

## 5. 结构终点与停止条件

主指标为 micro/macro pair-F1、P/R、micro/macro stem-F1；AUPRC/r@P0.8 仅解释机制。seed0 必须满足现行全部方向门，macro pair-F1 严格为正，目标边界 FN recall 相对 control 严格提高，并且 stacking_closure 相对两个 destroy 对照在 micro/macro pair-F1 与 stem-F1 上有优势。任一失败立即停止，不继续堆模块、不改成功定义。通过 seed0 也只授权预注册三种子确认；三种子任一失败即停止，不选冠军 seed。

## 6. 公平计算与登记

运行前冻结合同并登记每层参数、总参数、FLOPs、峰值显存和推理时延；候选、匹配 control 与 destroy 使用相同 batch、步数、初始化、checkpoint 规则和成本口径。逐块梯度检查点只用于显存。保存逐序列预测、目标子群指标、真实停止原因、代码/数据/权重/配置哈希。

## 7. 当前授权状态

本页仅满足新形式设计准入，**不是训练结果，不自动授权训练以外的任何外部评估**。下一步若实现，必须先生成独立冻结合同和参数/FLOPs/identity 预检；外部测试、严格 OOD 与 SOTA 保持关闭。

证据链：`claim=显式 stacking 可能修复边界 FN` → `required evidence=目标子群+全局结构门+匹配/destroy 对照` → `current artifact=stacking_fn_audit.json（仅描述 FN）` → `counterexample/limitation=孤立 pair、raw-only 改善、单 seed 或 destroy 同样改善均否定机制叙事`。
