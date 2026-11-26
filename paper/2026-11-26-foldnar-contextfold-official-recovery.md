# ContextFold 官方恢复记录（2026-11-26）

## 结论

当前不是 full-ready。已从作者官方页面的合法公共存档恢复 1.00 readme 和 additional resources，并核实官方默认模型、命令行入口及部分训练协议；但原始 `ContextFold_1_00.zip` 未恢复，因而没有可验证的 Java 实现、预训练参数、可执行资产或其 SHA256。不得使用同名第三方项目、重写实现、summary-only 数值或空预测补齐。

机器可读证据和逐项阻塞见 `manifests/foldnar_v2/contextfold_official_recovery_20261126.json`。原始证据保存在 `/teams/HoulabData_1788502323/liucunyu/foldnar/contextfold_official_recovery_20261126/evidence/`，未纳入 Git。

## 官方身份与证据

- 论文：Zakov、Goldberg、Elhadad、Ziv-Ukelson，*Rich Parameterization Improves RNA Structure Prediction*，DOI `10.1089/cmb.2011.0184`，PMID `22035327`；RECOMB 版本 DOI `10.1007/978-3-642-20036-6_48`。
- Crossref/PubMed 摘要明确指向作者发布页 `www.cs.bgu.ac.il/~negevcb/contextfold`，声明提供源代码和训练模型。
- Arquivo.pt 的 2024-10-29 原始 HTML 响应链接 `ContextFold_1_00.zip`、`readme.pdf`、`resources.pdf`，并指定默认 `StHighCoHigh`（约 70,000 个非零评分参数）。
- Common Crawl `CC-MAIN-2022-49` 保存了官方 readme/resources 的 HTTP 200 WARC 响应；readme SHA256 为 `c140e86b20a2b64e1bc76f83d2383599b5ac049fa6c91cb76423a69d8e43006b`，resources SHA256 为 `3b925b6641191a62ac714b3a352ebcf78e72eec9b8e50ded052584c687b4b98f`。

## 已核实的协议

官方 readme 要求 Java 6+，主入口为 `contextFold.app.Predict` 和 `contextFold.app.Train`。预测默认读取 `trained/StHighCoHigh.model`。每条原生输入由可选 `>`/`#` 标题、一行 A/C/G/U 序列及可选结构行组成；预测忽略输入结构并追加一条预测结构。

官方 resources 说明：

- 数据来自 Andronescu 等 RNA 2010 的 `S-Full`、`S-Full-Test`、`S-Full-Train`、`S-Strand2`。
- 论文测试预测由 `S-Full-Train` 上训练的模型产生；过滤绝对权重小于最大绝对权重 1% 的特征。
- 家族专用模型先基于 `S-Full` 训练 `StHighCoHigh`，再按家族重训；官方明确警示其家族过拟合。
- `S-Strand2` 与训练集存在一定重叠。
- readme 的 `iter:6 improve:0.5` 仅是重训示例，不能冒充发布模型的默认训练超参数。完整优化器和发布训练默认值仍未核实。

## 恢复阻塞

作者站点 HTTP/HTTPS 及原始 WARC IP 均超时。Arquivo.pt 没有 ZIP 捕获。所列 2012–2025 年度 Common Crawl 索引中，`ContextFold_1_00.zip` 精确查询最终均返回 404；Wayback/archive.org 在两个网络环境均连接超时。出版方元数据和页面未暴露替代官方 ZIP，GitHub 同名结果为无关项目。

因此，官方源码、二进制、模型和完整 ZIP 许可声明仍阻塞；已恢复文档本身没有给出软件/模型 SPDX 许可。

## bpRNA-new 5388 合同

实测 `/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl`：SHA256 `271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9`，5388 个唯一名称，长度 33–489 nt，纯 A/C/G/U，参考 pair 总数 135184。

`tools/run_contextfold_official_bprna_new.py` 仅调用 CPU Java，向官方程序提供名称和序列，不提供参考 pair/structure。官方资产清单未转为 ready 前会 fail closed。恢复后要求逐行输出 `{name,seq,pred_pairs}`，pair 为零基、闭区间、`i<j`，并验证完整 5388 名称/序列/顺序、伙伴唯一和无交叉；输出原子替换。当前 `predictions=null`，未运行完整推理，未产生任何伪预测，也未启动或修改 GPU 任务。
