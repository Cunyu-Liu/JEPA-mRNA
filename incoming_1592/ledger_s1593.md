### 15.93 PPT 第四轮 + NucleicBERT fine-tune 管线就绪（T-A96，2026-10-20 下午）

- **主图 5 v4**（用户两处反馈）：左板 7 个标注逐点手工排布+引线（上/下/左/右错开，无重叠）；右板改**绝对综合分对数轴**条形（原归一化后柱长视觉相同的问题根治），每根柱标"我们 N×于它"倍数——差距与倍数一一可见。
- **主图 4 重标定方法说明**（S29 图注 196 字）：SC→logit 空间两参数仿射 sigmoid(a·logit+b)，参数仅在 VL0 验证集网格搜索（a∈[0.05,10]×b∈[-6,1] 按 ECE 最优），测试 split 从不重拟合（严格 held-out，§15.63 协议）。
- **NucleicBERT fine-tune 管线部署**（`tools/eval_nucleicbert_ft.py`，353→脚本就绪）：404M 编码器全解冻+官方 SSP 头、分层 lr（enc 1e-5/head 1e-4）、bprna_tr1c 去污染语料、BCE+pos_weight=100、VL0 阈值、8-split 板评测、ckpt 出 /teams。**阻塞在权重**：用户给的 orico 外置盘路径被 macOS TCC 拦（Operation not permitted，Finder 亦超时），已给用户两个解锁方案（拷 /tmp 或授权完全磁盘访问）。权重到位即开跑（GPU0 有 7GB 余量不够 404M 微调，等卡或用 MIG 缝隙，见 GPU 监控）。
- 本地图集/脚本/PPT 双副本同步；集群 figs/main + tools 已推送。
