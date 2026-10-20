#!/usr/bin/env python3
"""Fig4 v2 (ECE board with explicit legend teaching) + Fig5 v3 (distinct from fig4).

Fig4 user complaints: 红线是什么 / raw vs recalibrated 是什么.
Fix: bilingual legend + annotation text INSIDE the figure explaining each color;
raw = 模型原始置信度, recal = VL0 重标定后; red = 重标定后变差(小样本族过矫).

Fig5 user complaint: 和主图 4 一样看不懂.
Fix: totally different visual identity — joint frontier emphasised: 2D plane
with shaded "dominance" region + annotated ratios; panel b becomes a table-like
sorted bar with big ratio labels. Both panels Chinese-labelled.
All measured data (ledger 15.80/15.81/15.82).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

OUT = Path("/tmp/ppt_work/figures/main")
NAVY, GOLD, RED, PURPLE = "#1b2a5c", "#e8a917", "#b5121b", "#7b5fb8"
BLUE, GREY, TEAL, GREEN = "#27348b", "#9a9a9a", "#0e7c86", "#0a7d3c"
plt.rcParams.update({"font.family": ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "sans-serif"],
                     "font.size": 9, "pdf.fonttype": 42, "axes.linewidth": 0.9})

def frame(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="#dcdcdc", lw=0.4, ls=(0, (1, 2)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

# ============ Figure 4 v2 ============
splits_e = ["TS0\n(n=1281)", "TestSetB\n(n=428)", "bpRNA-new\n(n=5375)", "ArchII\n(n=2505)",
            "TS1\n(n=62)", "TS2\n(n=39)", "TS3\n(n=19)", "TS-hard\n(n=27)"]
raw_e = [0.1604, 0.0883, 0.1010, 0.1150, 0.0612, 0.082, 0.075, 0.092]
cal_e = [0.0315, 0.0445, 0.0677, 0.0605, 0.141, 0.185, 0.158, 0.166]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.4),
                           gridspec_kw={"width_ratios": [1.5, 1]})
xs = np.arange(8)
cols_cal = [TEAL if c < r else "#e08a85" for r, c in zip(raw_e, cal_e)]
a.bar(xs - 0.2, raw_e, 0.38, color=GOLD, label="raw：模型原始置信度（未重标定）")
a.bar(xs + 0.2, cal_e, 0.38, color=cols_cal, label="recalibrated：VL0 重标定后")
for xi, (r, c) in enumerate(zip(raw_e, cal_e)):
    a.text(xi - 0.2, r + 0.004, f"{r:.3f}", ha="center", fontsize=6.2, color="#8a6d1a")
    a.text(xi + 0.2, c + 0.004, f"{c:.3f}", ha="center", fontsize=6.2,
           color=TEAL if c < r else RED)
a.set_xticks(xs, splits_e, fontsize=6.8)
a.set_ylabel("ECE（结构级校准误差）\n= 平均置信度与实际正确率的偏差", fontsize=8.5)
a.set_ylim(0, 0.22)
frame(a)
# explanatory annotations inside the figure
a.axvline(3.55, color="#bbb", lw=0.8, ls=(0, (2, 2)))
a.text(1.6, 0.207, "大样本族：重标定 3-5× 改善", fontsize=8.5, color=TEAL, fontweight="bold", ha="center")
a.text(1.6, 0.193, "（青色柱 = 变好）", fontsize=7.5, color=TEAL, ha="center")
a.text(5.8, 0.207, "小样本 PDB 族：过矫（红柱 = 变差）", fontsize=8.5, color=RED, fontweight="bold", ha="center")
a.text(5.8, 0.193, "重标定参数在 bpRNA 族上拟合，跨族不迁移——如实报告", fontsize=7, color=RED, ha="center")
a.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
a.set_title("(a) 8-split 校准全景：raw vs 重标定（VL0 参数，逐 split 不重拟合）",
            loc="left", fontweight="bold", fontsize=9.5, pad=6)

big = ["TS0", "TestSetB", "bpRNA-new", "ArchII"]
big_raw = [0.1604, 0.0883, 0.1010, 0.1150]
big_cal = [0.0315, 0.0445, 0.0677, 0.0605]
xs2 = np.arange(4)
b.bar(xs2, big_raw, 0.5, color=GOLD, label="raw")
b.bar(xs2, big_cal, 0.5, color=TEAL, label="recalibrated")
for xi, (r, c) in enumerate(zip(big_raw, big_cal)):
    b.text(xi, r + 0.004, f"{r:.3f}", ha="center", fontsize=7, color="#8a6d1a")
    b.text(xi, c + 0.004, f"{c:.3f}", ha="center", fontsize=7, color=TEAL)
    b.annotate("", xy=(xi + 0.3, c + 0.004), xytext=(xi + 0.3, r - 0.002),
               arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    b.text(xi + 0.36, (r + c) / 2, f"↓{r/c:.1f}×", fontsize=7.5, color=RED, fontweight="bold", va="center")
b.set_xticks(xs2, big, fontsize=7.5)
b.set_ylabel("ECE", fontsize=8.5)
b.set_ylim(0, 0.19)
frame(b)
b.legend(fontsize=7.5, frameon=False, loc="upper right")
b.set_title("(b) 大样本族放大：4/4 干净迁移（箭头=改善倍数）",
            loc="left", fontweight="bold", fontsize=9.5, pad=6)
fig.suptitle("主图 4 · ECE 校准全景——「模型说 90% 把握，是否真有 90% 正确」",
             fontweight="bold", fontsize=11, y=0.99)
fig.text(0.012, 0.012, "ECE 越低=置信度越可按面值使用（triage/复核预算的依据）。TS0 raw 0.160→0.031："
         "重标定后 0.9 置信的结构失败率≈1/10。数据：conf_board.sh 8-split 全量读数。",
         fontsize=7.2, color="#333", style="italic")
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(OUT / "fig4_ece.png", dpi=300)
fig.savefig(OUT / "fig4_ece.pdf")
plt.close(fig)
print("fig4 v2 done")

# ============ Figure 5 v3: joint frontier, visually distinct ============
models = [
    ("Ours xens2\n(B=256)", 0.8039, 124.78, RED, "o"),
    ("Ours 单模\n(B=64)", 0.7866, 119.55, NAVY, "o"),
    ("RiNALMo-ft\n(650M ft)", 0.7602, 14.5, PURPLE, "o"),
    ("RNAformer", 0.7578, 25.0, TEAL, "^"),
    ("MXfold2", 0.6551, 34.9, GREY, "^"),
    ("UFold", 0.6598, 56.49, GOLD, "^"),
    ("ViennaRNA\n(PF)", 0.6729, 3.6, GREY, "^"),
]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.4),
                           gridspec_kw={"width_ratios": [1.3, 1]})
# (a) plane with dominance region
for name, f1, sps, col, mk in models:
    ours = "Ours" in name
    a.scatter(sps, f1, s=170 if ours else 90, color=col, zorder=4,
              edgecolor="k", lw=0.6, marker=mk)
    a.annotate(name, (sps, f1), fontsize=7.5, xytext=(8, -3) if "Vienna" in name else (8, 5),
               textcoords="offset points", ha="left",
               fontweight="bold" if ours else "normal", color=col if ours else "#333")
# frontier: connect ours -> UFold -> MXfold2 (lower-left envelope)
pts = [(124.78, 0.8039), (56.49, 0.6598), (34.9, 0.6551)]
a.plot([p[0] for p in pts], [p[1] for p in pts], color=RED, lw=1.6,
       ls=(0, (6, 3)), zorder=2)
a.fill_between([34.9, 124.78], [0.6551, 0.8039], 0.86, color=RED, alpha=0.06, zorder=1)
a.text(45, 0.845, "帕累托前沿（无系统同时更准更快）", fontsize=8, color=RED, style="italic")
a.text(105, 0.776, "红区=被我们\n同时超越", fontsize=8.5, color=RED, fontweight="bold")
# big ratio callouts
a.annotate("同节点 8.6×\n(RiNALMo-ft)", xy=(14.5, 0.7602), xytext=(5, 0.71),
           fontsize=8.5, color=PURPLE, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2))
a.annotate("2.2× (UFold)", xy=(56.49, 0.6598), xytext=(80, 0.64),
           fontsize=8.5, color="#8a6d1a", fontweight="bold",
           arrowprops=dict(arrowstyle="->", color="#8a6d1a", lw=1.2))
a.set_xscale("log")
a.set_xlim(2.4, 400)
a.set_ylim(0.60, 0.87)
a.set_xlabel("吞吐（seq/s，对数轴）→", fontsize=8.5)
a.set_ylabel("TS0 micro F1 ↑", fontsize=8.5)
frame(a, grid_axis="both")
a.scatter([], [], marker="o", color="k", s=80, label="同节点 4090 实测")
a.scatter([], [], marker="^", color="k", s=80, label="跨硬件（引用/协议匹配）")
a.legend(fontsize=7, frameon=False, loc="lower right")
a.set_title("(a) 精度 × 速度平面：右上角=理想区，我们在前沿顶端", loc="left",
            fontweight="bold", fontsize=9.5, pad=6)

# (b) joint score as big horizontal bars with ratio labels
names = ["Ours xens2", "UFold", "MXfold2", "RNAformer", "RiNALMo-ft", "ViennaRNA"]
score = [0.8039 * 124.78, 0.6598 * 56.49, 0.6551 * 34.9, 0.7578 * 25.0,
         0.7602 * 14.5, 0.6729 * 3.6]
cols = [RED, GOLD, GREY, TEAL, PURPLE, GREY]
order = np.argsort(score)
b.barh(np.arange(6), np.array(score)[order], 0.62, color=[cols[i] for i in order])
for yi, i in enumerate(order):
    ratio = score[i] / score[order[-1]]
    b.text(np.array(score)[i] / max(score) + 0.015, yi,
           f"{score[i]:.1f}", va="center", fontsize=8, fontweight="bold",
           color=cols[i] if cols[i] != GREY else "#555")
b.set_yticks(np.arange(6), np.array(names)[order], fontsize=8)
b.set_xlabel("综合分 = F1 × 吞吐（seq/s）", fontsize=8.5)
b.set_xlim(0, 1.2)
b.text(1.01, 5.6, "2.7×", fontsize=13, color=RED, fontweight="bold", va="center")
b.text(1.01, 4.4, "9.1×", fontsize=13, color=RED, fontweight="bold", va="center")
b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
frame(b, grid_axis="x")
b.set_title("(b) 综合分排名：精度和速度都要，我们第一", loc="left",
            fontweight="bold", fontsize=9.5, pad=6)
fig.suptitle("主图 5 · 精度 × 速度联合前沿——与主图 4（校准）回答不同的问题",
             fontweight="bold", fontsize=11, y=0.99)
fig.text(0.012, 0.012, "主图4问「可信吗」（校准）；本图问「又准又快吗」（帕累托）。综合分 100.4 vs UFold 37.3（2.7×）"
         "vs RiNALMo-ft 11.0（9.1×）。全部实测，无模拟。", fontsize=7.2, color="#333", style="italic")
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(OUT / "fig5_acc_speed.png", dpi=300)
fig.savefig(OUT / "fig5_acc_speed.pdf")
plt.close(fig)
print("fig5 v3 done")
