#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ppt/.round39_work/source_data/appendix_leakage/round39_ece_ppt_source.json"
DECK = ROOT / "ppt/RNA-JEV_论文逻辑重构_20261031_导师讨论_round38.pptx"
BACKUP = ROOT / "ppt/.round39_work/round38_before_ece_update.pptx"
FIGURE = ROOT / "ppt/figures/round38_ece_evidence.png"
QA = ROOT / "ppt/.round39_work/round38_ece_update_qa.json"
SPLITS = ["bprna_ts0", "testsetb", "bprna_new", "archiveii_embok_clean"]
LABELS = {"bprna_ts0": "TS0", "testsetb": "TestSetB", "bprna_new": "bpRNA-new", "archiveii_embok_clean": "ArchiveII"}
NAMES = {"mxfold2": "MXfold2", "contrafold": "CONTRAfold", "eternafold": "EternaFold"}
BLUE = "#0F4D92"
TEAL = "#0E7C86"
GOLD = "#C98A22"
MUTED = "#667085"
GRID = "#DDE3EA"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_figure(data):
    rows = {cell["split"]: cell for cell in data["cells"]}
    assert set(rows) == set(SPLITS)
    assert data["status"] == "PASS_FIGURE_SOURCE_NOT_UNIVERSAL_ECE_LEADERBOARD"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.linewidth": 0.8, "figure.dpi": 300})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 6.4), gridspec_kw={"width_ratios": [1.02, 1]})
    x = np.arange(len(SPLITS))
    raw = [rows[split]["xens2_own_clean"]["raw_ece"] for split in SPLITS]
    calibrated = [rows[split]["xens2_own_clean"]["calibrated_ece"] for split in SPLITS]
    width = 0.34
    left.bar(x - width / 2, raw, width, color="#A7B0BA", label="Before VL0 affine")
    left.bar(x + width / 2, calibrated, width, color=TEAL, label="After VL0 affine")
    left.set_yscale("log")
    left.set_ylim(0.00035, 0.12)
    left.set_xticks(x, [LABELS[split] for split in SPLITS])
    left.set_ylabel("ECE (log scale; lower is better)")
    left.set_title("A  |  VL0 affine reduces ECE (own-clean)", loc="left", fontsize=12, fontweight="bold", color=BLUE, pad=13)
    for index, values in enumerate(zip(raw, calibrated)):
        for dx, value in zip([-width / 2, width / 2], values):
            left.text(index + dx, value * 1.12, f"{value:.4f}", ha="center", va="bottom", fontsize=7.5, color="#344054")
    left.legend(frameon=False, loc="upper right")
    left.grid(axis="y", color=GRID, linewidth=0.65)
    left.set_axisbelow(True)

    right.set_xscale("log")
    right.set_xlim(0.00018, 0.012)
    right.set_ylim(-0.75, 3.65)
    right.set_yticks(range(4), [LABELS[split] for split in reversed(SPLITS)])
    right.set_xlabel("ECE (log scale; lower is better)")
    right.set_title("B  |  Lower-ECE baselines remain (matched sensitivity)", loc="left", fontsize=12, fontweight="bold", color=BLUE, pad=13)
    for y, split in enumerate(reversed(SPLITS)):
        cell = rows[split]
        baseline = cell["baseline"]
        ours_value = baseline["matched_xens2"]["ece"]
        best_value = baseline["matched_baseline"]["ece"]
        system = baseline["system"]
        assert best_value < ours_value
        right.plot([best_value, ours_value], [y, y], color="#B5BDC7", linewidth=1.7, zorder=1)
        right.scatter([ours_value], [y], s=62, color=BLUE, zorder=3)
        right.scatter([best_value], [y], s=62, color=GOLD, zorder=3)
        right.text(ours_value * 1.05, y + 0.02, f"xens2  {ours_value:.6f}", va="center", fontsize=8.1, color=BLUE)
        right.text(best_value * 1.02, y + 0.16, f"{NAMES[system]}  {best_value:.6f}", va="center", ha="left", fontsize=8.1, color=GOLD)
        right.text(np.sqrt(best_value * ours_value), y - 0.24, f"matched n={baseline['matched_n']:,}", ha="center", fontsize=7.7, color=MUTED)
    right.scatter([], [], s=62, color=BLUE, label="xens2 (same matched set)")
    right.scatter([], [], s=62, color=GOLD, label="Lowest-ECE eligible baseline")
    right.legend(frameon=False, loc="upper right", fontsize=8.2)
    right.grid(axis="x", color=GRID, linewidth=0.65)
    right.set_axisbelow(True)
    for axis in (left, right):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.13, wspace=0.29)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=300, facecolor="white")
    plt.close(fig)
    from PIL import Image
    with Image.open(FIGURE) as image:
        assert image.size == (3840, 1920), image.size


def replace_text(slide, old, new):
    replaced = 0
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for paragraph in shape.text_frame.paragraphs:
            for run in paragraph.runs:
                if old in run.text:
                    run.text = run.text.replace(old, new)
                    replaced += 1
    assert replaced == 1, (old, replaced)


def update_deck():
    prs = Presentation(BACKUP)
    assert len(prs.slides) == 23
    slide = prs.slides[7]
    replace_text(slide, "主图 4｜大样本 split 校准改善，小 PDB split 不作过度承诺",
                 "主图 4｜VL0 校准改善，但不构成全板最低 ECE")
    replace_text(slide, "VL0 拟合两参数 logit 仿射，测试 split 不重拟合；ECE 与排序能力并列。",
                 "左：xens2 校准前后；右：各 split 最优基线与 matched n；概率校准与结构排序分开。")
    replace_text(slide, "TS0/TestSetB/bpRNA-new/ArchiveII 均改善；TS1 与 TS-hard 小幅改善，TS2/TS3 过校正，必须保留 n 与限制。",
                 "128 格注册概率证据已闭环；多个基线 ECE 更低，Spearman/AURC 需分开判断。")
    replace_text(slide, "台账 §15.80；structure confidence board",
                 "台账 §15.440；eligible_probability_assessment；matched 仅作敏感性")
    old_picture = next(shape for shape in slide.shapes if shape.shape_type == 13)
    position = (old_picture.left, old_picture.top, old_picture.width, old_picture.height)
    old_picture._element.getparent().remove(old_picture._element)
    slide.shapes.add_picture(str(FIGURE), *position)
    slide.notes_slide.notes_text_frame.text = (
        "主图 4｜VL0 校准改善，但不构成全板最低 ECE\n"
        "左侧是 xens2 在各自 own-clean 上的校准前后结果，VL0 仿射确实降低 ECE，测试集没有重新拟合。\n"
        "右侧是逐基线 matched-clean 附加敏感性，点和 n 一一对应：TS0 的 MXfold2 ECE=0.000463753，低于 xens2=0.000625610；TestSetB 的 CONTRAfold=0.001538720，低于 xens2=0.002159792；bpRNA-new 的 EternaFold=0.000954411，低于 xens2=0.004831027；ArchiveII 的 CONTRAfold=0.000491236，低于 xens2=0.001030768。\n"
        "注册概率证据共 128 格，pending=0；MXfold2 ArchiveII 的原生 BPP 超界，ECE 不定义，不能夹紧或填 0。不同系统的 clean n 不强行统一，matched 只作附加敏感性，own-clean F1 主榜仍是 5/8。\n"
        "ECE 与 Spearman/AURC 是不同问题。多个基线在部分 split 的概率误差和结构筛选更好，因此不能说全板最低 ECE 或最佳置信度。"
    )
    temporary = DECK.with_suffix(".ece_tmp.pptx")
    prs.save(temporary)
    temporary.replace(DECK)


def verify(before_text):
    prs = Presentation(DECK)
    assert len(prs.slides) == 23
    after_text = [[shape.text for shape in slide.shapes if shape.has_text_frame] for slide in prs.slides]
    unchanged = all(before_text[index] == after_text[index] for index in range(23) if index != 7)
    assert unchanged
    assert any(shape.shape_type == 13 for shape in prs.slides[7].shapes)
    payload = {"status": "PASS_ROUND38_ECE_PAGE_ONLY_PENDING_RENDER",
               "changed_slide": 8, "other_22_slide_text_unchanged": unchanged,
               "deck": str(DECK), "deck_sha256": sha256(DECK),
               "figure": str(FIGURE), "figure_sha256": sha256(FIGURE),
               "source": str(SOURCE), "source_sha256": sha256(SOURCE)}
    QA.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    before = Presentation(BACKUP)
    before_text = [[shape.text for shape in slide.shapes if shape.has_text_frame] for slide in before.slides]
    data = json.loads(SOURCE.read_text())
    make_figure(data)
    update_deck()
    verify(before_text)
    print(json.dumps({"status": "PASS_ROUND38_ECE_PAGE_ONLY_PENDING_RENDER", "figure": str(FIGURE),
                      "deck": str(DECK), "sha256": sha256(DECK)}, indent=2))


if __name__ == "__main__":
    main()
