from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent
DECK = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
BACKUP = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pre_round70_crossover.pptx"
VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round70_crossover_build_verification.json"
FROZEN = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_frozen_context_crossover_v1/full")
BASE_COMMIT = "b6f4d9c6452f302861ff0832e284de3512ac3bc1"
CROSSOVER_COMMIT = "810441e7480ea06a717097c57a49130c820c9636"
SUMMARY_SHA256 = "5d1984e9271e190cbc83c79d3c87f6571580fefc3ae84e0bbec5cca7db3fefd7"
VERIFICATION_SHA256 = "06322b53c93d6e5428b347176443e48a0b4ce0416c0ebc0b0c28557c58808054"

BURGUNDY = RGBColor(122, 32, 44)
DARK_RED = RGBColor(145, 36, 48)
INK = RGBColor(30, 25, 28)
MUTED = RGBColor(105, 96, 101)
ROSE = RGBColor(247, 241, 239)
PALE = RGBColor(250, 245, 246)
FAIL_PALE = RGBColor(252, 235, 237)
WHITE = RGBColor(255, 255, 255)
LINE = RGBColor(222, 192, 184)
SANS = "Source Han Sans SC"
SERIF = "Source Han Serif SC"
ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
DISPLAY_ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def signed(value: float) -> str:
    return f"{value:+.6f}"


def text(slide, x, y, w, h, value, size, color=INK, bold=False, font="Arial", align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_before = paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def rect(slide, x, y, w, h, fill, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line or fill
    return shape


def footer(slide, evidence, artifact):
    line = rect(slide, 0.78, 6.48, 11.78, 0.012, RGBColor(212, 165, 154))
    line.line.fill.background()
    text(slide, 0.78, 6.56, 11.65, 0.18, evidence, 7.6, MUTED)
    text(slide, 0.78, 6.76, 11.65, 0.18, artifact, 7.6, MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=DECK)
    parser.add_argument("--backup", type=Path, default=BACKUP)
    parser.add_argument("--verification", type=Path, default=VERIFICATION)
    parser.add_argument("--crossover-root", type=Path, default=FROZEN)
    args = parser.parse_args()
    deck = args.deck.resolve()
    backup = args.backup.resolve()
    verification_path = args.verification.resolve()
    frozen = args.crossover_root.resolve()
    summary_path = frozen / "summary.json"
    independent_path = frozen / "independent_verification.json"
    assert sha256(summary_path) == SUMMARY_SHA256
    assert sha256(independent_path) == VERIFICATION_SHA256
    summary = load(summary_path)
    independent = load(independent_path)
    assert summary["schema_version"] == "foldnar.gec_frozen_context_crossover.v1"
    assert independent["schema_version"] == "foldnar.gec_frozen_context_crossover.independent_verification.v1"
    assert summary["status"] == "COMPLETE" and summary["mode"] == "full"
    assert independent["status"] == "PASS" and independent["errors"] == []
    assert independent["independent_raw_truth_recomputation"] is True
    assert independent["official_evaluator_imported"] is False
    assert independent["official_aggregate_read"] is False
    assert summary["record_count"] == independent["record_count"] == 9143
    assert summary["panel_id"] == "tr1c_cluster80_dev_v1"
    assert summary["scientific_scope"] == independent["scientific_scope"] == "single_seed0_cluster_dev_frozen_causal_diagnostic_only"
    assert summary["gec_pass_claim"] is independent["gec_pass_claim"] is False
    assert summary["sota_claim"] is independent["sota_claim"] is False
    assert summary["seeds_1_2_authorized"] is independent["seeds_1_2_authorized"] is False
    assert summary["git_commit"] == CROSSOVER_COMMIT
    assert summary["decoder"]["search_allowed"] is False
    assert summary["models"] == list(ARMS) and summary["input_contexts"] == list(ARMS)
    assert len(summary["artifacts"]) == 16
    for arm in ARMS:
        native = summary["native_prediction_validation"][arm]
        assert native["pass"] is True and native["matched_records"] == native["record_count"] == 9143
        assert independent["native_prediction_validation"][arm] == native
        for context in ARMS:
            key = f"{arm}__{context}"
            assert key in summary["metrics"] and key in independent["metrics"]
            assert summary["metrics"][key]["micro_pair_f1"] == independent["metrics"][key]["micro_pair_f1"]
            assert summary["artifacts"][key]["sha256"]

    ensemble_metrics = {
        context: summary["metrics"][f"ensemble_context__{context}"]["micro_pair_f1"]
        for context in DISPLAY_ARMS
    }
    assert [fmt(ensemble_metrics[context]) for context in DISPLAY_ARMS] == ["0.699302", "0.681231", "0.687129"]
    assert ensemble_metrics["matched_control"] > ensemble_metrics["destroy_interval_shuffle"] > ensemble_metrics["ensemble_context"]
    independent_deltas = {}
    native_deltas = {}
    for arm in DISPLAY_ARMS:
        comparison = summary["comparisons"][arm]["each_context_s1_minus_matched_s1"]["independent_context"]
        independent_check = independent["comparisons"][arm]["each_context_s1_minus_matched_s1"]["independent_context"]
        delta = comparison["micro_pair_f1"]
        assert abs(delta - independent_check["micro_pair_f1"]) < 1e-12
        observed = summary["metrics"][f"{arm}__independent_context"]["micro_pair_f1"] - summary["metrics"][f"{arm}__matched_control"]["micro_pair_f1"]
        assert abs(delta - observed) < 1e-12 and delta < 0
        independent_deltas[arm] = delta
        native_delta = summary["comparisons"][arm]["native_s1_minus_shared_s0"]["micro_pair_f1"]
        assert native_delta == independent["comparisons"][arm]["native_s1_minus_shared_s0"]["micro_pair_f1"]
        native_deltas[arm] = native_delta
    assert [signed(independent_deltas[arm]) for arm in DISPLAY_ARMS] == ["-0.104102", "-0.193295", "-0.289777"]
    assert [signed(native_deltas[arm]) for arm in DISPLAY_ARMS] == ["+0.010192", "+0.001519", "+0.001206"]

    prs = Presentation(deck)
    if len(prs.slides) == 14 and backup.exists():
        prs = Presentation(backup)
    assert len(prs.slides) == 13, "Expected the preserved thirteen-slide Phase 0 deck; refusing duplicate append"
    if not backup.exists():
        descriptor, temporary = tempfile.mkstemp(prefix=backup.name + ".", suffix=".tmp", dir=backup.parent)
        os.close(descriptor)
        shutil.copy2(deck, temporary)
        os.replace(temporary, backup)
    assert len(Presentation(backup).slides) == 13
    prior_slide_xml = [slide.part.blob for slide in prs.slides]

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    text(slide, 0.78, 0.29, 11.8, 0.28, "33 · FOLDNAR PHASE 1 / FROZEN CONTEXT CROSSOVER", 10.5, BURGUNDY, True)
    text(slide, 0.78, 0.68, 11.9, 0.56, "上下文有因果作用，但真实系综输入并非一致最优", 24, INK, True, SERIF)
    text(slide, 0.78, 1.29, 11.9, 0.34,
         "MODEL-DEV ONLY · single seed0 / TR1c cluster80 dev · 16 组合 × 9,143 条 · 独立 raw-truth 重算 PASS",
         11.0, MUTED, font=SANS)

    cards = (
        (0.78, "matched 输入 · 三者最高", ensemble_metrics["matched_control"], "同一 ensemble 模型 / matched control"),
        (4.80, "真实 ensemble 输入 · native", ensemble_metrics["ensemble_context"], "原生上下文并未取得最高 micro F1"),
        (8.82, "destroy 输入", ensemble_metrics["destroy_interval_shuffle"], "扰动输入反而高于真实 ensemble"),
    )
    for x, label, value, note in cards:
        rect(slide, x, 1.78, 3.74, 1.18, PALE, LINE)
        text(slide, x + 0.22, 1.85, 3.30, 0.22, label, 10.0, BURGUNDY, True, SANS)
        text(slide, x + 0.22, 2.10, 3.30, 0.40, fmt(value), 23, DARK_RED, True)
        text(slide, x + 0.22, 2.57, 3.30, 0.20, note, 8.8, MUTED, font=SANS)
    text(slide, 0.78, 3.00, 11.70, 0.18, "ensemble_context 模型 · 仅切换输入上下文 · micro pair-F1", 8.8, MUTED, font=SANS)

    rect(slide, 0.78, 3.25, 5.74, 1.60, ROSE, LINE)
    text(slide, 1.00, 3.33, 5.30, 0.24, "independent 扰动：三个模型均显著退化", 11.0, BURGUNDY, True, SANS)
    text(slide, 1.00, 3.61, 5.30, 0.18, "micro pair-F1 · independent 输入 − matched 输入", 8.8, MUTED, font=SANS)
    for index, arm in enumerate(DISPLAY_ARMS):
        y = 3.86 + index * 0.30
        text(slide, 1.00, y, 3.25, 0.24, arm, 10.0, INK, True)
        text(slide, 4.22, y - 0.02, 2.04, 0.27, signed(independent_deltas[arm]), 14.0, DARK_RED, True, align=PP_ALIGN.RIGHT)

    rect(slide, 6.80, 3.25, 5.76, 1.60, PALE, LINE)
    text(slide, 7.02, 3.33, 5.32, 0.24, "原生历史差值复现：逐条预测全部吻合", 11.0, BURGUNDY, True, SANS)
    text(slide, 7.02, 3.61, 5.32, 0.18, "native s1 − shared matched s0 · micro pair-F1", 8.8, MUTED)
    for index, arm in enumerate(DISPLAY_ARMS):
        y = 3.86 + index * 0.30
        text(slide, 7.02, y, 3.25, 0.24, arm, 10.0, INK, True)
        text(slide, 10.24, y - 0.02, 2.08, 0.27, signed(native_deltas[arm]), 14.0, DARK_RED, True, align=PP_ALIGN.RIGHT)
    text(slide, 7.02, 4.69, 5.30, 0.15, "4 × 9,143 native prediction validation · PASS", 8.2, MUTED)

    rect(slide, 0.78, 5.07, 11.78, 1.23, FAIL_PALE, DARK_RED)
    text(slide, 1.00, 5.14, 3.20, 0.26, "边界与下一步 · 承接 structured margin", 11.0, DARK_RED, True, SANS)
    text(slide, 4.20, 5.13, 8.12, 0.29, "证明输入因果作用，不等于表示问题已解决", 12.5, DARK_RED, True, SANS)
    text(slide, 1.00, 5.50, 11.30, 0.31,
         "仅替换上下文即可改变同一模型预测，但真实 ensemble 并非一致最优；不得据此放行 GEC、宣称 SOTA 或启动 seeds 1/2。",
         9.8, INK, True, SANS)
    text(slide, 1.00, 5.87, 11.30, 0.26,
         "仅 seed0 cluster dev · 未读外部测试 · 固定 MEA decoder、无搜索；继续检验 structured margin，本页不构成其效果证据。",
         9.5, DARK_RED, True, SANS)
    footer(slide,
           f"base {BASE_COMMIT[:7]} · crossover {CROSSOVER_COMMIT[:7]} · 16/16 combinations · independent verification PASS · model development only",
           "frozen sources /teams/HoulabData_1788502323/liucunyu/foldnar/gec_frozen_context_crossover_v1/full/{summary.json,independent_verification.json}")

    descriptor, temporary = tempfile.mkstemp(prefix=deck.name + ".", suffix=".tmp.pptx", dir=deck.parent)
    os.close(descriptor)
    prs.save(temporary)
    os.replace(temporary, deck)
    reopened = Presentation(deck)
    assert len(reopened.slides) == 14
    assert [item.part.blob for item in list(reopened.slides)[:13]] == prior_slide_xml, "Prior slide XML changed"
    with zipfile.ZipFile(backup) as before, zipfile.ZipFile(deck) as after:
        for index in range(1, 14):
            part = f"ppt/slides/slide{index}.xml"
            assert before.read(part) == after.read(part), f"Prior slide part changed: {part}"
    new_slide = reopened.slides[13]
    new_text = []
    for shape in new_slide.shapes:
        assert shape.left >= 0 and shape.top >= 0
        assert shape.left + shape.width <= reopened.slide_width + 2
        assert shape.top + shape.height <= reopened.slide_height + 2
        if shape.has_text_frame:
            new_text.append(shape.text_frame.text)
    joined = "\n".join(new_text)
    for phrase in (
        "上下文有因果作用，但真实系综输入并非一致最优", "0.699302", "0.681231", "0.687129",
        "-0.104102", "-0.193295", "-0.289777", "+0.010192", "+0.001519", "+0.001206",
        "4 × 9,143 native prediction validation · PASS", "证明输入因果作用，不等于表示问题已解决",
        "不得据此放行 GEC、宣称 SOTA 或启动 seeds 1/2", "继续检验 structured margin，本页不构成其效果证据",
        "MODEL-DEV ONLY", "独立 raw-truth 重算 PASS",
    ):
        assert phrase in joined, phrase

    report = {
        "status": "PASS_BUILD_REOPEN_13_PRIOR_XML_ATOMIC_REPLACE_SOURCE_TEXT_AND_SHAPE_BOUNDS",
        "scientific_status": "CONTEXT_CAUSAL_BUT_REAL_ENSEMBLE_NOT_UNIFORMLY_OPTIMAL_MODEL_DEV_ONLY",
        "deck": str(deck),
        "deck_sha256": sha256(deck),
        "deck_bytes": deck.stat().st_size,
        "backup": str(backup),
        "backup_sha256": sha256(backup),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "base_commit": BASE_COMMIT,
        "crossover_commit": CROSSOVER_COMMIT,
        "sources": {
            str(summary_path): {"sha256": sha256(summary_path), "bytes": summary_path.stat().st_size},
            str(independent_path): {"sha256": sha256(independent_path), "bytes": independent_path.stat().st_size},
        },
        "source_values": {
            "record_count": summary["record_count"],
            "combination_count": len(summary["artifacts"]),
            "ensemble_model_micro_pair_f1_by_input": ensemble_metrics,
            "independent_minus_matched_micro_pair_f1": independent_deltas,
            "native_s1_minus_shared_s0_micro_pair_f1": native_deltas,
            "native_prediction_validated_records_per_arm": 9143,
            "native_prediction_validation_arms": ARMS,
            "scientific_scope": summary["scientific_scope"],
            "gec_pass_claim": False,
            "sota_claim": False,
            "representation_solved_claim": False,
            "decoder_search_allowed": False,
            "seeds_1_2_allowed": False,
            "next_hypothesis": "structured margin; this slide is not margin efficacy evidence",
        },
        "slide_count": len(reopened.slides),
        "deck_slide_index": 14,
        "added_slide_number": 33,
        "prior_slide_xml_preserved": 13,
        "atomic_replace": True,
        "shape_bounds_checks": len(new_slide.shapes),
        "render_visual_check": "PENDING_SEPARATE_LIBREOFFICE_RENDER_OCR_AND_IMAGE_REVIEW",
    }
    atomic_json(verification_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
