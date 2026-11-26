from pathlib import Path
import hashlib
import json
import shutil
import zipfile

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent
DECK = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
BACKUP = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pre_round69.pptx"
VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round69_build_verification.json"
FROZEN = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_residual_error_audit_v1")
MANIFEST = FROZEN / "artifact_manifest.json"
COMPARISON = FROZEN / "comparison.json"
AUDIT_VERIFICATION = FROZEN / "verification.json"
BASE_COMMIT = "82f6b4ef6f948c40aba0a53f73a31e1515af3efb"
RESIDUAL_COMMIT = "0589b2f489aa456f7a885b6d5dfe625b5de1b76b"

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


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text())


def signed(value):
    return f"{value:+.5f}"


manifest = load(MANIFEST)
comparison = load(COMPARISON)
audit = load(AUDIT_VERIFICATION)
assert manifest["schema_version"] == "foldnar.gec_residual_error_audit.v1"
assert manifest["git_head"] == RESIDUAL_COMMIT
assert comparison["schema_version"] == manifest["schema_version"]
assert comparison["status"] == "COMPLETE"
assert comparison["independent_seed0_failure_remains"] is True
assert audit["schema_version"] == "foldnar.gec_residual_error_audit.verification.v1"
assert audit["status"] == "PASS" and audit["errors"] == []
assert audit["external_test_sets"] is False
assert audit["comparison_sha256"] == sha256(COMPARISON)
for source in (COMPARISON, AUDIT_VERIFICATION):
    recorded = manifest["outputs"][source.name]
    assert recorded["path"] == str(source)
    assert source.stat().st_size == recorded["bytes"]
    assert sha256(source) == recorded["sha256"]

classes = audit["checks"]["ensemble_context"]["classes"]
nonpositive_fn = classes["nonpositive_score_margin"]
competition_fn = classes["positive_weight_mea_competition"]
total_fn = nonpositive_fn + competition_fn
nonpositive_pct = 100 * nonpositive_fn / total_fn
assert total_fn == 102903 and nonpositive_fn == 101164 and competition_fn == 1739
assert f"{nonpositive_pct:.2f}" == "98.31"
assert audit["checks"]["ensemble_context"]["sequences"] == 9143
assert audit["checks"]["ensemble_context"]["true_pairs"] == 247013

recall = comparison["recall_transitions_destroy_to_ensemble"]
transitions = recall["transitions"]["all"]
lost_to_nonpositive = transitions["selected->nonpositive_score_margin"]
lost_to_competition = transitions["selected->positive_weight_mea_competition"]
lost_total = lost_to_nonpositive + lost_to_competition
assert lost_total == 14471 and lost_to_nonpositive == 13993
assert recall["mea_weight_change"]["recall_lost"]["n"] == lost_total
assert recall["ensemble_forced_gap_on_changed_recall"]["recall_lost"]["n"] == lost_total
lost_nonpositive_pct = 100 * lost_to_nonpositive / lost_total

native = comparison["native_vs_counterfactual_context"]["ensemble_context"]
isolated = native["isolated:true"]["delta_native"]
long_span = native["span:ge128"]["delta_native"]
assert isolated["n"] == 10206 and long_span["n"] == 9866
assert isolated["mean"] < 0 and long_span["mean"] < 0
assert signed(isolated["mean"]) == "-0.74335"
assert signed(long_span["mean"]) == "-0.36870"

log = comparison["validation_logs"]["ensemble_context"]
assert log["best_step"] == 10500 and log["stop_step"] == 13000
assert log["stop_validation_nll"] > log["best_validation_nll"]
assert log["post_best_validation_nll_change"] > 0
assert log["training_loss_windows"]["last_500"]["mean"] < log["training_loss_windows"]["pre_best"]["mean"]
assert log["stop_reason_satisfied"] is True

prs = Presentation(DECK)
if len(prs.slides) == 13 and BACKUP.exists():
    prs = Presentation(BACKUP)
assert len(prs.slides) == 12, "Expected the preserved twelve-slide Phase 0 deck; refusing duplicate append"
if not BACKUP.exists():
    shutil.copy2(DECK, BACKUP)
assert len(Presentation(BACKUP).slides) == 12
prior_slide_xml = [slide.part.blob for slide in prs.slides]


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


slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid()
slide.background.fill.fore_color.rgb = WHITE
text(slide, 0.78, 0.29, 11.8, 0.28, "32 · FOLDNAR PHASE 1 / GEC RESIDUAL ERROR DECOMPOSITION", 10.5, BURGUNDY, True)
text(slide, 0.78, 0.68, 11.9, 0.56, "残差错误定位：FN 集中于非正 score margin", 25, INK, True, SERIF)
text(slide, 0.78, 1.29, 11.9, 0.34,
     "MODEL-DEV ONLY · 残差审计 0589b2f · 9,143 dev sequences / 247,013 true pairs · 关联分解，不作训练因果断言",
     11.0, MUTED, font=SANS)

cards = ((0.78, 3.74), (4.80, 3.74), (8.82, 3.74))
for x, width in cards:
    rect(slide, x, 1.76, width, 1.12, PALE, LINE)
text(slide, 1.00, 1.84, 3.30, 0.42, f"{nonpositive_pct:.2f}%", 25, DARK_RED, True)
text(slide, 1.00, 2.28, 3.30, 0.23, f"{nonpositive_fn:,} / {total_fn:,} ensemble FN", 10.5, INK, True)
text(slide, 1.00, 2.54, 3.30, 0.19, "nonpositive_score_margin；主错误类", 9.0, MUTED, font=SANS)
text(slide, 5.02, 1.86, 3.30, 0.38, f"{lost_to_nonpositive:,} / {lost_total:,}", 21, DARK_RED, True)
text(slide, 5.02, 2.28, 3.30, 0.23, "destroy → ensemble 丢失 recall", 10.3, INK, True, SANS)
text(slide, 5.02, 2.54, 3.30, 0.19, f"{lost_nonpositive_pct:.2f}% 转为非正 margin", 9.0, MUTED, font=SANS)
text(slide, 9.04, 1.86, 3.30, 0.38, f"{100 * competition_fn / total_fn:.2f}%", 21, BURGUNDY, True)
text(slide, 9.04, 2.28, 3.30, 0.23, f"仅 {competition_fn:,} / {total_fn:,} FN", 10.3, INK, True)
text(slide, 9.04, 2.54, 3.30, 0.19, "positive-weight MEA competition", 8.9, MUTED)

rect(slide, 0.78, 3.12, 5.74, 1.67, ROSE, LINE)
text(slide, 1.00, 3.22, 5.30, 0.25, "真值 delta_native 均值：isolated 与长 span 均为负", 11.0, BURGUNDY, True, SANS)
text(slide, 1.00, 3.60, 2.95, 0.28, "isolated true pairs", 11.0, INK, True)
text(slide, 4.02, 3.54, 2.24, 0.38, signed(isolated["mean"]), 19, DARK_RED, True, align=PP_ALIGN.RIGHT)
text(slide, 1.00, 3.91, 5.26, 0.20, f"n={isolated['n']:,} · negative fraction={100 * isolated['negative_fraction']:.2f}%", 8.8, MUTED)
text(slide, 1.00, 4.18, 2.95, 0.28, "pair span ≥128", 11.0, INK, True)
text(slide, 4.02, 4.12, 2.24, 0.38, signed(long_span["mean"]), 19, DARK_RED, True, align=PP_ALIGN.RIGHT)
text(slide, 1.00, 4.49, 5.26, 0.20, f"n={long_span['n']:,} · negative fraction={100 * long_span['negative_fraction']:.2f}%", 8.8, MUTED)

rect(slide, 6.80, 3.12, 5.76, 1.67, PALE, LINE)
text(slide, 7.02, 3.22, 5.32, 0.25, "后段过拟合存在，但不能解释 best=10500", 11.0, BURGUNDY, True, SANS)
text(slide, 7.02, 3.57, 5.32, 0.24,
     f"best {log['best_step']:,} → stop {log['stop_step']:,} · validation NLL {log['best_validation_nll']:.6f} → {log['stop_validation_nll']:.6f}",
     9.8, INK, True)
text(slide, 7.02, 3.87, 5.32, 0.23,
     f"ΔNLL={log['post_best_validation_nll_change']:+.6f} · last-500 train loss={log['training_loss_windows']['last_500']['mean']:.6f}",
     9.5, DARK_RED, True)
text(slide, 7.02, 4.17, 5.32, 0.43,
     "日志只证明 best 之后验证恶化；不能反向解释 best checkpoint 已发生的错误，也不据此调早停救援。",
     9.5, INK, True, SANS)

rect(slide, 0.78, 5.03, 11.78, 1.27, FAIL_PALE, DARK_RED)
text(slide, 1.00, 5.11, 3.32, 0.26, "下一可证伪假设 · 不是结论", 11.2, DARK_RED, True, SANS)
text(slide, 4.22, 5.08, 8.10, 0.31, "pair-Hamming cost-augmented structured max-margin", 14.2, DARK_RED, True)
text(slide, 1.00, 5.47, 11.30, 0.31,
     "若直接惩罚结构化 pair-Hamming 错误，应减少非正 margin FN，并改善 isolated / span≥128 recall；未改善则否决该假设。",
     10.0, INK, True, SANS)
text(slide, 1.00, 5.85, 11.30, 0.28,
     "仅 seed0 · 固定数据与 decoder · 禁止温度 / gamma / 阈值搜索 · 禁止 seeds 1/2 · 显存安全且冻结合同后才训练",
     9.6, DARK_RED, True, SANS)
footer(slide,
       "base 82f6b4e · residual audit 0589b2f · scope=registered TR1c cluster80 dev only · no external test · no decoder tuning",
       "frozen sources /teams/HoulabData_1788502323/liucunyu/foldnar/gec_residual_error_audit_v1/{comparison.json,verification.json,artifact_manifest.json}")

path_tmp = DECK.with_suffix(".round69_tmp.pptx")
prs.save(path_tmp)
path_tmp.replace(DECK)
reopened = Presentation(DECK)
assert len(reopened.slides) == 13
assert [slide.part.blob for slide in list(reopened.slides)[:12]] == prior_slide_xml, "Prior slide XML changed"
with zipfile.ZipFile(BACKUP) as before, zipfile.ZipFile(DECK) as after:
    for index in range(1, 13):
        part = f"ppt/slides/slide{index}.xml"
        assert before.read(part) == after.read(part), f"Prior slide part changed: {part}"
new_slide = reopened.slides[12]
new_text = []
for shape in new_slide.shapes:
    assert shape.left >= 0 and shape.top >= 0
    assert shape.left + shape.width <= reopened.slide_width + 2
    assert shape.top + shape.height <= reopened.slide_height + 2
    if shape.has_text_frame:
        new_text.append(shape.text_frame.text)
joined = "\n".join(new_text)
for phrase in (
    "98.31%", "101,164 / 102,903", "13,993 / 14,471", "96.70%",
    signed(isolated["mean"]), signed(long_span["mean"]), "best=10500",
    "不能反向解释 best checkpoint 已发生的错误", "pair-Hamming cost-augmented structured max-margin",
    "未改善则否决该假设", "禁止温度 / gamma / 阈值搜索", "禁止 seeds 1/2",
    "显存安全且冻结合同后才训练", "MODEL-DEV ONLY",
):
    assert phrase in joined, phrase

report = {
    "status": "PASS_BUILD_REOPEN_12_PRIOR_XML_ATOMIC_REPLACE_SOURCE_TEXT_AND_SHAPE_BOUNDS",
    "deck": str(DECK),
    "deck_sha256": sha256(DECK),
    "deck_bytes": DECK.stat().st_size,
    "backup": str(BACKUP),
    "backup_sha256": sha256(BACKUP),
    "builder": str(Path(__file__).resolve()),
    "builder_sha256": sha256(Path(__file__).resolve()),
    "base_commit": BASE_COMMIT,
    "residual_audit_commit": RESIDUAL_COMMIT,
    "sources": {
        str(MANIFEST): sha256(MANIFEST),
        str(COMPARISON): sha256(COMPARISON),
        str(AUDIT_VERIFICATION): sha256(AUDIT_VERIFICATION),
    },
    "source_values": {
        "ensemble_fn": total_fn,
        "nonpositive_score_margin_fn": nonpositive_fn,
        "nonpositive_score_margin_fn_percent": nonpositive_pct,
        "positive_weight_mea_competition_fn": competition_fn,
        "destroy_to_ensemble_recall_lost": lost_total,
        "recall_lost_to_nonpositive_score_margin": lost_to_nonpositive,
        "isolated_delta_native_mean": isolated["mean"],
        "span_ge128_delta_native_mean": long_span["mean"],
        "ensemble_validation_log": log,
        "best_checkpoint_error_explained_by_post_best_overfitting": False,
        "next_hypothesis": "pair-Hamming cost-augmented structured max-margin",
        "decoder_search_allowed": False,
        "seeds_1_2_allowed": False,
    },
    "slide_count": len(reopened.slides),
    "deck_slide_index": 13,
    "added_slide_number": 32,
    "prior_slide_xml_preserved": 12,
    "atomic_replace": True,
    "shape_bounds_checks": len(new_slide.shapes),
    "render_visual_check": "PENDING_SEPARATE_LIBREOFFICE_RENDER",
}
verification_tmp = VERIFICATION.with_suffix(".round69_tmp.json")
verification_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
verification_tmp.replace(VERIFICATION)
print(json.dumps(report, ensure_ascii=False, indent=2))
