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
BACKUP = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pre_round68.pptx"
VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round68_build_verification.json"
FROZEN = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3_independent_reeval_v1")
MANIFEST = FROZEN / "artifact_manifest.json"
REEVAL = FROZEN / "independent_reeval.json"
GATES = FROZEN / "gate_summary.json"
BASE_COMMIT = "5b80abc4e77743ee0d8cafe6f5fed3a65c78099f"
INDEPENDENT_COMMIT = "342a714"
CONVERGENCE_FIX_COMMIT = "6a1178e"

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
METRICS = ("micro_pair_f1", "macro_pair_f1", "micro_stem_f1", "macro_stem_f1")
METRIC_LABELS = {
    "micro_pair_f1": "micro pair-F1",
    "macro_pair_f1": "macro pair-F1",
    "micro_stem_f1": "micro stem-F1",
    "macro_stem_f1": "macro stem-F1",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text())


def fmt(value):
    return f"{value:.6f}"


def signed(value):
    return f"{value:+.6f}"


manifest = load(MANIFEST)
for source in (REEVAL, GATES):
    recorded = manifest["artifacts"][source.name]
    assert recorded["path"] == str(source)
    assert source.stat().st_size == recorded["bytes"]
    assert sha256(source) == recorded["sha256"]
reeval = load(REEVAL)
gates = load(GATES)
assert reeval["schema_version"] == "foldnar.gec_seed0_independent_reeval.v1"
assert reeval["status"] == gates["status"] == "FAIL_GEC_SEED0_INDEPENDENT_REEVAL"
assert reeval["overall_pass"] is gates["overall_pass"] is False
assert reeval["qualification_pass"] is gates["qualification_pass"] is False
assert reeval["code_identity"]["git_head"] == BASE_COMMIT
assert reeval["independence"] == {
    "aggregate_summary_read": False,
    "inputs": "validation_log.jsonl, train_log.jsonl, best.pt, resume.pt, dev_predictions.jsonl and frozen truth",
    "official_evaluator_imported": False,
    "run_meta_read": False,
}
panel = reeval["panel"]
assert panel["panel_id"] == "tr1c_cluster80_dev_v1"
assert panel["route_scope"] == "model development only; no external test data"
assert panel["exact_train_id_overlap"] is False and panel["train_cluster_overlap"] is False
arm_metrics = gates["arm_metrics"]
assert arm_metrics == reeval["arm_metrics"]
assert set(arm_metrics) == set(ARMS)
control_deltas = gates["matched_control_deltas"]
destroy_deltas = gates["destroy_deltas"]
for metric in METRICS:
    ensemble = arm_metrics["ensemble_context"][metric]
    assert control_deltas[metric]["delta"] < 0
    assert destroy_deltas[metric]["delta"] < 0
    assert abs(control_deltas[metric]["delta"] - (ensemble - arm_metrics["matched_control"][metric])) < 1e-12
    assert abs(destroy_deltas[metric]["delta"] - (ensemble - arm_metrics["destroy_interval_shuffle"][metric])) < 1e-12
metric_gates = gates["metric_gates"]
assert len(metric_gates["gates"]) == 16
passed_count = sum(metric_gates["gates"].values())
assert passed_count == 4 and len(metric_gates["failed_gates"]) == 12
assert metric_gates["pass"] is False
assert set(metric_gates["failed_gates"]) == {name for name, passed in metric_gates["gates"].items() if not passed}
targets = gates["targets"]
isolated = targets["isolated_pair:isolated"]
long_span = targets["pair_span:ge128"]
for target in (isolated, long_span):
    assert target["recall_delta"] < 0 and target["target_gate_pass"] is False
    assert target["diversity"]["recall_delta"] == target["recall_delta"]
assert reeval["qualification_gates"] == gates.get("qualification_gates", reeval["qualification_gates"])
qualification_failed = [name for name, passed in reeval["qualification_gates"].items() if not passed]
assert len(qualification_failed) == 4
assert all(name.endswith(":checkpoint_and_restored_convergence") for name in qualification_failed)

prs = Presentation(DECK)
if len(prs.slides) == 12 and BACKUP.exists():
    prs = Presentation(BACKUP)
assert len(prs.slides) == 11, "Expected the preserved eleven-slide Phase 0 deck; refusing duplicate append"
if not BACKUP.exists():
    shutil.copy2(DECK, BACKUP)
assert len(Presentation(BACKUP).slides) == 11
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


def table(slide, x, y, widths, row_height, headers, rows, emphasis_row=None, font_size=10.0):
    shape = slide.shapes.add_table(
        len(rows) + 1, len(headers), Inches(x), Inches(y),
        Inches(sum(widths)), Inches(row_height * (len(rows) + 1)),
    )
    tbl = shape.table
    for index, width in enumerate(widths):
        tbl.columns[index].width = Inches(width)
    for row_index in range(len(rows) + 1):
        tbl.rows[row_index].height = Inches(row_height)
        values = headers if row_index == 0 else rows[row_index - 1]
        for column_index, value in enumerate(values):
            cell = tbl.cell(row_index, column_index)
            cell.fill.solid()
            if row_index == 0:
                cell.fill.fore_color.rgb = BURGUNDY
            elif row_index == emphasis_row:
                cell.fill.fore_color.rgb = FAIL_PALE
            else:
                cell.fill.fore_color.rgb = ROSE if row_index % 2 == 0 else WHITE
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.01)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if column_index == 0 else PP_ALIGN.CENTER
            paragraph.space_before = paragraph.space_after = Pt(0)
            paragraph.font.name = SANS if column_index == 0 else "Arial"
            paragraph.font.size = Pt(font_size)
            paragraph.font.bold = row_index == 0 or row_index == emphasis_row
            paragraph.font.color.rgb = WHITE if row_index == 0 else (DARK_RED if row_index == emphasis_row else INK)
    return tbl


def footer(slide, evidence, artifact):
    line = rect(slide, 0.78, 6.48, 11.78, 0.012, RGBColor(212, 165, 154))
    line.line.fill.background()
    text(slide, 0.78, 6.56, 11.65, 0.18, evidence, 7.6, MUTED)
    text(slide, 0.78, 6.76, 11.65, 0.18, artifact, 7.6, MUTED)


slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid()
slide.background.fill.fore_color.rgb = WHITE
text(slide, 0.78, 0.29, 11.8, 0.28, "31 · FOLDNAR PHASE 1 / GEC SEED0 INDEPENDENT VERDICT", 10.5, BURGUNDY, True)
text(slide, 0.78, 0.68, 11.9, 0.56, "GEC seed0 独立验收失败：停止放行，不启动 seeds 1/2", 25, INK, True, SERIF)
text(slide, 0.78, 1.29, 11.9, 0.34,
     f"MODEL-DEV ONLY · {panel['n_sequences']:,} frozen dev sequences · 独立重算 {INDEPENDENT_COMMIT} · 不读取外部测试，不构成 SOTA 结论",
     11.2, MUTED, font=SANS)

arm_rows = []
arm_names = {
    "matched_control": "matched_control",
    "ensemble_context": "ensemble_context · 真实 GEC",
    "destroy_interval_shuffle": "destroy_interval_shuffle",
    "independent_context": "independent_context",
}
for arm in ARMS:
    arm_rows.append([arm_names[arm]] + [fmt(arm_metrics[arm][metric]) for metric in METRICS])
table(slide, 0.78, 1.75, (3.18, 2.15, 2.15, 2.15, 2.15), 0.34,
      ("四臂 / independent recomputation", "micro pair-F1", "macro pair-F1", "micro stem-F1", "macro stem-F1"),
      arm_rows, emphasis_row=2, font_size=9.8)
text(slide, 0.78, 3.49, 11.7, 0.22,
     "Δ = ensemble_context − 对照；负值表示真实-GEC 更差。destroy 相对 ensemble 的四项核心 F1 均更好。",
     9.5, DARK_RED, True, SANS)

delta_rows = [[METRIC_LABELS[metric], signed(control_deltas[metric]["delta"]), signed(destroy_deltas[metric]["delta"])] for metric in METRICS]
table(slide, 0.78, 3.80, (2.75, 2.18, 2.18), 0.31,
      ("核心指标", "Δ vs control", "Δ vs destroy"), delta_rows, emphasis_row=None, font_size=9.8)

rect(slide, 8.13, 3.80, 4.43, 1.55, PALE, LINE)
text(slide, 8.34, 3.88, 4.00, 0.24, "目标子群 recall Δ vs control", 10.6, BURGUNDY, True, SANS)
text(slide, 8.34, 4.18, 2.24, 0.28, "isolated pair", 10.2, INK, True)
text(slide, 10.54, 4.15, 1.78, 0.32, signed(isolated["recall_delta"]), 15, DARK_RED, True, align=PP_ALIGN.RIGHT)
text(slide, 8.34, 4.47, 3.98, 0.20,
     f"{isolated['n_pairs']:,} pairs · {isolated['n_sequences']:,} sequences · FAIL", 8.5, MUTED)
text(slide, 8.34, 4.72, 2.24, 0.28, "pair span ≥128", 10.2, INK, True)
text(slide, 10.54, 4.69, 1.78, 0.32, signed(long_span["recall_delta"]), 15, DARK_RED, True, align=PP_ALIGN.RIGHT)
text(slide, 8.34, 5.01, 3.98, 0.20,
     f"{long_span['n_pairs']:,} pairs · {long_span['n_sequences']:,} sequences · FAIL", 8.5, MUTED)

rect(slide, 0.78, 5.54, 11.78, 0.76, FAIL_PALE, DARK_RED)
text(slide, 1.00, 5.61, 2.10, 0.29, f"16 门仅 {passed_count} 过 / 12 失败", 15, DARK_RED, True, SANS)
text(slide, 3.18, 5.59, 9.10, 0.33,
     "总体 FAIL_GEC_SEED0_INDEPENDENT_REEVAL；qualification 亦未通过（4 项 checkpoint/restored convergence）。",
     10.0, DARK_RED, True, SANS)
text(slide, 1.00, 5.94, 11.28, 0.25,
     f"禁止包装成正结果、禁止启动 seeds 1/2、禁止调门救援；{CONVERGENCE_FIX_COMMIT} 仅修复后续状态保留，不改写本次历史裁决。",
     9.7, INK, True, SANS)
footer(slide,
       f"base {BASE_COMMIT[:7]} · independent reeval {INDEPENDENT_COMMIT} · metric gates 4/16 · panel={panel['panel_id']} · scope=model development only",
       "frozen sources /teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3_independent_reeval_v1/{independent_reeval.json,gate_summary.json,artifact_manifest.json}")

path_tmp = DECK.with_suffix(".round68_tmp.pptx")
prs.save(path_tmp)
path_tmp.replace(DECK)
reopened = Presentation(DECK)
assert len(reopened.slides) == 12
assert [slide.part.blob for slide in list(reopened.slides)[:11]] == prior_slide_xml, "Prior slide XML changed"
with zipfile.ZipFile(BACKUP) as before, zipfile.ZipFile(DECK) as after:
    for index in range(1, 12):
        part = f"ppt/slides/slide{index}.xml"
        assert before.read(part) == after.read(part), f"Prior slide part changed: {part}"
new_slide = reopened.slides[11]
new_text = []
for shape in new_slide.shapes:
    assert shape.left >= 0 and shape.top >= 0
    assert shape.left + shape.width <= reopened.slide_width + 2
    assert shape.top + shape.height <= reopened.slide_height + 2
    if shape.has_text_frame:
        new_text.append(shape.text_frame.text)
    if shape.has_table:
        new_text.extend(cell.text for row in shape.table.rows for cell in row.cells)
joined = "\n".join(new_text)
for arm in ARMS:
    assert arm in joined
    for metric in METRICS:
        assert fmt(arm_metrics[arm][metric]) in joined
for metric in METRICS:
    assert signed(control_deltas[metric]["delta"]) in joined
    assert signed(destroy_deltas[metric]["delta"]) in joined
for phrase in (
    "FAIL_GEC_SEED0_INDEPENDENT_REEVAL", "16 门仅 4 过 / 12 失败", "MODEL-DEV ONLY",
    "不启动 seeds 1/2", "禁止启动 seeds 1/2", signed(isolated["recall_delta"]),
    signed(long_span["recall_delta"]), "destroy 相对 ensemble 的四项核心 F1 均更好",
):
    assert phrase in joined
assert "PASS_GEC_SEED0" not in joined

report = {
    "status": "PASS_BUILD_REOPEN_11_PRIOR_XML_ATOMIC_REPLACE_SOURCE_TEXT_AND_SHAPE_BOUNDS",
    "deck": str(DECK),
    "deck_sha256": sha256(DECK),
    "deck_bytes": DECK.stat().st_size,
    "backup": str(BACKUP),
    "backup_sha256": sha256(BACKUP),
    "builder": str(Path(__file__).resolve()),
    "builder_sha256": sha256(Path(__file__).resolve()),
    "base_commit": BASE_COMMIT,
    "independent_reeval_commit": INDEPENDENT_COMMIT,
    "convergence_fix_commit": CONVERGENCE_FIX_COMMIT,
    "sources": {
        str(MANIFEST): sha256(MANIFEST),
        str(REEVAL): sha256(REEVAL),
        str(GATES): sha256(GATES),
    },
    "source_values": {
        "status": gates["status"],
        "arm_metrics": arm_metrics,
        "ensemble_minus_control": {metric: control_deltas[metric]["delta"] for metric in METRICS},
        "ensemble_minus_destroy": {metric: destroy_deltas[metric]["delta"] for metric in METRICS},
        "isolated_recall_delta": isolated["recall_delta"],
        "span_ge128_recall_delta": long_span["recall_delta"],
        "metric_gates_passed": passed_count,
        "metric_gates_total": len(metric_gates["gates"]),
        "qualification_pass": reeval["qualification_pass"],
        "panel": panel,
    },
    "slide_count": len(reopened.slides),
    "deck_slide_index": 12,
    "added_slide_number": 31,
    "prior_slide_xml_preserved": 11,
    "atomic_replace": True,
    "shape_bounds_checks": len(new_slide.shapes),
    "render_visual_check": "PENDING_SEPARATE_LIBREOFFICE_RENDER",
}
verification_tmp = VERIFICATION.with_suffix(".round68_tmp.json")
verification_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
verification_tmp.replace(VERIFICATION)
print(json.dumps(report, ensure_ascii=False, indent=2))
