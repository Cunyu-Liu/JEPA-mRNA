from pathlib import Path
import hashlib
import json
import shutil

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent
DECK = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
BACKUP = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pre_round64.pptx"
VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round64_build_verification.json"
GOV = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/data_governance_v1")
BASE = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2")
DIST = GOV / "distribution_and_quality.json"
SUMMARY = GOV / "audit_summary.json"
INDEP = GOV / "independent_verification.json"
MMSEQS = GOV / "mmseqs_evidence.json"
LEAKAGE = GOV / "leakage_matrix.json"
METRICS = BASE / "reports/bprna_new_raw_metrics_verification_v2.json"

BURGUNDY = RGBColor(122, 32, 44)
INK = RGBColor(30, 25, 28)
MUTED = RGBColor(105, 96, 101)
ROSE = RGBColor(247, 241, 239)
PALE = RGBColor(250, 245, 246)
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(34, 113, 84)
LINE = RGBColor(222, 192, 184)
SANS = "Source Han Sans SC"
SERIF = "Source Han Serif SC"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commas(value):
    return f"{value:,}"


def pct(value, total, digits=3):
    return f"{100 * value / total:.{digits}f}%"


def load(path):
    return json.loads(path.read_text())


dist = load(DIST)
summary = load(SUMMARY)
indep = load(INDEP)
mmseqs = load(MMSEQS)
leakage = load(LEAKAGE)
metrics = load(METRICS)
assert indep["status"] == "PASS" and indep["n_checks"] == 154
assert all(check["passed"] for check in indep["checks"])
assert metrics["status"] == "PASS"
assert summary["status"] == "COMPLETE_AUDIT_WITH_UNCHECKED_LAYERS"
raw = dist["scopes"]["raw"]["datasets"]

prs = Presentation(DECK)
if len(prs.slides) == 6 and BACKUP.exists():
    prs = Presentation(BACKUP)
assert len(prs.slides) == 3, "Expected the preserved three-slide Phase 0 deck; refusing duplicate append"
if not BACKUP.exists():
    shutil.copy2(DECK, BACKUP)


def text(slide, x, y, w, h, value, size, color=INK, bold=False, font="Arial", align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]
    p.alignment = align
    p.space_before = p.space_after = Pt(0)
    run = p.add_run()
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


def new_slide(number, section, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    text(slide, 0.78, 0.29, 11.8, 0.28, section, 10.5, BURGUNDY, True)
    text(slide, 0.78, 0.70, 11.9, 0.56, title, 26, INK, True, SERIF)
    text(slide, 0.78, 1.31, 11.9, 0.33, subtitle, 11.5, MUTED, font=SANS)
    text(slide, 11.96, 6.99, 0.6, 0.20, str(number), 10, BURGUNDY, True, align=PP_ALIGN.RIGHT)
    return slide


def table(slide, x, y, widths, row_height, headers, rows, font_size=10.5, header_size=None, aligns=None):
    shape = slide.shapes.add_table(
        len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_height * (len(rows) + 1))
    )
    tbl = shape.table
    for index, width in enumerate(widths):
        tbl.columns[index].width = Inches(width)
    for r in range(len(rows) + 1):
        tbl.rows[r].height = Inches(row_height)
        values = headers if r == 0 else rows[r - 1]
        for c, value in enumerate(values):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BURGUNDY if r == 0 else (ROSE if r % 2 == 0 else WHITE)
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(0.045)
            cell.margin_top = cell.margin_bottom = Inches(0.012)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = aligns[c] if aligns else (PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
            p.space_before = p.space_after = Pt(0)
            p.font.name = SANS if c == 0 else "Arial"
            p.font.size = Pt((header_size or font_size) if r == 0 else font_size)
            p.font.bold = r == 0
            p.font.color.rgb = WHITE if r == 0 else INK
    return tbl


def footer(slide, evidence, artifact):
    line = rect(slide, 0.78, 6.48, 11.78, 0.012, RGBColor(212, 165, 154))
    line.line.fill.background()
    text(slide, 0.78, 6.57, 11.5, 0.17, evidence, 7.8, MUTED)
    text(slide, 0.78, 6.77, 11.5, 0.17, artifact, 7.8, MUTED)


required = {23: [], 24: [], 25: []}
slide = new_slide(
    23,
    "23 · FOLDNAR PHASE 0 / DEVELOPMENT-ONLY DATA GOVERNANCE",
    "TR1c / CHANRG：pair QA 与分布底账",
    "仅 Train / Validation · 未访问 Test / GenA / GenC / GenF · 观测审计，不是采样、训练或模型结果",
)
quality_rows = []
for key, label in (("tr1c", "TR1c raw"), ("chanrg_train", "CHANRG train"), ("chanrg_validation", "CHANRG validation")):
    item = raw[key]
    length = item["length"]
    gc = item["gc_fraction"]
    qa = item["quality_assurance"]
    n_pairs = item["n_pairs"]
    row = (
        label,
        commas(item["n_records"]),
        f"{length['median']:.0f} [{length['p25']:.0f}–{length['p75']:.0f}] · {length['min']:.0f}–{length['max']:.0f}",
        f"{100 * gc['median']:.2f}% [{100 * gc['p25']:.2f}–{100 * gc['p75']:.2f}]",
        commas(n_pairs),
        f"{commas(qa['noncanonical_pairs'])} ({pct(qa['noncanonical_pairs'], n_pairs)})",
        f"{commas(qa['min_loop_span_lt4_pairs'])} ({pct(qa['min_loop_span_lt4_pairs'], n_pairs)})",
    )
    quality_rows.append(row)
    required[23].extend(row[1:])
table(
    slide, 0.78, 1.78, (1.65, 0.83, 2.12, 2.03, 1.18, 2.06, 1.91), 0.56,
    ("Dataset", "records", "length median [IQR] · range", "GC median [IQR]", "pairs", "noncanonical n (pair %)", "span < 4 n (pair %)"),
    quality_rows, font_size=8.9, header_size=8.5,
)
tr1c = raw["tr1c"]
ct = raw["chanrg_train"]
cv = raw["chanrg_validation"]
boxes = (
    (0.78, "非 canonical 并非边缘类别", f"CHANRG 含非 canonical 的记录：train {commas(ct['distributions']['canonical_status']['counts']['CONTAINS_NONCANONICAL'])} / {commas(ct['n_records'])}（{pct(ct['distributions']['canonical_status']['counts']['CONTAINS_NONCANONICAL'], ct['n_records'], 2)}）；validation {commas(cv['distributions']['canonical_status']['counts']['CONTAINS_NONCANONICAL'])} / {commas(cv['n_records'])}（{pct(cv['distributions']['canonical_status']['counts']['CONTAINS_NONCANONICAL'], cv['n_records'], 2)}）。TR1c 为 0。"),
    (4.80, "pair-level QA 通过", "三套 raw 数据均为 valid coordinates、无 crossing pairs、无 partner conflicts、无 pair-symmetry mismatch；这只说明解析与 pair 一致性，不等于实验质量已验证。"),
    (8.82, "ID 与实验 lineage", f"TR1c 有 {commas(tr1c['n_duplicate_id_rows_beyond_first'])} 条重复 ID 余量（{commas(tr1c['n_duplicate_id_groups'])} 组），但 normalized sequence 无重复；CHANRG 重复 ID 为 0。逐记录实验 lineage / 质量仍 UNRESOLVED。"),
)
for x, heading, body in boxes:
    rect(slide, x, 4.27, 3.74, 1.70, PALE, RGBColor(225, 206, 209))
    text(slide, x + 0.18, 4.39, 3.38, 0.27, heading, 12, BURGUNDY, True, SANS)
    text(slide, x + 0.18, 4.73, 3.38, 1.08, body, 9.8, INK, font=SANS)
required[23].extend(["85,334", "9,718", "3,496", "3,480"])
text(slide, 0.78, 6.10, 11.6, 0.22, "分布显示 TR1c 长度 19–500 nt，而 CHANRG 为 50–200 nt；分层与模型比较必须保留这一支持域差异。", 9.8, MUTED, font=SANS)
footer(slide, "evidence data_governance_v1 · independent source reread 154 / 154 PASS", "artifact /teams/HoulabData_1788502323/liucunyu/foldnar/data_governance_v1/distribution_and_quality.json")

slide = new_slide(
    24,
    "24 · FOLDNAR PHASE 0 / LEAKAGE BOUNDARIES",
    "去污边界：‘未检出’不等于独立或 clean",
    "exact sequence · MMseqs2 80% identity / 80% coverage · exact topology signature · 未检查层显式保留",
)
state_labels = {
    "tr1c_registered_train_vs_dev": "TR1c registered train ↔ dev",
    "chanrg_raw_train_vs_validation": "CHANRG raw train ↔ validation",
    "chanrg_matched_clean_accounting": "CHANRG matched-clean accounting",
    "tr1c_vs_chanrg_train": "TR1c ↔ CHANRG train",
    "tr1c_vs_chanrg_validation": "TR1c ↔ CHANRG validation",
}
leak_rows = []
for key, label in state_labels.items():
    state = summary["leakage_states"][key]
    row = (label, state["exact_sequence"], state["homology_80pct"], state["exact_structure_signature"], state["pretraining_exposure"], state["temporal_and_label_lineage"])
    leak_rows.append(row)
    required[24].extend(row[1:])
table(
    slide, 0.78, 1.77, (3.18, 1.85, 1.85, 1.55, 1.60, 1.75), 0.43,
    ("Comparison", "exact sequence", "80% homology", "exact topology", "pretraining", "time / label"),
    leak_rows, font_size=8.2, header_size=8.8,
)
account = summary["matched_clean_accounting"]
retained_train = raw["chanrg_train"]["n_records"] - account["n_train_excluded"]
retained_validation = raw["chanrg_validation"]["n_records"] - account["n_validation_excluded"]
shared_topology = leakage["comparisons"]["chanrg_matched_clean_accounting"]["evidence"]["exact_structure_signature"]["n_shared"]
rect(slide, 0.78, 4.55, 5.72, 1.49, ROSE, LINE)
text(slide, 1.00, 4.66, 5.28, 0.25, "CHANRG matched-clean：仅保守会计视图", 12.2, BURGUNDY, True, SANS)
account_text = (
    f"{commas(account['n_shared_clusters'])} 个 cross-split 80% clusters；双侧排除 train {commas(account['n_train_excluded'])}、validation {commas(account['n_validation_excluded'])} 条，"
    f"余量 {commas(retained_train)} / {commas(retained_validation)}。raw 80% homology 命中 {commas(mmseqs['homology_summary']['n_validation_records'])} 条 validation、{commas(mmseqs['homology_summary']['n_train_records'])} 条 train。"
)
text(slide, 1.00, 4.98, 5.28, 0.88, account_text, 10.1, INK, font=SANS)
rect(slide, 6.74, 4.55, 5.82, 1.49, PALE, RGBColor(225, 206, 209))
text(slide, 6.96, 4.66, 5.38, 0.25, "仍不能称为 clean", 12.2, BURGUNDY, True, SANS)
text(
    slide, 6.96, 4.97, 5.38, 0.93,
    f"• matched-clean 仍有 {commas(shared_topology)} 个 shared exact-topology signatures；签名相同不是结构相似性搜索。\n"
    "• exact / local-alignment / cluster non-hit 仅 NOT_DETECTED_UNKNOWN；更广 homology、预训练暴露、时间 / 标签 lineage 未闭环。\n"
    "• 不输出 ID 子集，不授权 sampling、training 或 route decision。",
    9.6, INK, font=SANS,
)
required[24].extend([commas(account["n_shared_clusters"]), commas(account["n_train_excluded"]), commas(account["n_validation_excluded"]), commas(retained_train), commas(retained_validation), commas(shared_topology)])
text(slide, 0.78, 6.13, 11.6, 0.20, "TR1c own-clean 仅为 supplied manifest 标签；本任务未打开 downstream sets。chemical modified-parent mapping 亦不可用。", 9.5, MUTED, font=SANS)
footer(slide, "evidence data_governance_v1 · exact / 80% / topology states kept distinct", "artifact /teams/HoulabData_1788502323/liucunyu/foldnar/data_governance_v1/audit_summary.json · leakage_matrix.json · mmseqs_evidence.json")

slide = new_slide(
    25,
    "25 · FOLDNAR PHASE 0 / BASELINE RECOVERY V2",
    "bpRNA-new 同序列 raw 基线与 BPfold 标签口径",
    "5,388 / 5,388 project coverage · strict exact-pair · zero-based inclusive · raw / 非去污，不作 SOTA 排名",
)
frozen = metrics["frozen_project_predictions"]
baseline_rows = []
for key, label in (("bpfold", "BPfold frozen replay"), ("ufold", "UFold MultiMolecule conversion")):
    m = frozen[key]["metrics"]
    row = (
        label,
        f"{commas(m['n_with_prediction'])} / {m['coverage']:.0%}",
        commas(m["tp"]), commas(m["fp"]), commas(m["fn"]),
        f"{m['micro_precision']:.5f}", f"{m['micro_recall']:.5f}", f"{m['micro_f1']:.5f}", f"{m['macro_f1']:.5f}",
    )
    baseline_rows.append(row)
    required[25].extend(row[1:])
table(
    slide, 0.78, 1.77, (2.65, 1.12, 0.98, 0.98, 0.98, 1.05, 1.05, 1.05, 1.92), 0.49,
    ("Prediction identity · project labels", "n / coverage", "TP", "FP", "FN", "micro P", "micro R", "micro F1", "macro F1"),
    baseline_rows, font_size=9.1, header_size=8.8,
)
official = metrics["bpfold_official_release"]
official_eval = official["official_reference_evaluation"]
all_official = official_eval["metrics_all_5401_official_reference"]
intersection = official_eval["metrics_5388_official_reference"]
sensitive = official["metrics_on_same_sequence_intersection"]
rect(slide, 0.78, 3.46, 5.72, 2.12, ROSE, LINE)
text(slide, 1.00, 3.58, 5.28, 0.27, "同一 official v0.2 release · official labels", 12.2, BURGUNDY, True, SANS)
text(slide, 1.00, 3.96, 5.28, 0.35, f"all {commas(all_official['n_dataset_sequences'])} · micro {all_official['micro_f1']:.5f} · macro {all_official['macro_f1']:.5f}", 15, GREEN, True)
text(
    slide, 1.00, 4.40, 5.28, 0.86,
    f"project-name intersection {commas(intersection['n_dataset_sequences'])}：micro {intersection['micro_f1']:.5f} · macro {intersection['macro_f1']:.5f}\n"
    f"all-{commas(all_official['n_dataset_sequences'])} macro 四舍五入为 {official_eval['readme_macro_f1_rounded']:.3f}，复现 official README 的 rounded value；13 条 official-only 不进入 project intersection。",
    10.1, INK, font=SANS,
)
rect(slide, 6.74, 3.46, 5.82, 2.12, PALE, RGBColor(205, 161, 170))
text(slide, 6.96, 3.58, 5.38, 0.27, "同一 release · project labels（仅敏感性审计）", 12.2, BURGUNDY, True, SANS)
text(slide, 6.96, 3.96, 5.38, 0.35, f"micro {sensitive['micro_f1']:.5f} · macro {sensitive['macro_f1']:.5f}", 16, BURGUNDY, True)
text(
    slide, 6.96, 4.36, 5.38, 0.48,
    f"仅 {commas(official_eval['n_project_label_sets_exactly_equal_official_reference'])} / {commas(official['n_project_sequences'])} label sets 完全相同；{commas(official_eval['n_project_label_sets_different_from_official_reference'])} 套不同。",
    10.1, INK, font=SANS,
)
text(slide, 6.96, 4.91, 5.38, 0.42, "0.96167 LABEL-SENSITIVE · EXCLUDED FROM SOTA CLAIMS", 11.5, BURGUNDY, True)
required[25].extend([
    f"{all_official['micro_f1']:.5f}", f"{all_official['macro_f1']:.5f}", f"{intersection['micro_f1']:.5f}", f"{intersection['macro_f1']:.5f}",
    f"{sensitive['micro_f1']:.5f}", f"{sensitive['macro_f1']:.5f}", "106", "5,282", "0.96167 LABEL-SENSITIVE · EXCLUDED FROM SOTA CLAIMS",
])
rect(slide, 0.78, 5.77, 11.78, 0.45, ROSE, LINE)
text(slide, 1.00, 5.84, 11.32, 0.28, "ContextFold · BLOCKED_NO_VERIFIED_PREDICTIONS · metrics / coverage = null（不是 0）；未用 summary-only 数值替代。", 10.2, BURGUNDY, True, SANS)
text(
    slide, 0.78, 6.26, 11.6, 0.18,
    f"UFold 为 MultiMolecule conversion，有效训练全集未重建；official release 与 frozen replay 有 {commas(official['n_pair_sets_different_from_frozen_project_replay'])} / {commas(official['n_project_sequences'])} pair sets 不同，不可混为同一预测身份。",
    9.1, MUTED, font=SANS,
)
required[25].extend(["ContextFold", "BLOCKED_NO_VERIFIED_PREDICTIONS", commas(official["n_pair_sets_different_from_frozen_project_replay"])])
footer(slide, "evidence baseline_recovery_v2 · frozen-prediction independent recomputation PASS", "artifact /teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2/reports/bprna_new_raw_metrics_verification_v2.json")

path_tmp = DECK.with_suffix(".round64_tmp.pptx")
prs.save(path_tmp)
path_tmp.replace(DECK)

reopened = Presentation(DECK)
assert len(reopened.slides) == 6
slide_text = []
bounds = []
for index, reopened_slide in enumerate(reopened.slides, start=20):
    texts = []
    for shape in reopened_slide.shapes:
        assert shape.left >= 0 and shape.top >= 0
        assert shape.left + shape.width <= reopened.slide_width + 2
        assert shape.top + shape.height <= reopened.slide_height + 2
        bounds.append({"slide_number": index, "shape": shape.name, "within_slide": True})
        if shape.has_text_frame:
            texts.append(shape.text_frame.text)
        if shape.has_table:
            for row in shape.table.rows:
                for cell in row.cells:
                    texts.append(cell.text)
    joined = "\n".join(texts)
    slide_text.append(joined)
    for needle in required.get(index, []):
        assert needle in joined, f"Slide {index} missing source-derived value: {needle}"

report = {
    "status": "PASS_BUILD_REOPEN_SOURCE_TEXT_AND_SHAPE_BOUNDS",
    "deck": str(DECK),
    "deck_sha256": sha256(DECK),
    "backup": str(BACKUP),
    "backup_sha256": sha256(BACKUP),
    "builder": str(Path(__file__).resolve()),
    "builder_sha256": sha256(Path(__file__).resolve()),
    "sources": {str(path): sha256(path) for path in (DIST, SUMMARY, INDEP, MMSEQS, LEAKAGE, METRICS)},
    "slide_count": len(reopened.slides),
    "added_slide_numbers": [23, 24, 25],
    "source_derived_text_checks": {str(key): len(value) for key, value in required.items()},
    "shape_bounds_checks": len(bounds),
    "governance_independent_checks": indep["n_checks"],
    "baseline_status": metrics["status"],
    "contextfold_status": metrics["contextfold"]["status"],
    "sota_exclusion": "0.96167 label-sensitive project-label result excluded from SOTA claims",
    "render_visual_check": "PENDING_SEPARATE_LIBREOFFICE_RENDER",
}
VERIFICATION.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
