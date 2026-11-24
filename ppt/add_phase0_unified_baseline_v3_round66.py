from pathlib import Path
import hashlib
import json
import shutil
import subprocess

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent
DECK = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
BACKUP = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pre_round66.pptx"
VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round66_build_verification.json"
ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar")
SPOT = ROOT / "baseline_recovery_v3_spot_rnaformer/reports/bprna_new_metrics_v3.json"
SPOT_DELIVERY = ROOT / "baseline_recovery_v3_spot_rnaformer/reports/delivery_verification_v3.json"
EFOLD = ROOT / "baseline_recovery_v3_efold_family/reports/bprna_new_efold_family_metrics_v3.json"
EFOLD_DELIVERY = ROOT / "baseline_recovery_v3_efold_family/reports/delivery_verification_v3.json"
FOUNDATION = ROOT / "baseline_recovery_v3_foundation/reports/foundation_bprna_new_metrics_verification_v3.json"
FOUNDATION_DELIVERY = ROOT / "baseline_recovery_v3_foundation/reports/delivery_verification_v3.json"
REPO = Path("/home/rna_jev_xens2_dualview_round56")
EXPECTED_PROJECT_COMMIT = "c3cfc7f8e30adad029abe9d3f2e9c8b5dce9d297"

BURGUNDY = RGBColor(122, 32, 44)
INK = RGBColor(30, 25, 28)
MUTED = RGBColor(105, 96, 101)
ROSE = RGBColor(247, 241, 239)
PALE = RGBColor(250, 245, 246)
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


spot = load(SPOT)
efold = load(EFOLD)
foundation = load(FOUNDATION)
deliveries = [load(path) for path in (SPOT_DELIVERY, EFOLD_DELIVERY, FOUNDATION_DELIVERY)]
assert all(item["status"] in {"PASS", "PASS_DELIVERY_VERIFICATION"} for item in deliveries)
project_commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
assert project_commit == EXPECTED_PROJECT_COMMIT
assert spot["status"] == "PASS_FULL_COVERAGE_INDEPENDENT_RECOMPUTATION"
assert efold["status"] == "PASS_WITH_EXPLICIT_PROVENANCE_GAPS"
assert foundation["status"] == "PASS_FIVE_FULL_PREDICTION_IDENTITIES_ERNIE_BLOCKED"
assert foundation["dataset"]["label_scope"] == "PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED"
assert spot["arms"]["spot_rna_registry_raw"]["metrics"]["n_empty_predictions"] == 5388
assert spot["arms"]["spot_rna_registry_raw"]["metrics"]["micro_precision"] is None
assert foundation["blocked"]["ernie_rna_author_bprna_new_protocol"]["metrics"] is None
assert efold["systems"]["e2efold"]["metrics"] is None and efold["systems"]["debfold"]["metrics"] is None

prs = Presentation(DECK)
if len(prs.slides) == 9 and BACKUP.exists():
    prs = Presentation(BACKUP)
assert len(prs.slides) == 8, "Expected the preserved eight-slide Phase 0 deck; refusing duplicate append"
if not BACKUP.exists():
    shutil.copy2(DECK, BACKUP)
prior_slide_xml = [slide.part.blob for slide in prs.slides]


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


def table(slide, x, y, widths, row_height, headers, rows):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_height * (len(rows) + 1)))
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
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.012)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c < 2 else PP_ALIGN.CENTER
            p.space_before = p.space_after = Pt(0)
            p.font.name = SANS if c == 1 else "Arial"
            p.font.size = Pt(10.3)
            p.font.bold = r == 0 or (c == 3 and r > 0)
            p.font.color.rgb = WHITE if r == 0 else (BURGUNDY if c == 3 else INK)
    return tbl


def footer(slide, evidence, artifact):
    line = rect(slide, 0.78, 6.48, 11.78, 0.012, RGBColor(212, 165, 154))
    line.line.fill.background()
    text(slide, 0.78, 6.57, 11.5, 0.17, evidence, 7.8, MUTED)
    text(slide, 0.78, 6.77, 11.5, 0.17, artifact, 7.8, MUTED)


slide = new_slide(
    28,
    "28 · FOLDNAR PHASE 0 / UNIFIED BASELINE RECOVERY V3",
    "bpRNA-new：可用协议的 raw full-coverage micro-F1",
    "5,388 / 5,388 coverage · exact zero-based inclusive pairs · PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED · 非 SOTA 排名",
)
entries = (
    ("eFold 0.1.3", "project replay · official source / weights", efold["systems"]["efold"]["metrics"]),
    ("SPOT-RNA", "MM conversion · separate postprocess", spot["arms"]["spot_rna_postprocessed"]["metrics"]),
    ("structRFM", "author bpRNA1m fine-tune", foundation["systems"]["structrfm_author_bprna1m_ssp"]["metrics"]),
    ("RNAformer 32M", "author bpRNA task head · frozen replay", spot["arms"]["rnaformer_bprna"]["metrics"]),
    ("RiNALMo", "author bpRNA fine-tune", foundation["systems"]["rinalmo_author_bprna_ft"]["metrics"]),
    ("NucleicBERT", "author secstr full fine-tune", foundation["systems"]["nucleicbert_author_secstr_full_ft"]["metrics"]),
)
expected = ("0.6495050", "0.5836013", "0.5437894", "0.4935795", "0.4489962", "0.1908860")
rows = []
for (name, protocol, metrics), expected_value in zip(entries, expected):
    assert metrics["coverage"] == 1.0 and metrics["n_with_prediction"] == 5388
    value = f"{metrics['micro_f1']:.7f}"
    assert value == expected_value
    rows.append((name, protocol, "100%", value))
table(slide, 0.78, 1.78, (2.20, 6.00, 1.45, 2.13), 0.405, ("System", "prediction / training identity", "coverage", "micro-F1"), rows)
spot_raw = spot["arms"]["spot_rna_registry_raw"]["metrics"]
rect(slide, 0.78, 4.76, 11.78, 0.43, ROSE, LINE)
text(slide, 1.00, 4.82, 11.32, 0.28, f"SPOT registry raw 是另一协议：{spot_raw['n_empty_predictions']:,} / {spot_raw['n_dataset_sequences']:,} all-empty · micro-F1 {spot_raw['micro_f1']:.1f} · precision null；不替代 postprocessed {spot['arms']['spot_rna_postprocessed']['metrics']['micro_f1']:.7f}。", 10.1, BURGUNDY, True, SANS)
text(slide, 0.78, 5.36, 11.6, 0.34, "Identity boundary · eFold 为 project replay；RiNALMo / NucleicBERT 为 author fine-tune；RNA-FM CRF 与 NucleicBERT TR1c head 是独立 project-head arms，不改标为 author release。", 9.7, MUTED, font=SANS)
rect(slide, 0.78, 5.86, 11.78, 0.39, PALE, RGBColor(205, 161, 170))
text(slide, 1.00, 5.91, 11.32, 0.27, "ERNIE-RNA / E2Efold / DEBFold · eligible full-project predictions unavailable · metrics = null（不是 0）；未用 summary / paper value 补齐。", 10.2, BURGUNDY, True, SANS)
footer(slide, f"source commit {project_commit} · three v3 delivery verifications PASS · independent exact-pair recomputation", "artifacts /teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v3_{spot_rnaformer,efold_family,foundation}/reports/")

path_tmp = DECK.with_suffix(".round66_tmp.pptx")
prs.save(path_tmp)
path_tmp.replace(DECK)
reopened = Presentation(DECK)
assert len(reopened.slides) == 9
assert [slide.part.blob for slide in list(reopened.slides)[:8]] == prior_slide_xml, "Prior slide XML changed"
new_slide_text = []
for index, reopened_slide in enumerate(reopened.slides, start=20):
    texts = []
    for shape in reopened_slide.shapes:
        assert shape.left >= 0 and shape.top >= 0
        assert shape.left + shape.width <= reopened.slide_width + 2
        assert shape.top + shape.height <= reopened.slide_height + 2
        if shape.has_text_frame:
            texts.append(shape.text_frame.text)
        if shape.has_table:
            texts.extend(cell.text for row in shape.table.rows for cell in row.cells)
    joined = "\n".join(texts)
    if index == 28:
        new_slide_text.append(joined)
        for value in expected:
            assert value in joined
        for phrase in ("all-empty", "metrics = null", "project replay", "author fine-tune", "project-head arms", "NON_DECONTAMINATED"):
            assert phrase in joined
        assert "0.96167" not in joined and "BPfold" not in joined and "GEC" not in joined

source_paths = (SPOT, SPOT_DELIVERY, EFOLD, EFOLD_DELIVERY, FOUNDATION, FOUNDATION_DELIVERY)
report = {
    "status": "PASS_BUILD_REOPEN_PRIOR_SLIDES_SOURCE_TEXT_AND_SHAPE_BOUNDS",
    "deck": str(DECK),
    "deck_sha256": sha256(DECK),
    "backup": str(BACKUP),
    "backup_sha256": sha256(BACKUP),
    "builder": str(Path(__file__).resolve()),
    "builder_sha256": sha256(Path(__file__).resolve()),
    "project_commit": project_commit,
    "sources": {str(path): sha256(path) for path in source_paths},
    "slide_count": len(reopened.slides),
    "added_slide_numbers": [28],
    "prior_slide_xml_preserved": 8,
    "verified_micro_f1": {row[0]: row[3] for row in rows},
    "unavailable_metrics": {"ERNIE-RNA": None, "E2Efold": None, "DEBFold": None},
    "spot_registry_raw": {"empty_predictions": spot_raw["n_empty_predictions"], "micro_f1": spot_raw["micro_f1"], "micro_precision": None},
    "shape_bounds_checks": sum(len(slide.shapes) for slide in reopened.slides),
    "render_visual_check": "PENDING_SEPARATE_LIBREOFFICE_RENDER",
}
VERIFICATION.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
