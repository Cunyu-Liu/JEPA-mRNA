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
BACKUP = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pre_round65.pptx"
VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round65_build_verification.json"
EXPOSURE = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/exposure_temporal_audit_v1")
LONG = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/long_sequence_audit_v1")
REPO = Path("/home/rna_jev_xens2_dualview_round56")
REGISTRY = REPO / "manifests/foldnar_v1"
EXP_SUMMARY = EXPOSURE / "audit_summary.json"
EXP_TEMPORAL = EXPOSURE / "temporal_label_lineage.json"
EXP_INDEP = EXPOSURE / "independent_verification.json"
LONG_REPORT = LONG / "audit_report.json"
LONG_DELIVERY = LONG / "delivery_verification_v1.json"
LONG_SMOKE = LONG / "cpu_smoke_v2/summary.json"
LONG_ASSETS = LONG / "long_chain_records_complete.jsonl"
RECONCILIATION = LONG / "registry_reconciliation_v1/reconciliation_report.json"
REGISTRY_DELIVERY = LONG / "registry_reconciliation_v1/delivery_verification.json"
DATA_REGISTRY = REGISTRY / "data_registry.json"
BASELINE_REGISTRY = REGISTRY / "baseline_registry.json"
EXPECTED_PROJECT_COMMIT = "1bb28399eedb8c528b8abb96aabaaf3b15407df3"

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


def load(path):
    return json.loads(path.read_text())


def commas(value):
    return f"{value:,}"


exp = load(EXP_SUMMARY)
temporal = load(EXP_TEMPORAL)
exp_indep = load(EXP_INDEP)
long_report = load(LONG_REPORT)
long_delivery = load(LONG_DELIVERY)
smoke = load(LONG_SMOKE)
reconciliation = load(RECONCILIATION)
registry_delivery = load(REGISTRY_DELIVERY)
data_registry = load(DATA_REGISTRY)
baseline_registry = load(BASELINE_REGISTRY)
project_commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
assert project_commit == EXPECTED_PROJECT_COMMIT
assert exp_indep["status"] == "PASS" and exp_indep["n_checks"] == 17 and all(exp_indep["checks"].values())
assert exp["full_historical_pretraining_state"] == "UNCHECKED"
assert exp["historical_nt"]["state"] == "UNCHECKED"
assert long_report["execution_policy"]["gec_exact_cubic_long_chain_support"] == "NOT_ESTABLISHED"
assert long_report["linearpartition"]["existing_prediction_count"] == 0
assert long_delivery["long_chain_inference_executed"] is False
assert smoke["status"] == "PASS_OFFICIAL_SHORT_CPU_PROTOCOL_SMOKE" and smoke["long_chain_inference_executed"] is False
assert reconciliation["status"] == "PASS_NARROW_LONG_SEQUENCE_REGISTRY_RECONCILIATION"
assert registry_delivery["status"] == "PASS_REGISTRY_RECONCILIATION_DELIVERY"
assert sha256(DATA_REGISTRY) == reconciliation["data_registry"]["after_sha256"]
assert sha256(BASELINE_REGISTRY) == reconciliation["baseline_registry"]["after_sha256"]
assert sha256(LONG_REPORT) == "e63f6e55ae5d580f334bd747ded805b7d833ecf344747af0b8c86a06cd2c2e0f"


def registry_row(registry, key, value):
    return next(row for row in registry["rows"] if row[key] == value)


long_7224 = registry_row(data_registry, "dataset_id", "longchain_7224")
long_19000 = registry_row(data_registry, "dataset_id", "longchain_19000")
linearpartition = registry_row(baseline_registry, "baseline_id", "linearpartition")
assert long_7224["n_chains_at_contract_length"] == 2 and long_19000["n_chains_at_contract_length"] == 1
assert linearpartition["official_source"]["commit"] == long_report["linearpartition"]["upstream"]["commit"]
assert linearpartition["inference"]["evaluation_status"] == "PLANNED_ONLY_NO_PREDICTIONS"
assert not linearpartition["prediction_ids"]

prs = Presentation(DECK)
if len(prs.slides) == 8 and BACKUP.exists():
    prs = Presentation(BACKUP)
assert len(prs.slides) == 6, "Expected the preserved six-slide Phase 0 deck; refusing duplicate append"
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


def table(slide, x, y, widths, row_height, headers, rows, font_size=10.5):
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
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.012)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c in (0, len(values) - 1) else PP_ALIGN.CENTER
            p.space_before = p.space_after = Pt(0)
            p.font.name = SANS if c in (0, len(values) - 1) else "Arial"
            p.font.size = Pt(font_size)
            p.font.bold = r == 0
            p.font.color.rgb = WHITE if r == 0 else INK
    return tbl


def footer(slide, evidence, artifact):
    line = rect(slide, 0.78, 6.48, 11.78, 0.012, RGBColor(212, 165, 154))
    line.line.fill.background()
    text(slide, 0.78, 6.57, 11.5, 0.17, evidence, 7.8, MUTED)
    text(slide, 0.78, 6.77, 11.5, 0.17, artifact, 7.8, MUTED)


required = {26: [], 27: []}
slide = new_slide(
    26,
    "26 · FOLDNAR PHASE 0 / EXPOSURE + TEMPORAL AUDIT",
    "已知源暴露已量化，历史训练暴露仍未闭环",
    "TR1c / CHANRG development only · RNAcentral 22 · Rfam 14.9 · Ensembl 109 / Bacteria 56 · normalized full-sequence EXACT",
)
for x, dataset, label in ((0.78, "tr1c", "TR1c · exact known-source membership"), (6.74, "chanrg_validation", "CHANRG Validation · exact known-source membership")):
    view = exp["views"][dataset]["raw"]
    exact = view["state_counts"]["EXACT"]
    total = view["n_records"]
    rect(slide, x, 1.86, 5.72 if x < 1 else 5.82, 1.52, ROSE, LINE)
    text(slide, x + 0.22, 2.00, 5.28, 0.25, label, 11.5, BURGUNDY, True, SANS)
    metric = f"{commas(exact)} / {commas(total)}"
    text(slide, x + 0.22, 2.35, 4.00, 0.48, metric, 25, GREEN, True)
    text(slide, x + 4.15, 2.43, 1.25, 0.30, f"{100 * exact / total:.2f}%", 15, GREEN, True, align=PP_ALIGN.RIGHT)
    required[26].extend([metric, f"{100 * exact / total:.2f}%"])
text(slide, 0.78, 3.55, 11.6, 0.25, "EXACT 仅指命名版本源中的序列成员关系；不是历史 RiNALMo 训练样本包含的证明。CHANRG homology 仍 UNCHECKED。", 10.5, MUTED, font=SANS)
rect(slide, 0.78, 3.94, 11.78, 0.94, PALE, RGBColor(205, 161, 170))
text(slide, 1.00, 4.05, 11.32, 0.26, "FULL HISTORICAL EXPOSURE · UNCHECKED", 13, BURGUNDY, True)
text(slide, 1.00, 4.39, 11.32, 0.30, "Historical nt snapshot 未固定且不可用；即使 known-source EXACT 为阳性，也不能推断完整预训练暴露或 clean 边界。", 10.5, INK, font=SANS)
all_records = sum(item["n_records"] for item in temporal["datasets"].values())
date_rows = sum(item["field_coverage"]["source_date"] for item in temporal["datasets"].values())
pdb_rows = sum(item["field_coverage"]["mmcif_metadata"] for item in temporal["datasets"].values())
for x, heading, body in (
    (0.78, "逐记录日期", f"source date {commas(date_rows)} / {commas(all_records)}；文件 mtime 与源 release timestamp 不是 label date。"),
    (4.80, "PDB / mmCIF lineage", f"metadata {commas(pdb_rows)} / {commas(all_records)}；modified-parent 与 residue observation 均 UNCHECKED。"),
    (8.82, "标签层级不升级", "TR1c 保持 DATABASE；CHANRG 保持 COMPARATIVE_INFERRED。逐记录实验 / comparative lineage 未解决。"),
):
    rect(slide, x, 5.12, 3.74, 1.02, PALE, RGBColor(225, 206, 209))
    text(slide, x + 0.18, 5.23, 3.38, 0.23, heading, 11.5, BURGUNDY, True, SANS)
    text(slide, x + 0.18, 5.52, 3.38, 0.48, body, 9.6, INK, font=SANS)
required[26].extend([f"{commas(date_rows)} / {commas(all_records)}", f"{commas(pdb_rows)} / {commas(all_records)}", "UNCHECKED", "DATABASE", "COMPARATIVE_INFERRED"])
footer(slide, "evidence exposure_temporal_audit_v1 · independent raw/source replay 17 / 17 PASS", "artifact /teams/HoulabData_1788502323/liucunyu/foldnar/exposure_temporal_audit_v1/audit_summary.json · temporal_label_lineage.json")

slide = new_slide(
    27,
    "27 · FOLDNAR PHASE 0 / LONG-SEQUENCE READINESS",
    "长序列资产与官方协议已冻结；预测仍为空",
    f"reconciled registries @ {project_commit[:12]} · full-length assets only · protocol / provenance verification，不是性能结果",
)
asset_rows = []
for target in long_report["targets"]:
    row = (target["name"], commas(target["length"]), commas(target["evaluation_plan"]["reference_helix_count"]), target["evaluation_plan"]["status"].replace("BLOCKED_NO_LONG_CHAIN_PREDICTIONS", "FULL-LENGTH ASSET VERIFIED"))
    asset_rows.append(row)
    required[27].extend(row[:2])
table(slide, 0.78, 1.79, (2.25, 1.60, 2.10, 5.83), 0.44, ("Chain", "length (nt)", "reference helices", "asset / prediction state"), asset_rows, font_size=10.0)
text(slide, 0.78, 3.66, 11.6, 0.22, "2 × 7,224 nt + 1 × 19,000 nt · 全长序列与标签身份已验证；无 truncation / nucleotide deletion。", 10.2, MUTED, font=SANS)
upstream = long_report["linearpartition"]["upstream"]
rect(slide, 0.78, 4.01, 7.36, 1.33, ROSE, LINE)
text(slide, 1.00, 4.12, 6.92, 0.25, "LinearPartition-C / V · official source + frozen protocol", 12.2, BURGUNDY, True, SANS)
text(slide, 1.00, 4.45, 6.92, 0.24, f"official commit {upstream['commit']}", 10.5, INK, True)
text(slide, 1.00, 4.76, 6.92, 0.40, "C：embedded feature weights；V：-V + ViennaRNA parameters · beam 100 · BPP cutoff 0.0 · dangles 2 · no sharpturn · optional MEA γ=3 · full-length FASTA", 9.8, INK, font=SANS)
rect(slide, 8.38, 4.01, 4.18, 1.33, PALE, RGBColor(205, 161, 170))
text(slide, 8.60, 4.15, 3.74, 0.30, "APPROXIMATE_SPARSE", 15, BURGUNDY, True)
text(slide, 8.60, 4.55, 3.74, 0.52, f"finite beam / sparse BPP ≠ exact global cubic DP\n仅 {smoke['rows'][0]['sequence_length']} nt official short CPU smoke PASS；不是长链结果。", 9.8, INK, font=SANS)
rect(slide, 0.78, 5.57, 11.78, 0.61, ROSE, LINE)
text(slide, 1.00, 5.65, 11.32, 0.40, "NO LONG-CHAIN PREDICTIONS / METRICS · exact cubic GEC support NOT ESTABLISHED · memory / FLOPs / latency NOT MEASURED", 11.0, BURGUNDY, True, SANS)
required[27].extend([upstream["commit"], "APPROXIMATE_SPARSE", "NO LONG-CHAIN PREDICTIONS / METRICS", "NOT ESTABLISHED", str(smoke["rows"][0]["sequence_length"])])
footer(slide, f"source commit {project_commit} · registry reconciliation PASS · prediction registry unchanged", "artifact /teams/HoulabData_1788502323/liucunyu/foldnar/long_sequence_audit_v1/audit_report.json · registry_reconciliation_v1/")

path_tmp = DECK.with_suffix(".round65_tmp.pptx")
prs.save(path_tmp)
path_tmp.replace(DECK)
reopened = Presentation(DECK)
assert len(reopened.slides) == 8
assert [slide.part.blob for slide in list(reopened.slides)[:6]] == prior_slide_xml, "Prior slide XML changed"
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
    for needle in required.get(index, []):
        assert needle in joined, f"Slide {index} missing source-derived value: {needle}"

source_paths = (EXP_SUMMARY, EXP_TEMPORAL, EXP_INDEP, LONG_REPORT, LONG_DELIVERY, LONG_SMOKE, LONG_ASSETS, RECONCILIATION, REGISTRY_DELIVERY, DATA_REGISTRY, BASELINE_REGISTRY)
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
    "added_slide_numbers": [26, 27],
    "prior_slide_xml_preserved": 6,
    "source_derived_text_checks": {str(key): len(value) for key, value in required.items()},
    "shape_bounds_checks": sum(len(slide.shapes) for slide in reopened.slides),
    "exposure_independent_checks": exp_indep["n_checks"],
    "full_historical_pretraining_state": exp["full_historical_pretraining_state"],
    "long_chain_predictions": long_report["linearpartition"]["existing_prediction_count"],
    "gec_exact_cubic_long_chain_support": long_report["execution_policy"]["gec_exact_cubic_long_chain_support"],
    "render_visual_check": "PENDING_SEPARATE_LIBREOFFICE_RENDER",
}
VERIFICATION.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
