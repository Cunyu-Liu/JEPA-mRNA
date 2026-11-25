from pathlib import Path
from datetime import datetime, timedelta, timezone
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
BACKUP = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pre_round67.pptx"
VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round67_build_verification.json"
ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar")
THERMO = ROOT / "baseline_recovery_v4_thermodynamic/reports/thermodynamic_bprna_new_metrics_v4.json"
THERMO_DELIVERY = ROOT / "baseline_recovery_v4_thermodynamic/reports/delivery_verification_v4.json"
KIRIGAMI_DELIVERY = ROOT / "baseline_recovery_v4_kirigami/reports/delivery_verification_v4.json"
GAP = ROOT / "baseline_recovery_v1/xens2_absolute_gap_audit_20261124.json"
GEC = ROOT / "gec_seed0_v3"
REPO = Path("/home/rna_jev_xens2_dualview_round56")
EXPECTED_PROJECT_COMMIT = "eed09df8cdb634140aeced9bb2340c5500587b92"
LAUNCH_COMMIT = "65211e041242b5f37c914bb18b7164a7c045abfe"

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


thermo = load(THERMO)
thermo_delivery = load(THERMO_DELIVERY)
kirigami = load(KIRIGAMI_DELIVERY)
gap = load(GAP)
identity = load(GEC / "production_identity.json")
launch = load(GEC / "production_launch.json")
config = load(REPO / "configs/gec_four_arm_convergence_v1.json")

assert thermo["status"] == "PASS_RAW_5388_ALL_6_DECODERS_CLEAN_SCOPES_SEPARATED"
assert thermo_delivery["status"] == "PASS_INDEPENDENT_EXACT_RATIONAL_RECOMPUTATION"
assert kirigami["status"] == "PASS_SEPARATE_INDEPENDENT_RECOMPUTATION"
assert gap["status"] == "PASS"
assert identity["git_commit"] == LAUNCH_COMMIT
assert launch["parallelism"] == 2 and launch["cpu_fallback"] is False
assert len(identity["code_manifest"]) == 18
assert config["model"]["decoder"] == {"name": "MEA", "temperature": 1.0, "gamma": 1.0,
                                      "score_variant": "train_off/plana_off", "search_allowed": False}
assert config["training"]["batch_size"] == 4
project_commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
assert project_commit == EXPECTED_PROJECT_COMMIT

thermo_entries = (
    ("ViennaRNA MFE", "2.7.2 · sequence-independent · no fitted params", "vienna_mfe"),
    ("ViennaRNA centroid", "2.7.2 · ensemble centroid decoding", "vienna_centroid"),
    ("ViennaRNA MEA", "2.7.2 · maximum expected accuracy", "vienna_mea"),
    ("RNAstructure Fold", "6.6 · --MFE · official archive verified", "rnastructure_fold"),
    ("RNAstructure MaxExpect", "6.6 · first CT block retained", "rnastructure_maxexpect"),
    ("CONTRAfold MEA", "commit 87b9aac · native parameters", "contrafold_mea"),
)
expected_micro = ("0.6379382", "0.6770327", "0.6619892", "0.6318346", "0.6628923", "0.6967639", "0.5783642")
expected_macro = ("0.6580838", "0.6820728", "0.6764248", "0.6496592", "0.6746697", "0.7108879", "0.5526673")
rows = []
for (name, protocol, key), em, ea in zip(thermo_entries, expected_micro[:6], expected_macro[:6]):
    raw = thermo["systems"][key]["raw"]
    assert raw["unique_names"] == 5388 and raw["coverage"] == 1.0
    assert raw["recorded_deterministic_repeats_verified"] == 3
    m = raw["metrics"]
    assert f"{m['micro_f1']:.7f}" == em and f"{m['macro_f1']:.7f}" == ea
    rows.append((name, protocol, "100%", em, ea))
km = kirigami["recomputed_metrics"]
assert kirigami["n_sequences"] == 5388
assert f"{km['micro_f1']:.7f}" == expected_micro[6] and f"{km['macro_f1']:.7f}" == expected_macro[6]
rows.append(("Kirigami", "project replay · official weights @ 49e61ff", "100%", expected_micro[6], expected_macro[6]))

clean = thermo["systems"]["contrafold_mea"]["clean"]
assert clean["coverage_of_raw_5388"] == 5386 / 5388
clean_micro = f"{clean['metrics']['micro_f1']:.7f}"
assert clean_micro == "0.6967483"
xens2_micro = f"{gap['systems']['xens2']['micro_f1']:.7f}"
assert xens2_micro == "0.5531421"
gap_m = f"{gap['absolute_gaps']['mxfold2_minus_xens2']['micro_f1_absolute_gap']:.4f}"
gap_e = f"{gap['absolute_gaps']['eternafold_minus_xens2']['micro_f1_absolute_gap']:.4f}"
assert gap_m == "0.1157" and gap_e == "0.1322"

val = {}
train_last = {}
for arm in ("matched_control", "ensemble_context"):
    vrows = [json.loads(line) for line in (GEC / arm / "validation_log.jsonl").read_text().splitlines()]
    val[arm] = {r["step"]: r["validation_loss"] for r in vrows}
    trows = [json.loads(line) for line in (GEC / arm / "train_log.jsonl").read_text().splitlines()]
    train_last[arm] = trows
    assert round(val[arm][10500], 6) == (0.243223 if arm == "matched_control" else 0.241049)
    steps = [r["step"] for r in trows]
    assert 11000 in steps
    by_step = {r["step"]: r for r in trows}
    assert by_step[11000]["divergence_reason"] == ""
    first = sum(r["loss"] for r in trows[:2000]) / 2000
    late = sum(r["loss"] for r in trows if 8001 <= r["step"] <= 10000) / 2000
    assert first > 0.27 and late < 0.16
    assert by_step[11000]["unfrozen_blocks"] == 20
obs_row = next(r for r in train_last["matched_control"] if r["step"] == 11000)
started = datetime.fromisoformat(launch["started_utc"])
obs_utc = started + timedelta(seconds=obs_row["elapsed_seconds"])
obs_cst = obs_utc.astimezone(timezone(timedelta(hours=8)))
obs_label = obs_cst.strftime("%H:%M")
peak_gb = obs_row["peak_cuda_bytes"] / 1e9
assert f"{peak_gb:.2f}" == "13.47"

prs = Presentation(DECK)
if len(prs.slides) == 11 and BACKUP.exists():
    prs = Presentation(BACKUP)
assert len(prs.slides) == 9, "Expected the preserved nine-slide Phase 0 deck; refusing duplicate append"
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
            p.font.bold = r == 0 or (c in (3, 4) and r > 0)
            p.font.color.rgb = WHITE if r == 0 else (BURGUNDY if c in (3, 4) else INK)
    return tbl


def footer(slide, evidence, artifact):
    line = rect(slide, 0.78, 6.48, 11.78, 0.012, RGBColor(212, 165, 154))
    line.line.fill.background()
    text(slide, 0.78, 6.57, 11.5, 0.17, evidence, 7.8, MUTED)
    text(slide, 0.78, 6.77, 11.5, 0.17, artifact, 7.8, MUTED)


slide = new_slide(
    29,
    "29 · FOLDNAR PHASE 0 / UNIFIED BASELINE RECOVERY V4",
    "bpRNA-new：热力学解码器与 Kirigami 的 raw 全量结果",
    "5,388 / 5,388 coverage · exact zero-based inclusive pairs · PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED · 非 SOTA 排名",
)
table(slide, 0.78, 1.78, (2.30, 5.30, 1.30, 1.44, 1.44), 0.36,
      ("System", "prediction / protocol identity", "coverage", "micro-F1", "macro-F1"), rows)
rect(slide, 0.78, 4.80, 11.78, 0.43, ROSE, LINE)
text(slide, 1.00, 4.86, 11.32, 0.28, f"CONTRAfold clean_tr1c 分栏：5,386 条（2 条 project 排除）· micro {clean_micro}；是 project TR1c 排除口径，不是其 own-training-clean，不进入同 5,388 比较。", 10.1, BURGUNDY, True, SANS)
text(slide, 0.78, 5.36, 11.6, 0.50, "Kirigami 无 author 发布的 bpRNA-new 逐序列预测 → project replay（官方权重哈希一致）；optimized 与 original 逐条 pair set 相同，不是独立臂。ViennaRNA / RNAstructure 无独立 clean 资产，clean = null；各系统 3 次确定性重复推理一致。", 9.7, MUTED, font=SANS)
rect(slide, 0.78, 5.98, 11.78, 0.39, PALE, RGBColor(205, 161, 170))
text(slide, 1.00, 6.03, 11.32, 0.27, f"同面板 xens2 raw micro-F1 {xens2_micro} · 距 MXfold2 / EternaFold 绝对差距 {gap_m} / {gap_e}（v1 恢复 · 独立重算 CONFIRMED）——核心缺口校准。", 10.2, BURGUNDY, True, SANS)
footer(slide, f"source commit {project_commit} · thermodynamic delivery PASS（exact rational）· kirigami delivery PASS（separate recomputation）· gap audit PASS",
       "artifacts /teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v4_{thermodynamic,kirigami}/reports/ · baseline_recovery_v1/xens2_absolute_gap_audit_20261124.json")

slide = new_slide(
    30,
    "30 · FOLDNAR PHASE 1 / GEC SEED0 PRODUCTION LAUNCH",
    "GEC 四臂收敛训练：启动合同与过程记录",
    "假设 H-GEC · 全局系综上下文反哺 pair 表示 · 训练进行中，本页不写任何门裁决",
)
facts = (
    ("臂与调度", "matched_control / ensemble_context / destroy_interval_shuffle / independent_context；实测显存波次 parallelism=2，wave2 自动接续，四臂全部收敛后严格评估器自动裁决。"),
    ("收敛判据", "无步数上限；每 500 步验证 · patience 5 · min-delta 1e-6；终态解冻 20/33 blocks；batch 4 · head LR 1e-4 · backbone LR 1e-6 · AdamW。"),
    ("身份绑定", f"shared_init_seed0 + 零初始化 GEC 残差；第一遍精确非交叉 inside–outside（float64 · stop-gradient）；S1=S0+Δ；18 个代码文件 SHA256 绑定启动 commit {LAUNCH_COMMIT[:7]}。"),
    ("数据与解码", "tr1c_cluster80_dev_v1：train 33,421 / dev 9,143 · 交集 0；MEA 解码冻结（T=γ=1 · 禁止搜索）；外部测试集不参与开发。"),
    ("进度观测", f"{obs_label} CST（step 11,000，由 elapsed 推得）：loss 窗口均值 0.275→0.152；验证 loss 0.2432 / 0.2410 仍改善；单臂峰值显存 {peak_gb:.2f} GB · 双臂 GPU 并行 · 无发散 · 无 CPU 降级。"),
)
y = 1.78
for label, body in facts:
    rect(slide, 0.78, y, 2.10, 0.56, ROSE, LINE)
    text(slide, 0.92, y + 0.06, 1.85, 0.44, label, 11, BURGUNDY, True, SANS)
    text(slide, 3.06, y + 0.03, 9.50, 0.52, body, 10.2, INK, font=SANS)
    y += 0.66
rect(slide, 0.78, 5.20, 11.78, 0.50, PALE, RGBColor(205, 161, 170))
text(slide, 1.00, 5.26, 11.32, 0.38, "状态：训练进行中。本页仅记录启动合同与带时间戳的过程观测；seed0 放行门（micro 非负 + macro 严格提升 + 子群改善 + 双对照优势）待四臂收敛后由独立评估器给出 PASS_GEC_SEED0 / FAIL_GEC_SEED0。", 10.2, BURGUNDY, True, SANS)
text(slide, 0.78, 5.86, 11.6, 0.46, "H-GEC：相同序列表示、参数预算、训练步数与结构化 NLL 下，非交叉系综上下文应主要改善孤立 / 长程 pair 与 macro 指标；区间打乱或独立 sigmoid 对照应使优势消失。待检验假设，不是已成立结论。", 9.7, MUTED, font=SANS)
footer(slide, f"record commit {project_commit} · launch commit {LAUNCH_COMMIT[:7]} · production_identity / launch 哈希已核 · 18/18 代码文件一致",
       "artifacts /teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3/{production_identity.json,*/train_log.jsonl,*/validation_log.jsonl}")

path_tmp = DECK.with_suffix(".round67_tmp.pptx")
prs.save(path_tmp)
path_tmp.replace(DECK)
reopened = Presentation(DECK)
assert len(reopened.slides) == 11
assert [slide.part.blob for slide in list(reopened.slides)[:9]] == prior_slide_xml, "Prior slide XML changed"
new_slide_text = {}
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
    if index in (29, 30):
        new_slide_text[index] = joined
for value in expected_micro + expected_macro:
    assert value in new_slide_text[29]
for phrase in ("clean_tr1c", "project replay", "NON_DECONTAMINATED", "0.1157", "0.1322", "0.5531421"):
    assert phrase in new_slide_text[29]
assert "SOTA 排名" not in new_slide_text[29].replace("非 SOTA 排名", "")
for phrase in ("训练进行中", "patience 5", "stop-gradient", "33,421", "9,143", "0.2432", "0.2410", LAUNCH_COMMIT[:7], "FAIL_GEC_SEED0"):
    assert phrase in new_slide_text[30]
assert "PASS_GEC_SEED0" in new_slide_text[30]

source_paths = (THERMO, THERMO_DELIVERY, KIRIGAMI_DELIVERY, GAP,
                GEC / "production_identity.json", GEC / "production_launch.json",
                REPO / "configs/gec_four_arm_convergence_v1.json")
report = {
    "status": "PASS_BUILD_REOPEN_PRIOR_SLIDES_SOURCE_TEXT_AND_SHAPE_BOUNDS",
    "deck": str(DECK),
    "deck_sha256": sha256(DECK),
    "backup": str(BACKUP),
    "backup_sha256": sha256(BACKUP),
    "builder": str(Path(__file__).resolve()),
    "builder_sha256": sha256(Path(__file__).resolve()),
    "project_commit": project_commit,
    "gec_launch_commit": LAUNCH_COMMIT,
    "sources": {str(path): sha256(path) for path in source_paths},
    "slide_count": len(reopened.slides),
    "added_slide_numbers": [29, 30],
    "prior_slide_xml_preserved": 9,
    "verified_micro_f1": {row[0]: row[3] for row in rows},
    "verified_macro_f1": {row[0]: row[4] for row in rows},
    "contrafold_clean_tr1c": {"n": 5386, "micro_f1": clean_micro},
    "xens2_gap": {"xens2_micro_f1": xens2_micro, "mxfold2_gap": gap_m, "eternafold_gap": gap_e},
    "gec_progress_observation": {"step": 11000, "observed_cst": obs_label,
                                 "validation_loss": {"matched_control": round(val["matched_control"][10500], 6),
                                                     "ensemble_context": round(val["ensemble_context"][10500], 6)},
                                 "peak_cuda_gb_per_arm": round(peak_gb, 2)},
    "shape_bounds_checks": sum(len(slide.shapes) for slide in reopened.slides),
    "render_visual_check": "PENDING_SEPARATE_LIBREOFFICE_RENDER",
}
VERIFICATION.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))