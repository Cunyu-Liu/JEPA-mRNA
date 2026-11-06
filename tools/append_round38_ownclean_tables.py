#!/usr/bin/env python3
import copy
import hashlib
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ppt/.round39_work/source_data/appendix_leakage/round39_own_clean_leaderboard.json"
DECK = ROOT / "ppt/RNA-JEV_论文逻辑重构_20261031_导师讨论_round38.pptx"
BACKUP = ROOT / "ppt/.round39_work/round38_before_ownclean_tables.pptx"
QA = ROOT / "ppt/.round39_work/round38_ownclean_tables_qa.json"
EXPECTED_SHA256 = "9da97573e4049122dc63f5e2323685c571dbb5e51e5e677cc7cdaaef93f7b326"
SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]
SPLIT_LABELS = {"bprna_ts0": "bpRNA TS0", "bprna_new": "bpRNA-new", "ref_pdb_ts1": "PDB TS1",
                "ref_pdb_ts2": "PDB TS2", "ref_pdb_ts3": "PDB TS3", "ref_pdb_ts_hard": "PDB TS-hard",
                "archiveii_embok_clean": "ArchiveII", "testsetb": "TestSetB"}
SYSTEM_LABELS = {"eternafold": "EternaFold", "bpfold": "BPfold", "mxfold2": "MXfold2",
                 "rnaformer_interfam": "RNAformer-inter", "rnaformer_bprna": "RNAformer-bprna",
                 "rnaformer_intrafam": "RNAformer-intra", "rnaformer_biophysical": "RNAformer-biophysical",
                 "rinalmo_ft": "RiNALMo-ft", "spotrna": "SPOT-RNA", "kirigami_fast": "Kirigami",
                 "nucleicbert_ft": "NucleicBERT-ft", "nucleicbert_frozen": "NucleicBERT-frozen", "xens2": "xens2"}
FONT = "微软雅黑"
BLUE = "0F4D92"
TEAL = "0E7C86"
INK = "17243A"
MUTED = "667085"
PALE = "F5F7FA"
PALE_TEAL = "EAF4F4"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(path=SOURCE):
    data = json.loads(path.read_text())
    assert data["status"] == "PASS_COMPLETE_OWN_CLEAN_LEADERBOARD"
    assert data["complete_cell_count"] == data["required_cell_count"] == 104
    assert len(data["systems"]) == 13 and sha256(path) == EXPECTED_SHA256
    assert all(len(cells) == 8 for cells in data["systems"].values())
    return data


def cell_values(system, split, cell):
    if cell["status"] == "EMPTY_AFTER_DELEAK":
        return {"system": SYSTEM_LABELS[system], "split": split, "clean_n": "0",
                "f1": "—", "tp": "—", "fp": "—", "fn": "—",
                "removed_exact": str(cell["removed_exact"]),
                "removed_homology": str(cell["removed_homology_only"]), "status": cell["status"]}
    assert cell["status"] == "PASS"
    assert cell["raw_n"] == cell["clean_n"] + cell["removed_exact"] + cell["removed_homology_only"]
    assert len(cell["f1_values"]) == 3
    return {"system": SYSTEM_LABELS[system], "split": split, "clean_n": f"{cell['clean_n']:,}",
            "f1": f"{cell['f1_mean']:.4f}", "tp": f"{cell['tp']:,}", "fp": f"{cell['fp']:,}",
            "fn": f"{cell['fn']:,}", "removed_exact": str(cell["removed_exact"]),
            "removed_homology": str(cell["removed_homology_only"]), "status": cell["status"]}


def all_values(data):
    values = []
    for split in SPLITS:
        for system, cells in data["systems"].items():
            values.append(cell_values(system, split, cells[split]))
    assert len(values) == 104
    assert sum(value["status"] == "EMPTY_AFTER_DELEAK" for value in values) == 2
    return values


def style_run(run, size, color=INK, bold=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        element = rpr.find(qn(tag))
        if element is None:
            element = rpr.makeelement(qn(tag), {})
            rpr.append(element)
        element.set("typeface", FONT)


def add_text(slide, text, x, y, width, height, size, color=INK, bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.02)
    frame.margin_top = frame.margin_bottom = Inches(0.01)
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = 1.5
    paragraph.space_before = Pt(0)
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = text
    style_run(run, size, color, bold)
    return shape


def set_cell(cell, text, size, color=INK, bold=False, align=PP_ALIGN.CENTER, fill=None):
    frame = cell.text_frame
    frame.clear()
    frame.word_wrap = False
    frame.margin_left = frame.margin_right = Inches(0.035)
    frame.margin_top = frame.margin_bottom = Inches(0.005)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = 1.5
    paragraph.space_before = Pt(0)
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = text
    style_run(run, size, color, bold)
    if fill:
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor.from_string(fill)


def add_split_slide(prs, data, split, page_number):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)
    raw_n = next(iter(data["systems"].values()))[split]["raw_n"]
    add_text(slide, "附录 D / own-clean 主榜", 0.52, 0.24, 6.2, 0.26, 9, TEAL, True)
    add_text(slide, f"{SPLIT_LABELS[split]}｜13 系统独立 clean 结果", 0.52, 0.57, 12.1, 0.55, 18, INK, True)
    add_text(slide, f"raw n={raw_n:,}；严格 micro-F1（三次推理均值，SD=0）；clean n 随各系统训练语料独立确定。",
             0.52, 1.12, 12.1, 0.30, 10, MUTED)
    rows, columns = 14, 8
    table_shape = slide.shapes.add_table(rows, columns, Inches(0.52), Inches(1.65), Inches(12.29), Inches(4.83))
    table = table_shape.table
    widths = [2.80, 1.00, 1.35, 1.40, 1.40, 1.40, 1.45, 1.49]
    for index, width in enumerate(widths):
        table.columns[index].width = Inches(width)
    table.rows[0].height = Inches(0.34)
    for index in range(1, rows):
        table.rows[index].height = Inches(0.345)
    headers = ["系统", "clean n", "F1", "TP", "FP", "FN", "删除 exact", "删除同源"]
    for index, header in enumerate(headers):
        set_cell(table.cell(0, index), header, 9.2, "FFFFFF", True, fill=BLUE)
    for row_index, (system, cells) in enumerate(data["systems"].items(), 1):
        value = cell_values(system, split, cells[split])
        fields = [value["system"], value["clean_n"], value["f1"], value["tp"], value["fp"],
                  value["fn"], value["removed_exact"], value["removed_homology"]]
        fill = PALE_TEAL if system == "xens2" else (PALE if row_index % 2 == 0 else "FFFFFF")
        for column_index, field in enumerate(fields):
            set_cell(table.cell(row_index, column_index), field, 8.6, INK, system == "xens2",
                     PP_ALIGN.LEFT if column_index == 0 else PP_ALIGN.CENTER, fill)
    note = "删除列为精确重叠/同源剔除数；EMPTY_AFTER_DELEAK 的 F1 与 TP/FP/FN 不定义，不记 0 分。"
    if split == "archiveii_embok_clean":
        note += " NucleicBERT-ft/frozen 保留原生首 598 nt，分别截断 39/38 条。"
    add_text(slide, note, 0.52, 6.60, 12.15, 0.32, 8.5, MUTED)
    add_text(slide, f"源：own_clean_leaderboard.json；SHA256={EXPECTED_SHA256[:12]}…；matched 交集不替代本表。",
             0.52, 7.08, 11.2, 0.18, 7.6, MUTED)
    add_text(slide, str(page_number), 12.18, 7.13, 0.62, 0.23, 8.5, BLUE, False, PP_ALIGN.RIGHT)
    notes_frame = slide.notes_slide.notes_text_frame
    if notes_frame is None:
        source_notes_shape = next(shape for shape in prs.slides[0].notes_slide.shapes
                                  if shape.is_placeholder and shape.placeholder_format.type == PP_PLACEHOLDER.BODY)
        slide.notes_slide.shapes._spTree.append(copy.deepcopy(source_notes_shape._element))
        notes_frame = slide.notes_slide.notes_text_frame
    notes_frame.text = (
        f"{SPLIT_LABELS[split]} own-clean 主榜。13 系统均独立去泄露、独立三次完整推理。"
        "F1 是 strict micro-F1 的三次均值，所有格 SD=0。删除列为 exact/homology 数量。"
        "EMPTY_AFTER_DELEAK 不计算 F1 或 TP/FP/FN，不填 0。不同系统 clean n 不强行相同。"
        f"完整源 SHA256={EXPECTED_SHA256}。"
    )


def main():
    data = load_source()
    values = all_values(data)
    prs = Presentation(BACKUP)
    assert len(prs.slides) == 23
    before_text = [[shape.text for shape in slide.shapes if shape.has_text_frame] for slide in prs.slides]
    for index, split in enumerate(SPLITS, 24):
        add_split_slide(prs, data, split, index)
    assert len(prs.slides) == 31
    temporary = DECK.with_suffix(".ownclean_tmp.pptx")
    prs.save(temporary)
    temporary.replace(DECK)
    after = Presentation(DECK)
    after_text = [[shape.text for shape in slide.shapes if shape.has_text_frame] for slide in list(after.slides)[:23]]
    assert before_text == after_text
    for page_index, split in enumerate(SPLITS, 23):
        tables = [shape.table for shape in after.slides[page_index].shapes if shape.has_table]
        assert len(tables) == 1
        table = tables[0]
        assert len(table.rows) == 14 and len(table.columns) == 8
        for row_index, (system, cells) in enumerate(data["systems"].items(), 1):
            expected = cell_values(system, split, cells[split])
            actual = [table.cell(row_index, column).text for column in range(8)]
            assert actual == [expected["system"], expected["clean_n"], expected["f1"], expected["tp"],
                              expected["fp"], expected["fn"], expected["removed_exact"], expected["removed_homology"]]
    payload = {"status": "PASS_OWN_CLEAN_104_TABLES_PENDING_NATIVE_RENDER", "slides": 31,
               "original_23_slide_text_unchanged": True, "table_cells_verified": len(values),
               "empty_cells_as_dash": 2, "source_sha256": EXPECTED_SHA256,
               "deck_sha256": sha256(DECK)}
    QA.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
