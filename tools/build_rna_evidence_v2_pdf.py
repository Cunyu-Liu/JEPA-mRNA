import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"command failed: {command}: {result.stdout}\n{result.stderr}")
    return result.stdout


def font(run, name, size, bold=None, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor(*color)
    properties = run._element.get_or_add_rPr()
    fonts = properties.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        properties.insert(0, fonts)
    fonts.set(qn("w:eastAsia"), "PingFang SC")


def set_table_grid(table, fractions):
    cells = table.rows[0].cells
    if len(cells) != len(fractions):
        raise AssertionError("unexpected table column count")
    widths = [int(9865 * fraction) for fraction in fractions]
    widths[-1] = 9865 - sum(widths[:-1])
    for element in list(table._tbl.tblGrid):
        table._tbl.tblGrid.remove(element)
    for width in widths:
        column = OxmlElement("w:gridCol")
        column.set(qn("w:w"), str(width))
        table._tbl.tblGrid.append(column)
    table.autofit = False
    properties = table._tbl.tblPr
    table_width = properties.find(qn("w:tblW"))
    if table_width is None:
        table_width = OxmlElement("w:tblW")
        properties.insert(0, table_width)
    table_width.set(qn("w:type"), "dxa")
    table_width.set(qn("w:w"), "9865")
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Mm(width * 174 / 9865)


def style_document(path):
    document = Document(path)
    section = document.sections[0]
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.left_margin = section.right_margin = Mm(18)
    section.top_margin = Mm(18)
    section.bottom_margin = Mm(20)
    section.header_distance = Mm(8)
    section.footer_distance = Mm(8)
    for name, size, bold in (("Normal", 10, False), ("Body Text", 10, False),
                             ("Title", 20, True), ("Heading 1", 15, True),
                             ("Heading 2", 12, True), ("Heading 3", 10.5, True)):
        if name not in document.styles:
            continue
        style = document.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.color.rgb = RGBColor(35, 48, 61)
        style.paragraph_format.line_spacing = 1.15
        style.paragraph_format.space_after = Pt(4)
        if name.startswith("Heading"):
            style.paragraph_format.space_before = Pt(10)
            style.paragraph_format.keep_with_next = True
        for run in style.element.xpath(".//w:r"):
            font(run, "Times New Roman", size, bold, (35, 48, 61))
    for paragraph in document.paragraphs:
        code_style = paragraph.style.name in ("Source Code", "Verbatim Char", "HTML Preformatted")
        for run in paragraph.runs:
            if code_style:
                font(run, "Courier New", 7.5)
            else:
                size = run.font.size.pt if run.font.size else 10
                font(run, "Times New Roman", size, run.bold, run.font.color.rgb if run.font.color and run.font.color.type else None)
    widths = [(0.32, 0.26, 0.42), (0.34, 0.33, 0.33), (0.72, 0.28),
              (0.50, 0.25, 0.25), (0.08, 0.92)]
    if len(document.tables) != len(widths):
        raise AssertionError("unexpected table count")
    for table, fractions in zip(document.tables, widths):
        set_table_grid(table, fractions)
        for row_index, row in enumerate(table.rows):
            properties = row._tr.get_or_add_trPr()
            if properties.find(qn("w:cantSplit")) is None:
                properties.append(OxmlElement("w:cantSplit"))
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(1)
                    paragraph.paragraph_format.line_spacing = 1.0
                    for run in paragraph.runs:
                        font(run, "Times New Roman", 8, row_index == 0 or run.bold)
    header = section.header.paragraphs[0]
    header.text = "RNA EVIDENCE V2  |  CONTAMINATION-AWARE EVALUATION"
    for run in header.runs:
        font(run, "Times New Roman", 8, False, (100, 112, 123))
    footer = section.footer.paragraphs[0]
    footer.alignment = 1
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    document.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", default="paper/rna_evidence_v2_manuscript_draft.md")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    manuscript = Path(args.manuscript).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(manuscript)
    if args.expected_sha256 and source_hash != args.expected_sha256:
        raise AssertionError("manuscript changed after evidence QA")
    pandoc = shutil.which("pandoc")
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    pdfinfo = shutil.which("pdfinfo")
    pdftotext = shutil.which("pdftotext")
    pdftoppm = shutil.which("pdftoppm")
    if not all((pandoc, soffice, pdfinfo, pdftotext, pdftoppm)):
        raise RuntimeError("PDF toolchain is incomplete")
    docx = output / "rna_evidence_v2_manuscript_draft.docx"
    pdf = output / "rna_evidence_v2_manuscript_draft.pdf"
    run([pandoc, str(manuscript), "--from=gfm", "--to=docx", f"--output={docx}"])
    style_document(docx)
    if [len(table.columns) for table in Document(docx).tables] != [3, 3, 2, 3, 2]:
        raise AssertionError("table grid was not written correctly")
    run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(output), str(docx)], cwd=output)
    if not pdf.is_file():
        raise AssertionError("PDF was not produced")
    info = run([pdfinfo, str(pdf)])
    pages = int(next(line.split(":", 1)[1] for line in info.splitlines() if line.startswith("Pages:")).strip())
    if pages < 2:
        raise AssertionError("unexpected PDF page count")
    text = " ".join(run([pdftotext, str(pdf), "-"]).split())
    for phrase in ("not submission-ready", "Q12", "Nussinov + Turner 堆叠能", "162,489",
                   "37,870 / 42,564", "dual-view comparison is limited to H0–H6 on model-dev"):
        if phrase not in text:
            raise AssertionError(f"PDF text is missing: {phrase}")
    previews = []
    for label, page in (("first", 1), ("middle", (pages + 1) // 2), ("last", pages)):
        prefix = output / f"preview_{label}"
        run([pdftoppm, "-f", str(page), "-l", str(page), "-r", "110", "-png", str(pdf), str(prefix)])
        candidates = sorted(output.glob(f"preview_{label}-*.png"))
        if len(candidates) != 1:
            raise AssertionError(f"preview missing: {label}")
        previews.append(str(candidates[0]))
    manifest = {"status": "PASS_PDF_BUILD_AND_TEXT_QA",
                "manuscript": str(manuscript), "manuscript_sha256": source_hash,
                "script_sha256": sha256(Path(__file__).resolve()),
                "docx": str(docx), "docx_sha256": sha256(docx),
                "pdf": str(pdf), "pdf_sha256": sha256(pdf), "pages": pages,
                "preview_pages": previews,
                "scope": "A4 single-column evidence draft; not submission-ready and not a scientific accuracy claim"}
    destination = output / "pdf_build_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
