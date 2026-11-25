from pathlib import Path
import hashlib
import html
import json
import os
import re
import subprocess
import shutil
import zipfile

OUT = Path(__file__).resolve().parent
DECK = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round68_render_verification.json"
WORK = OUT / ".round68_render_work"
WORK.mkdir(exist_ok=True)
SOFFICE = shutil.which("soffice") or "/opt/homebrew/bin/soffice"
PDFTOPPM = shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm"
PDFTOTEXT = shutil.which("pdftotext") or "/opt/homebrew/bin/pdftotext"
SERVER_DECK = os.environ.get("ROUND68_SERVER_DECK", str(DECK))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value):
    return re.sub(r"\s+", "", value)


with zipfile.ZipFile(DECK) as archive:
    pres = archive.read("ppt/presentation.xml").decode("utf-8")
    rels = archive.read("ppt/_rels/presentation.xml.rels").decode("utf-8")
    rid_to_target = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
    rids = re.findall(r'<p:sldId[^>]*r:id="(rId\d+)"', pres)
    slide_parts = ["ppt/" + rid_to_target[rid].lstrip("/") for rid in rids]
    slide_texts = []
    for part in slide_parts:
        xml = archive.read(part).decode("utf-8")
        texts = [html.unescape(value) for value in re.findall(r"<a:t>(.*?)</a:t>", xml, re.S)]
        slide_texts.append([value for value in texts if value.strip()])
    slide_part_hashes = {part: hashlib.sha256(archive.read(part)).hexdigest() for part in slide_parts}

assert len(slide_parts) == 12
pdf = WORK / (DECK.stem + ".pdf")
if pdf.exists():
    pdf.unlink()
result = subprocess.run(
    [SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", str(WORK), str(DECK)],
    capture_output=True, text=True, timeout=300,
)
assert pdf.is_file(), f"PDF export failed: {result.stdout} {result.stderr}"
version = subprocess.run([SOFFICE, "--version"], capture_output=True, text=True).stdout.strip()
for path in WORK.glob("page-*.png"):
    path.unlink()
subprocess.run([PDFTOPPM, "-r", "120", "-png", str(pdf), str(WORK / "page")], check=True, timeout=300)
pngs = sorted(WORK.glob("page-*.png"))
sizes = []
for path in pngs:
    output = subprocess.run(["file", str(path)], capture_output=True, text=True).stdout
    match = re.search(r"(\d+) x (\d+)", output)
    sizes.append([int(match.group(1)), int(match.group(2))] if match else None)
text = subprocess.run([PDFTOTEXT, str(pdf), "-"], capture_output=True, text=True, check=True).stdout
haystack = norm(text)
checks = 0
missing = []
for index, strings in enumerate(slide_texts, start=20):
    for value in strings:
        checks += 1
        if norm(value) not in haystack:
            missing.append({"slide": index, "text": value[:160]})
new_page_text = norm("".join(slide_texts[-1]))
for phrase in (
    "FAIL_GEC_SEED0_INDEPENDENT_REEVAL", "16门仅4过/12失败", "MODEL-DEVONLY",
    "不启动seeds1/2", "-0.009356", "-0.026472", "-0.019792", "-0.052808",
):
    assert norm(phrase) in new_page_text, phrase
report = {
    "status": "PASS" if not missing and len(pngs) == 12 else "FAIL_RENDER_OR_TEXT",
    "renderer": f"{version} headless PDF export; pdftoppm 120 dpi",
    "deck": SERVER_DECK,
    "deck_sha256": sha256(DECK),
    "slides_rendered": len(pngs),
    "deck_slide_count": len(slide_parts),
    "added_slide": {"deck_index": 12, "display_number": 31, "png": pngs[-1].name, "png_sha256": sha256(pngs[-1])},
    "raster_sizes": sizes,
    "ppt_to_pdf_complete_text_checks": checks,
    "missing_or_clipped_text": missing,
    "pdf_words_checked": len(text.split()),
    "pdf_sha256": sha256(pdf),
    "slide_part_sha256": slide_part_hashes,
    "render_host": "local_macOS_libreoffice",
    "visual_review": "PENDING_AGENT_IMAGE_REVIEW",
}
VERIFICATION.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
