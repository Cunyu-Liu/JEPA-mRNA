from pathlib import Path
import hashlib
import html
import json
import re
import subprocess
import shutil
import zipfile

OUT = Path(__file__).resolve().parent
DECK = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round67_render_verification.json"
WORK = OUT / ".round67_render_work"
WORK.mkdir(exist_ok=True)
SOFFICE = shutil.which("soffice") or "/opt/homebrew/bin/soffice"
PDFTOPPM = shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm"
PDFTOTEXT = shutil.which("pdftotext") or "/opt/homebrew/bin/pdftotext"
SERVER_DECK = str(DECK)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value):
    return re.sub(r"\s+", "", value)


with zipfile.ZipFile(DECK) as z:
    pres = z.read("ppt/presentation.xml").decode("utf-8")
    rels = z.read("ppt/_rels/presentation.xml.rels").decode("utf-8")
    rid_to_target = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
    rids = re.findall(r'<p:sldId[^>]*r:id="(rId\d+)"', pres)
    slide_parts = ["ppt/" + rid_to_target[r].lstrip("/") for r in rids]
    slide_texts = []
    for part in slide_parts:
        xml = z.read(part).decode("utf-8")
        texts = [html.unescape(t) for t in re.findall(r"<a:t>(.*?)</a:t>", xml, re.S)]
        slide_texts.append([t for t in texts if t.strip()])

pdf = WORK / (DECK.stem + ".pdf")
if pdf.exists():
    pdf.unlink()
r = subprocess.run([SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", str(WORK), str(DECK)],
                   capture_output=True, text=True, timeout=300)
assert pdf.is_file(), f"PDF export failed: {r.stdout} {r.stderr}"
version = subprocess.run([SOFFICE, "--version"], capture_output=True, text=True).stdout.strip()

for p in WORK.glob("page-*.png"):
    p.unlink()
subprocess.run([PDFTOPPM, "-r", "120", "-png", str(pdf), str(WORK / "page")], check=True, timeout=300)
pngs = sorted(WORK.glob("page-*.png"))
sizes = []
for p in pngs:
    out = subprocess.run(["file", str(p)], capture_output=True, text=True).stdout
    m = re.search(r"(\d+) x (\d+)", out)
    sizes.append([int(m.group(1)), int(m.group(2))] if m else None)

text = subprocess.run([PDFTOTEXT, str(pdf), "-"], capture_output=True, text=True, check=True).stdout
haystack = norm(text)
checks = 0
missing = []
for index, strings in enumerate(slide_texts, start=20):
    for s in strings:
        checks += 1
        if norm(s) not in haystack:
            missing.append({"slide": index, "text": s[:120]})

report = {
    "status": "PASS" if not missing and len(pngs) == len(slide_texts) else "FAIL_RENDER_OR_TEXT",
    "renderer": f"{version} headless PDF export; pdftoppm 120 dpi",
    "deck": SERVER_DECK,
    "deck_sha256": sha256(DECK),
    "slides_rendered": len(pngs),
    "raster_sizes": sizes,
    "ppt_to_pdf_complete_text_checks": checks,
    "missing_or_clipped_text": missing,
    "pdf_words_checked": len(text.split()),
    "pdf_sha256": sha256(pdf),
    "render_host": "libreoffice_headless",
}
VERIFICATION.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))