from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

OUT = Path(__file__).resolve().parent
DECK = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round70_crossover_render_verification.json"
WORK = OUT / ".round70_crossover_render_work"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value: str) -> str:
    return re.sub(r"\s+", "", value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=DECK)
    parser.add_argument("--work-dir", type=Path, default=WORK)
    parser.add_argument("--verification", type=Path, default=VERIFICATION)
    parser.add_argument("--server-deck", default=os.environ.get("ROUND70_CROSSOVER_SERVER_DECK", str(DECK)))
    args = parser.parse_args()
    deck = args.deck.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    soffice = shutil.which("soffice") or "/opt/homebrew/bin/soffice"
    pdftoppm = shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm"
    pdftotext = shutil.which("pdftotext") or "/opt/homebrew/bin/pdftotext"

    with zipfile.ZipFile(deck) as archive:
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

    assert len(slide_parts) == 14
    pdf = work / (deck.stem + ".pdf")
    if pdf.exists():
        pdf.unlink()
    result = subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(work), str(deck)],
        capture_output=True, text=True, timeout=300,
    )
    assert pdf.is_file(), f"PDF export failed: {result.stdout} {result.stderr}"
    version = subprocess.run([soffice, "--version"], capture_output=True, text=True).stdout.strip()
    for path in work.glob("page-*.png"):
        path.unlink()
    subprocess.run([pdftoppm, "-r", "150", "-png", str(pdf), str(work / "page")], check=True, timeout=300)
    pngs = sorted(work.glob("page-*.png"))
    sizes = []
    for path in pngs:
        output = subprocess.run(["file", str(path)], capture_output=True, text=True).stdout
        match = re.search(r"(\d+) x (\d+)", output)
        sizes.append([int(match.group(1)), int(match.group(2))] if match else None)
    pdf_text = subprocess.run([pdftotext, str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    haystack = norm(pdf_text)
    checks = 0
    missing = []
    for index, strings in enumerate(slide_texts, start=20):
        for value in strings:
            checks += 1
            if norm(value) not in haystack:
                missing.append({"slide": index, "text": value[:160]})
    new_page_text = norm("".join(slide_texts[-1]))
    for phrase in (
        "上下文有因果作用，但真实系综输入并非一致最优", "0.699302", "0.681231", "0.687129",
        "-0.104102", "-0.193295", "-0.289777", "+0.010192", "+0.001519", "+0.001206",
        "证明输入因果作用，不等于表示问题已解决", "继续检验structuredmargin，本页不构成其效果证据",
        "MODEL-DEVONLY", "独立raw-truth重算PASS",
    ):
        assert norm(phrase) in new_page_text, phrase
    passed = not missing and len(pngs) == 14
    report = {
        "status": "PASS" if passed else "FAIL_RENDER_OR_TEXT",
        "renderer": f"{version} headless PDF export; pdftoppm 150 dpi",
        "deck": args.server_deck,
        "deck_sha256": sha256(deck),
        "slides_rendered": len(pngs),
        "deck_slide_count": len(slide_parts),
        "added_slide": {
            "deck_index": 14,
            "display_number": 33,
            "png": pngs[-1].name,
            "png_sha256": sha256(pngs[-1]),
        },
        "raster_sizes": sizes,
        "ppt_to_pdf_complete_text_checks": checks,
        "missing_or_clipped_text": missing,
        "pdf_words_checked": len(pdf_text.split()),
        "pdf_sha256": sha256(pdf),
        "slide_part_sha256": slide_part_hashes,
        "render_host": "local_macOS_libreoffice",
        "visual_review": "PENDING_AGENT_IMAGE_REVIEW",
        "ocr_review": "PENDING_SEPARATE_VISION_OCR",
    }
    verification = args.verification.resolve()
    descriptor, temporary = tempfile.mkstemp(prefix=verification.name + ".", suffix=".tmp", dir=verification.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, verification)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
