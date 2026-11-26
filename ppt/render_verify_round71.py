#!/usr/bin/env python3
"""Render all 15 slides and independently check PDF text plus slide-34 OCR."""
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def slide_parts(deck: Path) -> tuple[list[str], list[list[str]]]:
    with zipfile.ZipFile(deck) as archive:
        presentation = archive.read("ppt/presentation.xml").decode("utf-8")
        relationships = archive.read("ppt/_rels/presentation.xml.rels").decode("utf-8")
        rid_to_target = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', relationships))
        rids = re.findall(r'<p:sldId[^>]*r:id="(rId\d+)"', presentation)
        parts = ["ppt/" + rid_to_target[rid].lstrip("/") for rid in rids]
        texts = []
        for part in parts:
            xml = archive.read(part).decode("utf-8")
            texts.append([html.unescape(value) for value in re.findall(r"<a:t>(.*?)</a:t>", xml, re.S)
                          if html.unescape(value).strip()])
    return parts, texts


def ocr(page: Path, work: Path) -> tuple[str, str]:
    tesseract = shutil.which("tesseract")
    if tesseract:
        result = subprocess.run([tesseract, str(page), "stdout", "-l", "eng+chi_sim"],
                                capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return result.stdout, "tesseract eng+chi_sim"
    swift = shutil.which("swift")
    helper = OUT / "ocr_round71.swift"
    if swift and helper.is_file():
        result = subprocess.run([swift, str(helper), str(page)], capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return result.stdout, "macOS Vision zh-Hans/en-US"
        raise RuntimeError(f"Vision OCR failed: {result.stderr}")
    raise RuntimeError("neither tesseract nor the macOS Vision OCR helper is available")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=DECK)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--expected-status", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--deck-label", default="")
    args = parser.parse_args(argv)
    deck = args.deck.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    deck_hash = sha256(deck)
    parts, slide_texts = slide_parts(deck)
    if len(parts) != 15:
        parser.error(f"expected 15 slides, found {len(parts)}")
    soffice = shutil.which("soffice") or "/opt/homebrew/bin/soffice"
    pdftoppm = shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm"
    pdftotext = shutil.which("pdftotext") or "/opt/homebrew/bin/pdftotext"
    pdf = work / (deck.stem + ".pdf")
    if pdf.exists():
        pdf.unlink()
    profile = work / "libreoffice-profile"
    result = subprocess.run([soffice, "--headless", f"-env:UserInstallation={profile.as_uri()}",
                             "--convert-to", "pdf", "--outdir", str(work), str(deck)],
                            capture_output=True, text=True, timeout=300)
    if not pdf.is_file():
        raise RuntimeError(f"PDF export failed: {result.stdout} {result.stderr}")
    version = subprocess.run([soffice, "--version"], capture_output=True, text=True).stdout.strip()
    for path in work.glob("page-*.png"):
        path.unlink()
    subprocess.run([pdftoppm, "-r", "120", "-png", str(pdf), str(work / "page")],
                   check=True, timeout=300)
    pages = sorted(work.glob("page-*.png"))
    subprocess.run([pdftoppm, "-f", "15", "-l", "15", "-singlefile", "-r", "220", "-png",
                    str(pdf), str(work / "slide34-ocr")], check=True, timeout=300)
    ocr_page = work / "slide34-ocr.png"
    ocr_text, ocr_engine = ocr(ocr_page, work)
    (work / "slide34-ocr.txt").write_text(ocr_text)
    pdf_text = subprocess.run([pdftotext, str(pdf), "-"], capture_output=True, text=True,
                              check=True, timeout=300).stdout
    haystack = norm(pdf_text)
    missing = []
    checks = 0
    for index, strings in enumerate(slide_texts, start=19):
        for value in strings:
            checks += 1
            if norm(value) not in haystack:
                missing.append({"slide": index, "text": value[:180]})
    new_page_text = norm("".join(slide_texts[-1]))
    required_pdf = ("MODEL-DEV ONLY", "QUALIFIED NLL", "ensemble", "destroy", "independent",
                    "isolated", "span≥128", "5,872−closure 769=5,103 NOT_DETECTED_UNKNOWN (not clean)",
                    "GEC INELIGIBLE", f"{args.expected_status}_STRUCTURED_MARGIN_INDEPENDENT_EVAL")
    missing_pdf = [value for value in required_pdf if norm(value) not in new_page_text]
    ocr_haystack = norm(ocr_text)
    required_ocr = ("FOLDNAR", "MODEL-DEV", "ensemble", "QUALIFIED", "5,103", "INELIGIBLE",
                    args.expected_status)
    missing_ocr = [value for value in required_ocr if norm(value) not in ocr_haystack]
    report = {
        "status": "PASS_RENDER_PDF_TEXT_AND_SLIDE34_OCR"
        if len(pages) == 15 and not missing and not missing_pdf and not missing_ocr else "FAIL_RENDER_OR_TEXT",
        "renderer": f"{version} headless PDF export; pdftoppm 120 dpi; slide34 OCR 220 dpi",
        "deck": args.deck_label or str(deck), "deck_sha256": deck_hash,
        "deck_unchanged": sha256(deck) == deck_hash, "slides_rendered": len(pages), "deck_slide_count": len(parts),
        "added_slide": {"deck_index": 15, "display_number": 34, "png": pages[-1].name if pages else None,
                        "ocr_png": ocr_page.name, "ocr_png_sha256": sha256(ocr_page)},
        "ppt_to_pdf_complete_text_checks": checks, "missing_or_clipped_text": missing,
        "missing_new_page_pdf_text": missing_pdf, "missing_new_page_ocr_text": missing_ocr,
        "ocr_engine": ocr_engine, "ocr_text_path": str(work / "slide34-ocr.txt"),
        "ocr_characters": len(ocr_text), "pdf_sha256": sha256(pdf),
        "scientific_status_prefix": args.expected_status,
        "visual_review": "PENDING_AGENT_IMAGE_REVIEW",
    }
    atomic_json(args.verification.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"].startswith("PASS") and report["deck_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
