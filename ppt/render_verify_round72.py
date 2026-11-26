#!/usr/bin/env python3
"""Render all 17 slides and independently check PDF text plus slide-35/36 OCR."""
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

OUT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def slide_parts(deck: Path) -> tuple[list[str], list[list[str]]]:
    import zipfile
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


def ocr(page: Path) -> tuple[str, str]:
    tesseract = shutil.which("tesseract")
    if tesseract:
        result = subprocess.run([tesseract, str(page), "stdout", "-l", "eng+chi_sim"],
                                capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return result.stdout, "tesseract eng+chi_sim"
    swift = shutil.which("swift")
    helper = OUT / "ocr_round72.swift"
    if swift and helper.is_file():
        result = subprocess.run([swift, str(helper), str(page)], capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return result.stdout, "macOS Vision zh-Hans/en-US"
        raise RuntimeError(f"Vision OCR failed: {result.stderr}")
    raise RuntimeError("neither tesseract nor the macOS Vision OCR helper is available")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--deck-label", default="SYNTHETIC_MARGIN_FIXTURE_REAL_FROZEN_EXTERNAL_RESULTS")
    args = parser.parse_args(argv)
    deck = args.deck.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    deck_hash = sha256(deck)
    parts, slide_texts = slide_parts(deck)
    if len(parts) != 17:
        parser.error(f"expected 17 slides, found {len(parts)}")
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
    ocr_results = {}
    required_ocr = {
        35: ("FOLDNAR", "structRFM", "0.4875747710", "0.4850647195", "0.7745179151",
             "0.6296755882", "0.4890166090", "BPfold", "CONTRAfold", "xens2",
             "RAW-EXPOSURE", "CLEAN", "SOTA", "4924e351"),
        36: ("FOLDNAR", "LinearPartition", "APPROXIMATE_SPARSE", "9MR4_5", "9Q7Q_5", "7AM2_1",
             "53/73", "8/73", "362.79", "8,481,693,696", "EXACT_GLOBAL_DP"),
    }
    for display_number in (35, 36):
        page_index = display_number - 19
        stem = f"slide{display_number}-ocr"
        subprocess.run([pdftoppm, "-f", str(page_index), "-l", str(page_index), "-singlefile",
                        "-r", "400", "-png", str(pdf), str(work / stem)], check=True, timeout=300)
        page = work / f"{stem}.png"
        ocr_text, engine = ocr(page)
        text_path = work / f"{stem}.txt"
        text_path.write_text(ocr_text)
        missing = [value for value in required_ocr[display_number] if norm(value) not in norm(ocr_text)]
        ocr_results[display_number] = {"deck_index": page_index, "png": page.name,
                                       "png_sha256": sha256(page), "text_path": str(text_path),
                                       "engine": engine, "characters": len(ocr_text), "missing": missing}
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
    required_pdf = {
        35: ("OFFICIAL_REFERENCE_5388", "0.4875747710", "0.4850647195", "0.7745179151",
             "0.6296755882", "0.4890166090", "RAW-EXPOSURE DESCRIPTIVE ≠ CLEAN SOTA",
             "28 full-coverage systems", "BPfold项目replay ≠ 官方release", "4924e351",
             "DERIVED_FROM_BPFOLD_PREDICTIONS"),
        36: ("APPROXIMATE_SPARSE", "EXACT_GLOBAL_DP", "53/73 · 72.60%", "8/73 · 10.96%",
             "362.79 s", "8,481,693,696 B RSS", "FLOPs NOT_MEASURED", "NOT_ESTABLISHED",
             "不用于模型选择", "span=j−i"),
    }
    missing_pdf = {}
    for display_number in (35, 36):
        page_text = norm("".join(slide_texts[display_number - 20]))
        absent = [value for value in required_pdf[display_number] if norm(value) not in page_text]
        if absent:
            missing_pdf[str(display_number)] = absent
    ocr_missing = {str(number): row["missing"] for number, row in ocr_results.items() if row["missing"]}
    passed = len(pages) == 17 and not missing and not missing_pdf and not ocr_missing
    report = {
        "status": "PASS_RENDER_PDF_TEXT_AND_SLIDE35_36_OCR" if passed else "FAIL_RENDER_OR_TEXT",
        "deck_label": args.deck_label, "deck": str(deck), "deck_sha256": deck_hash,
        "deck_unchanged": sha256(deck) == deck_hash, "slides_rendered": len(pages),
        "deck_slide_count": len(parts), "renderer": f"{version} headless PDF; pdftoppm 120 dpi; OCR 400 dpi",
        "ppt_to_pdf_complete_text_checks": checks, "missing_or_clipped_text": missing,
        "missing_new_page_pdf_text": missing_pdf, "missing_new_page_ocr_text": ocr_missing,
        "ocr": ocr_results, "pdf_sha256": sha256(pdf),
        "real_margin_data_used": False, "real_ppt_modified": False,
        "visual_review": "PENDING_AGENT_IMAGE_REVIEW",
    }
    atomic_json(args.verification.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed and report["deck_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
