#!/usr/bin/env python3
"""Render all 18 slides and independently check PDF text plus slide-37 OCR."""
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


def ocr_norm(value: str) -> str:
    return norm(value).replace("archiveli", "archiveii").replace("archivell", "archiveii")


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
    helper = OUT / "ocr_round73.swift"
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
    parser.add_argument("--deck-label", default="SYNTHETIC_17_SLIDE_DELIVERY_FIXTURE")
    args = parser.parse_args(argv)
    deck = args.deck.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    deck_hash = sha256(deck)
    parts, slide_texts = slide_parts(deck)
    if len(parts) != 18:
        parser.error(f"expected 18 slides, found {len(parts)}")
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
    stem = "slide37-ocr"
    subprocess.run([pdftoppm, "-f", "18", "-l", "18", "-singlefile", "-r", "400", "-png",
                    str(pdf), str(work / stem)], check=True, timeout=300)
    page = work / f"{stem}.png"
    ocr_text, engine = ocr(page)
    text_path = work / f"{stem}.txt"
    text_path.write_text(ocr_text)
    required_ocr = ("LABEL", "SOTA", "BPfold", "ArchiveII", "0.8156522070", "0.8999331355",
                    "0.8204522395", "PDB", "62", "91", "UNCHECKED", "5381/5388", "2048/2544",
                    "PROJECTED", "OFFICIAL", "7×1305", "7×430", "1302/1305", "429/430")
    ocr_missing = [value for value in required_ocr if ocr_norm(value) not in ocr_norm(ocr_text)]
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
    required_pdf = ("37 · LABEL PROTOCOL", "标签谱系决定SOTA证据资格", "DERIVED_FROM_BPFOLD_PREDICTIONS",
                    "bpRNA-new 5388", "clean 2544", "0.8156522070", "0.8999331355", "0.8204522395",
                    "+0.0842809284", "+0.0048000324", "strongest single", "62 rebuilt / 91 UNCHECKED",
                    "PROJECTED · TS0 1288 · TestSetB 428", "与 official 分开报告",
                    "OFFICIAL · TS0 7×1305 full · TestSetB 7×430 full",
                    "RNAstructure fold/MaxExpect：1302/1305 · 429/430 partial",
                    "不得排名", "5381/5388", "2048/2544", "UNKNOWN / UNCHECKED ≠ clean",
                    "完整 SOTA 仍未闭环", "四臂未收敛", "≠ clean SOTA")
    slide37 = norm("".join(slide_texts[17]))
    missing_pdf = [value for value in required_pdf if norm(value) not in slide37]
    obsolete_pdf = [value for value in ("58 rebuilt / 95 UNCHECKED", "PDB58/95",
                                        "全集无任何 full-coverage prediction")
                    if norm(value) in slide37]
    passed = len(pages) == 18 and not missing and not missing_pdf and not ocr_missing and not obsolete_pdf
    report = {
        "status": "PASS_RENDER_PDF_TEXT_AND_SLIDE37_OCR" if passed else "FAIL_RENDER_OR_TEXT",
        "deck_label": args.deck_label, "deck": str(deck), "deck_sha256": deck_hash,
        "deck_unchanged": sha256(deck) == deck_hash, "slides_rendered": len(pages),
        "deck_slide_count": len(parts), "renderer": f"{version} headless PDF; pdftoppm 120 dpi; OCR 400 dpi",
        "ppt_to_pdf_complete_text_checks": checks, "missing_or_clipped_text": missing,
        "missing_new_page_pdf_text": missing_pdf, "missing_new_page_ocr_text": ocr_missing,
        "obsolete_pdb58_95_text": obsolete_pdf,
        "ocr": {"deck_index": 18, "display_number": 37, "png": page.name,
                "png_sha256": sha256(page), "text_path": str(text_path), "engine": engine,
                "characters": len(ocr_text)},
        "pdf_sha256": sha256(pdf), "real_ppt_modified": False,
        "visual_review": "PENDING_AGENT_IMAGE_REVIEW",
    }
    atomic_json(args.verification.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed and report["deck_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
