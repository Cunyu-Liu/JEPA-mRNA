#!/usr/bin/env python3
"""Independently rebuild round72 and compare every ZIP part without importing its builder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from pptx import Presentation

OUT = Path(__file__).resolve().parent
DECK = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def zip_parts(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", type=Path, default=OUT / "add_external_results_round72.py")
    parser.add_argument("--source-deck", type=Path, required=True, help="hash-verified synthetic or real 15-slide margin deck")
    parser.add_argument("--margin-verification", type=Path, required=True)
    parser.add_argument("--margin-independent-verification", type=Path, required=True)
    parser.add_argument("--primary-deck", type=Path, required=True, help="built 17-slide deck")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--compare-previous-deck", type=Path,
                        help="optional pre-panel 17-slide deck; require page35 change and page36 XML identity")
    args = parser.parse_args(argv)
    source = args.source_deck.resolve()
    primary = args.primary_deck.resolve()
    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        parser.error("--work-dir must be new or empty")
    work.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    primary_hash = sha256(primary)
    independent = work / "independent_round72.pptx"
    shutil.copy2(source, independent)
    margin_report = json.loads(args.margin_verification.resolve().read_text())
    margin_report["deck"] = str(independent)
    rebound_margin_verification = work / "margin_verification_rebound.json"
    atomic_json(rebound_margin_verification, margin_report)
    command = [sys.executable, str(args.builder.resolve()), "--deck", str(independent),
               "--margin-verification", str(rebound_margin_verification),
               "--margin-independent-verification", str(args.margin_independent_verification.resolve()),
               "--backup", str(work / "independent.pre_round72.pptx"),
               "--verification", str(work / "independent_build.json")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode:
        parser.exit(result.returncode, f"independent builder failed:\n{result.stdout}\n{result.stderr}\n")
    before = zip_parts(primary)
    after = zip_parts(independent)
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    different = sorted(name for name in set(before) & set(after) if before[name] != after[name])
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(primary) as new:
        prior_xml = all(old.read(f"ppt/slides/slide{index}.xml")
                        == new.read(f"ppt/slides/slide{index}.xml") for index in range(1, 16))
    build_report = json.loads((work / "independent_build.json").read_text())
    page36_unchanged = None
    page35_changed = None
    if args.compare_previous_deck:
        with zipfile.ZipFile(args.compare_previous_deck) as previous, zipfile.ZipFile(independent) as current:
            page36_unchanged = (previous.read("ppt/slides/slide17.xml")
                                == current.read("ppt/slides/slide17.xml"))
            page35_changed = (previous.read("ppt/slides/slide16.xml")
                              != current.read("ppt/slides/slide16.xml"))
    report = {
        "status": "PASS_INDEPENDENT_CLEAN_BUILD_ALL_ZIP_PARTS_IDENTICAL"
        if not missing and not extra and not different and prior_xml
        and page36_unchanged is not False and page35_changed is not False else "FAIL_INDEPENDENT_BUILD",
        "primary_deck": str(primary), "independent_deck": str(independent),
        "primary_sha256": primary_hash, "independent_sha256": sha256(independent),
        "source_deck": str(source), "source_sha256_before_and_after": source_hash,
        "source_unchanged": sha256(source) == source_hash,
        "primary_unchanged": sha256(primary) == primary_hash,
        "margin_verification": str(args.margin_verification.resolve()),
        "margin_verification_sha256": sha256(args.margin_verification.resolve()),
        "margin_independent_verification": str(args.margin_independent_verification.resolve()),
        "margin_independent_verification_sha256": sha256(args.margin_independent_verification.resolve()),
        "exposure_sensitivity_path": build_report["exposure_sensitivity"]["path"],
        "exposure_sensitivity_sha256": build_report["exposure_sensitivity"]["sha256"],
        "exposure_embedded_matches": build_report["exposure_sensitivity"]["embedded_matches"],
        "exposure_cluster_dev_counts": build_report["exposure_sensitivity"]["cluster_dev_counts"],
        "zip_parts_compared": len(set(before) & set(after)), "missing_parts": missing,
        "extra_parts": extra, "different_parts": different,
        "slide_count": len(Presentation(primary).slides), "prior_slide_xml_preserved": 15 if prior_xml else 0,
        "original_phase0_xml_preserved": build_report["original_phase0_xml_preserved"],
        "added_slide_numbers": build_report["added_slide_numbers"],
        "compare_previous_deck": str(args.compare_previous_deck) if args.compare_previous_deck else None,
        "page35_xml_changed": page35_changed, "page36_xml_unchanged": page36_unchanged,
        "official_reference_panel_sha256": next(item["sha256"] for item in build_report["external_artifacts"]
                                                if item["path"].endswith("official_reference_5388.jsonl.gz")),
        "scope": build_report["scope"],
    }
    atomic_json(args.verification.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (report["status"].startswith("PASS") and report["source_unchanged"]
                 and report["primary_unchanged"] and report["slide_count"] == 17) else 1


if __name__ == "__main__":
    raise SystemExit(main())
