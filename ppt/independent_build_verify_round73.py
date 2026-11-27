#!/usr/bin/env python3
"""Independently rebuild round73 and compare every ZIP part without importing its builder."""
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
    parser.add_argument("--builder", type=Path, default=OUT / "add_label_protocol_round73.py")
    parser.add_argument("--source-deck", type=Path, required=True,
                        help="hash-verified synthetic or real 17-slide delivery")
    parser.add_argument("--source-verification", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--primary-deck", type=Path, required=True, help="built 18-slide temporary copy")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source_deck.resolve()
    primary = args.primary_deck.resolve()
    work = args.work_dir.resolve()
    if source == primary:
        parser.error("primary deck must be a temporary copy, not the source delivery")
    if work.exists() and any(work.iterdir()):
        parser.error("--work-dir must be new or empty")
    work.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    primary_hash = sha256(primary)
    independent = work / "independent_round73.pptx"
    shutil.copy2(source, independent)
    command = [sys.executable, str(args.builder.resolve()), "--deck", str(independent),
               "--source-deck", str(source), "--source-verification", str(args.source_verification.resolve()),
               "--backup", str(work / "independent.pre_round73.pptx"),
               "--verification", str(work / "independent_build.json")]
    if args.expected_source_sha256:
        command.extend(("--expected-source-sha256", args.expected_source_sha256))
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
                        == new.read(f"ppt/slides/slide{index}.xml") for index in range(1, 18))
    build_report = json.loads((work / "independent_build.json").read_text())
    passed = (not missing and not extra and not different and prior_xml
              and sha256(source) == source_hash and sha256(primary) == primary_hash
              and len(Presentation(primary).slides) == 18)
    report = {
        "status": "PASS_INDEPENDENT_CLEAN_BUILD_ALL_ZIP_PARTS_IDENTICAL" if passed else "FAIL_INDEPENDENT_BUILD",
        "primary_deck": str(primary), "independent_deck": str(independent),
        "primary_sha256": primary_hash, "independent_sha256": sha256(independent),
        "source_deck": str(source), "source_sha256_before_and_after": source_hash,
        "source_unchanged": sha256(source) == source_hash,
        "primary_unchanged": sha256(primary) == primary_hash,
        "source_verification": str(args.source_verification.resolve()),
        "source_verification_sha256": sha256(args.source_verification.resolve()),
        "zip_parts_compared": len(set(before) & set(after)), "missing_parts": missing,
        "extra_parts": extra, "different_parts": different,
        "slide_count": len(Presentation(primary).slides), "prior_slide_xml_preserved": 17 if prior_xml else 0,
        "added_slide_number": build_report["added_slide_number"], "scope": build_report["scope"],
        "real_ppt_modified": False,
    }
    atomic_json(args.verification.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
