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
BACKUP = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pre_round70_crossover.pptx"
FROZEN = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_frozen_context_crossover_v1/full")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def zip_parts(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", type=Path, default=OUT / "add_phase0_gec_context_crossover_round70.py")
    parser.add_argument("--source-deck", type=Path, default=BACKUP)
    parser.add_argument("--primary-deck", type=Path, default=DECK)
    parser.add_argument("--crossover-root", type=Path, default=FROZEN)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_deck.resolve()
    primary = args.primary_deck.resolve()
    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        parser.error("--work-dir must be new or empty")
    work.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    primary_hash = sha256(primary)
    independent = work / "independent_crossover_round70.pptx"
    shutil.copy2(source, independent)
    command = [
        sys.executable,
        str(args.builder.resolve()),
        "--deck", str(independent),
        "--crossover-root", str(args.crossover_root.resolve()),
        "--backup", str(work / "independent.pre_round70_crossover.pptx"),
        "--verification", str(work / "independent_build.json"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode:
        parser.exit(result.returncode, f"independent builder failed:\n{result.stdout}\n{result.stderr}\n")
    before = zip_parts(primary)
    after = zip_parts(independent)
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    different = sorted(name for name in set(before) & set(after) if before[name] != after[name])
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(primary) as new:
        prior_xml = all(
            old.read(f"ppt/slides/slide{index}.xml") == new.read(f"ppt/slides/slide{index}.xml")
            for index in range(1, 14)
        )
    build_report = json.loads((work / "independent_build.json").read_text())
    passed = not missing and not extra and not different and prior_xml
    report = {
        "status": "PASS_INDEPENDENT_CLEAN_BUILD_ALL_ZIP_PARTS_IDENTICAL" if passed else "FAIL_INDEPENDENT_BUILD",
        "primary_deck": str(primary),
        "independent_deck": str(independent),
        "primary_sha256": primary_hash,
        "independent_sha256": sha256(independent),
        "source_deck": str(source),
        "source_sha256_before_and_after": source_hash,
        "source_unchanged": sha256(source) == source_hash,
        "primary_unchanged": sha256(primary) == primary_hash,
        "zip_parts_compared": len(set(before) & set(after)),
        "missing_parts": missing,
        "extra_parts": extra,
        "different_parts": different,
        "slide_count": len(Presentation(primary).slides),
        "prior_slide_xml_preserved": 13 if prior_xml else 0,
        "scientific_status": build_report["scientific_status"],
        "summary_sha256": build_report["sources"][str((args.crossover_root.resolve() / "summary.json"))]["sha256"],
        "independent_verification_sha256": build_report["sources"][str((args.crossover_root.resolve() / "independent_verification.json"))]["sha256"],
        "base_commit": build_report["base_commit"],
        "crossover_commit": build_report["crossover_commit"],
    }
    atomic_json(args.verification.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed and report["source_unchanged"] and report["primary_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
