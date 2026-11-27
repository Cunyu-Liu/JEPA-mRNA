#!/usr/bin/env python3
"""Mirror final recovery evaluation for the already-running official publisher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

FINAL_FILES = ("independent_evaluation.json", "gate_summary.json", "input_manifest.json",
               "exposure_sensitivity.json")
FINAL_STATUSES = {"PASS_STRUCTURED_MARGIN_INDEPENDENT_EVAL", "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_report(source: Path) -> dict:
    report = json.loads((source / "independent_evaluation.json").read_text())
    if (report.get("status") not in FINAL_STATUSES
            or report.get("final_verdict_allowed") is not True
            or type(report.get("overall_pass")) is not bool):
        raise ValueError("recovery evaluation is not a final independent verdict")
    for name in FINAL_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    return report


def mirror_final_evaluation(source: Path, destination: Path, empty_backup: Path,
                            report_path: Path) -> dict:
    verdict = final_report(source)
    if not destination.is_dir() or any(destination.iterdir()):
        raise ValueError("publisher evaluation destination must be an empty directory")
    if empty_backup.exists() or report_path.exists():
        raise ValueError("mirror backup/report must not already exist")
    source_files = sorted(path for path in source.rglob("*") if path.is_file())
    before = {str(path.relative_to(source)): sha256_file(path) for path in source_files}
    temporary = destination.with_name(destination.name + f".mirror.tmp.{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"stale mirror temporary directory: {temporary}")
    shutil.copytree(source, temporary)
    after = {str(path.relative_to(source)): sha256_file(path) for path in source_files}
    if after != before:
        raise ValueError("recovery evaluation changed during publication mirroring")
    copied = {str(path.relative_to(temporary)): sha256_file(path)
              for path in sorted(temporary.rglob("*")) if path.is_file()}
    if copied != before:
        raise ValueError("publication mirror hash mismatch")
    os.rename(destination, empty_backup)
    os.rename(temporary, destination)
    result = {
        "schema_version": "foldnar.structured_margin_evaluation_publication_mirror.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_FINAL_EVALUATION_MIRRORED_FOR_EXISTING_PUBLISHER",
        "scientific_status": verdict["status"], "overall_pass": verdict["overall_pass"],
        "source": str(source), "destination": str(destination),
        "empty_destination_backup": str(empty_backup), "files": before,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--empty-backup", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    while True:
        try:
            result = mirror_final_evaluation(args.source, args.destination,
                                             args.empty_backup, args.report)
            print(json.dumps(result, indent=2), flush=True)
            return
        except (FileNotFoundError, json.JSONDecodeError):
            if not args.wait:
                raise
        if not args.wait:
            raise ValueError("final evaluation is not ready")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
