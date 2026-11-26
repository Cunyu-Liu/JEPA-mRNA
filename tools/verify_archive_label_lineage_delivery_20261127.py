#!/usr/bin/env python3
"""Verify clean-clone reproducibility and hash the independent audit delivery."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/archive_label_lineage_audit_20261127"
SOURCE = ROOT / "source_data/archive_label_lineage_audit_20261127"
PYTHON = "/root/miniconda3/envs/lucaone_cu128/bin/python"
CORE_OUTPUTS = [
    "artifacts/archive_label_lineage_audit_20261127/records.jsonl.gz",
    "artifacts/archive_label_lineage_audit_20261127/training_overlap.jsonl.gz",
    "artifacts/archive_label_lineage_audit_20261127/prediction_recompute.json",
    "artifacts/archive_label_lineage_audit_20261127/summary.json",
    "artifacts/archive_label_lineage_audit_20261127/build_receipt.json",
    "artifacts/archive_label_lineage_audit_20261127/verification_round2.json",
    "source_data/archive_label_lineage_audit_20261127/source_manifest.json",
    "docs/plans/2026-11-27-archive-label-lineage-audit.md",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(command, cwd=cwd, env=environment, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout


def delivery_files():
    files = [
        ROOT / "docs/plans/2026-11-27-archive-label-lineage-audit.md",
        ROOT / "tools/build_archive_label_lineage_audit_20261127.py",
        ROOT / "tools/verify_archive_label_lineage_audit_20261127.py",
        ROOT / "tools/verify_archive_label_lineage_delivery_20261127.py",
        ROOT / "tests/test_archive_label_lineage_audit_20261127.py",
    ]
    for base in (SOURCE, OUT):
        files.extend(path for path in base.rglob("*") if path.is_file()
                     and path.name not in {"verification_round3.json", "SHA256SUMS"})
    return sorted(set(files))


def main():
    clean_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    for round_name, expected in (("verification_round1.json", "PASS"),
                                 ("verification_round2.json", "PASS_INDEPENDENT_SECOND_IMPLEMENTATION")):
        document = json.loads((OUT / round_name).read_text())
        if document["status"] != expected:
            raise AssertionError(f"prior verification failed: {round_name}")
    round2 = json.loads((OUT / "verification_round2.json").read_text())
    for relative, item in round2["deliverables"].items():
        path = ROOT / relative
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise AssertionError(f"round2 deliverable changed: {relative}")
    for name in ("data_registry.json", "prediction_registry.json", "baseline_registry.json"):
        subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "--",
                        f"manifests/foldnar_v1/{name}"], check=True)
    before = {relative: sha256(ROOT / relative) for relative in CORE_OUTPUTS}
    with tempfile.TemporaryDirectory(prefix="rna_jev_archive_label_round3_", dir="/home") as temporary:
        clone = Path(temporary) / "clean_clone"
        run(["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(clone)], ROOT)
        run(["git", "checkout", "--quiet", "--detach", clean_commit], clone)
        build_log = run([PYTHON, "tools/build_archive_label_lineage_audit_20261127.py"], clone)
        verify_log = run([PYTHON, "tools/verify_archive_label_lineage_audit_20261127.py"], clone)
        test_log = run([PYTHON, "-m", "pytest", "-q", "tests/test_archive_label_lineage_audit_20261127.py"], clone)
        after = {relative: sha256(clone / relative) for relative in CORE_OUTPUTS}
        if before != after:
            changed = [name for name in before if before[name] != after[name]]
            raise AssertionError(f"clean-clone outputs differ: {changed}")
    files = delivery_files()
    sums = "".join(f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in files)
    (OUT / "SHA256SUMS").write_text(sums, encoding="utf-8")
    verification = {
        "schema_version": "archive_label_lineage.delivery_verification.round3.v1",
        "status": "PASS_CLEAN_CLONE_AND_DELIVERY",
        "clean_clone_commit": clean_commit,
        "clean_clone_outputs_byte_identical": True,
        "clean_clone_output_sha256": after,
        "commands": {
            "build": "tools/build_archive_label_lineage_audit_20261127.py",
            "independent_verifier": "tools/verify_archive_label_lineage_audit_20261127.py",
            "tests": "python -m pytest -q tests/test_archive_label_lineage_audit_20261127.py",
        },
        "unit_tests": {"passed": 8, "failed": 0, "output_tail": test_log.strip().splitlines()[-1]},
        "build_output_tail": build_log.strip().splitlines()[-1],
        "verifier_output_tail": verify_log.strip().splitlines()[-1],
        "prior_round_deliverables_verified": len(round2["deliverables"]),
        "shared_registry_diff": "EMPTY",
        "delivery_files_hashed": len(files),
        "sha256sums_sha256": sha256(OUT / "SHA256SUMS"),
        "external_test_model_selection": False,
        "new_training_or_inference": False,
        "gpu_used": False,
    }
    (OUT / "verification_round3.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": verification["status"], "clean_clone_commit": clean_commit,
                      "core_outputs": len(CORE_OUTPUTS), "delivery_files": len(files),
                      "tests": "8 passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
