#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

from contextfold_official_recovery_20261126 import ARTIFACT_ROOT, load_dataset, sha256

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "manifests/foldnar_v2/contextfold_official_recovery_20261126.json"
REGISTRY = REPO / "manifests/foldnar_v1/baseline_registry.json"


def verify_warc_pdf(root: Path, stem: str, expected_pdf_sha: str, expected_url: str) -> dict:
    compressed = root / f"{stem}.warc.gz"
    raw = gzip.decompress(compressed.read_bytes())
    headers = raw[:raw.index(b"%PDF-")].decode("latin1")
    if f"WARC-Target-URI: {expected_url}\r\n" not in headers or "HTTP/1.1 200 OK\r\n" not in headers:
        raise ValueError(f"Unexpected WARC identity for {stem}")
    length = int(re.findall(r"Content-Length: (\d+)", headers)[-1])
    start = raw.index(b"%PDF-")
    pdf = raw[start:start + length]
    if len(pdf) != length or not pdf.endswith(b"%%EOF") or raw[start + length:] != b"\r\n\r\n":
        raise ValueError(f"Invalid PDF response boundaries for {stem}")
    if (root / f"{stem}.pdf").read_bytes() != pdf:
        raise ValueError(f"Extracted PDF differs from WARC payload for {stem}")
    if hashlib.sha256(pdf).hexdigest() != expected_pdf_sha:
        raise ValueError(f"PDF SHA256 mismatch for {stem}")
    return {"warc_sha256": sha256(compressed), "pdf_sha256": expected_pdf_sha, "pdf_bytes": len(pdf)}


def verify(repo: Path = REPO, artifact_root: Path = ARTIFACT_ROOT) -> dict:
    document = json.loads((repo / MANIFEST.relative_to(REPO)).read_text())
    if document["full_ready"] is not False or not document["status"].startswith("BLOCKED_"):
        raise ValueError("This verifier covers the explicit blocked recovery state")
    evidence = document["official_evidence"]
    root = artifact_root / "evidence"
    expected_files = {
        "official_homepage_20241029125313.html": evidence["homepage"]["sha256"],
        "crossref_doi.json": document["paper"]["crossref_sha256"],
        "pubmed_22035327.xml": document["paper"]["pubmed_xml_sha256"],
        "readme.warc.gz": evidence["readme"]["warc_sha256"],
        "resources.warc.gz": evidence["resources"]["warc_sha256"],
    }
    for name, expected in expected_files.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Evidence SHA256 mismatch: {name}")
    homepage = (root / "official_homepage_20241029125313.html").read_text()
    if "ContextFold_1_00.zip" not in homepage or "StHighCoHigh" not in homepage:
        raise ValueError("Homepage no longer supports the official archive/model identity")
    warc = {
        stem: verify_warc_pdf(root, stem, evidence[stem]["pdf_sha256"], evidence[stem]["original_url"])
        for stem in ("readme", "resources")
    }
    expected_archive = Path(document["implementation"]["archive"]["expected_path"])
    if expected_archive.exists():
        raise ValueError("ZIP appeared after the blocked manifest was frozen; re-verify it instead of accepting stale status")
    if document["implementation"]["archive"]["sha256"] is not None:
        raise ValueError("Blocked manifest must not assert an archive SHA")
    contract = document["bprna_new_5388_contract"]
    if contract["predictions"] is not None or contract["prediction_sha256"] is not None:
        raise ValueError("Blocked recovery must not contain predictions")
    rows = load_dataset(contract["dataset_path"])
    registry = json.loads((repo / REGISTRY.relative_to(REPO)).read_text())
    row = next(row for row in registry["rows"] if row["baseline_id"] == "contextfold")
    if row["prediction_ids"] or row["n_frozen_prediction_rows"] or row["n_summary_only_rows"]:
        raise ValueError("Unified registry contains a ContextFold prediction substitute")
    if row["category"] != "STRUCTURED_DECODER" or row["official_weights"]["sha256"] is not None:
        raise ValueError("Unified registry identity mismatch")
    return {
        "status": "PASS_BLOCKED_OFFICIAL_RECOVERY_CONTRACT",
        "full_ready": False,
        "n_dataset_rows": len(rows),
        "evidence": warc,
        "predictions": None,
        "gpu_execution": False,
    }


def main() -> None:
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
