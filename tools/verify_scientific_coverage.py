#!/usr/bin/env python3
"""Independent consistency checks for scientific coverage audit artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ALLOWED_LEAKAGE = {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"}


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite verification: {args.out}")
    document = json.loads(args.document.read_text())
    checks = {
        "audit_not_sota": document["sota_claim"] is False and document["production_repair"] is False,
        "no_external_selection": document["old_external_panels_used_for_selection"] is False,
        "strict_ood_blocked": document["strict_ood_status"].startswith("BLOCKED"),
        "generator_hash": sha256_file(Path(__file__).resolve().with_name("audit_scientific_coverage.py")) == document["generator_sha256"],
    }
    registry_checks = {}
    for name, row in document["data_registry"].items():
        path = Path(row["path"])
        valid = path.exists() and path.stat().st_size > 0
        hash_ok = (sha256_file(path) == row["sha256"]) if valid else row["n_sequences"] == 0
        n_ok = True
        unique_ok = True
        if valid:
            sequences = []
            with path.open() as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            n_ok = len(rows) == row["n_sequences"]
            unique_ok = len({str(item.get("seq", item.get("sequence", ""))).upper().replace("T", "U")
                             for item in rows}) == row["n_unique_sequences"]
        distribution = row["distribution"]
        dist_ok = sum(distribution["length_histogram"].values()) == row["n_sequences"] and unique_ok
        if row["raw_expected_n_sequences"] is not None:
            dist_ok = dist_ok and abs(row["processed_over_raw_expected"] - row["n_sequences"] / row["raw_expected_n_sequences"]) < 1e-12
        registry_checks[name] = bool(hash_ok and n_ok and dist_ok)
    checks["data_registry_recomputed"] = all(registry_checks.values())
    cap_checks = {}
    for name, caps in document["truncation_theoretical_recall"].items():
        for cap, row in caps.items():
            denominator = row["n_pairs_original"]
            cap_checks[f"{name}:{cap}"] = (
                abs(row["task_truncation_max_recall"] - row["n_pairs_within_cap"] / denominator) < 1e-12 and
                abs(row["model_mask_max_recall"] - row["n_pairs_retained"] / denominator) < 1e-12
            ) if denominator else True
    checks["cap_recall_arithmetic"] = all(cap_checks.values())
    checks["leakage_contract_states"] = all(
        row["contract_status"] in ALLOWED_LEAKAGE and
        row["sequence"]["contract_status"] in ALLOWED_LEAKAGE and
        row["family"]["contract_status"] in ALLOWED_LEAKAGE and
        row["near_duplicate"]["contract_status"] in ALLOWED_LEAKAGE
        for row in list(document["leakage"].values()) + list(document["chanrg_leakage_vs_tr1c"].values()))
    checks["baseline_coverage_flags"] = all(
        panel["unified_panel_complete"] == all(entry["status"] == "AVAILABLE_AUDITED_FILE" for entry in panel["entries"])
        for panel in document["baseline_coverage"].values())
    checks["chanrg_not_full_panel"] = all(
        panel["full_29_predictor_panel_complete"] is False
        for panel in document["chanrg_result_coverage"].values())
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "document": str(args.document), "document_sha256": sha256_file(args.document),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "checks": checks, "registry_checks": registry_checks,
        "scope": "independent coverage-audit consistency; not a performance result",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
