#!/usr/bin/env python3
"""Verify that merged frozen tables exactly contain eight independently raw-verified partitions."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import pandas as pd

EXPECTED = ["archiveii_embok_clean", "bprna_new", "bprna_ts0", "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf", "rfam_fam"]
TABLES = ["decode_per_sequence", "decode_summary", "ranking_per_sequence", "ranking_summary"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-root", type=Path, required=True)
    ap.add_argument("--merged-root", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56"))
    args = ap.parse_args()
    checks = []

    def record(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    part_reports, part_tables = [], {name: [] for name in TABLES}
    part_routes = []
    for dataset in EXPECTED:
        part = args.parts_root / dataset
        path = part / "independent_external_raw_dp_verification.json"
        report = json.loads(path.read_text())
        passed = report["status"] == "PASS" and report["n_decode_rows_recomputed"] == 5760 and report["n_ranking_rows_recomputed"] == 1920 and report["n_ranking_summary_recomputed"] == 20
        record(f"independent_raw_recompute_{dataset}", passed,
               {"status": report["status"], "n_decode_rows": report["n_decode_rows_recomputed"], "n_ranking_rows": report["n_ranking_rows_recomputed"],
                "n_pooled_summary": report["n_ranking_summary_recomputed"]})
        part_reports.append({"dataset_id": dataset, "path": str(path), "sha256": sha256(path)})
        for name in TABLES:
            part_tables[name].append(pd.read_parquet(part / f"{name}.parquet"))
        part_routes.extend(json.loads((part / "route_summary.json").read_text())["rows"])
    for name in TABLES:
        expected = pd.concat(part_tables[name], ignore_index=True)
        observed = pd.read_parquet(args.merged_root / f"{name}.parquet")
        try:
            pd.testing.assert_frame_equal(observed, expected, check_exact=True)
            passed, detail = True, {"rows": len(observed), "exact": True}
        except AssertionError as exc:
            passed, detail = False, str(exc)[:1000]
        record(f"merged_exact_{name}", passed, detail)
    observed_routes = json.loads((args.merged_root / "route_summary.json").read_text())["rows"]
    record("merged_exact_routes", observed_routes == part_routes, {"observed": len(observed_routes), "expected": len(part_routes)})
    merged_frozen = json.loads((args.merged_root / "frozen_decode_config.json").read_text())
    same_config = all(json.loads((args.parts_root / d / "frozen_decode_config.json").read_text())["config"] == merged_frozen["config"] for d in EXPECTED)
    record("merged_frozen_config_identity", same_config and merged_frozen["external_datasets"] == EXPECTED,
           {"same_config": same_config, "external_datasets": merged_frozen["external_datasets"]})
    per_sequence = pd.read_parquet(args.merged_root / "decode_per_sequence.parquet")
    unique_sequences = per_sequence[["dataset_id", "name", "score_variant"]].drop_duplicates()
    record("complete_512_sequences_10_variants", len(unique_sequences) == 5120 and set(unique_sequences.dataset_id.unique()) == set(EXPECTED),
           {"unique_dataset_sequence_variants": len(unique_sequences), "datasets": sorted(unique_sequences.dataset_id.unique())})
    n_decode = sum(r["n_decode_rows_recomputed"] for r in [json.loads(Path(p["path"]).read_text()) for p in part_reports])
    n_ranking = sum(r["n_ranking_rows_recomputed"] for r in [json.loads(Path(p["path"]).read_text()) for p in part_reports])
    n_pooled = sum(r["n_ranking_summary_recomputed"] for r in [json.loads(Path(p["path"]).read_text()) for p in part_reports])
    n_failed = sum(not c["passed"] for c in checks)
    output = {"schema_version": 1, "status": "PASS" if n_failed == 0 else "FAIL", "n_checks": len(checks), "n_failed": n_failed,
              "verification_scope": "each partition independently recomputed from raw NPZ; merged tables required exact equality",
              "n_sequences": 512, "n_score_variants": 10, "n_decode_rows": n_decode, "n_ranking_rows": n_ranking,
              "n_pooled_summary": n_pooled, "partition_raw_verifications": part_reports, "checks": checks,
              "merged_table_sha256": {name: sha256(args.merged_root / f"{name}.parquet") for name in TABLES},
              "code": str(Path(__file__).resolve()), "code_sha256": sha256(Path(__file__).resolve()),
              "git_head": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()}
    (args.merged_root / "independent_external_raw_dp_verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ["status", "n_checks", "n_failed", "n_decode_rows", "n_ranking_rows", "n_pooled_summary"]}, indent=2))
    if n_failed:
        print(json.dumps([c for c in checks if not c["passed"]], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
