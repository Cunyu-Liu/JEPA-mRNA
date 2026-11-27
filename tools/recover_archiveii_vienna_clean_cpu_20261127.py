#!/usr/bin/env python3
"""Recover record-level Vienna predictions for the ArchiveII clean 2544 panel, CPU only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import recover_archiveii_full3966_cpu_baselines_20261127 as base

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "source_data/archiveii_official_reference_20261127/official_reference_clean_2544.jsonl.gz"
PLAN = ROOT / "source_data/archiveii_vienna_full3966_cpu_20261127/clean_recovery_plan.json"
DEFAULT_WORK = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_vienna_clean_cpu_20261127")
BASELINES = ("vienna_mfe", "vienna_centroid", "vienna_mea")


def recovery_rows():
    rows = [{"name": row["name"], "seq": row["seq"], "sequence_sha256": row["sequence_sha256"],
             "pair_set_sha256": row["pair_set_sha256"], "reference_member": row["bpfold_member"],
             "reference_member_sha256": row["bpfold_member_sha256"]} for row in base.read_gzip(CLEAN)]
    if len(rows) != 2544 or len({row["name"] for row in rows}) != 2544:
        raise ValueError("Vienna clean population changed")
    return rows


def build_plan():
    rows = recovery_rows()
    plan = {"schema_version": "archiveii.vienna_clean_cpu_recovery.v1.plan",
            "source_panel": str(CLEAN.relative_to(ROOT)), "source_panel_sha256": base.sha256(CLEAN),
            "n_records": len(rows), "non_acgu_records": sum(bool(set(row["seq"]) - set("ACGU")) for row in rows),
            "id_order_sha256": base.value_sha([row["name"] for row in rows]),
            "policy": {"compute": "CPU only; one thread per worker; no GPU visibility",
                       "non_acgu": "REJECTED_NON_ACGU without normalization or prediction",
                       "checkpoint": "one validated JSON per ID; resume skips only matching identity"},
            "baselines": list(BASELINES), "rows": rows}
    base.write_json(PLAN, plan)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=BASELINES)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--build-plan", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    plan = build_plan()
    if args.build_plan:
        print(json.dumps({"status": "PLAN_READY", "n_records": plan["n_records"],
                          "id_order_sha256": plan["id_order_sha256"]}, sort_keys=True))
        return
    if not args.baseline:
        parser.error("--baseline is required unless --build-plan")
    if args.status_only or args.consolidate_only:
        coverage = base.consolidate(args.work_root, args.baseline, plan["rows"])
        print(json.dumps(coverage, sort_keys=True))
        return
    coverage = base.run_baseline(args.baseline, args.work_root, plan["rows"], 1, args.limit)
    print(json.dumps(coverage, sort_keys=True))


if __name__ == "__main__":
    main()
