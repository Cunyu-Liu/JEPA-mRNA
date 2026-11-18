#!/usr/bin/env python3
"""Per-family CHANRG label attrition under the canonical nested task definition."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from pair_coverage_audit import dataset_rows, name_of, normalize_pairs, pair_geometry, seq_of, sha256  # noqa: E402

DATASETS = ("chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf")


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _family(row):
    return str(row.get("family") or row.get("rfam_id") or row.get("rfam") or "UNKNOWN")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=REPO / "manifests/foldnar_v1/data_registry.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registry = {row["dataset_id"]: row for row in json.loads(args.registry.read_text())["rows"]}
    args.out.mkdir(parents=True, exist_ok=True)
    per_sequence, family_rows, summaries = [], [], {}
    for dataset_id in DATASETS:
        spec = registry[dataset_id]
        grouped = {}
        counts = Counter()
        for line_number, row in dataset_rows(registry, dataset_id):
            seq = seq_of(row)
            pairs, duplicate, invalid = normalize_pairs(row.get("pairs"), len(seq))
            features = pair_geometry(seq, pairs)
            family = _family(row)
            clan = str(row.get("clan") or "UNKNOWN")
            values = Counter()
            values["n_records"] = 1
            values["n_raw_gt_pairs"] = len(pairs)
            values["n_duplicate_pairs"] = duplicate
            values["n_invalid_pairs"] = invalid
            for feature in features.values():
                canonical = feature["canonical"]
                nested = feature["nested"]
                one_to_one = not feature["partner_conflict"]
                values["n_canonical_pairs"] += int(canonical)
                values["n_canonical_nested_pairs"] += int(canonical and nested)
                values["n_canonical_nested_one_to_one_pairs"] += int(canonical and nested and one_to_one)
                values["n_task_legal_pairs"] += int(feature["legal_mask"] and nested and one_to_one)
                values["n_crossing_pairs"] += int(feature["crossing"])
                values["n_partner_conflict_pairs"] += int(feature["partner_conflict"])
                values["n_noncanonical_pairs"] += int(not canonical)
                values["n_min_loop_fail_pairs"] += int(not feature["min_loop"])
            values["n_task_removed_pairs"] = values["n_raw_gt_pairs"] - values["n_task_legal_pairs"]
            record = {"dataset_id": dataset_id, "name": name_of(row, line_number), "family": family,
                      "clan": clan, "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
                      "length": len(seq), "source": row.get("source"),
                      "n_dropped_crossing_recorded": row.get("n_dropped_crossing"),
                      "n_dropped_illegal_recorded": row.get("n_dropped_illegal"), **dict(values)}
            record["task_removed_fraction"] = _ratio(values["n_task_removed_pairs"], values["n_raw_gt_pairs"])
            record["task_native_theoretical_max_recall"] = _ratio(values["n_task_legal_pairs"], values["n_raw_gt_pairs"])
            per_sequence.append(record)
            counts.update(values)
            key = (family, clan)
            if key not in grouped:
                grouped[key] = Counter()
            grouped[key].update(values)
        for (family, clan), values in sorted(grouped.items()):
            raw = values["n_raw_gt_pairs"]
            family_rows.append({"dataset_id": dataset_id, "family": family, "clan": clan, **dict(values),
                                "task_removed_fraction": _ratio(values["n_task_removed_pairs"], raw),
                                "task_native_theoretical_max_recall_micro": _ratio(values["n_task_legal_pairs"], raw),
                                "canonical_nested_retained_fraction": _ratio(values["n_canonical_nested_one_to_one_pairs"], raw),
                                "unified_canonical_nested_denominator": values["n_task_legal_pairs"],
                                "unified_canonical_nested_theoretical_max_recall": 1.0 if values["n_task_legal_pairs"] else None})
        raw = counts["n_raw_gt_pairs"]
        summaries[dataset_id] = {**dict(counts), "n_families": len(grouped),
                                 "task_removed_fraction": _ratio(counts["n_task_removed_pairs"], raw),
                                 "task_native_theoretical_max_recall_micro": _ratio(counts["n_task_legal_pairs"], raw),
                                 "unified_canonical_nested_theoretical_max_recall": 1.0 if counts["n_task_legal_pairs"] else None,
                                 "path": spec["path"], "sha256": spec.get("sha256")}
    sequence_df = pd.DataFrame(per_sequence)
    family_df = pd.DataFrame(family_rows)
    sequence_df.to_parquet(args.out / "chanrg_per_sequence.parquet", index=False)
    family_df.to_csv(args.out / "chanrg_per_family.csv", index=False)
    report = {
        "schema_version": 1, "scope": "NO_MODEL_SELECTION_CHANRG_TASK_DEFINITION_AUDIT",
        "datasets": summaries,
        "metric_scopes": {
            "TASK_NATIVE_FROZEN_GT": "all unique valid pairs in the frozen JSONL remain in the denominator; ceiling=task_legal/raw_gt",
            "UNIFIED_CANONICAL_NESTED": "only canonical, nested, one-to-one, min-loop pairs remain; ceiling=1 within this altered denominator; removed pairs are reported separately",
        },
        "canonical_nested_definition": "canonical AU/UA/GC/CG/GU/UG, noncrossing, no partner conflict; task legal additionally requires j-i>3",
        "caution": "The two scopes answer different questions and must not be silently mixed. Parent-structure residue deletions are not inferred from JSONL.",
        "registry_sha256": sha256(args.registry), "n_sequence_rows": len(sequence_df),
        "n_family_rows": len(family_df),
    }
    (args.out / "chanrg_truncation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"datasets": summaries, "n_family_rows": len(family_df)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
