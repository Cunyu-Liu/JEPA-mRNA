#!/usr/bin/env python3
"""Audit data provenance, leakage layers, truncation, and baseline coverage.

This is an audit, not a benchmark run. No external performance value is used to
select a checkpoint, decoder, model, or route.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

CANONICAL = {"AU", "UA", "GC", "CG", "GU", "UG"}
MAIN_DATASETS = [
    "archiveii_embok_clean", "bprna_new", "bprna_ts0", "ref_pdb_ts1",
    "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "ref_synthetic_test",
]
ADDITIONAL_DATASETS = ["stransfer", "pdb669", "rfam_fam", "rfam_temporal"]
CHANRG_DATASETS = [
    "chanrg_train_train", "chanrg_train_validation", "chanrg_test",
    "chanrg_gena", "chanrg_genc", "chanrg_genf",
]
EXPECTED_RAW_CHANRG_COUNTS = {
    "chanrg_train_train": 123223, "chanrg_train_validation": 14070,
    "chanrg_test": 14070, "chanrg_gena": 12499, "chanrg_genc": 4424, "chanrg_genf": 1797,
}
CORE_BASELINES = ["vienna_mfe", "vienna_centroid", "vienna_mea", "nussinov_turner"]
OTHER_BASELINES = [
    "ufold", "eternafold", "mxfold2", "nucleicbert", "rnaformer",
    "rikenv2", "rnastructure", "contrafold", "probknot", "spotrna", "ipknot",
    "bpfold", "efold", "kirigami", "contextfold", "e2efold", "debfold",
    "rinalmo", "rnafm", "ernierna", "structrfm", "hyenarna",
    "xens2", "plan_a", "r2d", "h0",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sequence(value: str) -> str:
    return str(value).upper().replace("T", "U")


def record_name(record: dict) -> str:
    return str(record.get("name", record.get("id", "")))


def load_dataset(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def manifest_path(root: Path, name: str) -> Path:
    standard = root / f"{name}.manifest.json"
    if standard.exists():
        return standard
    split_manifest = root / f"{name}_manifest.json"
    if split_manifest.exists():
        return split_manifest
    if name in ("chanrg_train_train", "chanrg_train_validation"):
        return root / "chanrg_train_manifest.json"
    return standard


def dataset_distribution(name: str, records: list[dict]) -> dict:
    length_hist = Counter()
    gc_hist = Counter()
    pair_span_hist = Counter()
    pair_types = Counter()
    helix_len_hist = Counter()
    loop_types = Counter()
    junction_orders = Counter()
    exterior_roots = Counter()
    family_counts = Counter()
    clan_counts = Counter()
    total_length = total_gc = total_pairs = total_helices = 0
    n_multiloop_records = n_pseudoknot_records = n_crossing_remaining = 0
    for row in records:
        seq = normalize_sequence(row.get("seq", row.get("sequence", "")))
        n = len(seq)
        total_length += n
        gc = sum(base in "GC" for base in seq)
        total_gc += gc
        length_hist["<=100" if n <= 100 else "101-200" if n <= 200 else "201-400" if n <= 400 else
                    "401-600" if n <= 600 else "601-1024" if n <= 1024 else ">1024"] += 1
        fraction = gc / n if n else 0.0
        gc_hist["<0.25" if fraction < 0.25 else "0.25-0.50" if fraction < 0.50 else
                "0.50-0.75" if fraction < 0.75 else ">=0.75"] += 1
        pairs = {tuple(sorted((int(i), int(j)))) for i, j in row.get("pairs", [])
                 if 0 <= int(i) < int(j) < n}
        total_pairs += len(pairs)
        for i, j in pairs:
            span = j - i
            pair_span_hist["4-10" if span <= 10 else "11-20" if span <= 20 else "21-50" if span <= 50 else
                           "51-100" if span <= 100 else "101-200" if span <= 200 else "201+"] += 1
            pair_types[seq[i] + seq[j]] += 1
        for i, j in pairs:
            if (i - 1, j + 1) in pairs:
                continue
            stem_length = 0
            while (i + stem_length, j - stem_length) in pairs:
                stem_length += 1
            total_helices += 1
            helix_len_hist["1" if stem_length == 1 else "2" if stem_length == 2 else
                           "3" if stem_length == 3 else "4-5" if stem_length <= 5 else
                           "6-10" if stem_length <= 10 else "11+"] += 1
        ordered = sorted(pairs, key=lambda pair: (pair[1] - pair[0], pair[0]))
        children = {pair: [] for pair in ordered}
        roots = []
        for index, pair in enumerate(ordered):
            i, j = pair
            parent = next((outer for outer in ordered[index + 1:]
                           if outer[0] <= i and j <= outer[1]), None)
            if parent is None:
                roots.append(pair)
            else:
                children[parent].append(pair)
        exterior_roots[len(roots)] += 1
        has_multiloop = False
        for i, j in ordered:
            direct = sorted(children[(i, j)])
            if not direct:
                loop_type, order = "HAIRPIN", 0
            elif len(direct) == 1:
                a, b = direct[0]
                gaps = (a - i - 1) + (j - b - 1)
                loop_type = "STACK" if gaps == 0 else "BULGE" if a == i + 1 or b == j - 1 else "INTERNAL"
                order = 1
            else:
                loop_type, order = "MULTILOOP", len(direct) + 1
                has_multiloop = True
            loop_types[loop_type] += 1
            junction_orders[str(order)] += 1
        n_multiloop_records += has_multiloop
        n_pseudoknot_records += bool(row.get("is_pseudoknot", False))
        n_crossing_remaining += int(row.get("n_crossing", 0))
        if row.get("family") is not None:
            family_counts[str(row["family"])] += 1
        if row.get("clan") is not None:
            clan_counts[str(row["clan"])] += 1
    label_source = "SIMULATED" if "synthetic" in name else "DATABASE; EXPERIMENTAL_VS_COMPARATIVE_UNCHECKED_PER_RECORD"
    return {
        "length_histogram": dict(length_hist), "gc_histogram": dict(gc_hist),
        "gc_fraction_base_weighted": total_gc / total_length if total_length else None,
        "pair_density_per_nt": total_pairs / total_length if total_length else None,
        "pair_span_histogram": dict(pair_span_hist), "pair_type_counts": dict(pair_types),
        "n_helices": total_helices, "helix_length_histogram": dict(helix_len_hist),
        "loop_type_counts": dict(loop_types), "junction_order_counts": dict(junction_orders),
        "exterior_root_pair_count_histogram": dict(exterior_roots),
        "n_records_with_multiloop": n_multiloop_records,
        "n_records_with_pseudoknot_metadata": n_pseudoknot_records,
        "n_crossing_remaining": n_crossing_remaining,
        "n_families": len(family_counts), "n_clans": len(clan_counts),
        "top_families": family_counts.most_common(20), "top_clans": clan_counts.most_common(20),
        "label_source_class": label_source,
        "cluster_80_status": "UNCHECKED: no per-record 80% cluster assignments in JSONL",
        "experimental_quality": "UNCHECKED: mmCIF method/resolution/chain/modified-parent mapping is not in aggregated JSONL",
    }


def summarize_dataset(name: str, path: Path, records: list[dict]) -> dict:
    lengths = [len(normalize_sequence(row.get("seq", row.get("sequence", "")))) for row in records]
    manifest_file = manifest_path(path.parent, name)
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    actual_hash = sha256_file(path) if path.exists() else None
    expected_hash = manifest.get("out_sha256")
    if name == "chanrg_train_train":
        expected_hash = manifest.get("train_sha256")
    elif name == "chanrg_train_validation":
        expected_hash = manifest.get("val_sha256")
    provenance_status = (
        "BLOCKED_DATA" if not records else
        "BLOCKED_PROVENANCE" if not manifest else
        "PASS" if expected_hash == actual_hash else
        "UNVERIFIED_HASH" if expected_hash is None else "FAIL_HASH"
    )
    return {
        "dataset_id": name, "path": str(path), "sha256": actual_hash,
        "n_sequences": len(records),
        "n_unique_sequences": len({normalize_sequence(row.get("seq", row.get("sequence", ""))) for row in records}),
        "n_pairs": sum(len(row.get("pairs", [])) for row in records),
        "max_length": max(lengths, default=0),
        "n_over_512": sum(value > 512 for value in lengths),
        "n_over_600": sum(value > 600 for value in lengths),
        "n_over_1024": sum(value > 1024 for value in lengths),
        "sources": dict(Counter(str(row.get("source", "UNSPECIFIED")) for row in records)),
        "n_dropped_illegal_metadata": sum(int(row.get("n_dropped_illegal", 0)) for row in records),
        "n_dropped_crossing_metadata": sum(int(row.get("n_dropped_crossing", 0)) for row in records),
        "manifest_path": str(manifest_file) if manifest_file.exists() else None,
        "manifest": manifest, "provenance_status": provenance_status,
        "raw_expected_n_sequences": EXPECTED_RAW_CHANRG_COUNTS.get(name),
        "processed_over_raw_expected": (
            len(records) / EXPECTED_RAW_CHANRG_COUNTS[name] if name in EXPECTED_RAW_CHANRG_COUNTS else None),
        "distribution": dataset_distribution(name, records),
        "modified_parent_mmcif_qa": "BLOCKED_PROVENANCE: per-record parent mapping is not present in the aggregated JSONL",
    }


def leakage_audit(train: list[dict], test: list[dict], family_available: bool) -> dict:
    train_names = {record_name(row) for row in train}
    test_names = {record_name(row) for row in test}
    train_sequences = {normalize_sequence(row.get("seq", row.get("sequence", ""))) for row in train}
    test_sequences = {normalize_sequence(row.get("seq", row.get("sequence", ""))) for row in test}
    name_overlap = sorted(train_names & test_names)
    sequence_overlap = sorted(train_sequences & test_sequences)
    result = {
        "contract_status": "EXACT" if sequence_overlap else "UNCHECKED",
        "clean_view_policy": {
            "raw": "summarized separately", "own_clean": "not a substitute for matched-clean",
            "matched_clean": "NOT_RUN; no cross-clean subset ranking is permitted",
        },
        "name": {
            "status": "FAIL_NAME_OVERLAP" if name_overlap else "PASS_EXACT_NAME_CHECK",
            "n_overlap": len(name_overlap), "examples": name_overlap[:100],
        },
        "sequence": {
            "contract_status": "EXACT" if sequence_overlap else "NOT_DETECTED_UNKNOWN",
            "status": "FAIL_EXACT_SEQUENCE_OVERLAP" if sequence_overlap else "PASS_EXACT_SEQUENCE_CHECK",
            "n_overlap": len(sequence_overlap),
            "examples_sha256": [hashlib.sha256(seq.encode()).hexdigest() for seq in sequence_overlap[:100]],
        },
        "family": {"contract_status": "UNCHECKED", "status": "BLOCKED_PROVENANCE",
                   "reason": "structured family labels are absent from TR1c JSONL records"},
        "near_duplicate": {
            "contract_status": "UNCHECKED",
            "status": "BLOCKED_NOT_RUN_NO_ALIGNER",
            "reason": "no mmseqs/cd-hit-est/blastn/vsearch executable was found; exact matching is not a near-duplicate audit",
            "available_aligners": {name: shutil.which(name) for name in ("mmseqs", "cd-hit-est", "blastn", "vsearch")},
        },
        "pretraining_exposure": {
            "contract_status": "UNCHECKED",
            "reason": "RNAcentral/Rfam/Ensembl/nt exposure is not derivable from downstream JSONL exact matches",
        },
        "topology_similarity": {"contract_status": "UNCHECKED", "reason": "no structure-similarity search was run"},
        "temporal_and_label_provenance": {"contract_status": "UNCHECKED"},
    }
    if family_available:
        train_families = {str(row.get("family")) for row in train if row.get("family") is not None}
        test_families = {str(row.get("family")) for row in test if row.get("family") is not None}
        overlap = sorted(train_families & test_families)
        result["family"] = {
            "contract_status": "HOMOLOGY" if overlap else "NOT_DETECTED_UNKNOWN",
            "status": "FAMILY_OVERLAP_PRESENT" if overlap else "PASS_FAMILY_ID_CHECK",
            "n_train_families": len(train_families), "n_test_families": len(test_families),
            "n_overlap": len(overlap), "overlap": overlap,
            "note": "family overlap is descriptive for CHANRG random test; GenA/C/F holdouts are interpreted by their declared split, not this count alone",
        }
        if overlap and not sequence_overlap:
            result["contract_status"] = "HOMOLOGY"
    return result


def cap_pair_stats(records: list[dict], cap: int) -> dict:
    totals = Counter()
    for row in records:
        seq = normalize_sequence(row.get("seq", row.get("sequence", "")))
        pairs = {tuple(sorted((int(i), int(j)))) for i, j in row.get("pairs", [])}
        totals["n_pairs_original"] += len(pairs)
        within = set()
        for i, j in pairs:
            if i >= cap or j >= cap:
                totals["n_dropped_truncation"] += 1
                continue
            totals["n_pairs_within_cap"] += 1
            if not (0 <= i < j < len(seq)) or j - i < 4 or seq[i] + seq[j] not in CANONICAL:
                totals["n_dropped_invalid_or_minloop"] += 1
                continue
            within.add((i, j))
        ordered = sorted(within)
        retained = set()
        for index, (i, j) in enumerate(ordered):
            crossing = any((a < i < b < j) or (i < a < j < b)
                           for a, b in ordered[:index])
            if crossing:
                totals["n_dropped_crossing"] += 1
            else:
                retained.add((i, j))
        totals["n_pairs_retained"] += len(retained)
        totals["n_sequences_truncated"] += len(seq) > cap
    totals["task_truncation_max_recall"] = (
        totals["n_pairs_within_cap"] / totals["n_pairs_original"]
        if totals["n_pairs_original"] else None
    )
    totals["model_mask_max_recall"] = (
        totals["n_pairs_retained"] / totals["n_pairs_original"]
        if totals["n_pairs_original"] else None
    )
    return dict(totals)


def read_baseline_file(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def baseline_entry(path: Path, method: str, expected_n: int, identity: str) -> dict:
    base = {"method": method, "identity": identity, "path": str(path), "exists": path.exists()}
    if not path.exists():
        return {**base, "status": "MISSING_RESULT"}
    document = read_baseline_file(path)
    methods = document.get("baselines", {})
    observed_n = int(document.get("n_sequences", document.get("n", methods.get(method, {}).get("n_sequences", -1))))
    return {
        **base, "sha256": sha256_file(path), "observed_n_sequences": observed_n,
        "expected_n_sequences": expected_n,
        "status": "AVAILABLE_AUDITED_FILE" if observed_n == expected_n and method in methods else "MISMATCH_SPLIT_OR_METHOD",
        "protocol": document.get("protocol"),
        "source": methods.get(method, {}).get("source"),
        "note": "coverage only; values were not used for model or decoder selection",
    }


def baseline_coverage(root: Path, dataset: str, n_sequences: int) -> dict:
    entries = []
    core_path = root / f"baselines_{dataset}.json"
    for method in CORE_BASELINES:
        entries.append(baseline_entry(core_path, method, n_sequences, "core"))
    for method in OTHER_BASELINES:
        if method == "ufold" and dataset == "bprna_ts0":
            path = root / f"baselines_ufold_ref_bprna_ts0.json"
            entries.append(baseline_entry(path, method, n_sequences, "ref_split_mismatch_candidate"))
            continue
        if method == "rnaformer":
            exact = root / f"baselines_rnaformer_{dataset}.json"
            variants = sorted(path for path in root.glob(f"baselines_rnaformer*_{dataset}.json"))
            if exact.exists() and len(variants) == 1:
                entries.append(baseline_entry(exact, method, n_sequences, "rnaformer_exact"))
            elif variants:
                for path in variants:
                    entry = baseline_entry(path, method, n_sequences, path.stem)
                    entry["status"] = "BLOCKED_MODEL_IDENTITY_MULTIPLE_VARIANTS" if len(variants) > 1 else "NONUNIFORM_VARIANT"
                    entries.append(entry)
            else:
                entries.append(baseline_entry(exact, method, n_sequences, "rnaformer_exact"))
            continue
        path = root / f"baselines_{method}_{dataset}.json"
        entries.append(baseline_entry(path, method, n_sequences, "exact_method_dataset"))
    counts = Counter(entry["status"] for entry in entries)
    return {
        "dataset_id": dataset, "n_sequences": n_sequences, "entries": entries,
        "status_counts": dict(counts),
        "unified_panel_complete": all(entry["status"] == "AVAILABLE_AUDITED_FILE" for entry in entries),
        "note": "missing or nonuniform baseline coverage cannot be converted into a performance or SOTA conclusion",
    }


def frozen_split_consistency(path: Path, summaries: dict) -> dict:
    if not path.exists():
        return {"status": "BLOCKED_PROVENANCE", "path": str(path)}
    document = json.loads(path.read_text())
    rows = {}
    for name, expected in document.get("splits", {}).items():
        if name not in summaries:
            continue
        actual = summaries[name]
        match = actual["sha256"] == expected.get("sha256") and actual["n_unique_sequences"] == expected.get("n_unique")
        rows[name] = {
            "status": "MATCH" if match else "MISMATCH",
            "expected_n_unique": expected.get("n_unique"), "actual_n_unique": actual["n_unique_sequences"],
            "actual_n_records": actual["n_sequences"],
            "expected_sha256": expected.get("sha256"), "actual_sha256": actual["sha256"],
            "historical_overlap_with_training_corpora": expected.get("overlap_with_training_corpora"),
        }
    return {"status": "COMPLETE_WITH_MISMATCHES" if any(row["status"] != "MATCH" for row in rows.values()) else "MATCH",
            "path": str(path), "sha256": sha256_file(path), "splits": rows}


def chanrg_result_coverage(root: Path, summaries: dict) -> dict:
    output = {}
    for name in ("chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"):
        entries = []
        for path in sorted((root / name).glob("*/result.json")):
            document = json.loads(path.read_text())
            observed_n = int(document.get("n_sequences", document.get("n", -1)))
            processed_n = summaries[name]["n_sequences"]
            identity_ambiguous = any(key in document for key in ("plana_ckpt2", "w_plana", "plana_s1_weight"))
            entries.append({
                "predictor": path.parent.name, "path": str(path), "sha256": sha256_file(path),
                "observed_n_sequences": observed_n, "processed_n_sequences": processed_n,
                "raw_expected_n_sequences": EXPECTED_RAW_CHANRG_COUNTS[name],
                "status": ("HISTORICAL_NONUNIFORM_MODEL_IDENTITY" if identity_ambiguous else
                           "AVAILABLE_PROCESSED_SUBSET_ONLY" if observed_n == processed_n else "MISMATCH_SPLIT"),
                "decode": document.get("decode", {"mode": document.get("mode")}),
                "coverage_scope": "processed subset; not the raw official full panel",
            })
        output[name] = {
            "entries": entries, "n_result_files": len(entries),
            "full_29_predictor_panel_complete": False,
            "status": "INCOMPLETE_PREDICTORS_AND_RAW_COVERAGE",
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-root", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl"))
    parser.add_argument("--baseline-root", type=Path, default=Path("/root/rna_jepa_data/eval_decision_npz"))
    parser.add_argument("--eval-decision-root", type=Path, default=Path("/root/rna_jepa_data/eval_decision"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite audit: {args.out}")
    all_names = ["bprna_tr1c"] + MAIN_DATASETS + ADDITIONAL_DATASETS + CHANRG_DATASETS
    records = {name: load_dataset(args.jsonl_root / f"{name}.jsonl") for name in all_names}
    summaries = {name: summarize_dataset(name, args.jsonl_root / f"{name}.jsonl", rows)
                 for name, rows in records.items()}
    train = records["bprna_tr1c"]
    leakage = {name: leakage_audit(train, records[name], family_available=False)
               for name in MAIN_DATASETS + ADDITIONAL_DATASETS if records[name]}
    chanrg_train = records["chanrg_train_train"] + records["chanrg_train_validation"]
    for name in ("chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"):
        leakage[name] = leakage_audit(chanrg_train, records[name], family_available=True)
    chanrg_leakage_vs_tr1c = {
        name: leakage_audit(train, records[name], family_available=False)
        for name in ("chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf")
    }
    cap_audit = {name: {str(cap): cap_pair_stats(records[name], cap) for cap in (512, 600)}
                 for name in CHANRG_DATASETS + ["pdb669"] if records[name]}
    baselines = {name: baseline_coverage(args.baseline_root, name, summaries[name]["n_sequences"])
                 for name in MAIN_DATASETS}
    blocked = [name for name, summary in summaries.items() if summary["provenance_status"] != "PASS"]
    output = {
        "schema_version": 1, "status": "AUDIT_COMPLETE_WITH_BLOCKED_ITEMS",
        "scope": "DATA_BASELINE_COVERAGE_AUDIT_ONLY_NOT_PERFORMANCE_OR_SOTA",
        "old_external_panels_used_for_selection": False,
        "sota_claim": False, "production_repair": False,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "data_registry": summaries,
        "leakage": leakage,
        "chanrg_leakage_vs_tr1c": chanrg_leakage_vs_tr1c,
        "truncation_theoretical_recall": cap_audit,
        "baseline_coverage": baselines,
        "chanrg_result_coverage": chanrg_result_coverage(args.eval_decision_root, summaries),
        "frozen_split_consistency": frozen_split_consistency(args.jsonl_root.parent / "eval_splits_frozen.json", summaries),
        "additional_asset_coverage": {
            "d1_pdb_dssr_1805_entries": {"status": "BLOCKED_DATA", "reason": "no corresponding registered JSONL was audited"},
            "long_7224_19000_nt": {
                "status": "BLOCKED_DATA",
                "matching_files": [str(path) for path in args.jsonl_root.iterdir()
                                   if any(token in path.name.lower() for token in ("7224", "19000", "long"))],
                "note": "pdb669 max length is reported independently and does not substitute for 7224/19000 nt assets",
            },
        },
        "strict_ood_status": "BLOCKED_PENDING_FAMILY_NEAR_DUPLICATE_AND_UNIFIED_BASELINE_COMPLETION",
        "provenance_blocked_datasets": blocked,
        "interpretation": [
            "PASS_EXACT_SEQUENCE_CHECK applies only to exact normalized sequences.",
            "Exact decontamination in a manifest does not prove name, family, or near-duplicate decontamination.",
            "Task truncation recall and model-mask recall are separate pair-retention ceilings, not model performance.",
            "No performance panel was run or selected by this audit.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=False)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    concise = {
        "status": output["status"], "out": str(args.out),
        "n_datasets": len(summaries), "provenance_blocked_datasets": blocked,
        "baseline_panels_complete": sum(panel["unified_panel_complete"] for panel in baselines.values()),
        "strict_ood_status": output["strict_ood_status"],
        "chanrg_600": {name: cap_audit[name]["600"] for name in cap_audit if name.startswith("chanrg_")},
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
