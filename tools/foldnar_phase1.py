#!/usr/bin/env python3
"""Phase 1: RNA data QA, distributions, and four-layer leakage audit.

Exact sequence leakage and a deterministic 31-mer candidate screen are computed.
Existing independently checked known-source homology is hash-verified and joined
per sequence. Uncovered corpora, external baselines, and non-detections remain
explicit limitations; no clean verdict is inferred.
"""
from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rnajepa.fold_structure import StructureError, build_helix_tree, decompose_loops, extract_stems, topology_signature

JSONL = Path("/root/rna_jepa_data/ss_data/jsonl")
REPO = Path("/home/rna_jev_xens2_dualview_round56")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
SEARCH_TOOLS = ("minimap2", "cd-hit", "mmseqs", "blastn", "cmsearch", "vsearch", "usearch")


def normalize_sequence(seq: str) -> str:
    return str(seq).strip().upper().replace("T", "U")


def pair_problems(pairs: list, length: int) -> dict:
    problems = Counter()
    canonical = []
    used = Counter()
    for raw in pairs:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            problems["malformed_pair"] += 1
            continue
        try:
            i, j = sorted((int(raw[0]), int(raw[1])))
        except Exception:
            problems["malformed_pair"] += 1
            continue
        if i == j:
            problems["self_pair"] += 1
        if i < 0 or j >= length:
            problems["out_of_range"] += 1
        elif j - i < 4:
            problems["span_lt_4"] += 1
        if (i, j) in canonical:
            problems["duplicate_pair"] += 1
            continue
        canonical.append((i, j))
        used[i] += 1
        used[j] += 1
    problems["multi_paired_nucleotide"] = sum(v > 1 for v in used.values())
    ordered = sorted(canonical)
    crossing_ordered = [(i, j) for i, j in ordered if 0 <= i < j < length]
    crossing = set()
    for k, (i, j) in enumerate(crossing_ordered):
        for l, m in crossing_ordered[k + 1:]:
            if l >= j:
                break
            if i < l < j < m:
                crossing.add((i, j))
                crossing.add((l, m))
    problems["crossing_pair"] = len(crossing)
    problems["invalid_record"] = int(any(problems[key] for key in (
        "malformed_pair", "self_pair", "out_of_range", "duplicate_pair", "multi_paired_nucleotide"
    )))
    return {"problems": dict(problems), "pairs": ordered, "crossing": bool(crossing)}


def family_fields(row: dict) -> tuple[str | None, str | None]:
    family = row.get("family") or row.get("rfam_id") or row.get("rfam") or row.get("rfam_family")
    clan = row.get("clan") or row.get("rna_clan")
    return (str(family) if family not in (None, "") else None,
            str(clan) if clan not in (None, "") else None)


def topology_key(pairs: list) -> str:
    return json.dumps(topology_signature(build_helix_tree(pairs)), separators=(",", ":"))


def numeric_stats(values: list[float] | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "mean": None, "max": None}
    return {"n": int(len(arr)), "min": float(arr.min()), "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)), "p75": float(np.percentile(arr, 75)),
            "mean": float(arr.mean()), "max": float(arr.max())}


def read_dataset(spec: dict) -> tuple[dict, list[dict], list[tuple]]:
    path = Path(spec["path"])
    records = []
    qa = Counter()
    lengths, pair_counts, crossing_counts = [], [], []
    raw_pair_counts, gc_counts = [], []
    span_totals, span_counts, span_maxima = [], [], []
    helix_counts, helix_length_totals, helix_length_maxima = [], [], []
    multiloop_counts, multiloop_order_totals, multiloop_order_maxima = [], [], []
    family_counter, clan_counter, topology_counter, stem_counter, loop_counter = (Counter() for _ in range(5))
    per_record = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row.get("name", row.get("id", f"row:{line_number}")))
            seq = normalize_sequence(row.get("seq", row.get("sequence", "")))
            raw_pairs = row.get("pairs", [])
            checked = pair_problems(raw_pairs, len(seq))
            pairs = checked["pairs"]
            problems = checked["problems"]
            if not name:
                qa["invalid_id"] += 1
            if not seq:
                qa["empty_sequence"] += 1
            if any(base not in "ACGUN" for base in seq):
                qa["noncanonical_sequence"] += 1
            for key, value in problems.items():
                qa[key] += value
            qa["n_records"] += 1
            lengths.append(len(seq))
            pair_counts.append(len(pairs))
            raw_pair_counts.append(len(raw_pairs) if isinstance(raw_pairs, list) else 0)
            gc_counts.append(sum(base == "G" or base == "C" for base in seq))
            valid_spans = [j - i for i, j in pairs if 0 <= i < j < len(seq)]
            span_totals.append(sum(valid_spans))
            span_counts.append(len(valid_spans))
            span_maxima.append(max(valid_spans, default=0))
            helix_lengths, multiloop_orders = [], []
            crossing_counts.append(problems.get("crossing_pair", 0))
            family, clan = family_fields(row)
            if family:
                family_counter[family] += 1
            if clan:
                clan_counter[clan] += 1
            n_stems = -1
            if not checked["crossing"] and not problems.get("invalid_record") and seq:
                try:
                    stems = extract_stems(pairs)
                    loops = decompose_loops(pairs)
                    signature = topology_key(pairs)
                    n_stems = len(stems)
                    helix_lengths = [len(stem) for stem in stems]
                    multiloop_orders = [len(loop.children) for loop in loops if loop.kind == "MULTILOOP"]
                    topology_counter[signature] += 1
                    stem_counter[len(stems)] += 1
                    loop_counter.update(loop.kind for loop in loops)
                except StructureError:
                    qa["topology_error"] += 1
            else:
                signature = None
                qa["topology_not_applicable"] += 1
            helix_counts.append(len(helix_lengths))
            helix_length_totals.append(sum(helix_lengths))
            helix_length_maxima.append(max(helix_lengths, default=0))
            multiloop_counts.append(len(multiloop_orders))
            multiloop_order_totals.append(sum(multiloop_orders))
            multiloop_order_maxima.append(max(multiloop_orders, default=0))
            per_record.append((len(seq), len(pairs), problems.get("crossing_pair", 0), n_stems,
                               gc_counts[-1], raw_pair_counts[-1], span_totals[-1], span_counts[-1], span_maxima[-1],
                               helix_counts[-1], helix_length_totals[-1], helix_length_maxima[-1],
                               multiloop_counts[-1], multiloop_order_totals[-1], multiloop_order_maxima[-1]))
            raw_sequence = str(row.get("seq", row.get("sequence", "")))
            records.append({"id": name, "length": len(seq), "seq": seq,
                            "raw_sequence_sha256": hashlib.sha256(raw_sequence.encode()).hexdigest(),
                            "sha256": hashlib.sha256(seq.encode()).hexdigest(),
                            "family": family, "clan": clan, "topology": signature})
    n_unique_ids = len({r["id"] for r in records})
    n_duplicate_ids = qa["n_records"] - n_unique_ids
    if n_duplicate_ids:
        qa["duplicate_id"] = n_duplicate_ids
    arr = np.asarray(lengths, dtype=float)
    pairs_arr = np.asarray(pair_counts, dtype=float)
    gc_arr = np.asarray(gc_counts, dtype=float)
    gc_fraction = np.divide(gc_arr, arr, out=np.zeros_like(gc_arr), where=arr > 0)
    span_totals_arr, span_counts_arr = np.asarray(span_totals, dtype=float), np.asarray(span_counts, dtype=float)
    helix_counts_arr, helix_length_totals_arr = np.asarray(helix_counts, dtype=float), np.asarray(helix_length_totals, dtype=float)
    multiloop_counts_arr = np.asarray(multiloop_counts, dtype=float)
    multiloop_order_totals_arr = np.asarray(multiloop_order_totals, dtype=float)
    per_seq_mean_span = np.divide(span_totals_arr, span_counts_arr, out=np.zeros_like(span_totals_arr), where=span_counts_arr > 0)
    per_seq_mean_helix = np.divide(helix_length_totals_arr, helix_counts_arr, out=np.zeros_like(helix_counts_arr), where=helix_counts_arr > 0)
    per_seq_mean_multi_order = np.divide(multiloop_order_totals_arr, multiloop_counts_arr, out=np.zeros_like(multiloop_counts_arr), where=multiloop_counts_arr > 0)
    report = {
        "dataset_id": spec["dataset_id"], "role": spec["role"], "path": str(path),
        "scope": spec.get("scope"), "label_source": spec.get("label_source"),
        "contract_coverage_status": spec.get("contract_coverage_status"),
        "source_sha256": spec["sha256"], "n_records": qa["n_records"],
        "modified_parent_mapping": spec.get("modified_parent_mapping"),
        "experimental_metadata_status": "BLOCKED_PER_RECORD_MMCIF_ENTITY_MAPPING_REQUIRED" if spec.get("label_source") == "EXPERIMENTAL" else "NOT_APPLICABLE_OR_DATASET_LEVEL_ONLY",
        "chanrg_provenance": spec.get("chanrg_provenance"),
        "chanrg_dedup_status": spec.get("chanrg_dedup_status"),
        "n_unique_ids": n_unique_ids,
        "n_duplicate_ids": n_duplicate_ids,
        "quality_counts": dict(qa),
        "length": {"min": float(arr.min()) if len(arr) else 0, "mean": float(arr.mean()) if len(arr) else 0,
                   "median": float(np.median(arr)) if len(arr) else 0, "max": float(arr.max()) if len(arr) else 0},
        "pairs": {"total": int(pairs_arr.sum()), "mean_per_sequence": float(pairs_arr.mean()) if len(arr) else 0,
                  "paired_nucleotide_fraction": float(2 * pairs_arr.sum() / arr.sum()) if arr.sum() else 0},
        "pair_conservation": {
            "raw_pair_entries": int(sum(raw_pair_counts)), "canonical_pairs": int(pairs_arr.sum()),
            "malformed_pair_entries_dropped": int(qa["malformed_pair"]),
            "duplicate_pairs_collapsed": int(qa["duplicate_pair"]),
            "unaccounted_pair_deletion": int(sum(raw_pair_counts) - pairs_arr.sum() - qa["malformed_pair"] - qa["duplicate_pair"]),
            "note": "Only malformed entries drop out and exact duplicate canonical pairs collapse; invalid/crossing pairs remain flagged and are not silently repaired.",
            "min_loop_policy": spec.get("min_loop_policy"),
        },
        "gc_fraction": numeric_stats(gc_fraction),
        "pair_span": {**numeric_stats(span_maxima), "statistic": "per-sequence maximum span",
                      "per_sequence_mean_span": numeric_stats(per_seq_mean_span),
                      "weighted_mean_span": float(span_totals_arr.sum() / span_counts_arr.sum()) if span_counts_arr.sum() else None},
        "helices": {"count_per_sequence": numeric_stats(helix_counts_arr),
                    "per_sequence_mean_length": numeric_stats(per_seq_mean_helix),
                    "weighted_mean_length": float(helix_length_totals_arr.sum() / helix_counts_arr.sum()) if helix_counts_arr.sum() else None,
                    "max_length": int(max(helix_length_maxima, default=0))},
        "multiloop": {"count_per_sequence": numeric_stats(multiloop_counts_arr),
                      "mean_order_per_sequence": numeric_stats(per_seq_mean_multi_order),
                      "weighted_mean_order": float(multiloop_order_totals_arr.sum() / multiloop_counts_arr.sum()) if multiloop_counts_arr.sum() else None,
                      "max_order": int(max(multiloop_order_maxima, default=0)),
                      "junction_count": int(sum(multiloop_order_totals))},
        "family": {"n_labeled": sum(family_counter.values()), "n_unique": len(family_counter),
                   "top": family_counter.most_common(30)},
        "clan": {"n_labeled": sum(clan_counter.values()), "n_unique": len(clan_counter), "top": clan_counter.most_common(30)},
        "topology": {"n_valid_noncrossing": sum(topology_counter.values()), "n_unique": len(topology_counter),
                     "top": topology_counter.most_common(20)},
        "stem_count_histogram": dict(sorted(stem_counter.items())),
        "loop_counts": dict(loop_counter),
        "length_bins": {str(k): int(v) for k, v in Counter(
            "0-63" if x < 64 else "64-127" if x < 128 else "128-255" if x < 256 else "256-511" if x < 512 else "512+"
            for x in lengths).items()},
        "status": "COMPLETE_WITH_FLAGS" if any(qa[k] for k in (
            "invalid_id", "duplicate_id", "empty_sequence", "invalid_record", "crossing_pair",
            "noncanonical_sequence", "topology_error")) else "COMPLETE",
    }
    return report, records, per_record


def exact_overlap(datasets: dict[str, dict]) -> dict:
    train_hashes = defaultdict(list)
    test_hashes = defaultdict(list)
    for dataset_id, bundle in datasets.items():
        role = bundle["spec"]["role"]
        target = train_hashes if role == "train" else test_hashes if role in {"val", "test"} else None
        if target is None:
            continue
        for rec in bundle["records"]:
            target[rec["sha256"]].append((dataset_id, rec["id"]))
    shared = sorted(set(train_hashes) & set(test_hashes))
    pairs = []
    for digest in shared:
        pairs.append({"sequence_sha256": digest, "train": train_hashes[digest], "test": test_hashes[digest]})
    matrix = []
    train_ids = sorted({d for d, b in datasets.items() if b["spec"]["role"] == "train"})
    test_ids = sorted({d for d, b in datasets.items() if b["spec"]["role"] in {"val", "test"}})
    train_sets = {d: {r["sha256"] for r in datasets[d]["records"]} for d in train_ids}
    test_sets = {d: {r["sha256"] for r in datasets[d]["records"]} for d in test_ids}
    for test_id in test_ids:
        test_set = test_sets[test_id]
        for train_id in train_ids:
            overlap = len(test_set & train_sets[train_id])
            matrix.append({"train": train_id, "test": test_id, "n_overlap_sequences": overlap,
                           "n_test_unique_sequences": len(test_set),
                           "fraction_of_test": overlap / len(test_set) if test_set else 0,
                           "status": "EXACT_OVERLAP" if overlap else "NO_EXACT_SEQUENCE_OVERLAP"})
    return {"definition": "SHA256 of uppercase U-normalized full sequence", "n_shared_sequences": len(shared),
            "examples": pairs[:100], "matrix": matrix,
            "status": "EXACT_OVERLAP_DETECTED" if shared else "PASS_EXACT_ONLY"}


def length_bin(length: int) -> str:
    return "0-63" if length < 64 else "64-127" if length < 128 else "128-255" if length < 256 else "256-511" if length < 512 else "512+"


def matched_length_subset(train_records: list[dict], eval_records: list[dict]) -> dict:
    """Deterministic length-bin matching; selection never uses model outputs."""
    def grouped(records: list[dict]) -> dict[str, list[dict]]:
        out = defaultdict(list)
        for rec in records:
            out[length_bin(rec["length"])].append(rec)
        for values in out.values():
            values.sort(key=lambda rec: (rec["id"], rec["sha256"]))
        return out
    train_groups, eval_groups = grouped(train_records), grouped(eval_records)
    bins = []
    for name in sorted(set(train_groups) & set(eval_groups)):
        n = min(len(train_groups[name]), len(eval_groups[name]))
        def keys(records: list[dict]) -> list[dict]:
            return [{"id": rec["id"], "sha256": rec["sha256"], "length": rec["length"]} for rec in records[:n]]
        bins.append({"length_bin": name, "n_per_group": n,
                     "train": keys(train_groups[name]), "test": keys(eval_groups[name])})
    return {"method": "deterministic sorted-ID/hash selection within shared length bins; no model-score selection",
            "n_matched_per_group": sum(row["n_per_group"] for row in bins), "bins": bins}


def _seed_kmers(seq: str, k: int = 31, max_seeds: int = 9) -> set[str]:
    if len(seq) < k:
        return {seq} if seq else set()
    positions = np.linspace(0, len(seq) - k, num=min(max_seeds, len(seq) - k + 1), dtype=int)
    return {seq[int(pos):int(pos) + k] for pos in positions}


def _full_kmers(seq: str, k: int = 31, step: int = 3) -> set[str]:
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[pos:pos + k] for pos in range(0, len(seq) - k + 1, step)}


def near_duplicate_screen(datasets: dict[str, dict], max_candidates: int = 200) -> dict:
    """Supplemental exact 31-mer containment screen, not a homology search."""
    train_entries = []
    for dataset_id, bundle in datasets.items():
        if bundle["spec"]["role"] != "train":
            continue
        for index, rec in enumerate(bundle["records"]):
            train_entries.append((dataset_id, index, rec["id"], rec["seq"]))
    index = defaultdict(list)
    for train_index, (_, _, _, seq) in enumerate(train_entries):
        for kmer in _seed_kmers(seq):
            index[kmer].append(train_index)
    hits = []
    overflows = 0
    n_test = 0
    for dataset_id, bundle in datasets.items():
        if bundle["spec"]["role"] not in {"val", "test"}:
            continue
        for rec in bundle["records"]:
            n_test += 1
            candidates = Counter()
            for kmer in _seed_kmers(rec["seq"]):
                candidates.update(index.get(kmer, ()))
            selected = [key for key, count in candidates.items() if count >= 3]
            if len(selected) > max_candidates:
                overflows += 1
                selected = sorted(selected, key=lambda key: (-candidates[key], train_entries[key][2]))[:max_candidates]
            else:
                selected = sorted(selected, key=lambda key: (-candidates[key], train_entries[key][2]))
            test_kmers = _full_kmers(rec["seq"])
            for train_index in selected:
                train_dataset, _, train_id, train_seq = train_entries[train_index]
                length_ratio = min(len(rec["seq"]), len(train_seq)) / max(len(rec["seq"]), len(train_seq), 1)
                if length_ratio < 0.8:
                    continue
                train_kmers = _full_kmers(train_seq)
                intersection = len(test_kmers & train_kmers)
                denominator = min(len(test_kmers), len(train_kmers))
                containment = intersection / denominator if denominator else 0
                identity = (sum(a == b for a, b in zip(rec["seq"], train_seq)) / len(rec["seq"])
                            if len(rec["seq"]) == len(train_seq) and rec["seq"] else
                            difflib.SequenceMatcher(None, rec["seq"], train_seq, autojunk=False).ratio()) if containment >= 0.3 else 0
                if containment >= 0.8 or identity >= 0.9:
                    hits.append({"test_dataset": dataset_id, "test_id": rec["id"], "train_dataset": train_dataset,
                                 "train_id": train_id, "seed_matches": candidates[train_index],
                                 "kmer_containment": containment, "sequence_identity": identity, "length_ratio": length_ratio})
    return {"definition": "candidate: >=3 shared seeded exact 31-mers and length ratio >=0.8; hit: step-3 31-mer containment >=0.8 or SequenceMatcher identity >=0.9",
            "limitations": "not an alignment, not exhaustive for remote homology, and not a substitute for minimap2/MMseqs/BLAST/Infernal",
            "n_train_sequences": len(train_entries), "n_test_sequences": n_test, "n_candidate_overflow_test_sequences": overflows,
            "n_hits": len(hits), "examples": hits[:200],
            "status": "PARTIAL_CANDIDATE_OVERFLOW" if overflows else
                      "NEAR_DUPLICATE_CANDIDATES_DETECTED" if hits else "NO_CANDIDATES_WITHIN_LIMITED_SCREEN"}


def family_topology_audit(datasets: dict[str, dict]) -> dict:
    family_rows = []
    topology_rows = []
    train_ids = [d for d, b in datasets.items() if b["spec"]["role"] == "train"]
    test_ids = [d for d, b in datasets.items() if b["spec"]["role"] in {"val", "test"}]
    stats = {}
    for dataset_id, bundle in datasets.items():
        records = bundle["records"]
        stats[dataset_id] = {
            "families": {r["family"] for r in records if r["family"]},
            "topologies": {r["topology"] for r in records if r["topology"]},
            "family_coverage": sum(r["family"] is not None for r in records) / max(len(records), 1),
        }
    for test_id in test_ids:
        for train_id in train_ids:
            train_stats, test_stats = stats[train_id], stats[test_id]
            train_families, test_families = train_stats["families"], test_stats["families"]
            train_coverage, test_coverage = train_stats["family_coverage"], test_stats["family_coverage"]
            overlap = sorted(train_families & test_families)
            complete = train_coverage >= 0.95 and test_coverage >= 0.95
            family_rows.append({"train": train_id, "test": test_id, "train_label_coverage": train_coverage,
                                "test_label_coverage": test_coverage, "n_shared_families": len(overlap),
                                "examples": overlap[:30], "status": "FAMILY_OVERLAP" if overlap and complete else
                                "NO_FAMILY_OVERLAP" if complete else "BLOCKED_LABEL_UNAVAILABLE"})
            train_topologies, test_topologies = train_stats["topologies"], test_stats["topologies"]
            shared = train_topologies & test_topologies
            topology_rows.append({"train": train_id, "test": test_id, "n_shared_topology_signatures": len(shared),
                                  "n_test_unique_topologies": len(test_topologies),
                                  "fraction_of_test_topologies": len(shared) / len(test_topologies) if test_topologies else 0,
                                  "interpretation": "shared simple stem/loop topology is not by itself evidence of leakage"})
    return {"family_matrix": family_rows, "topology_matrix": topology_rows,
            "family_status": "PARTIAL_BY_LABEL_COVERAGE", "topology_status": "DESCRIPTIVE_NOT_LEAKAGE_VERDICT"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detected_count(value: dict) -> int:
    return int(value.get("n_detected_homolog", value.get("detected_by_any_source", 0)))


def _not_detected_count(value: dict) -> int:
    return int(value.get("n_not_detected", value.get("not_detected_by_any_source", 0)))


def load_per_sequence_homology(summary_path: Path) -> tuple[dict, dict]:
    """Hash-check and join QIDs to dataset/sequence hashes without assuming ID semantics."""
    summary = json.loads(summary_path.read_text())
    union_path = Path(summary["union_per_query"])
    query_manifest_path = union_path.parent / "query_manifest.json"
    query_manifest = json.loads(query_manifest_path.read_text())
    query_index_path = Path(query_manifest["query_index"])
    checks = {
        "query_manifest": _sha256(query_manifest_path) == summary["query_manifest_sha256"],
        "query_index": _sha256(query_index_path) == query_manifest["query_index_sha256"],
        "union_per_query": _sha256(union_path) == summary["union_per_query_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"homology per-sequence evidence hash mismatch: {checks}")
    by_qid = {}
    with gzip.open(union_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            detected = sorted(row.get("detected_sources", []))
            by_qid[row["query_id"]] = {
                "status": "HOMOLOGY" if detected else "NOT_DETECTED_UNKNOWN",
                "detected_sources": detected,
                "interpretation": row.get("interpretation"),
            }
    mapping = {}
    with gzip.open(query_index_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            value = by_qid[row["query_id"]]
            for dataset_id in row["source_sets"]:
                key = (dataset_id, row["sequence_sha256"])
                if key in mapping and mapping[key] != value:
                    raise RuntimeError(f"conflicting homology labels for {key}")
                mapping[key] = value
    metadata = {
        "status": "PASS_PER_SEQUENCE_HOMOLOGY_IDENTITY_JOIN",
        "checks": checks, "n_queries": len(by_qid), "n_dataset_sequence_links": len(mapping),
        "query_manifest": str(query_manifest_path), "query_index": str(query_index_path), "union_per_query": str(union_path),
        "scope": "known source union only; UNCHECKED and NOT_DETECTED_UNKNOWN are not clean labels",
    }
    return mapping, metadata


def existing_exposure_audit(artifact_root: Path, registry: list[dict]) -> dict:
    """Integrate, hash-check, and delimit the existing independent exposure audits."""
    summary_path = artifact_root / "artifacts/v2/reporting/homology_union_summary.json"
    qa_path = artifact_root / "artifacts/v2/reporting/homology_union_independent_qa.json"
    data_path = artifact_root / "manifests/v2/data_audit/data_exposure_audit.json"
    for path in (summary_path, qa_path, data_path):
        if not path.is_file():
            return {"status": "BLOCKED_REQUIRED_EXISTING_AUDIT_MISSING", "missing": str(path)}
    summary = json.loads(summary_path.read_text())
    qa = json.loads(qa_path.read_text())
    data_audit = json.loads(data_path.read_text())
    summary_hash = _sha256(summary_path)
    source_checks = []
    for source, evidence in sorted(summary["sources"].items()):
        path = Path(evidence["summary_path"])
        actual = _sha256(path) if path.is_file() else None
        source_checks.append({"source": source, "path": str(path), "expected_sha256": evidence["summary_sha256"],
                              "actual_sha256": actual, "match": actual == evidence["summary_sha256"]})
    query_checks = {name: _detected_count(value) + _not_detected_count(value) == value["n_queries"]
                    for name, value in summary["queries"].items()}
    qa_verified = (qa.get("summary_sha256") == summary_hash and qa.get("status") == "PASS_HOMOLOGY_UNION_INDEPENDENT_QA"
                   and all(row["match"] for row in source_checks) and all(query_checks.values()))
    by_id = {row["dataset_id"]: row for row in registry}
    query_rows = []
    for dataset_id, spec in sorted(by_id.items()):
        if spec.get("scope") in {"OUT_OF_SCOPE_NCRNA_CLASSIFICATION", "SYNTHETIC_ALGORITHM"} or spec.get("role") in {"out_of_scope_ncrna", "synthetic_algorithm", "derived_cache"}:
            continue
        value = summary["queries"].get(dataset_id)
        if spec.get("status") != "FROZEN":
            query_rows.append({"dataset_id": dataset_id, "status": "BLOCKED_DATASET_NOT_FROZEN", "reason": spec.get("block_reason")})
            continue
        if value is None:
            query_rows.append({"dataset_id": dataset_id, "status": "UNCHECKED", "reason": "not present in the known-source homology union summary"})
            continue
        detected = _detected_count(value)
        query_rows.append({"dataset_id": dataset_id, "status": "HOMOLOGY" if detected else "NOT_DETECTED_UNKNOWN",
                           **value, "detected_fraction": detected / value["n_queries"] if value["n_queries"] else None,
                           "scope": "known source union only; non-detection is not clean"})
    tr1c = data_audit["tr1c"]
    current_tr1c = by_id.get("bprna_tr1c", {})
    tr1c_hash_match = current_tr1c.get("sha256") == tr1c["sha256"]
    cluster_rows = []
    for filename, value in sorted(data_audit["evaluation_exact_overlap"].items()):
        dataset_id = Path(filename).stem
        exact_n = int(value.get("n_unique_exact_overlap", value.get("exact_overlap_with_tr1c", 0)))
        cluster_n = int(value.get("n_80pct_cluster_overlap", value.get("cluster_overlap_with_train", 0)))
        cluster_rows.append({"dataset_id": dataset_id, "train": "bprna_tr1c", **value,
                             "n_unique_exact_overlap": exact_n, "n_80pct_cluster_overlap": cluster_n,
                             "status": "EXACT" if exact_n else "NO_EXACT",
                             "cluster_status": "HOMOLOGY" if cluster_n else "NO_CLUSTER_OVERLAP_IN_THIS_AUDIT"})
    return {
        "status": "PASS_EXISTING_EVIDENCE_INTEGRATED_UNKNOWNS_REMAIN" if qa_verified and tr1c_hash_match else "BLOCKED_EXISTING_EVIDENCE_QA_MISMATCH",
        "qa_verified": qa_verified and tr1c_hash_match,
        "evidence": {"summary": {"path": str(summary_path), "sha256": summary_hash},
                     "independent_qa": {"path": str(qa_path), "sha256": _sha256(qa_path), "status": qa.get("status")},
                     "data_exposure_audit": {"path": str(data_path), "sha256": _sha256(data_path), "status": data_audit.get("status")}},
        "source_summary_checks": source_checks,
        "query_count_checks": query_checks,
        "pretraining": {"summary_status": summary["status"], "protocol": summary["scientific_scope"],
                        "n_queries": summary["n_queries"], "n_detected_homolog": _detected_count(summary),
                        "n_not_detected": _not_detected_count(summary), "datasets": query_rows,
                        "external_baseline_exposure": "BLOCKED_SEPARATE_TRAINING_CORPORA_REQUIRED"},
        "downstream_cluster_evidence": {"tr1c": tr1c, "tr1c_hash_matches_current_registry": tr1c_hash_match,
                                        "evaluation": cluster_rows, "limits": data_audit["limits"]},
        "upstream_cleaning_evidence": {"tr1c_attrition_totals": tr1c["attrition_totals"],
                                       "note": "These deletions occurred before the current frozen JSONL; current-file pair conservation is reported separately."},
        "limitations": [*summary["limitations"], *data_audit["limits"],
                        "Exact overlap, 80% cluster overlap, and pretraining homology are different evidence types and are never interchanged.",
                        "Datasets absent from the summary remain UNCHECKED; non-detected queries remain NOT_DETECTED_UNKNOWN."],
    }


def length_matched_analysis(datasets: dict[str, dict]) -> dict:
    pairs = []
    train_ids = sorted(name for name, bundle in datasets.items() if bundle["spec"]["role"] == "train")
    test_ids = sorted(name for name, bundle in datasets.items() if bundle["spec"]["role"] in {"val", "test"})
    for train_id in train_ids:
        # Use each dataset family's own training pool; CHANRG is not compared to TR1c by default.
        if train_id == "bprna_tr1c":
            targets = [name for name in test_ids if not name.startswith("chanrg_") and not name.startswith("interface_")]
        elif train_id == "chanrg_train_train":
            targets = [name for name in test_ids if name.startswith("chanrg_")]
        else:
            targets = []
        for test_id in targets:
            print(f"LENGTH MATCH {train_id} vs {test_id}", flush=True)
            result = matched_length_subset(datasets[train_id]["records"], datasets[test_id]["records"])
            pairs.append({"train": train_id, "test": test_id, **result})
    return {"status": "COMPLETE_DESCRIPTIVE_SUBSETS", "pairs": pairs,
            "note": "Length matching supports distribution comparisons only; it does not establish family or exposure independence."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--out-phase", default="phase1")
    parser.add_argument("--skip-near-screen", action="store_true")
    args = parser.parse_args()
    out = args.artifact_root / "artifacts/foldnar_v1" / args.out_phase
    final_paths = [out / name for name in ("data_quality_report.json", "distribution.npz", "distribution.json", "length_matched_subsets.json", "leakage_matrix.json", "per_sequence_audit.jsonl.gz")]
    if any(p.exists() for p in final_paths):
        raise FileExistsError("refusing to overwrite frozen Phase-1 outputs")
    out.mkdir(parents=True, exist_ok=True)
    registry = json.loads((args.repo / "manifests/foldnar_v1/data_registry.json").read_text())["rows"]
    exposure = existing_exposure_audit(args.artifact_root, registry)
    specs = [r for r in registry if r.get("role") in {"train", "val", "test", "synthetic_algorithm"} and r.get("status") == "FROZEN"]
    datasets = {}
    reports = []
    arrays = []
    for index, spec in enumerate(specs):
        print(f"QA {index + 1}/{len(specs)} {spec['dataset_id']}", flush=True)
        report, records, numeric = read_dataset(spec)
        reports.append(report)
        datasets[spec["dataset_id"]] = {"spec": spec, "records": records}
        arrays.extend((index, *values) for values in numeric)

    homology_map, homology_join = load_per_sequence_homology(Path(exposure["evidence"]["summary"]["path"]))
    homology_join["independent_qa_verified"] = exposure["qa_verified"]
    per_sequence_path = out / "per_sequence_audit.jsonl.gz"
    join_counts = {}
    with per_sequence_path.open("wb") as raw_handle, gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle:
        for spec in specs:
            dataset_id = spec["dataset_id"]
            for rec in datasets[dataset_id]["records"]:
                homology = homology_map.get((dataset_id, rec["sha256"]), {
                    "status": "UNCHECKED", "detected_sources": [], "interpretation": "not covered by the known-source query manifest",
                })
                status = homology["status"]
                join_counts[dataset_id] = join_counts.get(dataset_id, Counter())
                join_counts[dataset_id][status] += 1
                out_record = {
                    "dataset_id": dataset_id, "role": spec["role"], "id": rec["id"], "length": rec["length"],
                    "sha256": rec["sha256"], "raw_sequence_sha256": rec["raw_sequence_sha256"],
                    "family": rec["family"], "clan": rec["clan"], "topology": rec["topology"],
                    "pretraining_homology_status": status,
                    "pretraining_homology_sources": homology["detected_sources"],
                    "pretraining_homology_interpretation": homology["interpretation"],
                }
                handle.write((json.dumps(out_record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    homology_join.update({
        "artifact": str(per_sequence_path), "sha256": _sha256(per_sequence_path),
        "n_rows": sum(sum(counts.values()) for counts in join_counts.values()),
        "counts": {key: dict(value) for key, value in sorted(join_counts.items())},
    })

    # Independent reload count is the third count after registry JSON parsing and binary newlines.
    quality_payload = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                       "normalization": "uppercase; T->U; pairs zero-based inclusive", "datasets": reports,
                       "upstream_cleaning_evidence": exposure.get("upstream_cleaning_evidence", {"status": exposure.get("status")}),
                       "experimental_coordinate_audit": {
                           "status": "BLOCKED_PER_RECORD_MAPPING_INCOMPLETE_TEMPORAL_RAW_ASSETS_LOCATED",
                           "required": ["PDB ID", "entity ID", "model", "assembly", "chain mapping", "experimental method", "resolution", "modified residues", "ligands", "missing residues", "DSSR pair evidence"],
                           "located_temporal_evidence": [{"dataset_id": r["dataset_id"], "raw_dssr_root": r.get("raw_dssr_root"),
                                                          "n_dssr_entries": r.get("n_dssr_entries"), "block_reason": r.get("block_reason")}
                                                         for r in registry if r.get("raw_asset_status") == "LOCATED_QA_BLOCKED"],
                           "note": "Raw temporal mmCIF/DSSR assets do not substitute for a complete mapping to the contract PDB evaluation records."},
                       "required_asset_gaps": [{"dataset_id": r["dataset_id"], "status": r["status"], "reason": r.get("block_reason")}
                                               for r in registry if r.get("status") == "BLOCKED" and r.get("scope") != "OUT_OF_SCOPE_NCRNA_CLASSIFICATION"],
                       "verification": {"registry_rows": len(specs), "qa_rows": len(reports),
                                        "registry_sequence_total": sum(r["n_sequences"] for r in specs),
                                        "qa_sequence_total": sum(r["n_records"] for r in reports)}}
    quality_payload["verification"]["pass"] = (
        quality_payload["verification"]["registry_rows"] == quality_payload["verification"]["qa_rows"]
        and quality_payload["verification"]["registry_sequence_total"] == quality_payload["verification"]["qa_sequence_total"]
    )
    (out / "data_quality_report.json").write_text(json.dumps(quality_payload, indent=2, ensure_ascii=False) + "\n")
    arr = np.asarray(arrays, dtype=np.int64).reshape(-1, 16)
    np.savez_compressed(out / "distribution.npz", dataset_index=arr[:, 0], length=arr[:, 1], n_pairs=arr[:, 2],
                        n_crossing_pairs=arr[:, 3], n_stems=arr[:, 4], gc_count=arr[:, 5], raw_pair_entries=arr[:, 6],
                        pair_span_total=arr[:, 7], pair_span_count=arr[:, 8], max_pair_span=arr[:, 9],
                        n_helices=arr[:, 10], helix_length_total=arr[:, 11], max_helix_length=arr[:, 12],
                        n_multiloops=arr[:, 13], multiloop_order_total=arr[:, 14], max_multiloop_order=arr[:, 15],
                        dataset_ids=np.asarray([s["dataset_id"] for s in specs]))
    distribution = {"dataset_order": [s["dataset_id"] for s in specs], "n_rows": int(len(arr)),
                    "per_sequence_audit": homology_join,
                    "columns": ["dataset_index", "length", "n_pairs", "n_crossing_pairs", "n_stems", "gc_count",
                                "raw_pair_entries", "pair_span_total", "pair_span_count", "max_pair_span", "n_helices",
                                "helix_length_total", "max_helix_length", "n_multiloops", "multiloop_order_total", "max_multiloop_order"],
                    "summary": [{k: r[k] for k in ("dataset_id", "role", "n_records", "length", "pairs", "pair_conservation",
                                                   "gc_fraction", "pair_span", "helices", "multiloop", "length_bins",
                                                   "stem_count_histogram", "loop_counts", "topology")} for r in reports]}
    (out / "distribution.json").write_text(json.dumps(distribution, indent=2, ensure_ascii=False) + "\n")
    length_matched = length_matched_analysis(datasets)
    (out / "length_matched_subsets.json").write_text(json.dumps(length_matched, indent=2, ensure_ascii=False) + "\n")

    print("LEAKAGE exact sequence overlap", flush=True)
    exact = exact_overlap(datasets)
    near = {"status": "NOT_RUN_BY_REQUEST"} if args.skip_near_screen else near_duplicate_screen(datasets)
    print("LEAKAGE family/topology", flush=True)
    family = family_topology_audit(datasets)
    tool_paths = {name: shutil.which(name) for name in SEARCH_TOOLS}
    pretrain_expected = [
        args.artifact_root / "artifacts/v2/exposure_stratified/pretrain_bpRNA/label.json",
        args.artifact_root / "artifacts/v2/exposure_stratified/pretrain_bpRNA/manifest.json",
        args.artifact_root / "homology_verdict.json",
    ]
    known_cluster_rows = exposure.get("downstream_cluster_evidence", {}).get("evaluation", [])
    known_cluster_hits = sum(row["n_80pct_cluster_overlap"] > 0 for row in known_cluster_rows)
    leakage = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "included_roles": ["train", "val", "test"],
            "synthetic_algorithm_separate": [s["dataset_id"] for s in specs if s["role"] == "synthetic_algorithm"],
            "excluded": [{"dataset_id": r["dataset_id"], "reason": r["scope"]} for r in registry if r.get("scope") == "OUT_OF_SCOPE_NCRNA_CLASSIFICATION"],
        },
        "contract_coverage": [{"dataset_id": s["dataset_id"], "registered_n": s["n_sequences"],
                               "contract_expected_n": s["contract_expected_n"],
                               "status": s["contract_coverage_status"]} for s in specs if "contract_expected_n" in s],
        "overall_status": "BLOCKED_KNOWN_HOMOLOGY_AND_UNCHECKED_REMAIN",
        "training_authorization": {"authorized": False, "decision": "HOLD",
                                   "reason": "Known 80% cluster overlaps and known-source pretraining homology coexist with unchecked corpora/splits; E0 numerical GO_E1 is not permission to train."},
        "layers": {
            "direct_split_exact": exact,
            "direct_split_near_duplicate": {**near, "homology_tool_status": "BLOCKED" if not any(tool_paths.values()) else "AVAILABLE",
                                            "tool_paths": tool_paths},
            "direct_split_existing_80pct_clusters": {
                "status": "HOMOLOGY_DETECTED" if known_cluster_hits else "NO_CLUSTER_OVERLAP_IN_LIMITED_EXISTING_AUDIT",
                "n_evaluation_datasets_with_cluster_overlap": known_cluster_hits,
                "evidence": exposure.get("downstream_cluster_evidence", {"status": exposure.get("status")}),
            },
            "derived_split_lineage": {
                "status": "PARTIAL",
                "known": [{"source": "CHANRG", "datasets": [d for d in datasets if d.startswith("chanrg_")],
                           "provenance": "hf:multimolecule/chanrg@469835a2c1440bcd11dd40b7325090abe9fc22ad",
                           "dedup_status": "PROVENANCE_ONLY; current files are contract-size subsets/version mismatches"},
                          {"source": "Rfam", "datasets": [d for d in datasets if "rfam" in d],
                           "provenance": "split-specific manifests retained; family/temporal lineage not assumed"}],
                "exact_sequence_overlap_applies_to_all_derived_pairs": True,
            },
            "family_topology": family,
            "pretraining_exposure": {
                "status": exposure.get("status"),
                "reason": "Existing independently checked known-source homology evidence is integrated; absent expected paths no longer justify a blanket no-evidence claim.",
                "legacy_expected_artifacts": [{"path": str(p), "exists": p.exists()} for p in pretrain_expected],
                "evidence": exposure,
                "per_sequence_join": homology_join,
                "unknown": ["complete historical nt corpus", "unchecked CHANRG/interface/Rfam-temporal exposure", "non-detection does not mean clean", "external baseline training exposure"],
            },
        },
        "distribution_comparison": {"artifact": str(out / "length_matched_subsets.json"),
                                    "status": length_matched["status"], "n_pairs": len(length_matched["pairs"])},
        "unresolved_data_roles": [{"dataset_id": r["dataset_id"], "role": r["role"], "status": r["status"],
                                   "expected_n": r.get("contract_expected_n"), "expected_length": r.get("contract_expected_length")}
                                  for r in registry if r.get("status") == "BLOCKED" and r.get("scope") != "OUT_OF_SCOPE_NCRNA_CLASSIFICATION"],
        "required_remediation": [
            "Freeze full training/pretraining provenance and apply exact plus 80% homology decontamination separately to train, validation, and test.",
            "Re-run frozen baselines after any cleaning; do not reuse this audit as a clean-OOD or SOTA claim.",
            "Complete CHANRG/version coverage, temporal PDB candidates, D1/mmCIF/DSSR mapping, and long-chain assets before claiming those roles.",
        ],
        "decision_rule": "PASS requires direct, derived, family/topology, and pretraining evidence; exact-only, screen-only, or non-detection results cannot clear homology.",
    }
    (out / "leakage_matrix.json").write_text(json.dumps(leakage, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "COMPLETE_WITH_BLOCKED_LAYERS", "datasets": len(reports),
                      "exact_shared_sequences": exact["n_shared_sequences"], "near_screen_hits": near.get("n_hits"),
                      "known_cluster_datasets": known_cluster_hits,
                      "pretrain_known_source_detected": exposure.get("pretraining", {}).get("n_detected_homolog"),
                      "training_authorized": False, "out": str(out)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
