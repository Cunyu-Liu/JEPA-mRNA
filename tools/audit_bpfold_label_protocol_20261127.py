#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import math
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from data.ss.prepare_decision_data import project_to_legal, read_bpseq_text

RELEASE = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2/official_assets/bpfold/BPfold_test_results_v0.2.tar.gz")
REFERENCE = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz")
PROJECT = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
MANIFEST = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.manifest.json")
REPLAY = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark/bpfold/raw/bprna_new.json")
OUTPUT = Path("source_data/bpfold_label_protocol_audit_20261127")
CANONICAL = frozenset(("AU", "UA", "CG", "GC", "GU", "UG"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pair_sha256(pairs):
    return hashlib.sha256(canonical_json(sorted(pairs)).encode()).hexdigest()


def load_release(path):
    rows = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.isfile() and member.name.endswith(".bpseq") and "bprnanew" in member.name.lower().split("/"):
                name = Path(member.name).name
                assert name not in rows, name
                sequence, pairs = read_bpseq_text(archive.extractfile(member).read().decode())
                rows[name] = {"name": name, "seq": sequence, "pairs": sorted(pairs)}
    assert len(rows) == 5401
    return rows


def load_reference(path):
    rows = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.isfile() and member.name.startswith("BPfold_data/test_data/bpRNAnew/") and member.name.endswith(".bpseq"):
                name = Path(member.name).name
                assert name not in rows, name
                sequence, pairs = read_bpseq_text(archive.extractfile(member).read().decode())
                rows[name] = {"name": name, "seq": sequence, "pairs": sorted(pairs)}
    assert len(rows) == 5401
    return rows


def normalize(sequence):
    return sequence.upper().replace("T", "U")


def pair_set(pairs, length):
    result = set()
    occupied = set()
    for left, right in pairs:
        assert 0 <= left < right < length, (left, right, length)
        assert left not in occupied and right not in occupied, (left, right)
        occupied.update((left, right))
        result.add((left, right))
    return result


def crossing_pairs(pairs):
    ordered = sorted(pairs)
    return [
        (left, right)
        for index, left in enumerate(ordered)
        for right in ordered[index + 1:]
        if left[0] < right[0] < left[1] < right[1]
    ]


def noncanonical_count(sequence, pairs):
    return sum(sequence[left] + sequence[right] not in CANONICAL for left, right in pairs)


def score(labels, predictions):
    tp = fp = fn = 0
    f1_values = []
    for label in labels:
        name = label["name"]
        assert name in predictions, name
        truth = pair_set(label["pairs"], len(label["seq"]))
        predicted = pair_set(predictions[name]["pairs"], len(label["seq"]))
        a, b, c = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        tp, fp, fn = tp + a, fp + b, fn + c
        denominator = 2 * a + b + c
        f1_values.append(2 * a / denominator if denominator else 1.0)
    return {
        "n_sequences": len(labels),
        "coverage": 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "macro_f1": math.fsum(f1_values) / len(f1_values) if f1_values else None,
        "zero_pair_f1_convention": "F1=1 when truth and prediction both have zero pairs",
    }


def classify(reference_pairs, release_pairs, project_pairs):
    if reference_pairs == project_pairs:
        return "LABEL_SETS_EQUAL"
    if project_pairs == release_pairs:
        return "DIFFERENT_REFERENCE_VS_UNFILTERED_RELEASE"
    if reference_pairs == release_pairs:
        return "DIFFERENT_MIN_LOOP_OR_CROSSING_PROJECTION_ONLY"
    return "DIFFERENT_REFERENCE_AND_PROJECT_PROJECTION"


def audit_record(name, reference, release, project):
    reference_pairs = set(reference["pairs"])
    release_pairs = set(release["pairs"])
    length = len(release["seq"])
    bad_bases = sorted(set(release["seq"]) - set("ACGU"))
    project_pairs = None if project is None else {tuple(pair) for pair in project["pairs"]}
    row = {
        "name": name,
        "length": length,
        "project_included": project is not None,
        "project_exclusion_reason": None if project is not None else "bad_alphabet",
        "bad_alphabet": "".join(bad_bases),
        "official_reference_sequence_equals_release": reference["seq"] == release["seq"],
        "project_sequence_equals_release": None if project is None else normalize(project["seq"]) == release["seq"],
        "index_convention": "BPSEQ contiguous 1-based -> project zero-based inclusive; no offset transform",
        "n_reference_pairs": len(reference_pairs),
        "n_release_pairs": len(release_pairs),
        "n_project_pairs": None if project is None else len(project["pairs"]),
        "reference_pair_set_sha256": pair_sha256(reference_pairs),
        "release_pair_set_sha256": pair_sha256(release_pairs),
        "project_pair_set_sha256": None if project is None else pair_sha256(project["pairs"]),
        "reference_equals_release": reference_pairs == release_pairs,
        "reference_equals_project": None if project is None else reference_pairs == project_pairs,
        "n_reference_only_vs_project": None if project is None else len(reference_pairs - project_pairs),
        "n_project_only_vs_reference": None if project is None else len(project_pairs - reference_pairs),
        "n_reference_noncanonical_pairs": noncanonical_count(reference["seq"], reference_pairs),
        "n_release_noncanonical_pairs": noncanonical_count(release["seq"], release_pairs),
        "n_project_noncanonical_pairs": None if project is None else noncanonical_count(project["seq"], project["pairs"]),
        "n_reference_min_loop_pairs": sum(right - left <= 3 for left, right in reference_pairs),
        "n_release_min_loop_pairs": sum(right - left <= 3 for left, right in release_pairs),
        "n_project_min_loop_pairs": None if project is None else sum(right - left <= 3 for left, right in project["pairs"]),
        "reference_has_crossing": bool(crossing_pairs(reference_pairs)),
        "release_has_crossing": bool(crossing_pairs(release_pairs)),
        "project_has_crossing": None if project is None else bool(crossing_pairs(project["pairs"])),
    }
    if project is None:
        row.update({
            "classification": "OFFICIAL_ONLY_PROJECT_REJECTED_BAD_ALPHABET",
            "release_to_project_exact_projection": None,
            "n_release_to_project_min_loop_dropped": None,
            "n_release_to_project_crossing_dropped": None,
            "release_to_project_min_loop_dropped_pairs": None,
            "release_to_project_crossing_dropped_pairs": None,
        })
        return row
    kept, dropped_min_loop, dropped_crossing = project_to_legal(length, sorted(release_pairs), min_loop=3)
    row.update({
        "classification": classify(reference_pairs, release_pairs, project_pairs),
        "release_to_project_exact_projection": set(kept) == project_pairs,
        "n_release_to_project_min_loop_dropped": len(dropped_min_loop),
        "n_release_to_project_crossing_dropped": len(dropped_crossing),
        "release_to_project_min_loop_dropped_pairs": [list(pair) for pair in dropped_min_loop],
        "release_to_project_crossing_dropped_pairs": [list(pair) for pair in dropped_crossing],
    })
    return row


def input_row(path, role):
    return {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--replay", type=Path, default=REPLAY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    release = load_release(args.release)
    reference = load_reference(args.reference)
    assert set(release) == set(reference)
    project_rows = [json.loads(line) for line in args.project.read_text().splitlines() if line.strip()]
    assert len(project_rows) == 5388
    project = {row["name"]: row for row in project_rows}
    assert len(project) == len(project_rows)
    assert set(project) <= set(release)
    manifest = json.loads(args.manifest.read_text())
    replay_document = json.loads(args.replay.read_text())
    replay = {row["name"]: {"pairs": row["pred_pairs"]} for row in replay_document["per_sequence"]}
    assert list(replay) == [row["name"] for row in project_rows]

    records = [audit_record(name, reference[name], release[name], project.get(name)) for name in sorted(release)]
    accepted = [row for row in records if row["project_included"]]
    rejected = [row for row in records if not row["project_included"]]
    assert all(row["release_to_project_exact_projection"] for row in accepted)
    official_order = [reference[name] for name in sorted(reference)]
    project_order = [project[row["name"]] for row in project_rows]
    official_metrics = score(official_order, release)
    official_intersection_metrics = score([reference[row["name"]] for row in project_rows], release)
    release_project_metrics = score(project_order, release)
    replay_metrics = score(project_order, replay)
    assert (official_metrics["tp"], official_metrics["fp"], official_metrics["fn"]) == (86933, 59441, 41092)
    assert official_metrics["micro_f1"] == 0.633624758107719
    assert official_metrics["macro_f1"] == 0.6470223791166072
    assert (replay_metrics["tp"], replay_metrics["fp"], replay_metrics["fn"]) == (106998, 48176, 28186)
    assert replay_metrics["micro_f1"] == 0.7370074184282851
    assert (release_project_metrics["tp"], release_project_metrics["fp"], release_project_metrics["fn"]) == (135184, 10776, 0)
    assert release_project_metrics["micro_f1"] == 0.9616708875167174

    equal = sum(row["reference_equals_project"] is True for row in accepted)
    different = len(accepted) - equal
    assert (equal, different, len(rejected)) == (106, 5282, 13)
    class_counts = dict(sorted(Counter(row["classification"] for row in records).items()))
    summary = {
        "schema_version": "bpfold.label_protocol_audit.v1",
        "audit_date": "2026-11-27",
        "base_commit": "c183df8457b2e3d3695b6e9eb78bb6b56bff5d91",
        "scope": "CPU-only protocol audit of frozen external Test artifacts; no inference, training, calibration, model selection, or historical-result modification",
        "inputs": [
            input_row(args.release, "BPfold v0.2 author-released predicted structures"),
            input_row(args.reference, "BPfold v0.2 BPfold_data/test_data/bpRNAnew reference labels"),
            input_row(args.project, "frozen project bpRNA-new labels"),
            input_row(args.manifest, "project conversion manifest"),
            input_row(args.replay, "frozen project six-checkpoint BPfold replay predictions"),
        ],
        "label_set_accounting": {
            "official_release": len(release),
            "official_reference": len(reference),
            "project": len(project),
            "same_name_same_sequence_intersection": len(accepted),
            "official_only": len(rejected),
            "project_only": len(set(project) - set(release)),
            "exact_reference_project_label_sets": equal,
            "different_reference_project_label_sets": different,
        },
        "record_classification_counts": class_counts,
        "sequence_and_index": {
            "project_sequence_mismatches": sum(not row["project_sequence_equals_release"] for row in accepted),
            "reference_release_sequence_mismatches": sum(not row["official_reference_sequence_equals_release"] for row in records),
            "index_offset_transformations_required": 0,
            "evidence": "All 5388 project pair sets equal the historical projection of same-name release BPSEQ pairs after contiguous 1-based to zero-based conversion.",
        },
        "project_exclusions": {
            "count": len(rejected),
            "reason_counts": dict(Counter(row["project_exclusion_reason"] for row in rejected)),
            "names": [row["name"] for row in rejected],
        },
        "release_to_project_projection": {
            "exact_projection_records": sum(row["release_to_project_exact_projection"] is True for row in accepted),
            "records_with_min_loop_drops": sum(row["n_release_to_project_min_loop_dropped"] > 0 for row in accepted),
            "records_with_crossing_drops": sum(row["n_release_to_project_crossing_dropped"] > 0 for row in accepted),
            "records_with_any_drops": sum(row["n_release_to_project_min_loop_dropped"] + row["n_release_to_project_crossing_dropped"] > 0 for row in accepted),
            "min_loop_pairs_dropped": sum(row["n_release_to_project_min_loop_dropped"] for row in accepted),
            "crossing_pairs_dropped": sum(row["n_release_to_project_crossing_dropped"] for row in accepted),
            "min_loop_rule": "drop span<=3 before greedy crossing removal",
            "crossing_rule": "repeatedly remove the pair in the most crossings; ties go to the larger span",
        },
        "canonical_policy": {
            "intersection_scope": 5388,
            "official_only_release_noncanonical_pairs": sum(row["n_release_noncanonical_pairs"] for row in rejected),
            "official_only_reference_noncanonical_pairs": sum(row["n_reference_noncanonical_pairs"] for row in rejected),
            "standalone_canonical_filter_applied": False,
            "release_records_with_noncanonical_pairs": sum(row["n_release_noncanonical_pairs"] > 0 for row in accepted),
            "release_noncanonical_pairs": sum(row["n_release_noncanonical_pairs"] for row in accepted),
            "reference_records_with_noncanonical_pairs": sum(row["n_reference_noncanonical_pairs"] > 0 for row in accepted),
            "reference_noncanonical_pairs": sum(row["n_reference_noncanonical_pairs"] for row in accepted),
            "project_records_retaining_noncanonical_pairs": sum(row["n_project_noncanonical_pairs"] > 0 for row in accepted),
            "project_noncanonical_pairs_retained": sum(row["n_project_noncanonical_pairs"] for row in accepted),
            "note": "Canonical identity is descriptive here. Noncanonical pairs can still be removed by the min-loop or crossing rules; there is no separate canonical-pair deletion stage.",
        },
        "official_reference_protocol": {
            "pair_matching": "strict exact zero-based inclusive pairs; no crossing, canonical, or short-span filtering by the scorer",
            "release_on_official_labels_5401": official_metrics,
            "release_on_official_labels_project_intersection_5388": official_intersection_metrics,
            "readme_macro_rounding_reproduced": round(official_metrics["macro_f1"], 3) == 0.647,
        },
        "project_label_protocol_audits": {
            "project_replay_on_project_labels_5388": replay_metrics,
            "official_release_on_project_labels_5388": release_project_metrics,
            "warning": "The 0.9616708875167174 value scores release predictions against labels derived from those same release predictions after filtering; it is not an independent accuracy estimate or official BPfold result.",
        },
        "dssr_literature_attribution": {
            "status": "NOT_ESTABLISHED_BY_THESE_ARTIFACTS",
            "evidence": "The release API identifies BPfold_test_results as predicted structures and BPfold_data as datasets. The project converter and manifest establish the subsequent filtering. They contain no per-record DSSR or literature curation lineage, so no such root cause is assigned.",
        },
        "historical_manifest_crosscheck": {
            "n_accepted": manifest["n_accepted"],
            "n_rejected": manifest["n_rejected"],
            "reject_reasons": manifest["reject_reasons"],
            "n_dropped_illegal_pairs": manifest["n_dropped_illegal_pairs"],
            "n_dropped_crossing_pairs": manifest["n_dropped_crossing_pairs"],
            "n_records_with_drops": manifest["n_records_with_drops"],
            "source_directory": manifest["provenance"]["directory"],
        },
    }
    assert summary["release_to_project_projection"]["min_loop_pairs_dropped"] == manifest["n_dropped_illegal_pairs"]
    assert summary["release_to_project_projection"]["crossing_pairs_dropped"] == manifest["n_dropped_crossing_pairs"]
    assert summary["release_to_project_projection"]["records_with_any_drops"] == manifest["n_records_with_drops"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_record = args.output_dir / "per_record.jsonl.gz"
    payload = "".join(canonical_json(row) + "\n" for row in records).encode()
    per_record.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    uncompressed = args.output_dir / "per_record.jsonl"
    if uncompressed.exists():
        uncompressed.unlink()
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": "PASS",
        "per_record": str(per_record),
        "per_record_sha256": sha256(per_record),
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "class_counts": class_counts,
        "metrics": {
            "official_5401_micro": official_metrics["micro_f1"],
            "official_5401_macro": official_metrics["macro_f1"],
            "project_replay": replay_metrics["micro_f1"],
            "release_on_project_labels": release_project_metrics["micro_f1"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
