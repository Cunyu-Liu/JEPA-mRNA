#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]
DEFAULT_OUTPUT = ROOT / "source_data/bpfold_label_protocol_audit_20261127"
RELEASE = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2/official_assets/bpfold/BPfold_test_results_v0.2.tar.gz")
REFERENCE = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz")
PROJECT = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
MANIFEST = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.manifest.json")
REPLAY = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark/bpfold/raw/bprna_new.json")
CANONICAL = {"AU", "UA", "CG", "GC", "GU", "UG"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_digest(pairs):
    payload = json.dumps(sorted(pairs), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse(payload):
    bases = []
    partners = []
    for raw in payload.decode().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        index, base, partner = int(fields[0]), fields[1].upper().replace("T", "U"), int(fields[2])
        assert index == len(bases) + 1
        bases.append(base)
        partners.append(partner)
    pairs = set()
    for index, partner in enumerate(partners):
        if partner:
            assert 1 <= partner <= len(bases)
            assert partners[partner - 1] == index + 1
            pairs.add(tuple(sorted((index, partner - 1))))
    return "".join(bases), sorted(pairs)


def members(path, reference=False):
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            selected = member.name.startswith("BPfold_data/test_data/bpRNAnew/") if reference else "bprnanew" in member.name.lower().split("/")
            if member.isfile() and selected and member.name.endswith(".bpseq"):
                sequence, pairs = parse(archive.extractfile(member).read())
                result[Path(member.name).name] = (sequence, pairs)
    assert len(result) == 5401
    return result


def crossings(pairs):
    return [(a, c) for i, a in enumerate(sorted(pairs)) for c in sorted(pairs)[i + 1:] if a[0] < c[0] < a[1] < c[1]]


def project(length, pairs):
    illegal = sorted(pair for pair in pairs if pair[1] - pair[0] <= 3)
    kept = sorted(pair for pair in pairs if pair[1] - pair[0] > 3)
    removed = []
    while crossings(kept):
        counts = Counter(pair for edge in crossings(kept) for pair in edge)
        victim = next(iter(counts))
        for candidate in counts:
            if (counts[candidate], candidate[1] - candidate[0]) > (counts[victim], victim[1] - victim[0]):
                victim = candidate
        kept.remove(victim)
        removed.append(victim)
    return kept, illegal, removed


def strict_score(label_rows, prediction_map):
    totals = [0, 0, 0]
    sequence_f1 = []
    for name, sequence, label_pairs in label_rows:
        assert name in prediction_map
        truth = {tuple(pair) for pair in label_pairs}
        prediction = {tuple(pair) for pair in prediction_map[name]}
        assert all(0 <= i < j < len(sequence) for i, j in truth | prediction)
        assert len({i for pair in truth for i in pair}) == 2 * len(truth)
        assert len({i for pair in prediction for i in pair}) == 2 * len(prediction)
        counts = (len(truth & prediction), len(prediction - truth), len(truth - prediction))
        totals = [old + new for old, new in zip(totals, counts)]
        denominator = 2 * counts[0] + counts[1] + counts[2]
        sequence_f1.append(1.0 if denominator == 0 else 2 * counts[0] / denominator)
    tp, fp, fn = totals
    return {
        "n_sequences": len(label_rows),
        "coverage": 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "macro_f1": math.fsum(sequence_f1) / len(sequence_f1) if sequence_f1 else None,
        "zero_pair_f1_convention": "F1=1 when truth and prediction both have zero pairs",
    }


def expected_record(name, reference, release, project_row):
    reference_sequence, reference_list = reference
    release_sequence, release_list = release
    reference_pairs, release_pairs = set(reference_list), set(release_list)
    bad = "".join(sorted(set(release_sequence) - set("ACGU")))
    included = project_row is not None
    project_pairs = None if not included else set(tuple(pair) for pair in project_row["pairs"])
    row = {
        "name": name,
        "length": len(release_sequence),
        "project_included": included,
        "project_exclusion_reason": None if included else "bad_alphabet",
        "bad_alphabet": bad,
        "official_reference_sequence_equals_release": reference_sequence == release_sequence,
        "project_sequence_equals_release": None if not included else project_row["seq"].upper().replace("T", "U") == release_sequence,
        "index_convention": "BPSEQ contiguous 1-based -> project zero-based inclusive; no offset transform",
        "n_reference_pairs": len(reference_pairs),
        "n_release_pairs": len(release_pairs),
        "n_project_pairs": None if not included else len(project_pairs),
        "reference_pair_set_sha256": pair_digest(reference_pairs),
        "release_pair_set_sha256": pair_digest(release_pairs),
        "project_pair_set_sha256": None if not included else pair_digest(project_pairs),
        "reference_equals_release": reference_pairs == release_pairs,
        "reference_equals_project": None if not included else reference_pairs == project_pairs,
        "n_reference_only_vs_project": None if not included else len(reference_pairs - project_pairs),
        "n_project_only_vs_reference": None if not included else len(project_pairs - reference_pairs),
        "n_reference_noncanonical_pairs": sum(reference_sequence[i] + reference_sequence[j] not in CANONICAL for i, j in reference_pairs),
        "n_release_noncanonical_pairs": sum(release_sequence[i] + release_sequence[j] not in CANONICAL for i, j in release_pairs),
        "n_project_noncanonical_pairs": None if not included else sum(project_row["seq"][i] + project_row["seq"][j] not in CANONICAL for i, j in project_pairs),
        "n_reference_min_loop_pairs": sum(j - i <= 3 for i, j in reference_pairs),
        "n_release_min_loop_pairs": sum(j - i <= 3 for i, j in release_pairs),
        "n_project_min_loop_pairs": None if not included else sum(j - i <= 3 for i, j in project_pairs),
        "reference_has_crossing": bool(crossings(reference_pairs)),
        "release_has_crossing": bool(crossings(release_pairs)),
        "project_has_crossing": None if not included else bool(crossings(project_pairs)),
    }
    if not included:
        row.update({
            "classification": "OFFICIAL_ONLY_PROJECT_REJECTED_BAD_ALPHABET",
            "release_to_project_exact_projection": None,
            "n_release_to_project_min_loop_dropped": None,
            "n_release_to_project_crossing_dropped": None,
            "release_to_project_min_loop_dropped_pairs": None,
            "release_to_project_crossing_dropped_pairs": None,
        })
        return row
    kept, illegal, removed = project(len(release_sequence), release_pairs)
    if reference_pairs == project_pairs:
        classification = "LABEL_SETS_EQUAL"
    elif project_pairs == release_pairs:
        classification = "DIFFERENT_REFERENCE_VS_UNFILTERED_RELEASE"
    elif reference_pairs == release_pairs:
        classification = "DIFFERENT_MIN_LOOP_OR_CROSSING_PROJECTION_ONLY"
    else:
        classification = "DIFFERENT_REFERENCE_AND_PROJECT_PROJECTION"
    row.update({
        "classification": classification,
        "release_to_project_exact_projection": set(kept) == project_pairs,
        "n_release_to_project_min_loop_dropped": len(illegal),
        "n_release_to_project_crossing_dropped": len(removed),
        "release_to_project_min_loop_dropped_pairs": [list(pair) for pair in illegal],
        "release_to_project_crossing_dropped_pairs": [list(pair) for pair in removed],
    })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--replay", type=Path, default=REPLAY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    release = members(args.release)
    reference = members(args.reference, reference=True)
    project_list = [json.loads(line) for line in args.project.read_text().splitlines() if line.strip()]
    project = {row["name"]: row for row in project_list}
    replay_json = json.loads(args.replay.read_text())
    replay = {row["name"]: row["pred_pairs"] for row in replay_json["per_sequence"]}
    assert len(project_list) == len(project) == len(replay) == 5388
    actual_rows = [json.loads(line) for line in gzip.decompress((args.output_dir / "per_record.jsonl.gz").read_bytes()).decode().splitlines()]
    expected_rows = [expected_record(name, reference[name], release[name], project.get(name)) for name in sorted(release)]
    assert actual_rows == expected_rows
    assert len(actual_rows) == 5401
    accepted = [row for row in actual_rows if row["project_included"]]
    assert len(accepted) == 5388
    assert all(row["release_to_project_exact_projection"] for row in accepted)

    official_labels = [(name, reference[name][0], reference[name][1]) for name in sorted(reference)]
    intersection_labels = [(name, reference[name][0], reference[name][1]) for name in project]
    project_labels = [(row["name"], row["seq"], row["pairs"]) for row in project_list]
    release_predictions = {name: pairs for name, (_, pairs) in release.items()}
    metrics = {
        "official": strict_score(official_labels, release_predictions),
        "intersection": strict_score(intersection_labels, release_predictions),
        "release_project": strict_score(project_labels, release_predictions),
        "replay_project": strict_score(project_labels, replay),
    }
    expected_counts = {
        "official": (86933, 59441, 41092, 0.633624758107719, 0.6470223791166072),
        "release_project": (135184, 10776, 0, 0.9616708875167174, 0.9584497822657716),
        "replay_project": (106998, 48176, 28186, 0.7370074184282851, 0.7447389175263477),
    }
    for key, expected in expected_counts.items():
        actual = metrics[key]
        assert (actual["tp"], actual["fp"], actual["fn"]) == expected[:3]
        assert actual["micro_f1"] == expected[3]
        assert actual["macro_f1"] == expected[4]

    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["label_set_accounting"] == {
        "official_release": 5401,
        "official_reference": 5401,
        "project": 5388,
        "same_name_same_sequence_intersection": 5388,
        "official_only": 13,
        "project_only": 0,
        "exact_reference_project_label_sets": 106,
        "different_reference_project_label_sets": 5282,
    }
    assert summary["record_classification_counts"] == dict(sorted(Counter(row["classification"] for row in actual_rows).items()))
    assert summary["official_reference_protocol"]["release_on_official_labels_5401"] == metrics["official"]
    assert summary["official_reference_protocol"]["release_on_official_labels_project_intersection_5388"] == metrics["intersection"]
    assert summary["project_label_protocol_audits"]["official_release_on_project_labels_5388"] == metrics["release_project"]
    assert summary["project_label_protocol_audits"]["project_replay_on_project_labels_5388"] == metrics["replay_project"]
    assert summary["canonical_policy"]["standalone_canonical_filter_applied"] is False
    assert summary["dssr_literature_attribution"]["status"] == "NOT_ESTABLISHED_BY_THESE_ARTIFACTS"
    manifest = json.loads(args.manifest.read_text())
    assert manifest["n_rejected"] == 13 and manifest["reject_reasons"] == {"bad_alphabet": 13}
    assert manifest["n_dropped_illegal_pairs"] == sum(row["n_release_to_project_min_loop_dropped"] for row in accepted)
    assert manifest["n_dropped_crossing_pairs"] == sum(row["n_release_to_project_crossing_dropped"] for row in accepted)

    inputs = [args.release, args.reference, args.project, args.manifest, args.replay]
    for item, recorded in zip(inputs, summary["inputs"]):
        assert digest(item) == recorded["sha256"]
        assert item.stat().st_size == recorded["bytes"]
    deliverables = [
        ROOT / "tools/audit_bpfold_label_protocol_20261127.py",
        ROOT / "tools/verify_bpfold_label_protocol_20261127.py",
        ROOT / "tests/test_bpfold_label_protocol_20261127.py",
        ROOT / "docs/plans/2026-11-27-bpfold-label-protocol-audit.md",
        args.output_dir / "per_record.jsonl.gz",
        args.output_dir / "summary.json",
    ]
    verification = {
        "schema_version": "bpfold.label_protocol_audit.independent_verification.v1",
        "audit_date": "2026-11-27",
        "status": "PASS_INDEPENDENT_PROTOCOL_AUDIT",
        "independence": "Separate BPSEQ parser, projection, pair validation, and metric implementation; generator module is not imported",
        "records_compared_field_by_field": len(actual_rows),
        "metrics_recomputed": metrics,
        "checks": [
            "all 5401 records and all per-record fields",
            "5401 official names equal official reference names",
            "5388 project sequences and projected release pair identities",
            "13 bad-alphabet exclusions",
            "106 equal and 5282 different reference/project label sets",
            "four requested metric identities and pooled TP/FP/FN",
            "input SHA256 and byte counts",
            "no standalone canonical filter and no unsupported DSSR/literature attribution",
        ],
        "external_test_used_for_model_selection": False,
        "gpu_task_launched": False,
        "historical_results_modified": False,
        "deliverables": [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)} for path in deliverables],
    }
    output = args.output_dir / "independent_verification.json"
    output.write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": verification["status"], "records": len(actual_rows), "output": str(output), "sha256": digest(output)}, indent=2))


if __name__ == "__main__":
    main()
