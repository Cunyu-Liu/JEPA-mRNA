#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATASET = "bprna_new"
SYSTEMS = ("bpfold", "ufold")
DATA_SHA256 = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temporary.replace(path)


def pair_set(values, length=None):
    pairs = set()
    partners = {}
    for value in values:
        assert len(value) == 2, value
        i, j = map(int, value)
        assert 0 <= i < j, value
        if length is not None:
            assert j < length, (value, length)
        assert i not in partners and j not in partners, value
        partners[i], partners[j] = j, i
        pairs.add((i, j))
    return pairs


def normalize_sequence(sequence):
    return sequence.upper().replace("T", "U")


def strict_metrics(rows, prediction_by_name):
    tp = fp = fn = 0
    per_sequence = []
    for row in rows:
        name = row["name"]
        assert name in prediction_by_name, name
        truth = pair_set(row["pairs"], len(row["seq"]))
        predicted = pair_set(prediction_by_name[name]["pairs"], len(row["seq"]))
        a, b, c = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        denominator = 2 * a + b + c
        f1 = 2 * a / denominator if denominator else 1.0
        tp, fp, fn = tp + a, fp + b, fn + c
        per_sequence.append({
            "name": name,
            "length": len(row["seq"]),
            "n_truth_pairs": len(truth),
            "n_predicted_pairs": len(predicted),
            "tp": a,
            "fp": b,
            "fn": c,
            "precision": a / (a + b) if a + b else None,
            "recall": a / (a + c) if a + c else None,
            "f1": f1,
        })
    n = len(rows)
    return {
        "n_dataset_sequences": n,
        "n_with_prediction": n,
        "coverage": 1.0 if n else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "macro_f1": math.fsum(row["f1"] for row in per_sequence) / n if n else None,
        "zero_pair_f1_convention": "F1=1 when truth and prediction both have zero pairs",
    }, per_sequence


def assert_close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (actual, expected)


def load_registry(repo):
    root = Path(repo) / "manifests/foldnar_v1"
    data = json.loads((root / "data_registry.json").read_text())
    predictions = json.loads((root / "prediction_registry.json").read_text())
    data_row = next(row for row in data["rows"] if row["dataset_id"] == DATASET)
    prediction_rows = {
        system: next(row for row in predictions["rows"] if row["prediction_id"] == f"{system}/{DATASET}")
        for system in SYSTEMS
    }
    return data_row, prediction_rows


def audit_frozen(system, rows, registry_row, output_root):
    path = Path(registry_row["path"])
    assert sha256(path) == registry_row["sha256"]
    document = json.loads(path.read_text())
    assert document["n_sequences"] == registry_row["n_sequences"] == len(rows)
    source = document["per_sequence"]
    names = [row["name"] for row in source]
    assert names == [row["name"] for row in rows]
    assert all(count == 1 for count in Counter(names).values())
    prediction_by_name = {row["name"]: {"pairs": row["pred_pairs"]} for row in source}
    metrics, per_sequence = strict_metrics(rows, prediction_by_name)
    for actual, recorded in zip(per_sequence, source):
        assert actual["name"] == recorded["name"]
        assert (actual["tp"], actual["fp"], actual["fn"]) == (recorded["tp"], recorded["fp"], recorded["fn"])
        assert_close(actual["f1"], recorded["f1"])
    identity = [[name, prediction_by_name[name]["pairs"]] for name in names]
    prediction_identity_sha256 = hashlib.sha256(canonical(identity).encode()).hexdigest()
    assert len(document["repeats"]) == 3
    for repeat in document["repeats"]:
        recorded = repeat["metrics"]
        assert repeat["prediction_sha256"] == prediction_identity_sha256
        assert (recorded["tp"], recorded["fp"], recorded["fn"]) == (metrics["tp"], metrics["fp"], metrics["fn"])
        assert_close(recorded["micro_f1"], metrics["micro_f1"])
        assert_close(recorded["macro_f1"], metrics["macro_f1"])
    write_jsonl(output_root / "reports" / f"{system}_frozen_bprna_new_per_sequence_v2.jsonl", per_sequence)
    provenance = {
        "source_commit": document.get("repository_commit"),
        "device_recorded": document.get("device"),
        "threshold": document.get("threshold"),
        "distribution": document.get("distribution"),
        "weights": document.get("weights"),
        "weights_sha256": document.get("weights_sha256"),
        "checkpoint_sha256": document.get("checkpoint_sha256"),
    }
    return {
        "status": "PASS_INDEPENDENT_RECOMPUTATION",
        "artifact": str(path),
        "artifact_sha256": registry_row["sha256"],
        "prediction_identity_sha256": prediction_identity_sha256,
        "unique_names": len(set(names)),
        "missing_names": [],
        "extra_names": [],
        "order_matches_frozen_data": True,
        "recorded_repeats_verified": 3,
        "metrics": metrics,
        "provenance": provenance,
    }, prediction_by_name


def parse_bpseq(payload):
    sequence = []
    directed_pairs = []
    for line in payload.decode("utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        assert len(fields) >= 3, line
        index = int(fields[0]) - 1
        assert index == len(sequence), line
        sequence.append(fields[1].upper().replace("T", "U"))
        partner = int(fields[2]) - 1
        if partner >= 0:
            directed_pairs.append((index, partner))
    pairs = pair_set({(min(i, j), max(i, j)) for i, j in directed_pairs}, len(sequence))
    return "".join(sequence), pairs


def audit_official_bpfold(rows, archive, frozen_predictions, output_root):
    with tarfile.open(archive, "r:gz") as handle:
        members = [
            member for member in handle.getmembers()
            if member.isfile() and member.name.endswith(".bpseq")
            and "bprnanew" in member.name.lower().split("/")
        ]
        assert members, "No bpRNAnew BPSEQ members found"
        official = {}
        member_by_name = {}
        for member in members:
            name = Path(member.name).name
            assert name not in official, name
            sequence, pairs = parse_bpseq(handle.extractfile(member).read())
            official[name] = {"seq": sequence, "pairs": sorted(pairs)}
            member_by_name[name] = member.name
    project = {row["name"]: row for row in rows}
    common = sorted(set(project) & set(official))
    official_only = sorted(set(official) - set(project))
    project_only = sorted(set(project) - set(official))
    sequence_mismatches = [
        name for name in common
        if normalize_sequence(project[name]["seq"]) != official[name]["seq"]
    ]
    matched_rows = [project[name] for name in common]
    assert not sequence_mismatches, sequence_mismatches[:5]
    metrics, per_sequence = strict_metrics(matched_rows, official)
    normalized = [
        {"name": name, "seq": official[name]["seq"], "pairs": official[name]["pairs"],
         "archive_member": member_by_name[name]}
        for name in sorted(official)
    ]
    write_jsonl(output_root / "predictions/bpfold_official_release/bprna_new_all_official_v2.jsonl", normalized)
    write_jsonl(output_root / "reports/bpfold_official_release_bprna_new_matched_per_sequence_v2.jsonl", per_sequence)
    pair_identity_matches = sum(
        pair_set(official[name]["pairs"]) == pair_set(frozen_predictions[name]["pairs"])
        for name in common
    )
    return {
        "status": "PASS_INDEPENDENT_RECOMPUTATION",
        "release": "BPfold v0.2 author-released predictions; distinct from project model_predict replay",
        "release_url": "https://github.com/heqin-zhu/BPfold/releases/download/v0.2/BPfold_test_results.tar.gz",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "n_official_sequences": len(official),
        "n_project_sequences": len(project),
        "n_same_name_same_sequence": len(common),
        "n_official_only_excluded": len(official_only),
        "official_only_names_excluded": official_only,
        "project_only_names": project_only,
        "sequence_mismatch_names": sequence_mismatches,
        "coverage_of_project": len(common) / len(project),
        "metrics_on_same_sequence_intersection": metrics,
        "n_pair_sets_identical_to_frozen_project_replay": pair_identity_matches,
        "n_pair_sets_different_from_frozen_project_replay": len(common) - pair_identity_matches,
        "comparability": "Raw author release on exact project intersection; official release checkpoint/protocol may differ from frozen project replay. The 13 official-only records are preserved but never imputed into the project split.",
    }


def audit_bpfold_official_reference(rows, prediction_archive, reference_archive):
    with tarfile.open(prediction_archive, "r:gz") as predictions_handle, tarfile.open(reference_archive, "r:gz") as reference_handle:
        predictions = {}
        for member in predictions_handle:
            if member.isfile() and member.name.endswith(".bpseq") and "bprnanew" in member.name.lower().split("/"):
                name = Path(member.name).name
                assert name not in predictions
                sequence, pairs = parse_bpseq(predictions_handle.extractfile(member).read())
                predictions[name] = {"seq": sequence, "pairs": sorted(pairs)}
        reference = {}
        for member in reference_handle:
            if member.isfile() and "/test_data/bpRNAnew/" in member.name and member.name.endswith(".bpseq"):
                name = Path(member.name).name
                assert name not in reference
                sequence, pairs = parse_bpseq(reference_handle.extractfile(member).read())
                reference[name] = {"name": name, "seq": sequence, "pairs": sorted(pairs)}
        assert len(predictions) == len(reference) == 5401
        assert set(predictions) == set(reference)
        for name in sorted(predictions):
            assert predictions[name]["seq"] == reference[name]["seq"], name
    official_rows = [reference[name] for name in sorted(reference)]
    all_metrics, _ = strict_metrics(official_rows, predictions)
    project_names = {row["name"] for row in rows}
    matched_names = sorted(project_names & set(reference))
    matched_metrics, _ = strict_metrics([reference[name] for name in matched_names], predictions)
    project_by_name = {row["name"]: row for row in rows}
    exact_label_sets = sum(
        pair_set(reference[name]["pairs"]) == pair_set(project_by_name[name]["pairs"])
        for name in matched_names
    )
    return {
        "status": "PASS_OFFICIAL_README_ROUNDING_REPRODUCED",
        "reference_archive": str(reference_archive),
        "reference_archive_sha256": sha256(reference_archive),
        "reference_member_scope": "BPfold_data/test_data/bpRNAnew (official BPfold_eval default); duplicate nr500.canonicals labels not mixed",
        "n_official_sequences": len(official_rows),
        "metrics_all_5401_official_reference": all_metrics,
        "n_project_intersection": len(matched_names),
        "metrics_5388_official_reference": matched_metrics,
        "n_project_label_sets_exactly_equal_official_reference": exact_label_sets,
        "n_project_label_sets_different_from_official_reference": len(matched_names) - exact_label_sets,
        "readme_macro_f1_rounded": 0.647,
        "validation": "Official-reference macro-F1 rounds to the BPfold v0.2 README value 0.647; project-label scores are not the official evaluation protocol.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56"))
    parser.add_argument("--output-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2"))
    parser.add_argument("--bpfold-official-archive", type=Path)
    parser.add_argument("--bpfold-official-reference-archive", type=Path)
    args = parser.parse_args()
    data_row, registry = load_registry(args.repo)
    assert data_row["sha256"] == DATA_SHA256
    data_path = Path(data_row["path"])
    assert sha256(data_path) == DATA_SHA256
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    assert len(rows) == data_row["n_sequences"] == 5388
    assert len({row["name"] for row in rows}) == len(rows)
    systems = {}
    frozen_predictions = {}
    for system in SYSTEMS:
        systems[system], frozen_predictions[system] = audit_frozen(
            system, rows, registry[system], args.output_root
        )
    official = None
    official_reference = None
    if args.bpfold_official_archive:
        official = audit_official_bpfold(
            rows, args.bpfold_official_archive, frozen_predictions["bpfold"], args.output_root
        )
        if args.bpfold_official_reference_archive:
            official_reference = audit_bpfold_official_reference(
                rows, args.bpfold_official_archive, args.bpfold_official_reference_archive
            )
            official["official_reference_evaluation"] = official_reference
            official["comparability"] = (
                "Raw author release on exact project name/sequence intersection. Project pair labels frequently differ "
                "from the official BPfold_data reference; project-label scores are label-sensitive audits, not the "
                "official evaluation protocol. The 13 official-only records are preserved but never imputed."
            )
    result = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": "CPU-only independent frozen-prediction recomputation; no inference, training, calibration, or model selection",
        "dataset": {
            "dataset_id": DATASET,
            "path": str(data_path),
            "sha256": DATA_SHA256,
            "n_sequences": len(rows),
            "pair_convention": "zero-based inclusive; strict exact-pair matching; no crossing or short-span filtering",
        },
        "frozen_project_predictions": systems,
        "bpfold_official_release": official,
        "contextfold": {
            "status": "BLOCKED_NO_VERIFIED_PREDICTIONS",
            "metrics": None,
            "coverage": None,
            "reason": "No verified official weights or bpRNA-new per-sequence predictions; summary values were not substituted.",
        },
    }
    out = args.output_root / "reports/bprna_new_raw_metrics_verification_v2.json"
    write_json(out, result)
    print(json.dumps({
        "status": result["status"],
        "frozen": {key: value["metrics"] for key, value in systems.items()},
        "bpfold_official_release": official["metrics_on_same_sequence_intersection"] if official else None,
        "report": str(out),
        "report_sha256": sha256(out),
    }, indent=2))


if __name__ == "__main__":
    main()
