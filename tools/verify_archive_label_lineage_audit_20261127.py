#!/usr/bin/env python3
"""Independent second implementation for the archive label-lineage audit."""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

BASE = "bf463ccb4b5d0a8da2a2d2fed1696fbf63a8809c"
FIX = "e7ff4e76fe078178ab41d2052349627317b5a3e3"
SCHEMA = "archive_label_lineage.audit.v1"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/archive_label_lineage_audit_20261127"
SOURCE = ROOT / "source_data/archive_label_lineage_audit_20261127"
J = Path("/root/rna_jepa_data/ss_data/jsonl")
BD = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz")
BR = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2/official_assets/bpfold/BPfold_test_results_v0.2.tar.gz")
MX = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/mxfold2")
SPECS = {
    "archiveii_embok_clean": (2544, BD, "BPfold_data/test_data/archiveII/", MX / "archiveII.tar.gz", "data/archiveII/", BR, "BPfold_test_results/archiveII/"),
    "bprna_ts0": (1288, BD, "BPfold_data/bpRNA/TS0/", MX / "bpRNA.tar.gz", "data/bpRNA_dataset-canonicals/TS0/", BD, "BPfold_data/bpRNA/TS0/"),
    "testsetb": (428, MX / "Rivas.tar.gz", "data/TestSetB/", None, None, MX / "Rivas.tar.gz", "data/TestSetB/"),
}
EXPECTED_SHA = {
    BD: "819acdea53c6b093df14cab35bf079c4678325d2d5ea4296b904068d35959403",
    BR: "e8a3420e3458f7aad4c86b500f5d56d0e935b7da82b936675fefda64c5e03073",
    MX / "archiveII.tar.gz": "633bc50f8e9381f737b680fa6b4c79061a0a781773a5c38b342856eb1a6ac8a4",
    MX / "bpRNA.tar.gz": "69dc58fe6cfdf1ec178b1732b7da35b828dbbc7608e89d9debdde9d538937a55",
    MX / "Rivas.tar.gz": "4ec63e183ef21cdc7dc76561da5ad1543bdd53d1ec09c14d692cc0cd8da86db2",
    J / "archiveii_embok_clean.jsonl": "6c1fbaa4684cbe4644acdac2f31ae95e5f9b0eb36871f23d8148006f3d571399",
    J / "bprna_ts0.jsonl": "f21eb88c1e83fc25bc47cec823e1d4aec441b18a5305f20d5314411667d06882",
    J / "testsetb.jsonl": "78bba9255f36b301617c5017bdad79b7329190361c6822d2aec12f0043483a06",
    J / "bprna_tr0.jsonl": "93d328a49998435333413c61b8c1c530010d195a658f7a008e59bbfd3644a4cc",
    J / "bprna_tr1c.jsonl": "5f024ab2c488e790632bdc7383452f8ef00b92af8fb3b027623ca1b2a49ea854",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def norm(seq):
    return "".join(str(seq).strip().upper().split()).replace("T", "U")


def seq_sha(seq):
    return text_sha(norm(seq))


def pair_sha(pairs):
    body = json.dumps(sorted((min(i, j), max(i, j)) for i, j in pairs), separators=(",", ":"))
    return text_sha(body)


def bpseq(payload):
    seq, links = [], []
    for raw in payload.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line[0] == "#":
            continue
        columns = line.split()
        require(len(columns) >= 3, "BPSEQ column count")
        require(int(columns[0]) == len(seq) + 1, "BPSEQ index discontinuity")
        base = norm(columns[1])
        require(len(base) == 1, "BPSEQ base width")
        seq.append(base)
        links.append(int(columns[2]))
    require(seq, "empty BPSEQ")
    result = set()
    for i, link in enumerate(links):
        if link == 0:
            continue
        require(1 <= link <= len(seq) and links[link - 1] == i + 1, "BPSEQ partner mismatch")
        if link != i + 1:
            result.add(tuple(sorted((i, link - 1))))
    return "".join(seq), sorted(result)


def projection(pairs):
    kept = sorted(pair for pair in pairs if pair[1] - pair[0] > 3)
    short = sorted(pair for pair in pairs if pair[1] - pair[0] <= 3)
    crossing = []
    while True:
        degrees = {pair: 0 for pair in kept}
        relations = 0
        for i, left in enumerate(kept):
            a, b = left
            for right in kept[i + 1:]:
                c, d = right
                if a < c < b < d:
                    degrees[left] += 1
                    degrees[right] += 1
                    relations += 1
        if relations == 0:
            return kept, short, crossing
        victim = max(degrees, key=lambda pair: (degrees[pair], pair[1] - pair[0]))
        kept = [pair for pair in kept if pair != victim]
        crossing.append(victim)


def archive_records(path, prefix):
    answer = {}
    with tarfile.open(path, "r:gz") as package:
        for member in package:
            if not member.isfile() or not member.name.startswith(prefix) or not member.name.endswith(".bpseq"):
                continue
            payload = package.extractfile(member).read()
            sequence, pairs = bpseq(payload)
            name = member.name.rsplit("/", 1)[-1]
            require(name not in answer, "duplicate BPSEQ name")
            answer[name] = (sequence, pairs, member.name, hashlib.sha256(payload).hexdigest())
    require(answer, f"empty archive prefix {prefix}")
    return answer


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare(actual, expected, context):
    if actual != expected:
        if isinstance(actual, dict) and isinstance(expected, dict):
            differences = {key: (actual.get(key), expected.get(key)) for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)}
            raise AssertionError(f"{context}: {differences}")
        raise AssertionError(f"{context}: {actual!r} != {expected!r}")


def reference_compare(ref, mirror):
    if mirror is None:
        return {"status": "UNCHECKED_SINGLE_PACKAGE", "records": len(ref),
                "id_only_reference": None, "id_only_mirror": None,
                "sequence_mismatches": None, "pair_mismatches": None}, {}
    names = set(ref) | set(mirror)
    rows = {}
    sequence_bad = pair_bad = 0
    for name in names:
        left, right = ref.get(name), mirror.get(name)
        sequence_match = left is not None and right is not None and left[0] == right[0]
        pair_match = left is not None and right is not None and left[1] == right[1]
        sequence_bad += int(left is not None and right is not None and not sequence_match)
        pair_bad += int(left is not None and right is not None and not pair_match)
        rows[name] = {"sequence_match": sequence_match, "pair_match": pair_match,
                      "status": "EXACT" if sequence_match and pair_match else "MISMATCH"}
    status = "EXACT" if set(ref) == set(mirror) and not sequence_bad and not pair_bad else "MISMATCH"
    return {"status": status, "records": len(names),
            "id_only_reference": len(set(ref) - set(mirror)),
            "id_only_mirror": len(set(mirror) - set(ref)),
            "sequence_mismatches": sequence_bad, "pair_mismatches": pair_bad}, rows


def verify_records(summary):
    actual_rows = read_gzip(OUT / "records.jsonl.gz")
    actual = {}
    for row in actual_rows:
        key = row["dataset"], row["project_id"]
        require(key not in actual, f"duplicate output row {key}")
        actual[key] = row
    expected_count = 0
    for dataset, (expected_n, ref_path, ref_prefix, mirror_path, mirror_prefix, src_path, src_prefix) in SPECS.items():
        ref = archive_records(ref_path, ref_prefix)
        mirror = archive_records(mirror_path, mirror_prefix) if mirror_path else None
        src = ref if (ref_path, ref_prefix) == (src_path, src_prefix) else archive_records(src_path, src_prefix)
        mirror_summary, mirror_rows = reference_compare(ref, mirror)
        project = read_jsonl(J / f"{dataset}.jsonl")
        require(len(project) == expected_n, f"project n changed {dataset}")
        indexed = {row["name"]: (index, row) for index, row in enumerate(project)}
        require(len(indexed) == len(project), f"duplicate project IDs {dataset}")
        transform = (lambda value: "testsetb_" + value) if dataset == "testsetb" else (lambda value: value)
        source_names = {transform(name): name for name in set(ref) | set(src)}
        status_counts = collections.Counter()
        drops = collections.Counter()
        for project_id in sorted(set(source_names) | set(indexed)):
            source_id = source_names.get(project_id)
            r = ref.get(source_id)
            s = src.get(source_id)
            entry = indexed.get(project_id)
            index, row = entry if entry else (None, None)
            expected = {
                "schema_version": SCHEMA, "dataset": dataset, "source_id": source_id,
                "project_id": project_id, "project_row_index": index, "in_project": row is not None,
                "reference_member": r[2] if r else None, "reference_member_sha256": r[3] if r else None,
                "lineage_member": s[2] if s else None, "lineage_member_sha256": s[3] if s else None,
                "reference_cross_package": mirror_rows.get(source_id, {"status": "UNCHECKED_SINGLE_PACKAGE"}),
                "reference_sequence_sha256": seq_sha(r[0]) if r else None,
                "reference_pairs_sha256": pair_sha(r[1]) if r else None,
                "reference_pair_count": len(r[1]) if r else None,
                "reference_empty": not r[1] if r else None,
                "lineage_sequence_sha256": seq_sha(s[0]) if s else None,
                "lineage_pairs_sha256": pair_sha(s[1]) if s else None,
                "lineage_pair_count": len(s[1]) if s else None,
            }
            if row is None:
                if dataset == "archiveii_embok_clean":
                    status = "FILTERED_OUT_OF_CLEAN_SUBSET_UNCHECKED_RULE"
                elif r and set(r[0]) - set("ACGU"):
                    status = "PROJECT_REJECTED_BAD_ALPHABET"
                else:
                    status = "PROJECT_ABSENT_UNCHECKED"
                expected.update({"project_sequence_sha256": None, "project_pairs_sha256": None,
                                 "project_pair_count": None, "project_empty": None,
                                 "sequence_match_reference": None, "sequence_match_lineage": None,
                                 "pairs_equal_unfiltered_reference": None, "pairs_equal_projected_reference": None,
                                 "pairs_equal_projected_lineage": None, "lineage_min_loop_pairs_dropped": None,
                                 "lineage_crossing_pairs_dropped": None, "record_status": status})
                status_counts[status] += 1
            else:
                sequence, pairs = norm(row["seq"]), sorted(map(tuple, row["pairs"]))
                rp = projection(r[1]) if r else (None, None, None)
                sp = projection(s[1]) if s else (None, None, None)
                sequence_ref = r is not None and sequence == r[0]
                sequence_src = s is not None and sequence == s[0]
                equal_ref = r is not None and pairs == r[1]
                equal_ref_projection = r is not None and pairs == rp[0]
                equal_src_projection = s is not None and pairs == sp[0]
                if s is None:
                    status = "PROJECT_ONLY_UNCHECKED"
                elif not sequence_src or not equal_src_projection:
                    status = "LINEAGE_MISMATCH"
                elif dataset == "archiveii_embok_clean":
                    status = "EXACT_DERIVED_FROM_BPFOLD_PREDICTION_PROJECTION"
                else:
                    status = "EXACT_DERIVED_FROM_REFERENCE_PROJECTION"
                expected.update({
                    "project_sequence_sha256": seq_sha(sequence), "project_pairs_sha256": pair_sha(pairs),
                    "project_pair_count": len(pairs), "project_empty": not pairs,
                    "sequence_match_reference": sequence_ref, "sequence_match_lineage": sequence_src,
                    "pairs_equal_unfiltered_reference": equal_ref,
                    "pairs_equal_projected_reference": equal_ref_projection,
                    "pairs_equal_projected_lineage": equal_src_projection,
                    "lineage_min_loop_pairs_dropped": len(sp[1]),
                    "lineage_crossing_pairs_dropped": len(sp[2]), "record_status": status,
                })
                status_counts[status] += 1
                status_counts["unfiltered_reference_equal"] += int(equal_ref)
                status_counts["sequence_mismatch_reference"] += int(r is not None and not sequence_ref)
                status_counts["sequence_mismatch_lineage"] += int(not sequence_src)
                status_counts["project_empty"] += int(not pairs)
                status_counts["reference_empty_in_project"] += int(r is not None and not r[1])
                drops["min_loop"] += len(sp[1])
                drops["crossing"] += len(sp[2])
            key = dataset, project_id
            require(key in actual, f"missing record {key}")
            compare(actual[key], expected, f"record {key}")
            expected_count += 1
        item = summary["datasets"][dataset]
        dynamic = {
            "project_records": len(project), "official_reference_records": len(ref),
            "lineage_source_records": len(src), "project_pair_total": sum(len(row["pairs"]) for row in project),
            "reference_pair_total": sum(len(value[1]) for value in ref.values()),
            "record_status_counts": dict(sorted(status_counts.items())),
            "lineage_projection_drops": dict(drops),
            "reference_empty_records": sum(not value[1] for value in ref.values()),
            "project_empty_records": sum(not row["pairs"] for row in project),
            "reference_cross_package": mirror_summary,
        }
        for key, value in dynamic.items():
            compare(item[key], value, f"summary {dataset}.{key}")
        require(expected_count == len(actual_rows) or expected_count <= len(actual_rows), "record accounting")
    compare(len(actual_rows), expected_count, "total record rows")


def verify_training(summary):
    actual_rows = read_gzip(OUT / "training_overlap.jsonl.gz")
    actual = {(row["dataset"], row["project_row_index"], row["train_route"]): row for row in actual_rows}
    require(len(actual) == len(actual_rows), "duplicate training-overlap rows")
    train_data = {}
    for route in ("bprna_tr0", "bprna_tr1", "bprna_tr1c"):
        rows = read_jsonl(J / f"{route}.jsonl")
        mapping = collections.defaultdict(list)
        for row in rows:
            mapping[norm(row["seq"])].append(row["name"])
        train_data[route] = rows, mapping
    expected_count = 0
    for dataset, (expected_n, *_) in SPECS.items():
        project = read_jsonl(J / f"{dataset}.jsonl")
        for route, (train_rows, mapping) in train_data.items():
            audit = None
            if route != "bprna_tr1":
                short = route.replace("bprna_", "")
                path = ROOT / f"source_data/appendix_leakage/round39_clean_audits/{short}/{dataset}.audit.json"
                audit = json.loads(path.read_text())
                compare(audit["train_sha256"], file_sha(J / f"{route}.jsonl"), f"audit train SHA {dataset}/{route}")
                compare(audit["split_sha256"], file_sha(J / f"{dataset}.jsonl"), f"audit split SHA {dataset}/{route}")
            exact_declared = set(audit["removed_exact_indices"]) if audit else set()
            homology = set(audit["removed_homology_only_indices"]) if audit else set()
            exact = []
            for index, row in enumerate(project):
                matches = sorted(mapping.get(norm(row["seq"]), []))
                if matches:
                    exact.append(index)
                retained = index not in exact_declared and index not in homology
                expected = {
                    "dataset": dataset, "project_row_index": index, "project_id": row["name"],
                    "train_route": route, "project_sequence_sha256": seq_sha(row["seq"]),
                    "exact_sequence_match": bool(matches), "matched_train_ids": matches,
                    "frozen_homology_only_removal": index in homology,
                    "retained_in_frozen_clean_subset": retained if audit else None,
                    "homology_status": "EXACT_AUDIT_ARTIFACT" if audit else "UNCHECKED",
                }
                key = dataset, index, route
                require(key in actual, f"missing training row {key}")
                compare(actual[key], expected, f"training row {key}")
                expected_count += 1
            item = summary["training_overlap"][dataset][route]
            compare(item["train_records"], len(train_rows), f"train n {dataset}/{route}")
            compare(item["train_sha256"], file_sha(J / f"{route}.jsonl"), f"train SHA {dataset}/{route}")
            compare(item["exact_sequence_overlap_records"], len(exact), f"exact overlap {dataset}/{route}")
            compare(item["exact_sequence_overlap_indices"], exact, f"exact indices {dataset}/{route}")
            compare(item["homology_protocol_status"], "EXACT_AUDIT_ARTIFACT" if audit else "UNCHECKED", f"homology status {dataset}/{route}")
            if audit:
                union = exact_declared | homology
                retained_rows = [row for index, row in enumerate(project) if index not in union]
                clean_sha = hashlib.sha256("".join(json.dumps(row) + "\n" for row in retained_rows).encode()).hexdigest()
                compare(set(exact), exact_declared, f"declared exact {dataset}/{route}")
                compare(len(retained_rows), audit["clean_n"], f"clean n {dataset}/{route}")
                compare(clean_sha, audit["clean_sha256"], f"clean SHA {dataset}/{route}")
                for key, value in {"protocol": audit["protocol"], "removed_exact": audit["removed_exact"],
                                   "removed_homology_only": audit["removed_homology_only"],
                                   "clean_n": audit["clean_n"], "clean_sha256": audit["clean_sha256"],
                                   "clean_subset_reconstruction": "EXACT"}.items():
                    compare(item[key], value, f"clean summary {dataset}/{route}.{key}")
                compare(item["audit_artifact_sha256"], file_sha(item["audit_artifact"]), f"audit artifact SHA {dataset}/{route}")
    compare(len(actual_rows), expected_count, "total training rows")


def verify_predictions(summary):
    document = json.loads((OUT / "prediction_recompute.json").read_text())
    registry = json.loads((ROOT / "manifests/foldnar_v1/prediction_registry.json").read_text())
    expected = []
    for dataset, (panel_n, *_) in SPECS.items():
        for row in sorted((item for item in registry["rows"] if item.get("dataset_id") == dataset), key=lambda item: item["prediction_id"]):
            available = bool(row.get("exists") and row.get("status") == "FROZEN" and row.get("prediction_schema") in {"per_sequence", "predictions"})
            if available:
                action = "RESCORE_FROZEN_RECORD_LEVEL_PREDICTIONS"
            elif row.get("status") == "SUMMARY_ONLY":
                action = "BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED"
            else:
                action = "BLOCKED_REGISTERED_PREDICTION_MISSING"
            path = Path(row["path"]) if row.get("path") else None
            current = file_sha(path) if path and path.is_file() else None
            expected.append({
                "dataset": dataset, "prediction_id": row["prediction_id"], "registry_status": row.get("status"),
                "prediction_schema": row.get("prediction_schema"), "path": row.get("path"),
                "registry_sha256": row.get("sha256"), "current_sha256": current,
                "sha_matches_registry": current == row.get("sha256") if current else None,
                "registered_n_sequences": row.get("n_sequences"), "project_panel_n": panel_n,
                "record_level_rescore_available": available, "required_action": action,
                "target_labels": "UNFILTERED_OFFICIAL_REFERENCE_BPSEQ_ON_MATCHED_IDS",
                "existing_project_label_result_eligibility": "CIRCULAR_AUDIT_ONLY" if dataset == "archiveii_embok_clean" else "PROJECTED_REFERENCE_PROTOCOL_ONLY",
                "full_official_population": "REQUIRES_COVERAGE_ACCOUNTING_AND_PREDICTIONS_FOR_OFFICIAL_IDS_ABSENT_FROM_PROJECT_PANEL",
                "new_inference_executed_by_this_audit": False, "external_test_model_selection_allowed": False,
            })
    compare(document["rows"], expected, "prediction recompute rows")
    for dataset in SPECS:
        rows = [row for row in expected if row["dataset"] == dataset]
        item = summary["prediction_recompute"][dataset]
        compare(item["registered_prediction_identities"], len(rows), f"prediction count {dataset}")
        compare(item["action_counts"], dict(sorted(collections.Counter(row["required_action"] for row in rows).items())), f"prediction actions {dataset}")
        compare(item["sha_mismatches"], [row["prediction_id"] for row in rows if row["sha_matches_registry"] is False], f"prediction SHA mismatches {dataset}")


def verify_inputs_and_delivery(summary):
    manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    for item in manifest["inputs"]:
        path = Path(item["path"])
        require(path.is_file(), f"missing source input {path}")
        compare(path.stat().st_size, item["bytes"], f"source bytes {path}")
        compare(file_sha(path), item["sha256"], f"source SHA {path}")
    compare(manifest["base_commit"], BASE, "source base")
    require(subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", FIX, "HEAD"]).returncode == 0, "fix ancestry")
    for path, expected in EXPECTED_SHA.items():
        compare(file_sha(path), expected, f"pinned source {path}")
    compare(file_sha(SOURCE / "official_docs/bprna_PMC6009582.fullText.xml"), manifest["documentation"]["bprna_paper"]["full_text_sha256"], "bpRNA documentation SHA")
    registry_hashes = {name: file_sha(ROOT / name) for name in summary["shared_registry_sha256_before"]}
    compare(summary["shared_registry_sha256_before"], registry_hashes, "registry before")
    compare(summary["shared_registry_sha256_after"], registry_hashes, "registry after")
    require(summary["model_scores_computed"] is False and summary["ranking_performed"] is False, "external Test scoring/ranking boundary")
    require(summary["new_training_or_inference"] is False and summary["gpu_used"] is False, "compute boundary")
    receipt = json.loads((OUT / "build_receipt.json").read_text())
    for relative, item in receipt["outputs"].items():
        path = ROOT / relative
        compare(path.stat().st_size, item["bytes"], f"receipt bytes {relative}")
        compare(file_sha(path), item["sha256"], f"receipt SHA {relative}")


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    verify_records(summary)
    print("VERIFY_RECORDS_PASS 5701", flush=True)
    verify_training(summary)
    print("VERIFY_TRAINING_PASS 12780", flush=True)
    verify_predictions(summary)
    print("VERIFY_PREDICTIONS_PASS 97", flush=True)
    verify_inputs_and_delivery(summary)
    print("VERIFY_INPUTS_AND_DELIVERY_PASS", flush=True)
    files = [
        ROOT / "tools/build_archive_label_lineage_audit_20261127.py",
        ROOT / "tools/verify_archive_label_lineage_audit_20261127.py",
        ROOT / "tests/test_archive_label_lineage_audit_20261127.py",
        ROOT / "docs/plans/2026-11-27-archive-label-lineage-audit.md",
        SOURCE / "source_manifest.json",
        OUT / "records.jsonl.gz", OUT / "training_overlap.jsonl.gz",
        OUT / "prediction_recompute.json", OUT / "summary.json",
    ]
    verification = {
        "schema_version": "archive_label_lineage.independent_verification.round2.v1",
        "status": "PASS_INDEPENDENT_SECOND_IMPLEMENTATION",
        "generator_imported": False,
        "records_compared_field_by_field": 5701,
        "project_records_verified": 4260,
        "official_only_or_filtered_records_verified": 1441,
        "training_overlap_rows_compared_field_by_field": 12780,
        "prediction_recompute_rows_compared_field_by_field": 97,
        "checks": [
            "independent BPSEQ parser and crossing/min-loop projection",
            "ID, sequence, pair-set, archive-member and per-record SHA",
            "BPfold_data versus MXfold2 reference packages for ArchiveII and TS0",
            "TR0/TR1/TR1c exact sequence overlap and TR0/TR1c frozen clean JSONL SHA",
            "all registered prediction actions and current artifact SHA",
            "official documentation and read-only shared registry identities",
        ],
        "external_test_model_selection": False,
        "new_training_or_inference": False,
        "gpu_used": False,
        "deliverables": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": file_sha(path)} for path in files},
    }
    (OUT / "verification_round2.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": verification["status"], "records": 5701, "training_rows": 12780, "prediction_rows": 97}, sort_keys=True))


if __name__ == "__main__":
    main()
