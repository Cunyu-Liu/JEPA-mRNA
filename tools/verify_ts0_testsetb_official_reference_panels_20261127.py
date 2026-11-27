#!/usr/bin/env python3
"""Independent second implementation for TS0/TestSetB official-reference panels."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/ts0_testsetb_official_reference_20261127"
SOURCE = ROOT / "source_data/ts0_testsetb_official_reference_20261127"
REPORT = ROOT / "docs/plans/2026-11-27-ts0-testsetb-official-reference-panels.md"
PRIOR = ROOT / "artifacts/archive_label_lineage_audit_20261127/summary.json"
RECOVERY = ROOT / "source_data/ts0_official_coverage_recovery_20261127/recovery_manifest.json"
J = Path("/root/rna_jepa_data/ss_data/jsonl")
BD = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz")
MX = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/mxfold2")
SPECS = {
    "bprna_ts0": (1288, 1305, "f21eb88c1e83fc25bc47cec823e1d4aec441b18a5305f20d5314411667d06882", BD, "BPfold_data/bpRNA/TS0/", "819acdea53c6b093df14cab35bf079c4678325d2d5ea4296b904068d35959403", MX / "bpRNA.tar.gz", "data/bpRNA_dataset-canonicals/TS0/", "69dc58fe6cfdf1ec178b1732b7da35b828dbbc7608e89d9debdde9d538937a55", "bprna_ts0_official_reference_1305", "bprna_ts0_projected_reference_1288", "bprna_ts0_official_1305.jsonl.gz", "bprna_ts0_projected_1288.jsonl.gz", 26),
    "testsetb": (428, 430, "78bba9255f36b301617c5017bdad79b7329190361c6822d2aec12f0043483a06", MX / "Rivas.tar.gz", "data/TestSetB/", "4ec63e183ef21cdc7dc76561da5ad1543bdd53d1ec09c14d692cc0cd8da86db2", None, None, None, "testsetb_official_reference_430", "testsetb_projected_reference_428", "testsetb_official_430.jsonl.gz", "testsetb_projected_428.jsonl.gz", 20),
}
DETERMINISTIC = {"contrafold", "eternafold", "mxfold2", "rnastructure_fold", "rnastructure_maxexpect"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def compare(actual, expected, context):
    if actual != expected:
        if isinstance(actual, dict) and isinstance(expected, dict):
            diff = {key: (actual.get(key), expected.get(key)) for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)}
            raise AssertionError(f"{context}: {diff}")
        raise AssertionError(f"{context}: {actual!r} != {expected!r}")


def gzip_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pairs(values, length):
    answer, endpoints = set(), []
    for value in values:
        i, j = int(value[0]), int(value[1])
        require(0 <= i < j < length, "pair bounds")
        endpoints.extend((i, j))
        answer.add((i, j))
    require(len(endpoints) == len(set(endpoints)), "partner conflict")
    return answer


def bpseq(payload):
    bases, links = [], []
    for raw in payload.decode().splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.split()
        require(int(fields[0]) == len(bases) + 1, "BPSEQ index")
        bases.append(fields[1].upper().replace("T", "U"))
        links.append(int(fields[2]))
    result = set()
    for i, link in enumerate(links):
        if link:
            require(1 <= link <= len(bases) and links[link - 1] == i + 1, "BPSEQ partner")
            if link != i + 1:
                result.add((min(i, link - 1), max(i, link - 1)))
    return "".join(bases), pairs(result, len(bases))


def tar_load(path, prefix):
    answer = {}
    with tarfile.open(path, "r:gz") as package:
        for member in package:
            if not member.isfile() or not member.name.startswith(prefix) or not member.name.endswith(".bpseq"):
                continue
            payload = package.extractfile(member).read()
            sequence, pair_set = bpseq(payload)
            answer[member.name.rsplit("/", 1)[-1]] = (sequence, pair_set, member.name, hashlib.sha256(payload).hexdigest())
    return answer


def verify_panels(dataset):
    (project_n, official_n, project_sha, ref_path, ref_prefix, ref_sha, mirror_path,
     mirror_prefix, mirror_sha, full_id, project_id, full_file, project_file, available_n) = SPECS[dataset]
    compare(sha(J / f"{dataset}.jsonl"), project_sha, f"project SHA {dataset}")
    compare(sha(ref_path), ref_sha, f"reference SHA {dataset}")
    official = tar_load(ref_path, ref_prefix)
    mirror = tar_load(mirror_path, mirror_prefix) if mirror_path else None
    if mirror:
        compare(sha(mirror_path), mirror_sha, f"mirror SHA {dataset}")
    require(len(official) == official_n and (not mirror or official.keys() == mirror.keys()), f"official population {dataset}")
    transform = (lambda name: "testsetb_" + name) if dataset == "testsetb" else (lambda name: name)
    expected_full = []
    for source_id in sorted(official):
        ref = official[source_id]
        if mirror:
            compare((ref[0], ref[1]), (mirror[source_id][0], mirror[source_id][1]), f"mirror record {source_id}")
        core = {"name": transform(source_id), "source_id": source_id, "seq": ref[0],
                "pairs": [list(pair) for pair in sorted(ref[1])],
                "sequence_sha256": hashlib.sha256(ref[0].encode()).hexdigest(),
                "pair_set_sha256": value_sha(sorted(ref[1]))}
        row = {**core, "record_sha256": value_sha(core), "reference_member": ref[2],
               "reference_member_sha256": ref[3], "reference_cross_package_status": "EXACT" if mirror else "UNCHECKED_SINGLE_PACKAGE",
               "mirror_member": mirror[source_id][2] if mirror else None,
               "mirror_member_sha256": mirror[source_id][3] if mirror else None}
        expected_full.append(row)
    compare(gzip_rows(SOURCE / full_file), expected_full, f"full panel {dataset}")
    project = [json.loads(line) for line in (J / f"{dataset}.jsonl").read_text().splitlines() if line.strip()]
    require(len(project) == project_n, f"project n {dataset}")
    by_name = {row["name"]: row for row in expected_full}
    expected_project = [by_name[row["name"]] for row in project]
    compare(gzip_rows(SOURCE / project_file), expected_project, f"projected panel {dataset}")
    return expected_full, expected_project


def model_form(baseline_id):
    if baseline_id == "xens2":
        return "three_checkpoint_ensemble"
    if baseline_id == "bpfold":
        return "six_checkpoint_ensemble"
    return "deterministic_single_predictor" if baseline_id in DETERMINISTIC else "single_model"


def inventory(dataset):
    registry = json.loads((ROOT / "manifests/foldnar_v1/prediction_registry.json").read_text())
    result = []
    for row in sorted((item for item in registry["rows"] if item.get("dataset_id") == dataset), key=lambda item: item["prediction_id"]):
        available = bool(row.get("exists") and row.get("status") == "FROZEN" and row.get("prediction_schema") in {"per_sequence", "predictions"})
        baseline_id = row.get("baseline_id", row.get("model"))
        result.append({"prediction_id": row["prediction_id"], "baseline_id": baseline_id, "source_group": "REGISTERED",
                       "registry_status": row.get("status"), "path": row.get("path"), "sha256": row.get("sha256"),
                       "data_path": row.get("data_path"), "available": available, "model_form": model_form(baseline_id),
                       "prediction_origin": "project_bpfold_replay" if baseline_id == "bpfold" else "registered_frozen_prediction"})
    digests = {"bprna_ts0": {1: "5f670b2bd64a1091bb00cbd220ecb3228310add49b413a240cc5bcd093c5792f", 2: "77214cb915ce99d478f621e8161f01b158a68fe753d83879df84033a1a79b773"},
               "testsetb": {1: "fe3b7e261b1f38a6d61762f88f878c5f6e6bd2454733c1889aa29fe8d30fd893", 2: "588b99cebc99c5c7293d9768179074a29d748d4fa8f493df8878935d3a458f7c"}}
    for seed in (1, 2):
        result.append({"prediction_id": f"rinalmo_2d_s{seed}/{dataset}", "baseline_id": "r2d", "source_group": "INDEPENDENT_AUDIT_ASSET",
                       "registry_status": "UNREGISTERED_INDEPENDENT_AUDIT_ASSET",
                       "path": f"/root/rna_jepa_data/eval_decision/ow_rinalmo_r2dtr1c_b4_s{seed}_step20000/{dataset}/result.json",
                       "sha256": digests[dataset][seed], "data_path": str(J / f"{dataset}.jsonl"), "available": True,
                       "model_form": "single_model", "prediction_origin": f"additional_frozen_r2d_seed{seed}"})
    if dataset == "bprna_ts0":
        recovery = json.loads(RECOVERY.read_text())
        by_id = {row["prediction_id"]: index for index, row in enumerate(result)}
        for recovered in recovery["prediction_entries"]:
            if recovered["prediction_id"] in by_id:
                result[by_id[recovered["prediction_id"]]] = recovered
            else:
                result.append(recovered)
    require(len(result) == SPECS[dataset][-1] + 14 and sum(row["available"] for row in result) == SPECS[dataset][-1], f"inventory {dataset}")
    return result


def predictions(entry, dataset):
    path = Path(entry["path"])
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            document = json.load(handle)
    else:
        document = json.loads(path.read_text())
    if document.get("data"):
        compare(document["data"], entry["data_path"], f"data path {entry['prediction_id']}")
    if document.get("data_sha256"):
        require(document["data_sha256"] in {SPECS[dataset][2], sha(SOURCE / SPECS[dataset][11])}, f"data SHA {entry['prediction_id']}")
    if "per_sequence" in document:
        source = document["per_sequence"]
    elif entry["prediction_id"].startswith("rnastructure_fold/"):
        source = document["results"]["fold"]["per_sequence"]
    elif entry["prediction_id"].startswith("rnastructure_maxexpect/"):
        source = document["results"]["maxexpect"]["per_sequence"]
    else:
        source = document.get("predictions")
    require(source is not None, "record-level predictions")
    result = {}
    for row in source:
        pair_values = row.get("pred_pairs", row.get("pairs"))
        require(pair_values is not None and row["name"] not in result, "prediction identity")
        result[row["name"]] = (row.get("seq"), pair_values)
    return result


def exposure(entry, dataset, baselines, prior):
    training = baselines.get(entry["baseline_id"], {}).get("training_data", {})
    project_ids = training.get("project_training_dataset_ids", [])
    corpora = project_ids or training.get("external_training_corpora", [])
    answer = {"baseline_id": entry["baseline_id"], "training_provenance_status": training.get("status", "UNCHECKED"),
              "training_corpora": corpora, "route": None, "exact_sequence_overlap": None,
              "homology_only_overlap": None, "clean_n": None, "clean_sha256": None,
              "evidence_status": "UNCHECKED_EXTERNAL_TRAINING_OVERLAP", "full_project_panel_exposure_status": "UNCHECKED"}
    route = "bprna_tr1c" if "bprna_tr1c" in project_ids else "bprna_tr0" if entry["baseline_id"] == "rinalmo_ft" else None
    if route:
        evidence = prior["training_overlap"][dataset][route]
        answer.update({"route": route, "exact_sequence_overlap": evidence["exact_sequence_overlap_records"],
                       "homology_only_overlap": evidence["removed_homology_only"], "clean_n": evidence["clean_n"],
                       "clean_sha256": evidence["clean_sha256"], "evidence_status": "FROZEN_EXACT_AND_MMSEQS_AUDIT_RECONSTRUCTED",
                       "full_project_panel_exposure_status": "EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT" if evidence["exact_sequence_overlap_records"] else "HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN" if evidence["removed_homology_only"] else "UNCHECKED"})
        return answer
    if training.get("status") == "NOT_APPLICABLE_NOT_SEQUENCE_FITTED":
        answer.update({"route": "NOT_SEQUENCE_FITTED", "evidence_status": "NOT_APPLICABLE_NOT_SEQUENCE_FITTED",
                       "full_project_panel_exposure_status": "NOT_SEQUENCE_FITTED"})
        return answer
    if entry["baseline_id"] == "mxfold2":
        summary = json.loads((ROOT / "source_data/appendix_leakage/round39_mxfold2_clean_trainsetab_summary.json").read_text())
        audit = json.loads(Path(summary["splits"][dataset]["clean_audit"]).read_text())
        answer.update({"route": "mxfold2_trainsetab", "exact_sequence_overlap": audit["removed_exact"],
                       "homology_only_overlap": audit["removed_homology_only"], "clean_n": audit["clean_n"],
                       "clean_sha256": audit["clean_sha256"], "evidence_status": "OFFICIAL_TRAINSETAB_AUDIT_ARTIFACT; FULL_EXPOSURE_NOT_PROVEN",
                       "full_project_panel_exposure_status": "AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED"})
        return answer
    if entry["baseline_id"] == "nucleicbert_frozen_official":
        summary = json.loads((ROOT / "source_data/appendix_leakage/round39_nucleicbert_frozen_clean_audit_summary.json").read_text())
        evidence = summary["per_split"][dataset]
        answer.update({"route": "rnastralign_tr0_style", "exact_sequence_overlap": evidence["removed_exact"],
                       "homology_only_overlap": evidence["removed_homology_only"], "clean_n": evidence["clean_n"],
                       "clean_sha256": evidence["clean_sha256"], "evidence_status": "AUTHOR_TR0_STYLE_AUDIT_ARTIFACT; FULL_EXPOSURE_NOT_PROVEN",
                       "full_project_panel_exposure_status": "AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED"})
    return answer


def score(panel, prediction_rows, dataset, panel_id, prediction_id, actual_records):
    totals = [0, 0, 0]
    values, f1_by_name, missing = [], {}, []
    for label in panel:
        name = label["name"]
        if name not in prediction_rows:
            missing.append(name)
            continue
        sequence, raw_pairs = prediction_rows[name]
        if sequence is not None:
            compare(sequence.upper().replace("T", "U"), label["seq"], f"prediction seq {name}")
        truth, predicted = pairs(label["pairs"], len(label["seq"])), pairs(raw_pairs, len(label["seq"]))
        tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        if not truth and not predicted:
            precision = recall = f1 = 1.0
        else:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        values.append((precision, recall, f1))
        f1_by_name[name] = f1
        expected = {"dataset": dataset, "panel_id": panel_id, "prediction_id": prediction_id, "name": name,
                    "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
                    "truth_empty": not truth, "prediction_empty": not predicted, "empty_empty": not truth and not predicted}
        compare(actual_records[(dataset, panel_id, prediction_id, name)], expected, f"record {dataset}/{prediction_id}/{name}")
    n = len(values)
    tp, fp, fn = totals
    subset = [actual_records[(dataset, panel_id, prediction_id, row["name"])] for row in panel if row["name"] in prediction_rows]
    metrics = {"n_scored": n, "tp": tp, "fp": fp, "fn": fn,
               "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
               "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
               "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
               "macro_precision": math.fsum(value[0] for value in values) / n if n else None,
               "macro_recall": math.fsum(value[1] for value in values) / n if n else None,
               "macro_f1": math.fsum(value[2] for value in values) / n if n else None,
               "truth_empty_records": sum(row["truth_empty"] for row in subset),
               "empty_empty_records": sum(row["empty_empty"] for row in subset)}
    return metrics, f1_by_name, missing


def interval(deltas):
    rng = np.random.default_rng(20261127)
    samples = []
    for start in range(0, 10000, 200):
        size = min(200, 10000 - start)
        index = rng.integers(0, len(deltas), size=(size, len(deltas)))
        samples.append(np.asarray(deltas)[index].mean(axis=1))
    return [float(value) for value in np.percentile(np.concatenate(samples), [2.5, 97.5])]


def verify_dataset(dataset, full, projected, actual_records, baselines, prior):
    (project_n, official_n, project_sha, ref_path, ref_prefix, ref_sha, mirror_path, mirror_prefix,
     mirror_sha, full_id, project_id, full_file, project_file, available_n) = SPECS[dataset]
    document = json.loads((OUT / f"{dataset}_system_metrics.json").read_text())
    rows = document["rows"]
    actual_by_id = {row["prediction_id"]: row for row in rows}
    require(len(rows) == len(actual_by_id) == available_n + 14, f"system identities {dataset}")
    project_f1 = {}
    for entry in inventory(dataset):
        actual = actual_by_id[entry["prediction_id"]]
        for key, value in entry.items():
            compare(actual[key], value, f"inventory {entry['prediction_id']}.{key}")
        compare(actual["training_exposure"], exposure(entry, dataset, baselines, prior), f"exposure {entry['prediction_id']}")
        if not entry["available"]:
            status = "BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED" if entry["registry_status"] == "SUMMARY_ONLY" else "BLOCKED_REGISTERED_PREDICTION_MISSING"
            for panel_id, panel in ((full_id, full), (project_id, projected)):
                compare(actual["panels"][panel_id], {"status": status, "n_panel_sequences": len(panel), "n_with_prediction": 0,
                        "coverage": 0.0, "metrics": None, "delta_vs_xens2": None, "paired_macro_f1_delta_ci95_vs_xens2": None}, f"blocked {entry['prediction_id']}")
            continue
        compare(sha(entry["path"]), entry["sha256"], f"prediction SHA {entry['prediction_id']}")
        prediction_rows = predictions(entry, dataset)
        compare(actual["n_prediction_records"], len(prediction_rows), f"prediction n {entry['prediction_id']}")
        for panel_id, panel in ((full_id, full), (project_id, projected)):
            metrics, f1_by_name, missing = score(panel, prediction_rows, dataset, panel_id, entry["prediction_id"], actual_records)
            extra = sorted(set(prediction_rows) - {row["name"] for row in panel})
            expected = {"status": "PASS_FULL_COVERAGE" if not missing else "PARTIAL_COVERAGE_NOT_RANKED",
                        "n_panel_sequences": len(panel), "n_with_prediction": metrics["n_scored"],
                        "coverage": metrics["n_scored"] / len(panel), "missing_count": len(missing), "missing_names": missing,
                        "extra_count": len(extra), "extra_names": extra, "metrics": metrics,
                        "delta_vs_xens2": None if panel_id == full_id else actual["panels"][panel_id]["delta_vs_xens2"],
                        "paired_macro_f1_delta_ci95_vs_xens2": None if panel_id == full_id else actual["panels"][panel_id]["paired_macro_f1_delta_ci95_vs_xens2"],
                        "missing_predictions_imputed_as_zero": False}
            compare(actual["panels"][panel_id], expected, f"panel {entry['prediction_id']}/{panel_id}")
            if panel_id == project_id and not missing:
                project_f1[entry["prediction_id"]] = f1_by_name
    require(len(project_f1) == available_n, f"project coverage {dataset}")
    names = [row["name"] for row in projected]
    xens2_id = f"xens2/{dataset}"
    xens2 = np.asarray([project_f1[xens2_id][name] for name in names])
    reference = actual_by_id[xens2_id]["panels"][project_id]["metrics"]
    for prediction_id, f1_by_name in project_f1.items():
        actual = actual_by_id[prediction_id]["panels"][project_id]
        metrics = actual["metrics"]
        expected_delta = {key: metrics[key] - reference[key] for key in ("micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1")}
        compare(actual["delta_vs_xens2"], expected_delta, f"delta {prediction_id}")
        deltas = np.asarray([f1_by_name[name] for name in names]) - xens2
        expected_ci = {"observed_delta": float(deltas.mean()), "ci95": interval(deltas),
                       "method": "paired sequence-level percentile bootstrap", "n_bootstrap": 10000,
                       "seed": 20261127, "unit": "sequence", "applicability": f"full common {dataset} projected subset coverage only"}
        compare(actual["paired_macro_f1_delta_ci95_vs_xens2"], expected_ci, f"CI {prediction_id}")
    eligible = [actual_by_id[prediction_id] for prediction_id in project_f1]
    for key, subset in (("strongest_overall", eligible), ("strongest_ensemble", [row for row in eligible if "ensemble" in row["model_form"]]),
                        ("strongest_single_predictor", [row for row in eligible if "ensemble" not in row["model_form"]])):
        strongest = max(subset, key=lambda row: row["panels"][project_id]["metrics"]["micro_f1"])
        expected = {"prediction_id": strongest["prediction_id"], "model_form": strongest["model_form"],
                    "metrics": strongest["panels"][project_id]["metrics"], "delta_vs_xens2": strongest["panels"][project_id]["delta_vs_xens2"],
                    "training_exposure": strongest["training_exposure"],
                    "ranking_scope": "DESCRIPTIVE_FROZEN_PROJECTED_SUBSET_AUDIT_ONLY_NOT_MODEL_SELECTION"}
        compare(document[key], expected, f"{dataset}.{key}")
    compare(document["training_exposure_summary"], prior["training_overlap"][dataset], f"training summary {dataset}")
    full_eligible = [row for row in rows if row["panels"][full_id]["status"] == "PASS_FULL_COVERAGE"]
    if full_eligible:
        for key, subset in (("strongest_overall", full_eligible), ("strongest_ensemble", [row for row in full_eligible if "ensemble" in row["model_form"]]),
                            ("strongest_single_predictor", [row for row in full_eligible if "ensemble" not in row["model_form"]])):
            strongest = max(subset, key=lambda row: row["panels"][full_id]["metrics"]["micro_f1"])
            expected = {"prediction_id": strongest["prediction_id"], "model_form": strongest["model_form"],
                        "metrics": strongest["panels"][full_id]["metrics"], "delta_vs_xens2": None,
                        "training_exposure": strongest["training_exposure"],
                        "ranking_scope": "DESCRIPTIVE_FROZEN_OFFICIAL_FULL_AUDIT_ONLY_NOT_MODEL_SELECTION"}
            compare(document["official_full_summary"][key], expected, f"{dataset}.official_full.{key}")
    else:
        compare(document["official_full_summary"], {key: "UNCHECKED_NO_FULL_COVERAGE_PREDICTION" for key in
                ("strongest_overall", "strongest_ensemble", "strongest_single_predictor")}, f"{dataset}.official_full.empty")
    return document


def verify_coverage_and_delivery(documents, panels, actual_record_count):
    gaps = json.loads((OUT / "official_coverage_gaps.json").read_text())["datasets"]
    for dataset, document in documents.items():
        full, projected = panels[dataset]
        spec = SPECS[dataset]
        full_id = spec[9]
        project_names = {row["name"] for row in projected}
        missing = [row for row in full if row["name"] not in project_names]
        expected = {"dataset": dataset, "official_n": len(full), "project_n": len(projected),
                    "official_missing_from_project_count": len(missing),
                    "official_missing_from_project": [{"name": row["name"], "source_id": row["source_id"],
                        "reason": "PROJECT_REJECTED_BAD_ALPHABET", "non_acgu_characters": sorted(set(row["seq"]) - set("ACGU")),
                        "sequence_sha256": row["sequence_sha256"], "pair_set_sha256": row["pair_set_sha256"],
                        "reference_member": row["reference_member"], "reference_member_sha256": row["reference_member_sha256"]} for row in missing],
                    "per_prediction_official_coverage": [{"prediction_id": row["prediction_id"],
                        "status": row["panels"][full_id]["status"], "n_with_prediction": row["panels"][full_id]["n_with_prediction"],
                        "coverage": row["panels"][full_id]["coverage"], "missing_count": row["panels"][full_id].get("missing_count"),
                        "missing_names": row["panels"][full_id].get("missing_names")} for row in document["rows"]]}
        compare(gaps[dataset], expected, f"coverage gaps {dataset}")
    compare(actual_record_count, 84243, "record rows")
    recommendation = json.loads((OUT / "registry_recommendation.json").read_text())
    require(recommendation["recommendation_only"] and not recommendation["shared_registry_modified"], "recommendation boundary")
    for dataset, document in documents.items():
        compare(recommendation["datasets"][dataset]["suggested_prediction_overlay"]["rows"], document["rows"], f"recommendation rows {dataset}")
        compare(recommendation["datasets"][dataset]["suggested_prediction_overlay"]["sha256"], sha(OUT / f"{dataset}_system_metrics.json"), f"recommendation SHA {dataset}")
    manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    for relative, digest in manifest["shared_registry_sha256_before_after"].items():
        compare(sha(ROOT / relative), digest, f"registry {relative}")
    for dataset, item in manifest["datasets"].items():
        compare(sha(item["project"]["path"]), item["project"]["sha256"], f"manifest project {dataset}")
        compare(sha(item["reference"]["path"]), item["reference"]["sha256"], f"manifest reference {dataset}")
        if item["mirror"]["path"]:
            compare(sha(item["mirror"]["path"]), item["mirror"]["sha256"], f"manifest mirror {dataset}")
        for prediction in item["predictions"]:
            path = Path(prediction["path"]) if prediction["path"] else None
            if path and path.is_file():
                compare(sha(path), prediction["sha256"], f"manifest prediction {prediction['prediction_id']}")
            else:
                require(not prediction["available"] and prediction["sha256"] is None, "missing available prediction")
    receipt = json.loads((OUT / "build_receipt.json").read_text())
    compare(receipt["per_record_metric_rows"], 84243, "receipt rows")
    compare(receipt["prediction_identities"], 74, "receipt identities")
    for relative, item in receipt["outputs"].items():
        path = ROOT / relative
        compare(path.stat().st_size, item["bytes"], f"receipt bytes {relative}")
        compare(sha(path), item["sha256"], f"receipt SHA {relative}")
    text = REPORT.read_text()
    require("EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT" in text and "UNCHECKED_NO_FULL_COVERAGE_PREDICTION" in text, "report exposure/full coverage")


def main():
    actual_records_list = gzip_rows(OUT / "per_record_metrics.jsonl.gz")
    actual_records = {(row["dataset"], row["panel_id"], row["prediction_id"], row["name"]): row for row in actual_records_list}
    require(len(actual_records) == len(actual_records_list), "duplicate metric records")
    baselines = {row["baseline_id"]: row for row in json.loads((ROOT / "manifests/foldnar_v1/baseline_registry.json").read_text())["rows"]}
    prior = json.loads(PRIOR.read_text())
    documents, panels = {}, {}
    for dataset in SPECS:
        full, projected = verify_panels(dataset)
        panels[dataset] = (full, projected)
        print(f"VERIFY_PANELS_PASS {dataset} {len(full)}/{len(projected)}", flush=True)
        documents[dataset] = verify_dataset(dataset, full, projected, actual_records, baselines, prior)
        print(f"VERIFY_DATASET_PASS {dataset}", flush=True)
    verify_coverage_and_delivery(documents, panels, len(actual_records_list))
    print("VERIFY_COVERAGE_AND_DELIVERY_PASS", flush=True)
    files = [ROOT / "tools/build_ts0_testsetb_official_reference_panels_20261127.py",
             ROOT / "tools/verify_ts0_testsetb_official_reference_panels_20261127.py",
             ROOT / "tests/test_ts0_testsetb_official_reference_panels_20261127.py", REPORT,
             SOURCE / "source_manifest.json", OUT / "per_record_metrics.jsonl.gz",
             OUT / "official_coverage_gaps.json", OUT / "registry_recommendation.json", OUT / "build_receipt.json"]
    files += [SOURCE / SPECS[dataset][11] for dataset in SPECS] + [SOURCE / SPECS[dataset][12] for dataset in SPECS]
    files += [OUT / f"{dataset}_system_metrics.json" for dataset in SPECS]
    verification = {
        "schema_version": "ts0_testsetb.official_reference.independent_verification.round2.v1",
        "status": "PASS_INDEPENDENT_SECOND_IMPLEMENTATION", "generator_imported": False,
        "panel_records_compared_field_by_field": 1305 + 1288 + 430 + 428,
        "per_record_metrics_compared_field_by_field": 84243, "prediction_identities_verified": 74,
        "projected_full_coverage": {"bprna_ts0": 26, "testsetb": 20},
        "official_full_coverage": {"bprna_ts0": 7, "testsetb": 0},
        "checks": ["official reference packages and four panel identities", "all prediction schemas and strict pair metrics",
                   "official missing ID lists and no zero imputation", "xens2 deltas and paired macro CI",
                   "per-model training exposure columns and provenance status", "registry recommendation and source hashes"],
        "new_training_or_inference": False, "gpu_used": False, "external_test_model_selection": False,
        "deliverables": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in files},
    }
    (OUT / "verification_round2.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": verification["status"], "records": 84243, "identities": 74}, sort_keys=True))


if __name__ == "__main__":
    main()
