#!/usr/bin/env python3
"""Independent verification of ArchiveII official-reference panels and metrics."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/archiveii_official_reference_20261127"
SOURCE = ROOT / "source_data/archiveii_official_reference_20261127"
REPORT = ROOT / "docs/plans/2026-11-27-archiveii-official-reference-panel.md"
PROJECT = Path("/root/rna_jepa_data/ss_data/jsonl/archiveii_embok_clean.jsonl")
BD = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz")
BR = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2/official_assets/bpfold/BPfold_test_results_v0.2.tar.gz")
MX = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/mxfold2/archiveII.tar.gz")
FULL = "archiveii_official_reference_3966"
CLEAN = "archiveii_official_reference_clean_2544"
XENS2 = "xens2/archiveii_embok_clean"
RELEASE = "bpfold_official_release/archiveii"
S1 = "rinalmo_2d_s1/archiveii_embok_clean"
S2 = "rinalmo_2d_s2/archiveii_embok_clean"
TRUNC = "rinalmo_2d_s0/archiveii_embok_clean_trunc598"
DETERMINISTIC = {"contrafold/archiveii_embok_clean", "eternafold/archiveii_embok_clean",
                 "mxfold2/archiveii_embok_clean", "rnastructure_fold/archiveii_embok_clean",
                 "rnastructure_maxexpect/archiveii_embok_clean"}


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
        require(0 <= i < j < length, f"pair bounds {(i, j)}/{length}")
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
    result = {}
    with tarfile.open(path, "r:gz") as package:
        for member in package:
            if not member.isfile() or not member.name.startswith(prefix) or not member.name.endswith(".bpseq"):
                continue
            payload = package.extractfile(member).read()
            sequence, pair_set = bpseq(payload)
            name = member.name.rsplit("/", 1)[-1]
            result[name] = (sequence, pair_set, member.name, hashlib.sha256(payload).hexdigest())
    return result


def verify_panels(document):
    compare(sha(BD), "819acdea53c6b093df14cab35bf079c4678325d2d5ea4296b904068d35959403", "BPfold_data SHA")
    compare(sha(MX), "633bc50f8e9381f737b680fa6b4c79061a0a781773a5c38b342856eb1a6ac8a4", "MXfold2 SHA")
    compare(sha(PROJECT), "6c1fbaa4684cbe4644acdac2f31ae95e5f9b0eb36871f23d8148006f3d571399", "project SHA")
    official = tar_load(BD, "BPfold_data/test_data/archiveII/")
    mirror = tar_load(MX, "data/archiveII/")
    require(len(official) == len(mirror) == 3966 and official.keys() == mirror.keys(), "reference population")
    expected_full = []
    for name in sorted(official):
        left, right = official[name], mirror[name]
        compare((left[0], left[1]), (right[0], right[1]), f"reference packages {name}")
        core = {"name": name, "seq": left[0], "pairs": [list(pair) for pair in sorted(left[1])],
                "sequence_sha256": hashlib.sha256(left[0].encode()).hexdigest(),
                "pair_set_sha256": value_sha(sorted(left[1]))}
        expected_full.append({**core, "record_sha256": value_sha(core),
                              "bpfold_member": left[2], "bpfold_member_sha256": left[3],
                              "mxfold2_member": right[2], "mxfold2_member_sha256": right[3]})
    actual_full = gzip_rows(SOURCE / "official_reference_3966.jsonl.gz")
    compare(actual_full, expected_full, "full official panel")
    project = [json.loads(line) for line in PROJECT.read_text().splitlines() if line.strip()]
    by_name = {row["name"]: row for row in expected_full}
    expected_clean = [by_name[row["name"]] for row in project]
    compare(gzip_rows(SOURCE / "official_reference_clean_2544.jsonl.gz"), expected_clean, "clean official panel")
    require(sum(not row["pairs"] for row in expected_full) == 5, "full empty count")
    require(sum(not row["pairs"] for row in expected_clean) == 4, "clean empty count")
    for panel_id, panel, filename in ((FULL, expected_full, "official_reference_3966.jsonl.gz"),
                                      (CLEAN, expected_clean, "official_reference_clean_2544.jsonl.gz")):
        path = SOURCE / filename
        expected_identity = {
            "dataset_id": panel_id, "path": str(path.relative_to(ROOT)), "sha256": sha(path),
            "bytes": path.stat().st_size, "n_sequences": len(panel),
            "id_order_sha256": value_sha([row["name"] for row in panel]),
            "empty_reference_records": sum(not row["pairs"] for row in panel),
            "source_archives": {"BPfold_data": {"path": str(BD), "sha256": sha(BD)},
                                "MXfold2_archiveII": {"path": str(MX), "sha256": sha(MX)}},
            "pair_protocol": "unfiltered official BPSEQ; no min-loop/crossing/canonical projection",
        }
        compare(document["label_panels"][panel_id], expected_identity, f"panel identity {panel_id}")
    return expected_full, expected_clean


def inventory():
    registry = json.loads((ROOT / "manifests/foldnar_v1/prediction_registry.json").read_text())
    result = []
    for row in sorted((item for item in registry["rows"] if item.get("dataset_id") == "archiveii_embok_clean"), key=lambda item: item["prediction_id"]):
        available = bool(row.get("exists") and row.get("status") == "FROZEN" and row.get("prediction_schema") in {"per_sequence", "predictions"})
        form = "three_checkpoint_ensemble" if row["prediction_id"] == XENS2 else "six_checkpoint_ensemble" if row["prediction_id"] == "bpfold/archiveii_embok_clean" else "deterministic_single_predictor" if row["prediction_id"] in DETERMINISTIC else "single_model"
        result.append({"prediction_id": row["prediction_id"], "source_group": "REGISTERED",
                       "registry_status": row.get("status"), "path": row.get("path"), "sha256": row.get("sha256"),
                       "data_path": row.get("data_path"), "available": available, "model_form": form,
                       "prediction_origin": "project_bpfold_replay" if row["prediction_id"] == "bpfold/archiveii_embok_clean" else "registered_frozen_prediction"})
    additions = [
        (RELEASE, str(BR), "e8a3420e3458f7aad4c86b500f5d56d0e935b7da82b936675fefda64c5e03073", "six_checkpoint_ensemble", "bpfold_author_release_v0.2"),
        (S1, "/root/rna_jepa_data/eval_decision/ow_rinalmo_r2dtr1c_b4_s1_step20000/archiveii_embok_clean/result.json", "a8a0875fb63cdeb4782d6b6cfa28f0d87766d2e5989085d8f0f743b6827d8151", "single_model", "additional_frozen_r2d_seed1"),
        (S2, "/root/rna_jepa_data/eval_decision/ow_rinalmo_r2dtr1c_b4_s2_step20000/archiveii_embok_clean/result.json", "376278c46e29733d611a9343381bdd6f2b5ae4cbb4d593cf32bf0cfee5459caa", "single_model", "additional_frozen_r2d_seed2"),
        (TRUNC, "/root/rna_jepa_data/eval_decision/nb_jev_r2dtr1c_s0_step20000/archiveii_embok_clean_trunc598/result.json", "89990cecfaf8c5c98cdf6608044d089a5b9236285067605ccb78d53ae76f19d8", "single_model", "first598_protocol_not_full_length"),
    ]
    for prediction_id, path, digest, form, origin in additions:
        result.append({"prediction_id": prediction_id, "source_group": "INDEPENDENT_AUDIT_ASSET",
                       "registry_status": "UNREGISTERED_INDEPENDENT_AUDIT_ASSET", "path": path, "sha256": digest,
                       "data_path": str(PROJECT) if prediction_id != TRUNC else str(PROJECT.with_name("archiveii_embok_clean_trunc598.jsonl")),
                       "available": prediction_id != TRUNC, "model_form": form, "prediction_origin": origin})
    require(len(result) == 36 and sum(row["available"] for row in result) == 20, "inventory")
    return result


def predictions(entry):
    if entry["prediction_id"] == RELEASE:
        rows = tar_load(BR, "BPfold_test_results/archiveII/")
        return {name: {"seq": value[0], "pairs": value[1]} for name, value in rows.items()}, {"sequence_source": "BPSEQ sequence in author release"}
    document = json.loads(Path(entry["path"]).read_text())
    if document.get("data"):
        compare(document["data"], entry["data_path"], f"data path {entry['prediction_id']}")
    if document.get("data_sha256"):
        compare(document["data_sha256"], sha(PROJECT), f"data SHA {entry['prediction_id']}")
    if entry["prediction_id"].startswith("rnastructure_fold/"):
        source = document["results"]["fold"]["per_sequence"]
    elif entry["prediction_id"].startswith("rnastructure_maxexpect/"):
        source = document["results"]["maxexpect"]["per_sequence"]
    else:
        source = document.get("per_sequence", document.get("predictions"))
    require(source is not None, f"no predictions {entry['prediction_id']}")
    result = {}
    for row in source:
        pair_values = row.get("pred_pairs", row.get("pairs"))
        require(pair_values is not None and row["name"] not in result, "prediction rows")
        result[row["name"]] = {"seq": row.get("seq"), "pairs": pair_values}
    return result, {"sequence_source": "frozen source data path/SHA plus per-ID reference join; direct seq checked when present"}


def score(panel, prediction_rows, panel_id, prediction_id, actual_records):
    totals = [0, 0, 0]
    values, f1_by_name, missing = [], {}, []
    for label in panel:
        name = label["name"]
        if name not in prediction_rows:
            missing.append(name)
            continue
        prediction = prediction_rows[name]
        if prediction["seq"] is not None:
            compare(prediction["seq"].upper().replace("T", "U"), label["seq"], f"prediction sequence {name}")
        truth = pairs(label["pairs"], len(label["seq"]))
        predicted = pairs(prediction["pairs"], len(label["seq"]))
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
        expected_record = {"panel_id": panel_id, "prediction_id": prediction_id, "name": name,
                           "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
                           "truth_empty": not truth, "prediction_empty": not predicted,
                           "empty_empty": not truth and not predicted}
        compare(actual_records[(panel_id, prediction_id, name)], expected_record, f"record metric {panel_id}/{prediction_id}/{name}")
    n = len(values)
    tp, fp, fn = totals
    metrics = {"n_scored": n, "tp": tp, "fp": fp, "fn": fn,
               "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
               "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
               "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
               "macro_precision": math.fsum(value[0] for value in values) / n if n else None,
               "macro_recall": math.fsum(value[1] for value in values) / n if n else None,
               "macro_f1": math.fsum(value[2] for value in values) / n if n else None}
    # These counts are independently recomputed from the compared record rows.
    subset = [actual_records[(panel_id, prediction_id, row["name"])] for row in panel if row["name"] in prediction_rows]
    metrics["empty_empty_records"] = sum(row["empty_empty"] for row in subset)
    metrics["truth_empty_records"] = sum(row["truth_empty"] for row in subset)
    return metrics, f1_by_name, missing


def interval(deltas):
    rng = np.random.default_rng(20261127)
    samples = []
    for start in range(0, 10000, 200):
        size = min(200, 10000 - start)
        index = rng.integers(0, len(deltas), size=(size, len(deltas)))
        samples.append(np.asarray(deltas)[index].mean(axis=1))
    return [float(value) for value in np.percentile(np.concatenate(samples), [2.5, 97.5])]


def verify_metrics(document, full, clean):
    rows = document["rows"]
    actual_by_id = {row["prediction_id"]: row for row in rows}
    require(len(actual_by_id) == len(rows) == 36, "system rows")
    record_rows = gzip_rows(OUT / "per_record_metrics.jsonl.gz")
    actual_records = {(row["panel_id"], row["prediction_id"], row["name"]): row for row in record_rows}
    require(len(actual_records) == len(record_rows) == 103182, "per-record accounting")
    clean_f1 = {}
    for entry in inventory():
        actual = actual_by_id[entry["prediction_id"]]
        for key, value in entry.items():
            compare(actual[key], value, f"inventory {entry['prediction_id']}.{key}")
        if not entry["available"]:
            status = "BLOCKED_PROTOCOL_MISMATCH_FIRST598_NOT_FULL_LENGTH" if entry["prediction_id"] == TRUNC else "BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED" if entry["registry_status"] == "SUMMARY_ONLY" else "BLOCKED_REGISTERED_PREDICTION_MISSING"
            for panel_id, panel in ((FULL, full), (CLEAN, clean)):
                compare(actual["panels"][panel_id], {"status": status, "n_panel_sequences": len(panel),
                        "n_with_prediction": 0, "coverage": 0.0, "metrics": None,
                        "delta_vs_xens2": None, "paired_macro_f1_delta_ci95_vs_xens2": None}, f"blocked {entry['prediction_id']}/{panel_id}")
            continue
        compare(sha(entry["path"]), entry["sha256"], f"prediction SHA {entry['prediction_id']}")
        prediction_rows, join = predictions(entry)
        compare(actual["sequence_join"], join, f"sequence join {entry['prediction_id']}")
        compare(actual["n_prediction_records"], len(prediction_rows), f"prediction n {entry['prediction_id']}")
        for panel_id, panel in ((FULL, full), (CLEAN, clean)):
            metrics, f1_by_name, missing = score(panel, prediction_rows, panel_id, entry["prediction_id"], actual_records)
            extra = sorted(set(prediction_rows) - {row["name"] for row in panel})
            expected = {"status": "PASS_FULL_COVERAGE" if not missing else "PARTIAL_COVERAGE_NOT_RANKED",
                        "n_panel_sequences": len(panel), "n_with_prediction": metrics["n_scored"],
                        "coverage": metrics["n_scored"] / len(panel), "missing_count": len(missing),
                        "missing_names": missing, "extra_count": len(extra), "extra_names": extra,
                        "metrics": metrics, "delta_vs_xens2": None if panel_id == FULL else actual["panels"][panel_id]["delta_vs_xens2"],
                        "paired_macro_f1_delta_ci95_vs_xens2": None if panel_id == FULL else actual["panels"][panel_id]["paired_macro_f1_delta_ci95_vs_xens2"],
                        "missing_predictions_imputed_as_zero": False}
            compare(actual["panels"][panel_id], expected, f"panel metrics {entry['prediction_id']}/{panel_id}")
            if panel_id == CLEAN and not missing:
                clean_f1[entry["prediction_id"]] = f1_by_name
    require(len(clean_f1) == 20, "clean full coverage")
    names = [row["name"] for row in clean]
    xens2 = np.asarray([clean_f1[XENS2][name] for name in names])
    reference = actual_by_id[XENS2]["panels"][CLEAN]["metrics"]
    for prediction_id, f1_by_name in clean_f1.items():
        actual = actual_by_id[prediction_id]["panels"][CLEAN]
        metrics = actual["metrics"]
        expected_delta = {key: metrics[key] - reference[key] for key in ("micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1")}
        compare(actual["delta_vs_xens2"], expected_delta, f"delta {prediction_id}")
        values = np.asarray([f1_by_name[name] for name in names])
        deltas = values - xens2
        expected_ci = {"observed_delta": float(deltas.mean()), "ci95": interval(deltas),
                       "method": "paired sequence-level percentile bootstrap", "n_bootstrap": 10000,
                       "seed": 20261127, "unit": "sequence", "applicability": "full common clean 2544 coverage only"}
        compare(actual["paired_macro_f1_delta_ci95_vs_xens2"], expected_ci, f"CI {prediction_id}")
    eligible = [actual_by_id[prediction_id] for prediction_id in clean_f1]
    for key, subset in (("strongest_overall", eligible), ("strongest_ensemble", [row for row in eligible if "ensemble" in row["model_form"]]),
                        ("strongest_single_predictor", [row for row in eligible if "ensemble" not in row["model_form"]])):
        strongest = max(subset, key=lambda row: row["panels"][CLEAN]["metrics"]["micro_f1"])
        expected = {"prediction_id": strongest["prediction_id"], "model_form": strongest["model_form"],
                    "metrics": strongest["panels"][CLEAN]["metrics"], "delta_vs_xens2": strongest["panels"][CLEAN]["delta_vs_xens2"],
                    "ranking_scope": "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION"}
        compare(document[key], expected, key)
    full_eligible = [row for row in rows if row["panels"][FULL]["status"] == "PASS_FULL_COVERAGE"]
    require(len(full_eligible) == 1 and full_eligible[0]["prediction_id"] == RELEASE, "full coverage identity")


def verify_delivery(document):
    recommendation = json.loads((OUT / "registry_recommendation.json").read_text())
    require(recommendation["recommendation_only"] and not recommendation["shared_registry_modified"], "registry recommendation boundary")
    compare(recommendation["legacy_project_labels"]["disposition"], "circular-audit-only", "legacy disposition")
    compare(recommendation["suggested_prediction_overlay"]["sha256"], sha(OUT / "system_metrics.json"), "recommendation metrics SHA")
    compare(recommendation["suggested_prediction_overlay"]["rows"], document["rows"], "recommendation rows")
    manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    for item in manifest["official_archives"]:
        compare(sha(item["path"]), item["sha256"], f"source archive {item['path']}")
    for item in manifest["prediction_inputs"]:
        path = Path(item["path"]) if item["path"] else None
        if path and path.is_file():
            compare(sha(path), item["sha256"], f"source prediction {item['prediction_id']}")
        else:
            require(not item["available"] and item["sha256"] is None, f"unexpected missing available prediction {item['prediction_id']}")
    for relative, digest in manifest["shared_registry_sha256_before_after"].items():
        compare(sha(ROOT / relative), digest, f"shared registry {relative}")
    compare(document["shared_registry_sha256_before"], manifest["shared_registry_sha256_before_after"], "registry before/after")
    require(not document["external_test_model_selection"] and not document["new_training_or_inference"] and not document["gpu_used"], "compute boundary")
    receipt = json.loads((OUT / "build_receipt.json").read_text())
    compare(receipt["per_record_metric_rows"], 103182, "receipt record rows")
    for relative, item in receipt["outputs"].items():
        path = ROOT / relative
        compare(path.stat().st_size, item["bytes"], f"receipt bytes {relative}")
        compare(sha(path), item["sha256"], f"receipt SHA {relative}")
    text = REPORT.read_text()
    require("empty/empty=0" in text and "UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR" in text, "report empty/full-coverage boundary")


def main():
    document = json.loads((OUT / "system_metrics.json").read_text())
    full, clean = verify_panels(document)
    print("VERIFY_PANELS_PASS 3966/2544", flush=True)
    verify_metrics(document, full, clean)
    print("VERIFY_METRICS_PASS 103182 records / 36 identities", flush=True)
    verify_delivery(document)
    print("VERIFY_DELIVERY_PASS", flush=True)
    files = [ROOT / "tools/build_archiveii_official_reference_panel_20261127.py",
             ROOT / "tools/verify_archiveii_official_reference_panel_20261127.py",
             ROOT / "tests/test_archiveii_official_reference_panel_20261127.py", REPORT,
             SOURCE / "official_reference_3966.jsonl.gz", SOURCE / "official_reference_clean_2544.jsonl.gz",
             SOURCE / "source_manifest.json", OUT / "per_record_metrics.jsonl.gz",
             OUT / "system_metrics.json", OUT / "registry_recommendation.json", OUT / "build_receipt.json"]
    verification = {
        "schema_version": "archiveii.official_reference.independent_verification.round2.v1",
        "status": "PASS_INDEPENDENT_SECOND_IMPLEMENTATION", "generator_imported": False,
        "panel_records_compared_field_by_field": 6510,
        "per_record_metrics_compared_field_by_field": 103182,
        "prediction_identities_verified": 36,
        "clean_full_coverage_identities": 20, "official_full_coverage_identities": 1,
        "paired_bootstrap_replicates_per_identity": 10000,
        "checks": ["two official archives and panel SHA", "all prediction schemas and sequence joins",
                   "strict pair metrics and empty/empty convention", "coverage/missing/extra IDs and no zero imputation",
                   "xens2 deltas and paired macro CI", "release/replay/R2D identity boundaries",
                   "registry recommendation and read-only shared registries"],
        "new_training_or_inference": False, "gpu_used": False, "external_test_model_selection": False,
        "deliverables": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in files},
    }
    (OUT / "verification_round2.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": verification["status"], "records": 103182, "identities": 36}, sort_keys=True))


if __name__ == "__main__":
    main()
