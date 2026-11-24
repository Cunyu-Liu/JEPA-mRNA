#!/usr/bin/env python3
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v4_thermodynamic")
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
CLEAN_DATA = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark/clean_splits/tr1c/bprna_new.jsonl")
SYSTEMS = ("vienna_mfe", "vienna_centroid", "vienna_mea", "rnastructure_fold", "rnastructure_maxexpect", "contrafold_mea")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pairs(values, length):
    result = set()
    used = set()
    for i, j in values:
        assert isinstance(i, int) and isinstance(j, int) and 0 <= i < j < length
        assert i not in used and j not in used
        used.update((i, j))
        result.add((i, j))
    return result


def rescore(labels, predictions):
    assert len(labels) == len(predictions)
    by_name = {row["name"]: row for row in predictions}
    assert len(by_name) == len(predictions)
    assert [row["name"] for row in predictions] == [row["name"] for row in labels]
    tp = fp = fn = 0
    macro = Fraction(0, 1)
    counts = []
    for label in labels:
        name = label["name"]
        truth = pairs(label["pairs"], len(label["seq"]))
        predicted = pairs(by_name[name]["pairs"], len(label["seq"]))
        a = len(truth & predicted)
        b = len(predicted - truth)
        c = len(truth - predicted)
        f1 = Fraction(2 * a, 2 * a + b + c) if 2 * a + b + c else Fraction(1, 1)
        tp += a
        fp += b
        fn += c
        macro += f1
        counts.append((name, a, b, c, f1))
    n = len(labels)
    return {
        "n_dataset_sequences": n, "n_with_prediction": len(by_name),
        "coverage": Fraction(len(by_name), n), "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": Fraction(tp, tp + fp) if tp + fp else None,
        "micro_recall": Fraction(tp, tp + fn) if tp + fn else None,
        "micro_f1": Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "macro_f1": macro / n,
    }, counts


def compare_metrics(recorded, exact):
    for key in ("n_dataset_sequences", "n_with_prediction", "tp", "fp", "fn"):
        assert recorded[key] == exact[key], (key, recorded[key], exact[key])
    for key in ("coverage", "micro_precision", "micro_recall", "micro_f1", "macro_f1"):
        assert math.isclose(recorded[key], float(exact[key]), rel_tol=0.0, abs_tol=1e-12), (key, recorded[key], float(exact[key]))


def verify_scope(recorded, label_path, expected_n, expected_data_sha):
    labels = jsonl(label_path)
    assert len(labels) == expected_n
    assert digest(label_path) == expected_data_sha
    prediction_path = Path(recorded["prediction_artifact"])
    metric_path = Path(recorded["per_sequence_metrics_artifact"])
    assert digest(prediction_path) == recorded["prediction_artifact_sha256"]
    assert digest(metric_path) == recorded["per_sequence_metrics_sha256"]
    predictions = jsonl(prediction_path)
    exact, counts = rescore(labels, predictions)
    compare_metrics(recorded["metrics"], exact)
    per_sequence = jsonl(metric_path)
    assert len(per_sequence) == len(counts)
    for saved, (name, a, b, c, f1) in zip(per_sequence, counts):
        assert (saved["name"], saved["tp"], saved["fp"], saved["fn"]) == (name, a, b, c)
        assert math.isclose(saved["f1"], float(f1), rel_tol=0.0, abs_tol=1e-12)
    identity = [[row["name"], row["pairs"]] for row in predictions]
    identity_text = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(identity_text.encode()).hexdigest() == recorded["prediction_identity_sha256"]
    source = Path(recorded["source_artifact"])
    assert digest(source) == recorded["source_artifact_sha256"]
    return {"n": expected_n, "coverage": float(exact["coverage"]), "tp": exact["tp"], "fp": exact["fp"], "fn": exact["fn"], "micro_f1": float(exact["micro_f1"]), "macro_f1": float(exact["macro_f1"]), "prediction_sha256": digest(prediction_path), "per_sequence_sha256": digest(metric_path), "source_sha256": digest(source)}


def main():
    summary_path = ROOT / "reports/thermodynamic_bprna_new_metrics_v4.json"
    document = json.loads(summary_path.read_text())
    assert document["status"] == "PASS_RAW_5388_ALL_6_DECODERS_CLEAN_SCOPES_SEPARATED"
    assert tuple(document["systems"]) == SYSTEMS
    assert document["execution"] == {"device": "CPU_ONLY", "gpu_used": False, "cuda_visible_devices": ""}
    results = {}
    for system in SYSTEMS:
        record = document["systems"][system]
        results[f"{system}/raw"] = verify_scope(record["raw"], DATA, 5388, document["dataset"]["sha256"])
        if system == "contrafold_mea":
            results[f"{system}/clean_tr1c"] = verify_scope(record["clean"], CLEAN_DATA, 5386, document["clean_scope_exclusions"]["label_sha256"])
            assert record["clean"]["coverage_of_raw_5388"] == 5386 / 5388
            assert record["clean"]["scope_interpretation"] == "PROJECT_TR1C_EXCLUSION_NOT_VERIFIED_CONTRAFOLD_OWN_TRAIN_CLEAN"
        else:
            clean = record["clean"]
            assert clean["status"] == "NOT_AVAILABLE_DISTINCT_CLEAN_SCOPE"
            for key in ("scope", "source_artifact", "source_artifact_sha256", "prediction_artifact", "prediction_artifact_sha256", "metrics", "coverage"):
                assert clean[key] is None, (system, key, clean[key])
    provenance = document["systems"]["vienna_mfe"]["provenance"]
    assert provenance["version"] == "2.7.2" and provenance["binary"] is None and provenance["binary_sha256"] is None
    assert digest(provenance["runtime_extension"]) == provenance["runtime_extension_sha256"]
    assert provenance["parameters"]["path"] is None and provenance["parameters"]["sha256"] is None
    provenance = document["systems"]["rnastructure_fold"]["provenance"]
    assert digest(provenance["official_archive"]) == provenance["official_archive_sha256"]
    for binary in provenance["binaries"].values():
        assert digest(binary["path"]) == binary["sha256"]
    assert digest(provenance["parameters"]["manifest"]) == provenance["parameters"]["manifest_sha256"]
    provenance = document["systems"]["contrafold_mea"]["provenance"]
    assert provenance["version"] is None
    assert digest(provenance["binary"]) == provenance["binary_sha256"]
    for path, expected in provenance["parameters"]["files_sha256"].items():
        assert digest(path) == expected
    replay = document["replay"]
    assert replay["status"] == "PASS_CPU_REPLAY_ALL_SELECTED_PAIRS_EXACT" and replay["n_indices"] == 12
    assert replay["gpu_used"] is False
    assert set(replay["systems"]) == set(SYSTEMS)
    assert all(len(rows) == 12 and all(row["pairs_match"] for row in rows) for rows in replay["systems"].values())
    assert len(document["clean_scope_exclusions"]["names"]) == 2
    repo_files = [
        REPO / "tools/recover_thermodynamic_baselines_v4_20261124.py",
        REPO / "tools/verify_thermodynamic_recovery_v4_20261124.py",
        REPO / "tests/test_recover_thermodynamic_baselines_v4_20261124.py",
        REPO / "docs/thermodynamic_baseline_recovery_v4_20261124.md",
    ]
    assert all(path.is_file() for path in repo_files)
    artifact_report = ROOT / "reports/thermodynamic_baseline_recovery_v4_20261124.md"
    assert artifact_report.read_bytes() == repo_files[-1].read_bytes()
    excluded = {ROOT / "artifact_manifest_v4.json", ROOT / "reports/delivery_verification_v4.json"}
    artifact_files = sorted(path for path in ROOT.rglob("*") if path.is_file() and path not in excluded)
    manifest = {
        "schema_version": 4, "root": str(ROOT),
        "artifacts": {str(path): {"bytes": path.stat().st_size, "sha256": digest(path)} for path in artifact_files},
        "repo_files": {str(path): {"bytes": path.stat().st_size, "sha256": digest(path)} for path in repo_files},
    }
    manifest_path = ROOT / "artifact_manifest_v4.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    verification = {
        "schema_version": 4, "status": "PASS_INDEPENDENT_EXACT_RATIONAL_RECOMPUTATION",
        "method": "Standalone exact Fraction scoring; recovery module and shared evaluator not imported",
        "summary": str(summary_path), "summary_sha256": digest(summary_path),
        "artifact_manifest": str(manifest_path), "artifact_manifest_sha256": digest(manifest_path),
        "raw_5388_decoders_verified": len(SYSTEMS), "separate_clean_5386_decoders_verified": 1,
        "cpu_replay_decoders_verified": len(SYSTEMS), "cpu_replay_sequences_per_decoder": 12,
        "unavailable_clean_assets_preserved_as_null": 5,
        "results": results,
    }
    output = ROOT / "reports/delivery_verification_v4.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(verification, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": verification["status"], "raw_decoders": 6, "raw_n": 5388, "clean_decoders": 1, "clean_n": 5386, "out": str(output), "sha256": digest(output)}, indent=2))


if __name__ == "__main__":
    main()
