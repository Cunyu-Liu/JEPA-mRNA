#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v3_efold_family")
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
DATA_SHA256 = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
EFOLD_COMMIT = "4e146fe92c574104b2518d7350e0fa4a39f32030"
E2EFOLD_COMMIT = "2aab9e78326ddaa27d14c49eb34001c0003a1960"
DEBFOLD_COMMIT = "ff693cdf4cb177e50d8a6a15c7a6292af802b75e"


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


def strict_metrics(rows, prediction_by_name):
    names = [row["name"] for row in rows]
    assert len(names) == len(set(names))
    assert set(names) == set(prediction_by_name), "Prediction and label name scopes differ"
    tp = fp = fn = 0
    per_sequence = []
    for row in rows:
        name = row["name"]
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


def legacy_efold_prediction_sha256(rows, prediction_by_name):
    predictions = [
        {"name": row["name"], "pairs": prediction_by_name[row["name"]]["pairs"]}
        for row in sorted(rows, key=lambda row: len(row["seq"]))
    ]
    return hashlib.sha256(canonical(predictions).encode()).hexdigest()


def git_commit(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def load_frozen_rows():
    assert sha256(DATA) == DATA_SHA256
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    assert len(rows) == 5388
    assert all(count == 1 for count in Counter(row["name"] for row in rows).values())
    return rows


def audit_efold(rows):
    path = ROOT / "predictions/efold_project_replay_bprna_new_frozen_v3.json"
    assert json.loads((ROOT / "official_assets/efold/releases_api_v3.json").read_text()) == []
    document = json.loads(path.read_text())
    assert document["status"] == "PASS"
    assert document["system"] == "eFold"
    assert document["data_sha256"] == DATA_SHA256
    assert document["commit"] == EFOLD_COMMIT == git_commit(ROOT / "source/eFold")
    weights = ROOT / "official_assets/efold/efold_weights.pt"
    assert sha256(weights) == document["weights_sha256"]
    source = document["per_sequence"]
    assert document["n_sequences"] == len(source) == len(rows) == 5388
    assert [row["name"] for row in source] == [row["name"] for row in rows]
    predictions = {row["name"]: {"pairs": row["pred_pairs"]} for row in source}
    metrics, per_sequence = strict_metrics(rows, predictions)
    for actual, recorded in zip(per_sequence, source):
        assert actual["name"] == recorded["name"]
        assert (actual["tp"], actual["fp"], actual["fn"]) == (recorded["tp"], recorded["fp"], recorded["fn"])
        assert_close(actual["f1"], recorded["f1"])
    identity = [[row["name"], predictions[row["name"]]["pairs"]] for row in rows]
    prediction_identity_sha256 = hashlib.sha256(canonical(identity).encode()).hexdigest()
    legacy_prediction_sha256 = legacy_efold_prediction_sha256(rows, predictions)
    assert len(document["repeats"]) == 3
    for repeat in document["repeats"]:
        recorded = repeat["metrics"]
        assert repeat["prediction_sha256"] == legacy_prediction_sha256
        assert (recorded["tp"], recorded["fp"], recorded["fn"]) == (metrics["tp"], metrics["fp"], metrics["fn"])
        assert_close(recorded["micro_f1"], metrics["micro_f1"])
        assert_close(recorded["macro_f1"], metrics["macro_f1"])
    write_jsonl(ROOT / "reports/efold_project_replay_bprna_new_per_sequence_v3.jsonl", per_sequence)
    return {
        "status": "PASS_INDEPENDENT_RECOMPUTATION",
        "evidence_class": "PROJECT_REPLAY_WITH_OFFICIAL_SOURCE_AND_WEIGHTS",
        "official_released_bprna_new_predictions": None,
        "prediction_artifact": str(path),
        "prediction_artifact_sha256": sha256(path),
        "prediction_identity_sha256": prediction_identity_sha256,
        "legacy_replay_prediction_sha256": legacy_prediction_sha256,
        "legacy_hash_scope": "Length-group-ordered name/pair objects from the original replay; the dataset-order identity hash is reported separately",
        "label_scope": "Frozen project bpRNA-new JSONL labels; DATABASE scope, not an author-released reference-label scope",
        "sequence_scope": "Exact ordered 5,388-name/sequence project scope",
        "metrics": metrics,
        "recorded_repeats_verified": 3,
        "provenance": {
            "repository": "https://github.com/rouskinlab/eFold",
            "source_commit": document["commit"],
            "version": document["efold_version"],
            "weights": str(weights),
            "weights_sha256": document["weights_sha256"],
            "protocol": document["protocol"],
            "native_protocol_note": document["native_protocol_note"],
            "historical_replay_device": document["device"],
            "historical_replay_started": document["started"],
            "batch_size": document["batch_size"],
            "current_recovery": "CPU parsing and exact set arithmetic only; no model execution",
        },
        "official_release_boundary": "GitHub releases API is empty and no author-released bpRNA-new per-sequence prediction asset was found; do not relabel this project replay as an official release.",
    }


def unavailable_metrics(reason):
    return {
        "status": "BLOCKED_PROVENANCE",
        "reason": reason,
        "label_scope": "Frozen project bpRNA-new labels would be required, but no eligible predictions exist",
        "per_sequence_predictions": None,
        "metrics": None,
        "coverage": None,
    }


def audit_e2efold():
    source = ROOT / "source/e2efold"
    assert json.loads((ROOT / "official_assets/e2efold/releases_api_v3.json").read_text()) == []
    assert git_commit(source) == E2EFOLD_COMMIT
    checkpoint_files = sorted(path.name for path in (source / "models_ckpt").iterdir() if path.name != "README.md")
    assert checkpoint_files == []
    result = unavailable_metrics("No official or project bpRNA-new per-sequence predictions; official pretrained checkpoint and released-result downloads remain unavailable")
    result.update({
        "system": "E2Efold",
        "repository": "https://github.com/ml4bio/e2efold",
        "source_commit": E2EFOLD_COMMIT,
        "official_checkpoint_channel": "https://drive.google.com/drive/folders/1m038Fw0HBGEzsvhS0mRxd0U7cGXqLAVt",
        "official_results_channel": "https://drive.google.com/open?id=1ASAiXmiH-d6zdg1wNB0if-o-d1TU9lpb",
        "current_download_observation": "Both official Google Drive endpoints timed out after 60 seconds on 2026-11-24; no response payload was saved as a model or result",
        "official_protocol_scope": "Source README documents RNAStralign/ArchiveII test protocols, not bpRNA-new; these label scopes must not be merged",
        "missing_provenance": [
            "bpRNA-new per-sequence prediction artifact and hash",
            "eligible official pretrained checkpoint file and hash",
            "bpRNA-new inference/postprocessing record linked to a checkpoint",
        ],
        "substitution_policy": "No third-party checkpoint, retrained model, paper summary, zero score, or another dataset's result is substituted",
    })
    return result


def audit_debfold():
    assert json.loads((ROOT / "official_assets/debfold/releases_api_v3.json").read_text()) == []
    commit_document = json.loads((ROOT / "official_assets/debfold/main_commit_api_v3.json").read_text())
    tree_document = json.loads((ROOT / "official_assets/debfold/git_tree_main_v3.json").read_text())
    assert commit_document["sha"] == DEBFOLD_COMMIT
    paths = sorted(row["path"] for row in tree_document["tree"])
    assert paths == ["README", "TestSet_alpha.zip", "TestSet_beta.zip", "TestSet_gamma.zip", "TrainingValidationSet.zip"]
    training_archive = ROOT / "official_assets/debfold/TrainingValidationSet.zip"
    assert sha256(training_archive) == "20d4404ffa2c1d2b973f230de859bbfadd47cf3de46c3164e7b8c550b891c6a2"
    result = unavailable_metrics("No official or project bpRNA-new per-sequence predictions; official repository provides datasets and a web-service link but no local code or checkpoint")
    result.update({
        "system": "DEBFold",
        "repository": "https://github.com/cobisLab/DEBFold",
        "source_commit": DEBFOLD_COMMIT,
        "official_repository_files": paths,
        "official_web_service": "https://cobis.bme.ncku.edu.tw/DEBFold/",
        "recovered_training_archive": str(training_archive),
        "recovered_training_archive_sha256": sha256(training_archive),
        "training_label_scope": "Official DEBFold TrainingValidationSet reference CT structures; training provenance only, never test labels or predictions",
        "missing_provenance": [
            "bpRNA-new per-sequence prediction artifact and hash",
            "official local inference source and checkpoint with hashes",
            "batch bpRNA-new protocol linked to released predictions or a checkpoint",
        ],
        "substitution_policy": "Reference CT structures and web-service availability are not predictions; no paper summary or zero score is substituted",
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    assert args.root == ROOT, "Use the isolated v3 artifact root"
    rows = load_frozen_rows()
    created = datetime.now(timezone.utc).isoformat()
    dataset = {
        "dataset": "bprna_new",
        "path": str(DATA),
        "sha256": DATA_SHA256,
        "n_sequences": len(rows),
        "unique_names": len({row["name"] for row in rows}),
        "n_truth_pairs": sum(len(pair_set(row["pairs"], len(row["seq"]))) for row in rows),
        "pair_convention": "Zero-based inclusive i<j; exact pair-set equality; no crossing or short-span filtering",
        "label_scope": "Frozen project JSONL DATABASE labels only",
    }
    systems = {
        "efold": audit_efold(rows),
        "e2efold": audit_e2efold(),
        "debfold": audit_debfold(),
    }
    metrics_document = {
        "schema_version": 3,
        "created_utc": created,
        "status": "PASS_WITH_EXPLICIT_PROVENANCE_GAPS",
        "recovery_mode": "CPU parsing/download only; no inference and no GPU calls",
        "dataset": dataset,
        "systems": systems,
    }
    write_json(ROOT / "reports/bprna_new_efold_family_metrics_v3.json", metrics_document)
    provenance = {
        "schema_version": 3,
        "created_utc": created,
        "recovery_mode": "CPU parsing/download only; no inference and no GPU calls",
        "source_commits": {"efold": EFOLD_COMMIT, "e2efold": E2EFOLD_COMMIT, "debfold": DEBFOLD_COMMIT},
        "prediction_classes": {
            "efold": "project replay using official source/weights; not an author-released prediction set",
            "e2efold": None,
            "debfold": None,
        },
        "label_scopes_separate": True,
        "missing_metrics_are_null_not_zero": True,
        "shared_registries_modified": False,
        "running_configs_or_evaluator_modified": False,
        "ppt_modified": False,
        "failed_download_artifacts": [
            "official_assets/e2efold/checkpoint_folder_curl_v3.log",
            "official_assets/e2efold/released_results_curl_v3.log",
            "official_assets/debfold/DEBFold_main_download_attempt_v3.partial (partial codeload response; not a validated source archive)",
        ],
        "systems": systems,
    }
    write_json(ROOT / "reports/official_provenance_and_missing_assets_v3.json", provenance)
    print(json.dumps(metrics_document, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
