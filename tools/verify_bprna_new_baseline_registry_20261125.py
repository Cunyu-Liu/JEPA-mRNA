#!/usr/bin/env python3
"""Independently recompute and cross-check the twenty-asset recovery registry."""
import argparse
import hashlib
import json
import math
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json"
RAW_LABELS = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
CLEAN_LABELS = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark/clean_splits/tr1c/bprna_new.jsonl")
OFFICIAL_LABELS = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def pairs(values, length):
    result = set()
    used = set()
    for i, j in values:
        i, j = int(i), int(j)
        assert 0 <= i < j < length
        assert i not in used and j not in used
        used.update((i, j))
        result.add((i, j))
    return result


def predictions(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    else:
        document = json.loads(path.read_text())
        if "per_sequence" in document:
            rows = [{"name": row["name"], "pairs": row["pred_pairs"]} for row in document["per_sequence"]]
        else:
            rows = document["predictions"]
    names = [row["name"] for row in rows]
    assert len(names) == len(set(names))
    return rows


def jsonl(path, expected_sha):
    assert digest(path) == expected_sha
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    assert len(rows) == len({row["name"] for row in rows})
    return rows


def bpseq(payload):
    sequence, links = [], []
    for line in payload.decode().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        assert int(fields[0]) == len(sequence) + 1
        sequence.append(fields[1].upper().replace("T", "U"))
        partner = int(fields[2]) - 1
        if partner >= 0:
            links.append((len(sequence) - 1, partner))
    normalized = {(min(i, j), max(i, j)) for i, j in links}
    return "".join(sequence), sorted(map(list, pairs(normalized, len(sequence))))


def official_labels(expected_sha):
    assert digest(OFFICIAL_LABELS) == expected_sha
    result = []
    with tarfile.open(OFFICIAL_LABELS, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.startswith("BPfold_data/test_data/bpRNAnew/") and m.name.endswith(".bpseq")]
        assert len(members) == 5401
        for member in members:
            sequence, pair_values = bpseq(archive.extractfile(member).read())
            result.append({"name": Path(member.name).name, "seq": sequence, "pairs": pair_values})
    assert len({row["name"] for row in result}) == 5401
    return result


def rescore(labels, predicted_rows):
    truth_by_name = {row["name"]: row for row in labels}
    prediction_by_name = {row["name"]: row for row in predicted_rows}
    assert set(truth_by_name) == set(prediction_by_name)
    tp = fp = fn = 0
    macro = []
    identity = []
    for name, truth_row in truth_by_name.items():
        prediction = prediction_by_name[name]
        if "seq" in prediction:
            assert prediction["seq"].upper().replace("T", "U") == truth_row["seq"].upper().replace("T", "U")
        truth = pairs(truth_row["pairs"], len(truth_row["seq"]))
        predicted = pairs(prediction["pairs"], len(truth_row["seq"]))
        a, b, c = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        tp += a
        fp += b
        fn += c
        macro.append(2 * a / (2 * a + b + c) if 2 * a + b + c else 1.0)
        identity.append([name, sorted(map(list, predicted))])
    return {
        "n_dataset_sequences": len(labels), "n_with_prediction": len(predicted_rows), "coverage": 1.0,
        "missing_names": [], "extra_names": [], "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "macro_f1": math.fsum(macro) / len(macro),
        "zero_pair_f1_convention": "F1=1 when truth and prediction both have zero pairs",
        "order_matches_labels": [row["name"] for row in predicted_rows] == [row["name"] for row in labels],
        "prediction_identity_sha256": hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest(),
    }


def assert_metrics(actual, recorded):
    for key, expected in recorded.items():
        if isinstance(expected, float):
            assert math.isclose(actual[key], expected, rel_tol=0.0, abs_tol=1e-12), (key, actual[key], expected)
        else:
            assert actual[key] == expected, (key, actual[key], expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_registry_update_20261125"))
    args = parser.parse_args()
    document = json.loads(MANIFEST.read_text())
    assert document["status"] == "PASS_20_EXISTING_ASSETS_SHA_ID_METRICS"
    assert document["n_assets"] == len(document["rows"]) == 20
    assert len({row["asset_id"] for row in document["rows"]}) == 20
    assert len({row["prediction_id"] for row in document["rows"]}) == 20
    raw_labels = clean_labels = bp_labels = None
    checks = []
    for row in document["rows"]:
        assert digest(row["prediction_path"]) == row["prediction_sha256"], row["asset_id"]
        assert digest(row["source_path"]) == row["source_sha256"], row["asset_id"]
        if row["label_protocol"] == "PROJECT_DATABASE_RAW_5388":
            if raw_labels is None:
                raw_labels = jsonl(RAW_LABELS, row["label_sha256"])
            labels = raw_labels
        elif row["label_protocol"] == "PROJECT_TR1C_CLEAN_5386":
            if clean_labels is None:
                clean_labels = jsonl(CLEAN_LABELS, row["label_sha256"])
            labels = clean_labels
        else:
            assert row["label_protocol"] == "BPFOLD_OFFICIAL_REFERENCE_5401"
            if bp_labels is None:
                bp_labels = official_labels(row["label_sha256"])
            labels = bp_labels
        actual = rescore(labels, predictions(row["prediction_path"]))
        assert_metrics(actual, row["metrics"])
        assert actual["order_matches_labels"] == row["order_matches_labels"]
        assert actual["prediction_identity_sha256"] == row["prediction_identity_sha256"]
        assert actual["n_dataset_sequences"] == row["n_sequences"]
        checks.append({"asset_id": row["asset_id"], "status": "PASS", "n_sequences": row["n_sequences"],
                       "micro_f1": actual["micro_f1"], "macro_f1": actual["macro_f1"],
                       "prediction_identity_sha256": actual["prediction_identity_sha256"]})
    by_id = {row["asset_id"]: row for row in document["rows"]}
    assert by_id["bpfold_project_six_checkpoint_raw"]["model_form"] == "six_checkpoint_ensemble"
    assert by_id["bpfold_author_release_official_reference"]["model_form"] == "six_checkpoint_ensemble"
    assert by_id["bpfold_project_six_checkpoint_raw"]["prediction_identity_sha256"] != by_id["bpfold_author_release_official_reference"]["prediction_identity_sha256"]
    assert by_id["spot_rna_registry_raw_single"]["metrics"]["tp"] == 0
    assert by_id["spot_rna_postprocessed_single_raw"]["metrics"]["tp"] == 77326
    assert "full fine-tune" in by_id["nucleicbert_author_secstr_full_ft_single_raw"]["training_identity"]
    assert "encoder frozen" in by_id["nucleicbert_project_frozen_encoder_tr1c_single_raw"]["training_identity"]
    assert by_id["contrafold_mea_single_clean_tr1c"]["n_sequences"] == 5386
    assert not any("chanrg" in row["asset_id"].lower() for row in document["rows"])

    prediction_doc = json.loads((REPO / "manifests/foldnar_v1/prediction_registry.json").read_text())
    recovered_links = {}
    for prediction in prediction_doc["rows"]:
        if "recovery_asset_id" in prediction:
            assert prediction["recovery_asset_id"] not in recovered_links
            recovered_links[prediction["recovery_asset_id"]] = prediction
        if "recovered_bprna_new" in prediction:
            link = prediction["recovered_bprna_new"]
            assert link["recovery_asset_id"] not in recovered_links
            recovered_links[link["recovery_asset_id"]] = link
    assert set(recovered_links) == set(by_id)
    for asset_id, link in recovered_links.items():
        assert link["sha256"] == by_id[asset_id]["prediction_sha256"]
        assert link["label_protocol"] == by_id[asset_id]["label_protocol"]
        assert link["model_form"] == by_id[asset_id]["model_form"]

    baseline_doc = json.loads((REPO / "manifests/foldnar_v1/baseline_registry.json").read_text())
    baselines = {row["baseline_id"]: row for row in baseline_doc["rows"]}
    for row in document["rows"]:
        recovery = baselines[row["baseline_id"]]["bprna_new_recovery"]
        assert row["asset_id"] in recovery["asset_ids"]
        assert row["prediction_id"] in baselines[row["baseline_id"]]["prediction_ids"]
    assert "full fine-tune" in baselines["nucleicbert_frozen_official"]["version_protocol"]
    assert "encoder frozen" in baselines["nucleicbert_ft"]["version_protocol"]
    data_doc = json.loads((REPO / "manifests/foldnar_v1/data_registry.json").read_text())
    assert any(row["dataset_id"] == "bpfold_official_bprna_new_reference" and row["n_sequences"] == 5401 for row in data_doc["rows"])
    dataset_ids = {row["dataset_id"] for row in data_doc["rows"]}
    for baseline in baselines.values():
        for dataset_id in baseline.get("training_data", {}).get("project_training_dataset_ids", []):
            assert dataset_id in dataset_ids, (baseline["baseline_id"], dataset_id)

    raw = [row for row in document["rows"] if row["label_protocol"] == "PROJECT_DATABASE_RAW_5388"]
    best = max(raw, key=lambda row: row["metrics"]["micro_f1"])
    single = max((row for row in raw if row["model_form"] != "six_checkpoint_ensemble"), key=lambda row: row["metrics"]["micro_f1"])
    assert best["asset_id"] == document["strongest_project_raw"]["asset_id"] == "bpfold_project_six_checkpoint_raw"
    assert single["asset_id"] == document["strongest_project_raw_single_predictor"]["asset_id"] == "contrafold_mea_single_raw"
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    code_files = [REPO / "tools/register_bprna_new_recovered_baselines_20261125.py", REPO / "tools/verify_bprna_new_baseline_registry_20261125.py",
                  MANIFEST, REPO / "manifests/foldnar_v1/data_registry.json", REPO / "manifests/foldnar_v1/prediction_registry.json", REPO / "manifests/foldnar_v1/baseline_registry.json"]
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_INDEPENDENT_20_ASSET_RECOMPUTATION_AND_REGISTRY_LINKS",
        "git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "git_branch": subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip(),
        "n_assets": len(checks), "checks": checks,
        "strongest_project_raw": document["strongest_project_raw"],
        "strongest_project_raw_single_predictor": document["strongest_project_raw_single_predictor"],
        "remaining_blocked": document["remaining_blocked"],
        "files": [{"path": str(path), "sha256": digest(path)} for path in code_files],
    }
    output = args.artifact_root / "independent_registry_verification.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "n_assets": len(checks), "report": str(output), "sha256": digest(output)}, indent=2))


if __name__ == "__main__":
    main()
