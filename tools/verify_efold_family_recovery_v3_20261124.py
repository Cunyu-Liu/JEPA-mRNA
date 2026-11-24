#!/usr/bin/env python3
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v3_efold_family")
METRICS = ROOT / "reports/bprna_new_efold_family_metrics_v3.json"
PROVENANCE = ROOT / "reports/official_provenance_and_missing_assets_v3.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (actual, expected)


def independent_recompute():
    data_path = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
    prediction_path = ROOT / "predictions/efold_project_replay_bprna_new_frozen_v3.json"
    labels = [json.loads(line) for line in data_path.read_text().splitlines()]
    source = json.loads(prediction_path.read_text())["per_sequence"]
    assert len(labels) == len(source) == 5388
    total_tp = total_fp = total_fn = 0
    f1_values = []
    for label, prediction in zip(labels, source):
        assert label["name"] == prediction["name"]
        truth = {tuple(pair) for pair in label["pairs"]}
        predicted = {tuple(pair) for pair in prediction["pred_pairs"]}
        tp = len(truth & predicted)
        fp = len(predicted - truth)
        fn = len(truth - predicted)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        f1_values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0)
    return total_tp, total_fp, total_fn, math.fsum(f1_values) / len(f1_values)


def main():
    document = json.loads(METRICS.read_text())
    provenance = json.loads(PROVENANCE.read_text())
    assert document["status"] == "PASS_WITH_EXPLICIT_PROVENANCE_GAPS"
    assert document["recovery_mode"] == "CPU parsing/download only; no inference and no GPU calls"
    assert document["dataset"]["n_sequences"] == 5388
    assert document["dataset"]["unique_names"] == 5388
    assert document["dataset"]["n_truth_pairs"] == 135184
    assert document["dataset"]["sha256"] == "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
    efold = document["systems"]["efold"]
    metrics = efold["metrics"]
    assert efold["status"] == "PASS_INDEPENDENT_RECOMPUTATION"
    assert efold["evidence_class"] == "PROJECT_REPLAY_WITH_OFFICIAL_SOURCE_AND_WEIGHTS"
    assert efold["official_released_bprna_new_predictions"] is None
    assert efold["recorded_repeats_verified"] == 3
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (100583, 73955, 34601)
    close(metrics["micro_precision"], 0.5762813828507259)
    close(metrics["micro_recall"], 0.7440451532725766)
    close(metrics["micro_f1"], 0.6495050400036162)
    close(metrics["macro_f1"], 0.6556973548256184)
    assert metrics["n_with_prediction"] == metrics["n_dataset_sequences"] == 5388
    assert metrics["coverage"] == 1.0
    assert efold["prediction_artifact_sha256"] == "17e2652ea8ea622574efc2ff8d4e277509d8d9d61e04040fb9c1c8a244a15e3e"
    assert efold["legacy_replay_prediction_sha256"] == "567520183e007c2b3a50c67884162cf556190c6dc46ec03bf51aaa7aba2a70a2"
    assert efold["prediction_identity_sha256"] == "1b7e8a50c8ce1b1afdd34423d89c6c28e93f0901dfb4ee111c2e8aedfdcfd7f5"
    assert efold["provenance"]["weights_sha256"] == "fcf06c6a7724501e0fb1c5f12bc21ab5bfe7d6131e643d41f139be776aa1dca9"
    tp, fp, fn, macro_f1 = independent_recompute()
    assert (tp, fp, fn) == (metrics["tp"], metrics["fp"], metrics["fn"])
    close(macro_f1, metrics["macro_f1"])
    per_sequence_path = ROOT / "reports/efold_project_replay_bprna_new_per_sequence_v3.jsonl"
    per_sequence = [json.loads(line) for line in per_sequence_path.read_text().splitlines()]
    assert len(per_sequence) == 5388
    assert len({row["name"] for row in per_sequence}) == 5388
    assert sum(row["tp"] for row in per_sequence) == 100583
    assert sum(row["fp"] for row in per_sequence) == 73955
    assert sum(row["fn"] for row in per_sequence) == 34601
    close(math.fsum(row["f1"] for row in per_sequence) / 5388, metrics["macro_f1"])
    for system in ("e2efold", "debfold"):
        blocked = document["systems"][system]
        assert blocked["status"] == "BLOCKED_PROVENANCE"
        assert blocked["metrics"] is None
        assert blocked["coverage"] is None
        assert blocked["per_sequence_predictions"] is None
        assert len(blocked["missing_provenance"]) == 3
    assert provenance["label_scopes_separate"] is True
    assert provenance["missing_metrics_are_null_not_zero"] is True
    assert provenance["shared_registries_modified"] is False
    assert provenance["running_configs_or_evaluator_modified"] is False
    assert provenance["ppt_modified"] is False
    assert not list(ROOT.rglob("*.tmp"))
    core_files = [
        REPO / "tools/recover_efold_family_baselines_v3_20261124.py",
        REPO / "tests/test_recover_efold_family_baselines_v3_20261124.py",
        REPO / "tools/verify_efold_family_recovery_v3_20261124.py",
        REPO / "reports/baseline_recovery_v3_efold_family_20261124.md",
        ROOT / "scripts/recover_efold_family_baselines_v3_20261124.py",
        ROOT / "scripts/test_recover_efold_family_baselines_v3_20261124.py",
        ROOT / "scripts/verify_efold_family_recovery_v3_20261124.py",
        ROOT / "reports/baseline_recovery_v3_efold_family_20261124.md",
        METRICS,
        PROVENANCE,
        per_sequence_path,
        ROOT / "predictions/efold_project_replay_bprna_new_frozen_v3.json",
        ROOT / "official_assets/efold/efold_weights.pt",
        ROOT / "official_assets/efold/releases_api_v3.json",
        ROOT / "official_assets/e2efold/checkpoint_folder_curl_v3.log",
        ROOT / "official_assets/e2efold/released_results_curl_v3.log",
        ROOT / "official_assets/e2efold/releases_api_v3.json",
        ROOT / "official_assets/debfold/main_commit_api_v3.json",
        ROOT / "official_assets/debfold/git_tree_main_v3.json",
        ROOT / "official_assets/debfold/TrainingValidationSet.zip",
        ROOT / "official_assets/debfold/releases_api_v3.json",
    ]
    assert all(path.is_file() for path in core_files)
    result = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "checks": [
            "second standalone recomputation from frozen labels and eFold predictions",
            "exact 5,388-row coverage, counts, micro metrics, and macro-F1",
            "three recorded repeats and both legacy/dataset-order prediction hashes",
            "official source/weight hashes and release/provenance boundaries",
            "E2Efold and DEBFold null metrics/coverage with explicit missing provenance",
            "main-repo and artifact deliverable presence and hashes",
            "no temporary output files and no inference in recovery",
        ],
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in core_files
        ],
    }
    output = ROOT / "reports/delivery_verification_v3.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "n_core_files": len(core_files), "report": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
