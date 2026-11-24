#!/usr/bin/env python3
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2")
METRICS = ROOT / "reports/bprna_new_raw_metrics_verification_v2.json"
PROVENANCE = ROOT / "reports/official_provenance_and_source_attempts_v2.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected):
    assert abs(actual - expected) <= 1e-12, (actual, expected)


def main():
    document = json.loads(METRICS.read_text())
    provenance = json.loads(PROVENANCE.read_text())
    assert document["status"] == "PASS"
    assert document["dataset"]["n_sequences"] == 5388
    expected = {
        "bpfold": (106998, 48176, 28186, 0.7370074184282851, 0.7447389175263477),
        "ufold": (80572, 84827, 54612, 0.5361048362681855, 0.5589247295596474),
    }
    for system, values in expected.items():
        metrics = document["frozen_project_predictions"][system]["metrics"]
        assert (metrics["tp"], metrics["fp"], metrics["fn"]) == values[:3]
        close(metrics["micro_f1"], values[3])
        close(metrics["macro_f1"], values[4])
        assert metrics["n_with_prediction"] == metrics["n_dataset_sequences"] == 5388
        assert metrics["coverage"] == 1.0
        rows = (ROOT / f"reports/{system}_frozen_bprna_new_per_sequence_v2.jsonl").read_text().splitlines()
        assert len(rows) == 5388
    release = document["bpfold_official_release"]
    assert release["n_official_sequences"] == 5401
    assert release["n_same_name_same_sequence"] == 5388
    assert release["n_official_only_excluded"] == 13
    assert release["project_only_names"] == []
    project_metrics = release["metrics_on_same_sequence_intersection"]
    assert (project_metrics["tp"], project_metrics["fp"], project_metrics["fn"]) == (135184, 10776, 0)
    close(project_metrics["micro_f1"], 0.9616708875167174)
    official = release["official_reference_evaluation"]
    all_metrics = official["metrics_all_5401_official_reference"]
    assert (all_metrics["tp"], all_metrics["fp"], all_metrics["fn"]) == (86933, 59441, 41092)
    close(all_metrics["micro_f1"], 0.633624758107719)
    close(all_metrics["macro_f1"], 0.6470223791166072)
    assert round(all_metrics["macro_f1"], 3) == 0.647
    intersection = official["metrics_5388_official_reference"]
    assert (intersection["tp"], intersection["fp"], intersection["fn"]) == (86731, 59229, 40929)
    close(intersection["micro_f1"], 0.6339521964768657)
    close(intersection["macro_f1"], 0.6472307861307366)
    assert official["n_project_label_sets_exactly_equal_official_reference"] == 106
    assert official["n_project_label_sets_different_from_official_reference"] == 5282
    normalized = ROOT / "predictions/bpfold_official_release/bprna_new_all_official_v2.jsonl"
    normalized_rows = [json.loads(line) for line in normalized.read_text().splitlines()]
    assert len(normalized_rows) == 5401
    assert len({row["name"] for row in normalized_rows}) == 5401
    assert document["contextfold"]["metrics"] is None
    assert document["contextfold"]["coverage"] is None
    assert provenance["contextfold"]["per_sequence_predictions"] is None
    assert provenance["shared_registries_modified"] is False
    assert provenance["running_artifacts_modified"] is False
    assert not list(ROOT.rglob("*.tmp"))
    core_files = [
        ROOT / "scripts/recover_bprna_new_baselines_v2_20261124.py",
        ROOT / "scripts/test_recover_bprna_new_baselines_v2_20261124.py",
        ROOT / "scripts/verify_baseline_recovery_v2_20261124.py",
        ROOT / "reports/baseline_recovery_v2_report_20261124.md",
        METRICS,
        PROVENANCE,
        ROOT / "official_assets/bpfold/BPfold_test_results_v0.2.tar.gz",
        ROOT / "official_assets/bpfold/release_v0.2_api.json",
        normalized,
    ]
    result = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "checks": [
            "frozen BPfold and UFold metrics, coverage, and per-sequence row counts",
            "official BPfold 5401 predictions and 5388 exact project intersections",
            "official BPfold reference metrics reproduce README rounded macro-F1 0.647",
            "official/project label mismatch counts and 13 official-only exclusions",
            "ContextFold null metrics/coverage and no prediction substitution",
            "no temporary files and core deliverable hashes",
        ],
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in core_files
        ],
    }
    out = ROOT / "reports/delivery_verification_v2.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "n_core_files": len(core_files), "report": str(out), "sha256": sha256(out)}, indent=2))


if __name__ == "__main__":
    main()
