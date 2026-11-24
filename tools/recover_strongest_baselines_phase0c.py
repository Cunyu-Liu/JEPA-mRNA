#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import tarfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATASET = "bprna_new"
SYSTEMS = ("mxfold2", "eternafold")
EXPECTED = {
    "data_sha256": "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9",
    "mxfold2": {
        "prediction_sha256": "9cc688e0c57f188213c323f553039c76c896029b50e2446f8eeeceafc2b94e19",
        "weights_sha256": "8ba1283e70c1c073033161adc120f67cdc467176476358d26a12cadaf63262ec",
        "wheel_sha256": "a9ea044a6555903e019fc016d8409ba0907bfb143bbf2eda51b1d27aae763c72",
    },
    "eternafold": {
        "prediction_sha256": "aa2ef3577f150e42d90fb8954383d2ccc59169e8164209ef83cdcf48b073f04b",
        "parameters_sha256": "1421e89cc8df24b53a320eff6c72b8acfdb2771faba29ab9c798f7bdfc853a8a",
        "commit": "87b9aac55cee14fd562049d08f7b92d3131f10ce",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
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


def strict_metrics(rows, predictions):
    assert len(rows) == len(predictions)
    tp = fp = fn = 0
    macro = []
    per_sequence = []
    for row, prediction in zip(rows, predictions):
        assert row["name"] == prediction["name"]
        truth = pair_set(row["pairs"], len(row["seq"]))
        predicted = pair_set(prediction["pairs"], len(row["seq"]))
        a = len(truth & predicted)
        b = len(predicted - truth)
        c = len(truth - predicted)
        denominator = 2 * a + b + c
        f1 = 2 * a / denominator if denominator else 1.0
        tp += a
        fp += b
        fn += c
        macro.append(f1)
        per_sequence.append({"name": row["name"], "tp": a, "fp": b, "fn": c, "f1": f1})
    return {
        "n_sequences": len(rows),
        "n_with_prediction": len(predictions),
        "coverage": len(predictions) / len(rows) if rows else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "macro_f1": math.fsum(macro) / len(macro) if macro else None,
        "per_sequence": per_sequence,
    }


def close_enough(actual, expected):
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def prediction_rows(document, system):
    source = document["predictions"] if system == "mxfold2" else document["per_sequence"]
    key = "pairs" if system == "mxfold2" else "pred_pairs"
    return [{"name": row["name"], "pairs": row[key]} for row in source]


def check_recorded_metrics(document, system, metrics):
    if system == "mxfold2":
        assert len(document["runs"]) == 3
        for run in document["runs"]:
            assert run["n_sequences"] == metrics["n_sequences"]
            assert (run["tp"], run["fp"], run["fn"]) == (metrics["tp"], metrics["fp"], metrics["fn"])
            assert close_enough(run["strict_micro_f1"], metrics["micro_f1"])
            assert close_enough(run["strict_macro_f1"], metrics["macro_f1"])
        assert len({run["predictions_sha256"] for run in document["runs"]}) == 1
    else:
        assert len(document["repeats"]) == 3
        for run in document["repeats"]:
            recorded = run["metrics"]
            assert recorded["n_sequences"] == metrics["n_sequences"]
            assert (recorded["tp"], recorded["fp"], recorded["fn"]) == (metrics["tp"], metrics["fp"], metrics["fn"])
            assert close_enough(recorded["micro_f1"], metrics["micro_f1"])
            assert close_enough(recorded["macro_f1"], metrics["macro_f1"])
        assert len({run["prediction_sha256"] for run in document["repeats"]}) == 1
        for actual, recorded in zip(metrics["per_sequence"], document["per_sequence"]):
            assert actual["name"] == recorded["name"]
            assert (actual["tp"], actual["fp"], actual["fn"]) == (recorded["tp"], recorded["fp"], recorded["fn"])
            assert close_enough(actual["f1"], recorded["f1"])


def parse_bpseq(payload):
    sequence = []
    pairs = []
    for line in payload.decode("utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        assert len(fields) >= 3
        index = int(fields[0]) - 1
        assert index == len(sequence)
        sequence.append(fields[1].upper().replace("T", "U"))
        partner = int(fields[2]) - 1
        if partner >= 0:
            pairs.append((index, partner))
    normalized = []
    for i, j in pairs:
        assert 0 <= j < len(sequence)
        normalized.append((min(i, j), max(i, j)))
    return "".join(sequence), pair_set(set(normalized), len(sequence))


def audit_official_release(rows, archive):
    with tarfile.open(archive, "r:gz") as handle:
        members = [m for m in handle.getmembers() if m.isfile() and m.name.endswith(".bpseq")]
        official = {}
        for member in members:
            name = Path(member.name).name
            assert name not in official
            official[name] = parse_bpseq(handle.extractfile(member).read())
    project = {row["name"]: row for row in rows}
    project_names = set(project)
    official_names = set(official)
    common = sorted(project_names & official_names)
    sequence_matches = sum(
        official[name][0] == project[name]["seq"].upper().replace("T", "U") for name in common
    )
    exact_pair_matches = sum(official[name][1] == pair_set(project[name]["pairs"]) for name in common)
    return {
        "status": "REFERENCE_BPSEQ_ONLY_NOT_PREDICTIONS",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "release_url": "https://github.com/mxfold/mxfold2/releases/download/v0.1.1/bpRNAnew.tar.gz",
        "n_official_bpseq": len(official),
        "n_project": len(project),
        "n_common_names": len(common),
        "n_common_sequence_exact_after_T_to_U": sequence_matches,
        "n_common_pair_sets_exact": exact_pair_matches,
        "official_only_names": sorted(official_names - project_names),
        "project_only_names": sorted(project_names - official_names),
        "note": "The 13 official-only BPSEQ records are outside the frozen project split; no coverage or score is imputed for them.",
    }


def registry_entries(repo):
    root = Path(repo) / "manifests/foldnar_v1"
    data = json.loads((root / "data_registry.json").read_text())
    predictions = json.loads((root / "prediction_registry.json").read_text())
    baselines = json.loads((root / "baseline_registry.json").read_text())
    data_row = next(row for row in data["rows"] if row["dataset_id"] == DATASET)
    prediction_rows_by_system = {
        system: next(row for row in predictions["rows"] if row["prediction_id"] == f"{system}/{DATASET}")
        for system in SYSTEMS
    }
    baseline_rows = {row["baseline_id"]: row for row in baselines["rows"] if row["baseline_id"] in SYSTEMS}
    return data_row, prediction_rows_by_system, baseline_rows


def audit_system(system, rows, registry_row, recovery_root):
    prediction_path = recovery_root / "predictions" / system / f"{DATASET}.json"
    assert sha256(prediction_path) == registry_row["sha256"] == EXPECTED[system]["prediction_sha256"]
    document = json.loads(prediction_path.read_text())
    predictions = prediction_rows(document, system)
    names = [row["name"] for row in predictions]
    counts = Counter(names)
    assert len(predictions) == len(rows) == registry_row["n_sequences"]
    assert all(count == 1 for count in counts.values())
    assert names == [row["name"] for row in rows]
    metrics = strict_metrics(rows, predictions)
    check_recorded_metrics(document, system, metrics)
    public_metrics = {key: value for key, value in metrics.items() if key != "per_sequence"}
    if system == "mxfold2":
        wheel = recovery_root / "mxfold2/mxfold2-0.1.2-cp310-cp310-manylinux_2_17_x86_64.whl"
        assert sha256(wheel) == EXPECTED[system]["wheel_sha256"]
        with zipfile.ZipFile(wheel) as handle:
            weights = handle.read("mxfold2/models/TrainSetAB.pth")
            metadata = handle.read("mxfold2-0.1.2.dist-info/METADATA").decode()
            license_text = handle.read("mxfold2-0.1.2.dist-info/LICENSE").decode()
        assert hashlib.sha256(weights).hexdigest() == EXPECTED[system]["weights_sha256"]
        assert document["protocol"]["model_report"]["weights_sha256"] == EXPECTED[system]["weights_sha256"]
        assert "Version: 0.1.2" in metadata and "License: MIT" in metadata
        assert license_text.startswith("MIT License")
        provenance = {
            "version": "0.1.2",
            "commit": "51b213676708bebd664f0c40873a46e09353e1ee",
            "wheel": str(wheel),
            "wheel_sha256": EXPECTED[system]["wheel_sha256"],
            "weights_sha256": EXPECTED[system]["weights_sha256"],
            "license": "MIT",
            "official_distribution_match": True,
        }
    else:
        params = recovery_root / "eternafold/EternaFoldParams.v1"
        assert sha256(params) == document["parameters_sha256"] == EXPECTED[system]["parameters_sha256"]
        assert document["commit"] == EXPECTED[system]["commit"]
        assert document["data_sha256"] == EXPECTED["data_sha256"]
        assert "official serial CPU binary and MEA decode" in document["protocol"]
        provenance = {
            "version": "EternaFoldParams.v1; source commit " + document["commit"],
            "commit": document["commit"],
            "parameters": str(params),
            "parameters_sha256": document["parameters_sha256"],
            "binary_sha256": document["binary_sha256"],
            "license": "BSD-3-Clause",
            "official_parameter_match": True,
        }
    return {
        "status": "PASS",
        "prediction_path": str(prediction_path),
        "prediction_sha256": registry_row["sha256"],
        "metrics": public_metrics,
        "prediction_identity": {
            "unique_names": len(counts),
            "duplicate_names": sorted(name for name, count in counts.items() if count != 1),
            "missing_names": [],
            "extra_names": [],
            "order_matches_frozen_data": True,
        },
        "provenance": provenance,
    }


def correction_document(systems, release_audit):
    return {
        "schema_version": 1,
        "scope": "Phase 0C; bpRNA-new raw frozen predictions only; no model development",
        "target_registries": [
            "manifests/foldnar_v1/baseline_registry.json",
            "manifests/foldnar_v1/prediction_registry.json",
        ],
        "integration_mode": "FIELD_LEVEL_CORRECTION; shared registries intentionally not modified",
        "corrections": {
            "mxfold2": {
                "official_weights": {
                    "status": "OFFICIAL_RELEASE_VERIFIED",
                    "version": "0.1.2",
                    "commit": "51b213676708bebd664f0c40873a46e09353e1ee",
                    "distribution_url": "https://github.com/mxfold/mxfold2/releases/download/v0.1.2/mxfold2-0.1.2-cp310-cp310-manylinux_2_17_x86_64.whl",
                    "distribution_sha256": EXPECTED["mxfold2"]["wheel_sha256"],
                    "member": "mxfold2/models/TrainSetAB.pth",
                    "sha256": EXPECTED["mxfold2"]["weights_sha256"],
                },
                "license": {"status": "VERIFIED_OFFICIAL_DISTRIBUTION", "spdx": "MIT"},
                "provenance_status": "RECOVERED_WEIGHTS_LICENSE_PROTOCOL; TRAINING_OVERLAP_STILL_UNRESOLVED",
                "verified_metrics": systems["mxfold2"]["metrics"],
            },
            "eternafold": {
                "official_weights": {
                    "status": "OFFICIAL_PARAMETER_FILE_VERIFIED",
                    "version": "EternaFoldParams.v1",
                    "commit": EXPECTED["eternafold"]["commit"],
                    "distribution_url": "https://raw.githubusercontent.com/WaymentSteeleLab/EternaFold/87b9aac55cee14fd562049d08f7b92d3131f10ce/parameters/EternaFoldParams.v1",
                    "path": systems["eternafold"]["provenance"]["parameters"],
                    "sha256": EXPECTED["eternafold"]["parameters_sha256"],
                },
                "license": {"status": "VERIFIED_OFFICIAL_SOURCE_AND_PARAMETERS", "spdx": "BSD-3-Clause"},
                "provenance_status": "RECOVERED_PARAMETERS_LICENSE_PROTOCOL; EXACT_TRAINING_UNION_STILL_UNRESOLVED",
                "verified_metrics": systems["eternafold"]["metrics"],
            },
        },
        "remaining_blocked_provenance": [
            {
                "baseline_id": "mxfold2",
                "field": "training_data.homology_overlap",
                "status": "BLOCKED_PROVENANCE",
                "reason": "Official TrainSetA/B identity is known, but this slice did not reconstruct and compare the complete training union against bpRNA-new; raw results remain non-decontaminated.",
            },
            {
                "baseline_id": "eternafold",
                "field": "training_data.exact_union",
                "status": "BLOCKED_PROVENANCE",
                "reason": "Official repository describes multitask Eterna data and ships data lists, but the exact effective training union and bpRNA-new homology overlap were not reconstructed in this slice.",
            },
        ],
        "official_source_attempts": [
            {
                "source": "MXfold2 v0.1.2 bpRNAnew.tar.gz",
                "url": "https://github.com/mxfold/mxfold2/releases/download/v0.1.2/bpRNAnew.tar.gz",
                "outcome": "HTTP_404; asset is attached to v0.1.1 instead",
            },
            {
                "source": "MXfold2 v0.1.1 bpRNAnew.tar.gz",
                "outcome": "RECOVERED; 5401 reference BPSEQ files and no prediction structures",
                "audit": release_audit,
            },
            {
                "source": "WaymentSteeleLab/EternaFold pinned source tree",
                "commit": EXPECTED["eternafold"]["commit"],
                "outcome": "Parameters and licenses recovered; repository tree has no bpRNA-new per-sequence prediction asset",
            },
            {
                "source": "eternagame/EternaBench",
                "outcome": "Inspected scoring_data and repository tree; benchmark score tables are not per-sequence frozen structures and were not substituted",
            },
        ],
        "scientific_use": "Same 5388 raw sequences, strict zero-based pair scoring; not clean/homology-controlled and not a claim about the 13 release-only records.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56"))
    parser.add_argument("--recovery-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v1"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--correction-out", type=Path)
    args = parser.parse_args()
    out = args.out or args.recovery_root / "phase0c_verification.json"
    correction_out = args.correction_out or args.repo / "manifests/foldnar_v1/baseline_recovery_phase0c_correction.json"
    data_row, prediction_registry, _ = registry_entries(args.repo)
    assert data_row["sha256"] == EXPECTED["data_sha256"]
    data_path = Path(data_row["path"])
    assert sha256(data_path) == data_row["sha256"]
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    assert len(rows) == data_row["n_sequences"] == 5388
    assert len({row["name"] for row in rows}) == len(rows)
    systems = {
        system: audit_system(system, rows, prediction_registry[system], args.recovery_root)
        for system in SYSTEMS
    }
    release_audit = audit_official_release(rows, args.recovery_root / "mxfold2/bpRNAnew-v0.1.1.tar.gz")
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": "independent frozen-prediction recomputation; no model training or test-set development",
        "dataset": {
            "dataset_id": DATASET,
            "path": str(data_path),
            "sha256": data_row["sha256"],
            "n_sequences": len(rows),
            "pair_convention": "zero-based inclusive; strict exact-pair matching",
        },
        "systems": systems,
        "mxfold2_official_release_identity_audit": release_audit,
    }
    write_json(out, result)
    correction = correction_document(systems, release_audit)
    write_json(correction_out, correction)
    print(json.dumps({"status": result["status"], "systems": {k: v["metrics"] for k, v in systems.items()}}, indent=2))
    print(f"WROTE {out} sha256={sha256(out)}")
    print(f"WROTE {correction_out} sha256={sha256(correction_out)}")


if __name__ == "__main__":
    main()
