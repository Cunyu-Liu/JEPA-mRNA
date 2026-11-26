#!/usr/bin/env python3
import gzip
import hashlib
import json
import math
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "source_data/bprna_official_reference_5388"
MANIFESTS = ROOT / "manifests/foldnar_v1"
OFFICIAL = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz")
PROJECT = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_panel():
    return [json.loads(line) for line in gzip.decompress((OUT / "official_reference_5388.jsonl.gz").read_bytes()).decode().splitlines()]


def parse_official():
    result = {}
    with tarfile.open(OFFICIAL, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith("BPfold_data/test_data/bpRNAnew/") or not member.name.endswith(".bpseq"):
                continue
            bases, partners = [], []
            for raw in archive.extractfile(member).read().decode().splitlines():
                if not raw.strip() or raw.startswith("#"):
                    continue
                fields = raw.split()
                assert int(fields[0]) == len(bases) + 1
                bases.append(fields[1].upper().replace("T", "U"))
                partners.append(int(fields[2]))
            pairs = set()
            for i, partner in enumerate(partners):
                if partner:
                    assert partners[partner - 1] == i + 1
                    pairs.add((min(i, partner - 1), max(i, partner - 1)))
            result[Path(member.name).name] = ("".join(bases), pairs)
    return result


def verify_panel(panel):
    project_names = [json.loads(line)["name"] for line in PROJECT.read_text().splitlines() if line.strip()]
    official = parse_official()
    assert len(official) == 5401 and len(project_names) == len(panel) == 5388
    expected = []
    for name in project_names:
        sequence, pair_set = official[name]
        core = {
            "name": name, "seq": sequence, "pairs": [list(pair) for pair in sorted(pair_set)],
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "pair_set_sha256": value_hash(sorted(pair_set)),
        }
        expected.append({**core, "record_sha256": value_hash(core)})
    assert panel == expected


def prediction_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        source = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        document = json.loads(path.read_text())
        if "per_sequence" in document:
            source = document["per_sequence"]
        elif "predictions" in document:
            source = document["predictions"]
        else:
            raise ValueError("NO_PER_SEQUENCE_PREDICTIONS")
    result = {}
    for row in source:
        pairs = row.get("pairs", row.get("pred_pairs"))
        assert pairs is not None and row["name"] not in result
        result[row["name"]] = (row.get("seq"), pairs)
    return result


def pairs(values, length):
    result = set()
    endpoints = []
    for value in values:
        i, j = map(int, value)
        assert 0 <= i < j < length
        endpoints.extend((i, j))
        result.add((i, j))
    assert len(set(endpoints)) == len(endpoints)
    return result


def rescore(panel, predictions):
    panel_by_name = {row["name"]: row for row in panel}
    missing = sorted(set(panel_by_name) - set(predictions))
    extra = sorted(set(predictions) - set(panel_by_name))
    totals = [0, 0, 0]
    values = []
    per_name = {}
    for row in panel:
        name = row["name"]
        if name not in predictions:
            continue
        sequence, raw_pairs = predictions[name]
        if sequence is not None:
            assert sequence.upper().replace("T", "U") == row["seq"].upper().replace("T", "U")
        truth = pairs(row["pairs"], len(row["seq"]))
        predicted = pairs(raw_pairs, len(row["seq"]))
        tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        if not truth and not predicted:
            precision = recall = f1 = 1.0
        else:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        totals = [totals[0] + tp, totals[1] + fp, totals[2] + fn]
        values.append((precision, recall, f1))
        per_name[name] = f1
    tp, fp, fn = totals
    n = len(values)
    metrics = {
        "n_scored": n, "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
        "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "macro_precision": math.fsum(value[0] for value in values) / n if n else None,
        "macro_recall": math.fsum(value[1] for value in values) / n if n else None,
        "macro_f1": math.fsum(value[2] for value in values) / n if n else None,
    }
    return metrics, missing, extra, per_name


def interval(deltas):
    rng = np.random.default_rng(20261127)
    samples = []
    for _ in range(50):
        index = rng.integers(0, len(deltas), size=(200, len(deltas)))
        samples.append(deltas[index].mean(axis=1))
    flattened = np.concatenate(samples)
    return [float(x) for x in np.percentile(flattened, [2.5, 97.5])]


def verify_registries(document, panel_hash):
    data = json.loads((MANIFESTS / "data_registry.json").read_text())
    raw = next(row for row in data["rows"] if row["dataset_id"] == "bprna_new")
    assert raw["label_source"] == "DERIVED_FROM_BPFOLD_PREDICTIONS"
    assert raw["label_lineage"]["disposition"] == "circular-audit-only"
    assert raw["independent_sota_eligible"] is False
    panel = next(row for row in data["rows"] if row["dataset_id"] == "bpfold_official_bprna_new_reference_5388")
    assert panel["sha256"] == panel_hash and panel["n_sequences"] == 5388
    recovered = json.loads((MANIFESTS / "bprna_new_recovered_prediction_registry.json").read_text())
    assert recovered["historical_raw_ranking_disposition"].startswith("CIRCULAR_AUDIT_ONLY")
    assert recovered["strongest_project_raw"]["micro_f1"] == 0.7370074184282851
    assert recovered["strongest_project_raw"]["independent_sota_eligible"] is False
    assert recovered["official_reference_5388"]["strongest_overall"]["prediction_id"] == document["strongest_overall"]["prediction_id"]
    for row in recovered["rows"]:
        if row.get("legacy_label_protocol_before_20261127") in {"PROJECT_DATABASE_RAW_5388", "PROJECT_TR1C_CLEAN_5386"}:
            assert row["label_lineage"]["independent_sota_eligible"] is False
            assert "circular-audit-only" in row["sota_use"]
    predictions = json.loads((MANIFESTS / "prediction_registry.json").read_text())
    for row in predictions["rows"]:
        if row.get("data_path") == str(PROJECT) or row.get("dataset_id") == "bprna_new":
            assert row["label_protocol"] == "DERIVED_FROM_BPFOLD_PREDICTIONS_PROJECTED_5388"
            assert row["independent_sota_eligible"] is False
            assert row["historical_metrics_retained"] is True
    baselines = json.loads((MANIFESTS / "baseline_registry.json").read_text())
    assert baselines["official_reference_5388_overlay"].endswith("bprna_new_official_reference_5388_prediction_registry.json")
    overlay_text = (MANIFESTS / "bprna_new_official_reference_5388_prediction_registry.json").read_text()
    assert "0.9616708875167174" not in json.dumps(document)
    assert "0.9616708875167174" not in overlay_text


def main():
    panel = read_panel()
    verify_panel(panel)
    document = json.loads((OUT / "system_metrics.json").read_text())
    assert document["label_panel"]["sha256"] == file_hash(OUT / "official_reference_5388.jsonl.gz")
    all_f1 = {}
    for row in document["rows"]:
        if row["status"].startswith("BLOCKED"):
            assert row["metrics"] is None and row["missing_count"] == 5388
            continue
        assert file_hash(row["path"]) == row["sha256"]
        metrics, missing, extra, per_name = rescore(panel, prediction_rows(row["path"]))
        assert row["metrics"] == metrics, row["prediction_id"]
        assert row["missing_names"] == missing and row["missing_count"] == len(missing)
        assert row["extra_names"] == extra
        assert row["n_prediction_records"] == len(prediction_rows(row["path"]))
        assert row["n_with_prediction"] == 5388 - len(missing)
        assert row["coverage"] == (5388 - len(missing)) / 5388
        if not missing:
            assert row["status"] == "PASS_FULL_COVERAGE"
            all_f1[row["prediction_id"]] = np.asarray([per_name[name] for name in [item["name"] for item in panel]])
        else:
            assert row["status"] == "PARTIAL_COVERAGE_NOT_RANKED"
    assert len(all_f1) == 28
    xens2 = all_f1["xens2/bprna_new"]
    for row in document["rows"]:
        if row["status"] != "PASS_FULL_COVERAGE":
            continue
        deltas = all_f1[row["prediction_id"]] - xens2
        ci = row["paired_macro_f1_delta_ci95_vs_xens2"]
        assert ci["observed_delta"] == float(deltas.mean())
        assert ci["ci95"] == interval(deltas)
        assert ci["n_bootstrap"] == 10000 and ci["seed"] == 20261127
    eligible = [row for row in document["rows"] if row["status"] == "PASS_FULL_COVERAGE"]
    single = max((row for row in eligible if "ensemble" not in row["model_form"]), key=lambda row: row["metrics"]["micro_f1"])
    ensemble = max((row for row in eligible if "ensemble" in row["model_form"]), key=lambda row: row["metrics"]["micro_f1"])
    assert document["strongest_single_predictor"]["prediction_id"] == single["prediction_id"]
    assert document["strongest_ensemble"]["prediction_id"] == ensemble["prediction_id"]
    verify_registries(document, document["label_panel"]["sha256"])
    deliverables = [
        ROOT / "tools/build_bprna_official_reference_panel_20261127.py",
        ROOT / "tools/verify_bprna_official_reference_panel_20261127.py",
        ROOT / "tests/test_bprna_official_reference_panel_20261127.py",
        ROOT / "docs/plans/2026-11-27-bprna-official-reference-5388-panel.md",
        OUT / "official_reference_5388.jsonl.gz", OUT / "system_metrics.json",
        MANIFESTS / "bprna_new_official_reference_5388_prediction_registry.json",
        MANIFESTS / "bprna_new_recovered_prediction_registry.json",
        MANIFESTS / "prediction_registry.json", MANIFESTS / "data_registry.json", MANIFESTS / "baseline_registry.json",
    ]
    verification = {
        "schema_version": "bprna.official_reference_5388.independent_verification.v1",
        "audit_date": "2026-11-27", "status": "PASS_INDEPENDENT_SECOND_IMPLEMENTATION",
        "panel_records_verified": len(panel), "full_coverage_systems_verified": len(all_f1),
        "paired_ci_verified": len(all_f1), "bootstrap_replicates": 10000,
        "checks": [
            "official archive to 5388 panel IDs/sequences/pairs/per-record SHA",
            "every available prediction SHA, sequence, pair set, coverage and micro/macro P/R/F1",
            "paired sequence bootstrap macro-F1 CI versus xens2",
            "strongest single and ensemble descriptive ranking",
            "derived-label lineage and historical raw metric retention",
            "0.9616708875167174 absent from official-reference results and ranking",
        ],
        "new_inference": False, "model_selection": False, "gpu_task_launched": False,
        "deliverables": [{"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)} for path in deliverables],
    }
    write_path = OUT / "independent_verification.json"
    write_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": verification["status"], "panel": len(panel), "full_coverage": len(all_f1), "sha256": file_hash(write_path)}, indent=2))


if __name__ == "__main__":
    main()
