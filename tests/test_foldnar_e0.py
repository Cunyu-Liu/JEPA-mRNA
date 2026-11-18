from __future__ import annotations

import json
import gzip
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from foldnar_e0 import (decide_gate, evaluate_join, flatten_row, pair_span_confusion,
                        select_preregistered_helix_junction_cases)
from foldnar_e0_stratified import attach_strata, error_category_analysis, pair_span_analysis, summarize


def test_evaluate_join_and_same_set_gap(tmp_path: Path) -> None:
    data = [
        {"name": "simple", "seq": "ACGUAC", "pairs": [[0, 5], [1, 4]]},
        {"name": "cross", "seq": "ACGUACGU", "pairs": [[0, 5], [2, 7]]},
        {"name": "not_predicted", "seq": "ACGU", "pairs": []},
    ]
    data_path = tmp_path / "toy.jsonl"
    data_path.write_text("".join(json.dumps(x) + "\n" for x in data))
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps({"n_sequences": 2, "per_sequence": [
        {"name": "simple", "pred_pairs": [[0, 5], [1, 4]]},
        {"name": "cross", "pred_pairs": []},
    ]}))
    spec = {"prediction_id": "toy/toy", "model": "toy", "split": "toy", "repeat": None, "path": str(pred_path)}
    result = evaluate_join(spec, str(data_path))
    assert result["summary"]["n_joined_names"] == 2
    assert result["summary"]["n_structure_valid"] == 1
    assert result["summary"]["coverage_of_truth"] == pytest.approx(2 / 3)
    assert result["summary"]["macro"]["pair_f1"] == pytest.approx(0.5)
    assert result["summary"]["macro"]["topology_f1"] == 1
    # The gap uses only the one sequence where both pair and topology are defined.
    assert result["summary"]["macro"]["pair_f1_minus_topology_f1"] == 0
    cross = next(r for r in result["rows"] if r["name"] == "cross")
    assert cross["status"] == "STRUCTURE_UNDEFINED"
    assert cross["pair"]["fn"] == 2
    assert cross["pair_span_confusion"]["4_9"]["fn"] == 2
    flat = flatten_row(result["rows"][0])
    assert "pair_f1" in flat and "topology_ged" in flat
    frame = pd.DataFrame([flatten_row(row) for row in result["rows"]])
    simple = frame[frame["name"] == "simple"].iloc[0]
    records = {("toy", "simple", simple["sequence_sha256"]): ("toy-topology", None, "NOT_DETECTED_UNKNOWN")}
    joined = attach_strata(frame, records, set(), {"topologies": {"toy-topology": ["train"]}, "families": set()})
    joined_simple = joined[joined["name"] == "simple"].iloc[0]
    joined_cross = joined[joined["name"] == "cross"].iloc[0]
    assert joined_simple["train_topology_coverage"] == "COVERED"
    assert joined_simple["pretraining_homology_status"] == "NOT_DETECTED_UNKNOWN"
    assert joined_cross["audit_join_status"] == "MISSING"
    labels = ["0_3", "4_9", "10_24", "25_49", "50_99", "100_199", "200_plus"]
    spans = pair_span_analysis(joined, labels)
    span_4_9 = next(row for row in spans if row["pair_span_bin"] == "4_9")
    assert (span_4_9["tp"], span_4_9["fn"]) == (1, 2)
    span_0_3 = next(row for row in spans if row["pair_span_bin"] == "0_3")
    assert (span_0_3["tp"], span_0_3["fn"]) == (1, 0)
    assert summarize(joined)["macro"]["wiring_f1"] == 1
    categories = error_category_analysis(joined)
    assert next(row for row in categories if row["category"] == "wiring")["fn"] == 0

    artifact_root = tmp_path / "artifact_root"
    e0_dir = artifact_root / "artifacts/foldnar_v1/e0_expanded"
    phase1_dir = artifact_root / "artifacts/foldnar_v1/phase1"
    e0_dir.mkdir(parents=True)
    phase1_dir.mkdir(parents=True)
    frame.to_parquet(e0_dir / "error_decomposition.parquet", index=False)
    (e0_dir / "e0_error_decomposition.json").write_text(json.dumps({"pair_span_bins": [{"label": label} for label in labels]}))
    audit_path = phase1_dir / "per_sequence_audit.jsonl.gz"
    audit_rows = []
    for row in frame.to_dict("records"):
        audit_rows.append({"dataset_id": "toy", "role": "test", "id": row["name"], "length": row["length"],
                           "sha256": row["sequence_sha256"], "raw_sequence_sha256": row["sequence_sha256"],
                           "family": None, "clan": None, "topology": "toy-topology" if row["name"] == "simple" else None,
                           "pretraining_homology_status": "NOT_DETECTED_UNKNOWN" if row["name"] == "simple" else "UNCHECKED"})
    audit_rows.append({"dataset_id": "train", "role": "train", "id": "train", "length": 6, "sha256": "train-sha",
                       "raw_sequence_sha256": "train-sha", "family": "F1", "clan": "C1", "topology": "toy-topology",
                       "pretraining_homology_status": "UNCHECKED"})
    with gzip.open(audit_path, "wt") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row) + "\n")
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    (phase1_dir / "distribution.json").write_text(json.dumps({"per_sequence_audit": {
        "artifact": str(audit_path), "sha256": audit_sha, "n_rows": len(audit_rows)}}))
    tools = Path(__file__).resolve().parents[1] / "tools"
    main_run = subprocess.run([sys.executable, str(tools / "foldnar_e0_stratified.py"), "--artifact-root", str(artifact_root)],
                              capture_output=True, text=True)
    assert main_run.returncode == 0, main_run.stdout + main_run.stderr
    verify_run = subprocess.run([sys.executable, str(tools / "foldnar_verify_stratified.py"), "--artifact-root", str(artifact_root)],
                                capture_output=True, text=True)
    assert verify_run.returncode == 0, verify_run.stdout + verify_run.stderr
    report = json.loads((e0_dir / "independent_stratified_verification.json").read_text())
    assert report["status"] == "PASS" and report["n_mismatches"] == 0


def _run(model: str, split: str, topology: float = 0.7) -> dict:
    return {"model": model, "split": split, "repeat": None, "coverage_of_truth": 1,
            "n_structure_valid": 10,
            "macro": {"topology_f1": topology, "stem_f1": 0.7, "pair_f1_minus_topology_f1": 0.1,
                      "multiloop_f1": 0.6, "junction_f1": 0.6}}


def test_gate_requires_evidence_and_two_split_trigger(tmp_path: Path) -> None:
    runs = []
    for model in ("xens2", "mxfold2"):
        runs += [_run(model, "bprna_new"), _run(model, "bprna_ts0")]
    gate = decide_gate(runs, tmp_path / "missing_leakage.json")
    assert gate["evidence"]["sufficient"] is True
    assert gate["trigger"]["distinct_rule1_splits"] == 2
    assert gate["e0_decision"] == "GO_E1"
    assert gate["execution_decision"] == "PENDING_PHASE1_LEAKAGE_AUDIT"


def test_gate_blocks_when_second_system_missing(tmp_path: Path) -> None:
    gate = decide_gate([_run("xens2", "bprna_new"), _run("xens2", "bprna_ts0")], tmp_path / "none.json")
    assert gate["evidence"]["sufficient"] is False
    assert gate["e0_decision"] == "BLOCKED_INSUFFICIENT_EVIDENCE"
    assert gate["execution_decision"] == "DO_NOT_TRAIN"


def test_gate_no_trigger_does_not_train(tmp_path: Path) -> None:
    runs = []
    for model in ("xens2", "mxfold2"):
        for split in ("bprna_new", "bprna_ts0"):
            run = _run(model, split, topology=1)
            run["macro"].update({"stem_f1": 1, "pair_f1_minus_topology_f1": 0,
                                 "multiloop_f1": 1, "junction_f1": 1})
            runs.append(run)
    gate = decide_gate(runs, tmp_path / "none.json")
    assert gate["e0_decision"] == "NO_GO"
    assert gate["execution_decision"] == "DO_NOT_TRAIN"


def test_blank_names_require_verified_positional_alignment(tmp_path: Path) -> None:
    data = [{"id": "a", "seq": "ACGUAC", "pairs": [[0, 5]]},
            {"id": "b", "seq": "ACGU", "pairs": []}]
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("".join(json.dumps(x) + "\n" for x in data))
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps({"per_sequence": [
        {"name": "", "length": 6, "n_gt": 1, "f1": 1.0, "pred_pairs": [[0, 5]]},
        {"name": "", "length": 4, "n_gt": 0, "f1": 1.0, "pred_pairs": []},
    ]}))
    spec = {"prediction_id": "blank/test", "model": "blank", "split": "test", "repeat": None, "path": str(pred_path)}
    result = evaluate_join(spec, str(data_path))
    assert result["summary"]["n_joined_names"] == 2
    assert result["summary"]["coverage_of_truth"] == 1
    assert result["summary"]["join"]["method"] == "positional_verified_by_length_n_gt_and_f1"
    assert [r["name"] for r in result["rows"]] == ["a", "b"]


def test_blank_names_reject_unverified_alignment(tmp_path: Path) -> None:
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(json.dumps({"id": "a", "seq": "ACGUAC", "pairs": [[0, 5]]}) + "\n")
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps({"per_sequence": [
        {"name": "", "length": 6, "n_gt": 1, "f1": 0.0, "pred_pairs": [[0, 5]]},
    ]}))
    spec = {"prediction_id": "blank/test", "model": "blank", "split": "test", "repeat": None, "path": str(pred_path)}
    with pytest.raises(ValueError, match="F1 mismatch"):
        evaluate_join(spec, str(data_path))


def test_pair_span_confusion_reconciles_pair_counts():
    truth = [(0, 4), (0, 14), (0, 29)]
    pred = [(0, 4), (0, 24), (2, 3)]
    counts = pair_span_confusion(pred, truth)
    assert counts["0_3"] == {"tp": 0, "fp": 1, "fn": 0}
    assert counts["4_9"] == {"tp": 1, "fp": 0, "fn": 0}
    assert counts["10_24"] == {"tp": 0, "fp": 1, "fn": 1}
    assert counts["25_49"] == {"tp": 0, "fp": 0, "fn": 1}
    expected = {"tp": 1, "fp": 2, "fn": 2}
    for field in ("tp", "fp", "fn"):
        assert sum(item[field] for item in counts.values()) == expected[field]


def test_preregistered_case_selection_is_deterministic_and_capped():
    candidates = []
    for split in ("a", "b"):
        for index in range(5):
            candidates.append({
                "split": split, "model": "m", "name": f"n{index}", "repeat": None,
                "metrics": {"stem": {"f1": 0.9}, "junction": {"f1": index / 10}, "topology": {"ged": index}},
            })
    candidates.append({"split": "a", "model": "m", "name": "repeat", "repeat": 1,
                       "metrics": {"stem": {"f1": 0.9}, "junction": {"f1": 0.0}, "topology": {"ged": 9}}})
    selected, rule = select_preregistered_helix_junction_cases(candidates)
    assert len(selected) == 6
    assert all(item["repeat"] in (None, 0) for item in selected)
    assert [item["name"] for item in selected[:2]] == ["n0", "n0"]
    assert rule["allocation"] == "round-robin by sorted split; at most 3 per split; at most 30 total"
