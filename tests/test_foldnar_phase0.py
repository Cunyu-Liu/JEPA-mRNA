from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import foldnar_phase0 as phase0
from foldnar_phase0 import (
    CONTRACT_COUNTS, build_baseline_registry, build_data_registry,
    build_prediction_registry, data_specs, dataset_contract_metadata,
)


def test_scientific_scope_roles() -> None:
    specs = {name: role for name, role, description in data_specs()}
    assert specs["ref_syn_train"] == "synthetic_algorithm"
    assert specs["ncrna_rfamnovel_train"] == "out_of_scope_ncrna"
    assert specs["ncrna_rfamnovel_val"] == "out_of_scope_ncrna"
    assert specs["ncrna_rfamnovel_test"] == "out_of_scope_ncrna"
    assert specs["chanrg_test"] == "test"
    assert specs["chanrg_train_validation"] == "val"


def test_dataset_contract_provenance_and_expected_counts() -> None:
    synthetic = dataset_contract_metadata("ref_syn_train", "synthetic_algorithm")
    assert synthetic["label_source"] == "SIMULATED"
    assert synthetic["scope"] == "SYNTHETIC_ALGORITHM"
    excluded = dataset_contract_metadata("ncrna_rfamnovel_test", "out_of_scope_ncrna")
    assert excluded["scope"] == "OUT_OF_SCOPE_NCRNA_CLASSIFICATION"
    assert CONTRACT_COUNTS["chanrg_test"] == 14070
    assert dataset_contract_metadata("chanrg_test", "test")["contract_expected_n"] == 14070
    assert dataset_contract_metadata("ref_pdb_ts1", "test")["label_source"] == "EXPERIMENTAL"
    assert dataset_contract_metadata("chanrg_gena", "test")["label_source"] == "COMPARATIVE_INFERRED"


def test_missing_contract_assets_and_systems_are_explicitly_blocked(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(phase0, "JSONL", tmp_path)
    monkeypatch.setattr(phase0, "file_info", lambda path, content_hash=True: {"path": str(path), "exists": False})
    def fake_prediction(model, split, path, repeat=None):
        return {"prediction_id": f"{model}/{split}" + (f"/repeat{repeat}" if repeat is not None else ""),
                "model": model, "split": split, "repeat": repeat, "status": "BLOCKED"}
    monkeypatch.setattr(phase0, "prediction_row", fake_prediction)
    data = build_data_registry(cache_path=tmp_path / "missing-cache")
    by_id = {row["dataset_id"]: row for row in data}
    assert len(data) == 31
    for name in ("pdb_d1_dssr", "temporal_pdb_candidates_1805", "longchain_7224", "longchain_19000"):
        assert by_id[name]["status"] == "BLOCKED"
    assert by_id["temporal_pdb_candidates_1805"]["contract_expected_n"] == 1805
    assert by_id["longchain_7224"]["contract_expected_length"] == 7224
    predictions = build_prediction_registry()
    assert len(predictions) == 89
    blocked = {(row["model"], row["split"]) for row in predictions if row["status"] == "BLOCKED"}
    for model in ("best_plan_a", "h0"):
        assert (model, "bprna_new") in blocked
        assert (model, "archiveii_embok_clean") in blocked
    assert sum(row["model"] == "bpfold" for row in predictions) == 9
    best = next(row for row in predictions if row["prediction_id"] == "best_plan_a/bprna_new")
    assert best["selection_evidence"].endswith("xens2_plana_distill_s0_vl0/qa_summary.json")
    baselines = build_baseline_registry()
    assert len(baselines) == 18
    assert {row["model"] for row in baselines} >= {"best_plan_a", "h0", "bpfold"}


def test_existing_cluster_audit_is_registered_not_relabelled_clean() -> None:
    row = dataset_contract_metadata("bprna_vl0", "val")
    assert row["cluster_80_status"].startswith("AVAILABLE_IN_V2_AUDIT")
    assert row["cluster_80_evidence_path"].endswith("data_exposure_audit.json")
    assert "no unannounced minimum-loop deletion" in row["min_loop_policy"]
