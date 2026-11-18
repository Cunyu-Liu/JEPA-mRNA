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
    def fake_prediction(model, split, path, repeat=None, variant=None):
        return {"prediction_id": f"{model}/{split}" + (f"/repeat{repeat}" if repeat is not None else ""),
                "model": model, "split": split, "repeat": repeat, "variant": variant, "status": "BLOCKED"}
    monkeypatch.setattr(phase0, "prediction_row", fake_prediction)
    def fake_summary(model, split, path, repeat, variant):
        return {"prediction_id": f"{model}/{split}/repeat{repeat}", "model": model,
                "split": split, "repeat": repeat, "variant": variant, "status": "SUMMARY_ONLY"}
    monkeypatch.setattr(phase0, "summary_prediction_row", fake_summary)
    data = build_data_registry(cache_path=tmp_path / "missing-cache")
    by_id = {row["dataset_id"]: row for row in data}
    assert len(data) == 31
    for name in ("pdb_d1_dssr", "temporal_pdb_candidates_1805", "longchain_7224", "longchain_19000"):
        assert by_id[name]["status"] == "BLOCKED"
    assert by_id["temporal_pdb_candidates_1805"]["contract_expected_n"] == 1805
    assert by_id["longchain_7224"]["contract_expected_length"] == 7224
    predictions = build_prediction_registry()
    assert len(predictions) == 276
    assert sum(row["status"] == "SUMMARY_ONLY" for row in predictions) == 96
    blocked = {(row["model"], row["split"]) for row in predictions if row["status"] == "BLOCKED"}
    for model in ("best_plan_a", "h0"):
        assert (model, "bprna_new") in blocked
        assert (model, "archiveii_embok_clean") in blocked
    assert sum(row["model"] == "bpfold" for row in predictions) == 9
    best = next(row for row in predictions if row["prediction_id"] == "best_plan_a/bprna_new")
    assert best["selection_evidence"].endswith("xens2_plana_distill_s0_vl0/qa_summary.json")
    baselines = build_baseline_registry(predictions)
    assert len(baselines) == 63
    assert {row["model"] for row in baselines} >= {"best Plan-A", "H0", "BPfold"}
    assert {row["baseline_id"] for row in baselines} >= {"efold_0_1_2", "efold_0_1_3", "rinalmo_giga_unet"}


def test_baseline_cohorts_and_required_provenance_fields() -> None:
    baselines = build_baseline_registry()
    metadata = phase0.baseline_metadata(baselines)
    assert metadata["cohort_sizes"] == {"HISTORICAL_25": 25, "MAIN_BOARD_13": 13, "CHANRG_29": 29}
    assert metadata["source_membership_total"] == 67
    assert metadata["core_union_count"] == 54
    assert metadata["supplemental_row_count"] == 9
    assert len(metadata["legacy_18_disposition"]) == 18
    assert {"vienna", "linearfold", "nucleicbert_frozen"}.isdisjoint(row["baseline_id"] for row in baselines)
    for row in baselines:
        assert {"official_weights", "training_data", "license", "inference", "comparability"} <= set(row)
        assert row["unknown_fields_are_not_zero"] is True


def test_summary_and_nested_variant_prediction_schemas(tmp_path) -> None:
    import json
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"vienna_version": "ViennaRNA 2.7.2", "baselines": {
        "vienna_mfe": {"n_sequences": 2, "micro_f1": 0.5}}}))
    summary = phase0.summary_prediction_row("vienna_mfe", "toy", summary_path, 0, "vienna_mfe")
    assert summary["status"] == "SUMMARY_ONLY"
    assert summary["n_sequences"] == 2
    nested_path = tmp_path / "nested.json"
    nested_path.write_text(json.dumps({"n_sequences": 2, "results": {
        "fold": {"binary": "Fold", "per_sequence": [{"name": "a"}, {"name": "b"}]}}}))
    nested = phase0.prediction_row("rnastructure_fold", "toy", nested_path, variant="fold")
    assert nested["status"] == "FROZEN"
    assert nested["n_sequences"] == 2
    assert nested["model_metadata"]["binary"] == "Fold"


def test_bidirectional_registry_links() -> None:
    data = [{"dataset_id": name} for name in (
        "bprna_new", "bprna_tr1c", "bprna_vl0", "chanrg_train_train", "chanrg_train_validation")]
    predictions = [{"prediction_id": "xens2/bprna_new", "model": "xens2", "split": "bprna_new", "status": "FROZEN"}]
    baselines = build_baseline_registry(predictions)
    errors = phase0.link_registries(data, predictions, baselines)
    assert errors == []
    assert predictions[0]["baseline_id"] == "xens2"
    assert data[0]["prediction_ids"] == ["xens2/bprna_new"]
    assert "xens2" in next(row for row in data if row["dataset_id"] == "bprna_tr1c")["training_baseline_ids"]
    assert "rinalmo_giga_unet" in next(row for row in data if row["dataset_id"] == "chanrg_train_train")["training_baseline_ids"]


def test_final_decision_schema_compatibility() -> None:
    from foldnar_verify_phase0 import final_e1_allowed
    assert final_e1_allowed({"final_decision": {"e1_allowed": False}}) is False
    assert final_e1_allowed({"e1_allowed": True}) is True


def test_existing_cluster_audit_is_registered_not_relabelled_clean() -> None:
    row = dataset_contract_metadata("bprna_vl0", "val")
    assert row["cluster_80_status"].startswith("AVAILABLE_IN_V2_AUDIT")
    assert row["cluster_80_evidence_path"].endswith("data_exposure_audit.json")
    assert "no unannounced minimum-loop deletion" in row["min_loop_policy"]
