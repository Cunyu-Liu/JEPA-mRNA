from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from foldnar_phase0 import CONTRACT_COUNTS, data_specs, dataset_contract_metadata


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
