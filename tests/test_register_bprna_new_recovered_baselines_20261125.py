import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools/register_bprna_new_recovered_baselines_20261125.py"
SPEC = importlib.util.spec_from_file_location("baseline_registry_update", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_pair_metrics_and_empty_convention():
    labels = [{"name": "a", "seq": "AAAAAA", "pairs": [[0, 5]]}, {"name": "b", "seq": "AAAA", "pairs": []}]
    predictions = [{"name": "a", "pairs": [[0, 5], [1, 4]]}, {"name": "b", "pairs": []}]
    result = MODULE.score(labels, predictions)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["micro_f1"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(5 / 6)


def test_missing_id_is_not_dropped():
    with pytest.raises(AssertionError):
        MODULE.score([{"name": "a", "seq": "AAAA", "pairs": []}], [])


def test_partner_conflict_is_rejected():
    with pytest.raises(AssertionError):
        MODULE.pair_set([[0, 3], [0, 2]], 4)


def test_twenty_assets_have_distinct_protocol_identities():
    assert len(MODULE.ASSETS) == 20
    assert len({row["asset_id"] for row in MODULE.ASSETS}) == 20
    assert len({row["prediction_id"] for row in MODULE.ASSETS}) == 20
    by_id = {row["asset_id"]: row for row in MODULE.ASSETS}
    assert by_id["bpfold_project_six_checkpoint_raw"]["model_form"] == "six_checkpoint_ensemble"
    assert by_id["bpfold_author_release_official_reference"]["label_protocol"] == "BPFOLD_OFFICIAL_REFERENCE_5401"
    assert by_id["spot_rna_registry_raw_single"]["prediction_id"] != by_id["spot_rna_postprocessed_single_raw"]["prediction_id"]
    assert "full fine-tune" in by_id["nucleicbert_author_secstr_full_ft_single_raw"]["training_identity"]
    assert "encoder frozen" in by_id["nucleicbert_project_frozen_encoder_tr1c_single_raw"]["training_identity"]
    assert by_id["contrafold_mea_single_clean_tr1c"]["raw_clean_scope"] == "clean_tr1c"


def test_committed_registry_summary_does_not_mix_clean_or_official_labels():
    path = Path(__file__).parents[1] / "manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json"
    document = json.loads(path.read_text())
    assert document["n_assets"] == 20
    assert document["strongest_project_raw"]["asset_id"] == "bpfold_project_six_checkpoint_raw"
    assert document["strongest_project_raw_single_predictor"]["asset_id"] == "contrafold_mea_single_raw"
    assert document["scope_contract"]["raw_is_not_clean"] is True
    assert document["scope_contract"]["chanrg_empty_name_records_processed"] is False
