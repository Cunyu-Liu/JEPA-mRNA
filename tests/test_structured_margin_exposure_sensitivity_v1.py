from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/structured_margin_exposure_sensitivity_v1.py"
SPEC = importlib.util.spec_from_file_location("structured_margin_exposure_sensitivity_v1", MODULE_PATH)
E = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = E
SPEC.loader.exec_module(E)
I = E.independent


def exposure_records(n_per_state=30):
    truth = ((10, 26), (100, 132), (200, 264), (400, 528), (600, 602))
    return [
        I.Record(
            f"s{index}", f"RF{index % 10}", index, 700,
            hashlib.sha256(f"sequence-{index}".encode()).hexdigest(), truth,
        )
        for index in range(3 * n_per_state)
    ]


def assignments_for(records):
    return {record.sequence_id: E.STATES[index // (len(records) // 3)]
            for index, record in enumerate(records)}


def arms_for(records):
    drop = {"matched_control": 0, "destroy_interval_shuffle": 1, "independent_context": 2}
    return {
        arm: [replace(record, predicted_pairs=(
            record.true_pairs if arm == "ensemble_context"
            else record.true_pairs[:drop[arm]] + record.true_pairs[drop[arm] + 1:]
        )) for record in records]
        for arm in I.ARMS
    }


def exposure_fixture(tmp_path, records, omit_state=None, corrupt_sequence=False, official_state="UNCHECKED"):
    root = tmp_path / "exposure"
    root.mkdir()
    assignments = assignments_for(records)
    rows = []
    for index, record in enumerate(records):
        state = assignments[record.sequence_id]
        if state == omit_state:
            continue
        rows.append({
            "query_id": record.sequence_id, "population_id": "tr1c_cluster80_dev",
            "dataset_id": "bprna_tr1c", "role": "development", "selection_use": True,
            "sequence_sha256": "0" * 64 if corrupt_sequence and index == 0 else record.sequence_sha256,
            "normalized_length": record.length, "known_source_exact_or_homology_state": state,
            "known_source_exact_state": "EXACT" if state == "EXACT" else "NOT_DETECTED_UNKNOWN",
            "complete_official_pretraining_exposure_state": "EXACT" if state == "EXACT" else "UNCHECKED",
            "source_matches": [{"source": "fixture"}] if state == "EXACT" else [],
            "homology_query_id": f"Q{index:06d}",
        })
    with gzip.open(root / "sequence_registry.jsonl.gz", "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    counts = Counter(row["known_source_exact_or_homology_state"] for row in rows)
    dev_counts = {state: counts.get(state, 0) for state in E.STATES}
    summary = {
        "schema_version": "rinalmo.pretrain_exposure.summary.v1",
        "complete_official_pretraining_coverage_state": official_state,
        "populations": {
            "tr1c_cluster80_dev": {
                "n_input_rows": len(rows),
                "known_source_exact_or_homology_state": {
                    **{state: 0 for state in (*E.STATES, "UNCHECKED")}, **dev_counts,
                },
            },
            "bprna_new": {
                "n_input_rows": 5388,
                "known_source_exact_state": {"EXACT": 5381},
                "known_source_exact_or_homology_state": {"HOMOLOGY": 7},
            },
        },
    }
    assets = {
        "schema_version": "rinalmo.pretrain_exposure.assets.v1",
        "official_pretraining_declaration": {"complete_official_training_set_state": official_state},
    }
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "asset_registry.json").write_text(json.dumps(assets))
    verification = {
        "schema_version": "rinalmo.pretrain_exposure.independent_verification.v1",
        "status": "PASS", "no_model_inference_or_training": True,
        "official_complete_coverage": official_state,
        "artifact_sha256": {name: I.sha256(root / name) for name in (
            "sequence_registry.jsonl.gz", "summary.json", "asset_registry.json",
        )},
    }
    (root / "independent_verification.json").write_text(json.dumps(verification))
    contract = copy.deepcopy(E.EXPOSURE_CONTRACT)
    contract["source_commit"] = "synthetic-fixture"
    contract["dev_counts"] = dev_counts
    contract["artifact_sha256"] = {name: I.sha256(root / name) for name in (
        "sequence_registry.jsonl.gz", "summary.json", "asset_registry.json", "independent_verification.json",
    )}
    return root, contract, assignments


def test_pinned_registry_binds_real_cluster_dev_without_gpu():
    frozen, _, _, _ = I.load_frozen_panel(
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[1] / "configs/gec_structured_margin_convergence_v1.json",
    )
    assignments, provenance = E.bind_exposure_registry(frozen)
    assert len(assignments) == 9143
    assert provenance["source_commit"] == "4247d1e4a971ceb8c63e3f8338e026a23c18941b"
    assert provenance["cluster_dev_counts"] == {"EXACT": 7978, "HOMOLOGY": 539, "NOT_DETECTED_UNKNOWN": 626}
    assert provenance["bprna_new_anchor"] == {"rows": 5388, "exact": 5381, "homology": 7}
    assert provenance["complete_official_pretraining_coverage_state"] == "UNCHECKED"
    assert not provenance["not_detected_unknown_is_clean"]


def test_synthetic_registry_binds_every_sequence(tmp_path):
    records = exposure_records()
    root, contract, expected = exposure_fixture(tmp_path, records)
    assignments, provenance = E.bind_exposure_registry(records, root, contract)
    assert assignments == expected
    assert provenance["assignment_sha256"] == I.canonical_hash(expected)


@pytest.mark.parametrize("name", [
    "sequence_registry.jsonl.gz", "summary.json", "asset_registry.json", "independent_verification.json",
])
def test_missing_registry_layer_fails_closed(tmp_path, name):
    records = exposure_records()
    root, contract, _ = exposure_fixture(tmp_path, records)
    (root / name).unlink()
    with pytest.raises(OSError):
        E.bind_exposure_registry(records, root, contract)


@pytest.mark.parametrize("name", [
    "sequence_registry.jsonl.gz", "summary.json", "asset_registry.json", "independent_verification.json",
])
def test_wrong_registry_layer_hash_fails_closed(tmp_path, name):
    records = exposure_records()
    root, contract, _ = exposure_fixture(tmp_path, records)
    with (root / name).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        E.bind_exposure_registry(records, root, contract)


def test_missing_exposure_stratum_fails_closed(tmp_path):
    records = exposure_records()
    root, contract, _ = exposure_fixture(tmp_path, records, omit_state="HOMOLOGY")
    with pytest.raises(ValueError, match="exactly cover"):
        E.bind_exposure_registry(records, root, contract)


def test_wrong_sequence_hash_fails_closed(tmp_path):
    records = exposure_records()
    root, contract, _ = exposure_fixture(tmp_path, records, corrupt_sequence=True)
    with pytest.raises(ValueError, match="sequence identity mismatch"):
        E.bind_exposure_registry(records, root, contract)


def test_checked_official_corpus_claim_fails_closed(tmp_path):
    records = exposure_records()
    root, contract, _ = exposure_fixture(tmp_path, records, official_state="CHECKED")
    with pytest.raises(ValueError, match="must remain UNCHECKED"):
        E.bind_exposure_registry(records, root, contract)


def test_full_synthetic_sensitivity_has_all_metrics_ci_and_concentration():
    records = exposure_records()
    assignments = assignments_for(records)
    result = E.evaluate_exposure_sensitivity(records, arms_for(records), assignments, {
        "source_commit": "synthetic-fixture", "complete_official_pretraining_coverage_state": "UNCHECKED",
    }, resamples=20)
    assert result["role"] == "PREREGISTERED_SENSITIVITY_ONLY"
    assert result["gate_use"] == result["checkpoint_selection_use"] == "FORBIDDEN"
    for state in E.STATES:
        stratum = result["strata"][state]
        assert set(stratum["per_arm"]) == set(I.ARMS)
        for kind in ("pair", "stem"):
            assert set(stratum["per_arm"]["ensemble_context"][kind]) == {
                "tp", "fp", "fn", "precision", "recall", "f1",
            }
            comparison = stratum["ensemble_vs_controls"]["matched_control"][kind]["recall"]
            assert comparison["delta_ensemble_minus_control"] > 0
            assert len(comparison["paired_ci95"]) == 2
            assert len(comparison["family_cluster_sensitivity_ci95"]) == 2
        for target in I.TARGETS:
            assert stratum["support"]["targets"][target]["support_pass"]
            assert "recall" in stratum["per_arm"]["ensemble_context"]["targets"][target]
            assert "paired_ci95" in stratum["ensemble_vs_controls"]["matched_control"]["targets"][target]["recall"]
            assert "known_family_hhi" in stratum["family_concentration"]["targets"][target]


def test_small_exposure_stratum_fails_closed():
    records = exposure_records()
    assignments = assignments_for(records)
    del assignments[records[29].sequence_id]
    records = records[:29] + records[30:]
    with pytest.raises(ValueError, match="insufficient support for EXACT/pair"):
        E.evaluate_exposure_sensitivity(records, arms_for(records), assignments, {}, resamples=20)


def test_small_target_support_fails_closed():
    records = exposure_records()
    records = [record if index == 0 else replace(
        record, true_pairs=record.true_pairs + tuple((i + 1, j - 1) for i, j in record.true_pairs),
    ) for index, record in enumerate(records)]
    with pytest.raises(ValueError, match="isolated_pair:isolated"):
        E.evaluate_exposure_sensitivity(
            records, arms_for(records), assignments_for(records), {}, resamples=20,
        )


def test_missing_arm_and_unpaired_rows_fail_closed():
    records = exposure_records()
    assignments = assignments_for(records)
    arms = arms_for(records)
    del arms["matched_control"]
    with pytest.raises(ValueError, match="four-arm coverage"):
        E.evaluate_exposure_sensitivity(records, arms, assignments, {}, resamples=20)
    arms = arms_for(records)
    arms["ensemble_context"][0], arms["ensemble_context"][1] = arms["ensemble_context"][1], arms["ensemble_context"][0]
    with pytest.raises(ValueError, match="unpaired prediction ordering"):
        E.evaluate_exposure_sensitivity(records, arms, assignments, {}, resamples=20)


def test_small_bootstrap_and_unmapped_family_concentration_fail_or_null_cleanly():
    records = exposure_records()
    with pytest.raises(ValueError, match="bootstrap resamples"):
        E.evaluate_exposure_sensitivity(records, arms_for(records), assignments_for(records), {}, resamples=10)
    unmapped = [replace(record, family_id=None) for record in records[:2]]
    result = E.family_concentration(unmapped, [record.true_pairs for record in unmapped])
    assert result["status"] == "NO_KNOWN_FAMILY_MAPPING"
    assert result["known_family_hhi"] is None
    assert result["top_known_family_fraction_of_all"] is None
    assert result["unmapped_or_ambiguous_family_fraction"] == 1.0
