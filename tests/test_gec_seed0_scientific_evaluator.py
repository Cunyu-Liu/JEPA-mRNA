from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/gec_seed0_scientific_evaluator.py"
SPEC = importlib.util.spec_from_file_location("gec_seed0_scientific_evaluator", MODULE_PATH)
G = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = G
SPEC.loader.exec_module(G)


def _passing_evaluation():
    comparisons = {}
    for control in ("matched_control", "destroy_interval_shuffle", "independent_context"):
        comparisons[control] = {
            "metrics": {
                metric: {"delta": 0.02}
                for metric in G.GLOBAL_METRICS
            }
        }
    return {
        "ci_contract_pass": True,
        "comparisons": comparisons,
        "targets": {"pair_span:ge128": {"target_gate_pass": True}},
        "auxiliary_raw_ranking": {"ensemble_context": {"auprc": 1.0}},
    }


@pytest.mark.parametrize("gate_name", [
    "paired_ci_contract",
    "micro_pair_f1_nonnegative",
    "macro_pair_f1_strictly_positive",
    "micro_precision_loss_at_most_0.005",
    "macro_precision_loss_at_most_0.005",
    "micro_stem_f1_loss_at_most_0.005",
    "macro_stem_f1_loss_at_most_0.005",
    "strictly_improved_diverse_target",
] + [
    f"{control}:{metric}_strict_advantage"
    for control in ("destroy_interval_shuffle", "independent_context")
    for metric in ("micro_pair_f1", "macro_pair_f1", "micro_stem_f1", "macro_stem_f1")
])
def test_every_atomic_gate_can_fail(gate_name):
    evaluation = _passing_evaluation()
    assert G.verify_gates(evaluation)["pass"]
    if gate_name == "paired_ci_contract":
        evaluation["ci_contract_pass"] = False
    elif gate_name == "strictly_improved_diverse_target":
        evaluation["targets"]["pair_span:ge128"]["target_gate_pass"] = False
    elif gate_name == "micro_pair_f1_nonnegative":
        evaluation["comparisons"]["matched_control"]["metrics"]["micro_pair_f1"]["delta"] = -1e-12
    elif gate_name == "macro_pair_f1_strictly_positive":
        evaluation["comparisons"]["matched_control"]["metrics"]["macro_pair_f1"]["delta"] = 0.0
    elif "loss_at_most" in gate_name:
        metric = {
            "micro_precision_loss_at_most_0.005": "micro_pair_precision",
            "macro_precision_loss_at_most_0.005": "macro_pair_precision",
            "micro_stem_f1_loss_at_most_0.005": "micro_stem_f1",
            "macro_stem_f1_loss_at_most_0.005": "macro_stem_f1",
        }[gate_name]
        evaluation["comparisons"]["matched_control"]["metrics"][metric]["delta"] = -0.0050001
    else:
        control, metric = gate_name.split(":", 1)
        metric = metric.removesuffix("_strict_advantage")
        evaluation["comparisons"][control]["metrics"][metric]["delta"] = 0.0
    verdict = G.verify_gates(evaluation)
    assert not verdict["pass"]
    assert verdict["failed_gates"] == [gate_name]


def test_loss_threshold_boundaries_and_raw_scores_cannot_rescue():
    evaluation = _passing_evaluation()
    for metric in ("micro_pair_precision", "macro_pair_precision", "micro_stem_f1", "macro_stem_f1"):
        evaluation["comparisons"]["matched_control"]["metrics"][metric]["delta"] = -0.005
    evaluation["comparisons"]["matched_control"]["metrics"]["micro_pair_f1"]["delta"] = 0.0
    assert G.verify_gates(evaluation)["pass"]
    evaluation["comparisons"]["matched_control"]["metrics"]["micro_pair_f1"]["delta"] = -1e-8
    evaluation["auxiliary_raw_ranking"] = {"auprc_delta": 100.0, "recall_at_p0.8_delta": 100.0}
    assert not G.verify_gates(evaluation)["pass"]


def test_hand_checked_pair_stem_and_isolated_metrics():
    truth = G.pairs([(0, 11), (1, 10), (4, 15)])
    predicted = G.pairs([(0, 11), (2, 9), (4, 15)])
    assert G.confusion(truth, predicted) == (2, 1, 1)
    assert G.prf((2, 1, 1)) == pytest.approx((2 / 3, 2 / 3, 2 / 3))
    true_stems = G.stems(truth)
    predicted_stems = G.stems(predicted)
    assert len(true_stems) == 2 and len(predicted_stems) == 3
    assert G.confusion(true_stems, predicted_stems) == (1, 2, 1)
    assert G.prf((1, 2, 1)) == pytest.approx((1 / 3, 1 / 2, 0.4))
    true_isolated = G.environment_pairs(truth, "isolated_pair", "isolated")
    pred_isolated = G.environment_pairs(predicted, "isolated_pair", "isolated")
    assert true_isolated == {(4, 15)}
    assert pred_isolated == set(predicted)
    assert G.confusion(truth, predicted, true_isolated, pred_isolated) == (1, 2, 0)
    assert G.prf(G.confusion(truth, predicted, true_isolated, pred_isolated)) == pytest.approx((1 / 3, 1, 0.5))


def test_isolated_true_pair_becoming_stacked_is_not_an_environment_tp():
    truth = G.pairs([(4, 15)])
    predicted = G.pairs([(4, 15), (5, 14)])
    selected_truth = G.environment_pairs(truth, "isolated_pair", "isolated")
    selected_pred = G.environment_pairs(predicted, "isolated_pair", "isolated")
    assert selected_truth == {(4, 15)} and selected_pred == set()
    counts = G.confusion(truth, predicted, selected_truth, selected_pred)
    assert counts == (0, 0, 1)
    assert G.prf(counts) == (0.0, 0.0, 0.0)


def test_empty_convention_and_span_boundaries():
    assert G.prf((0, 0, 0)) == (0.0, 0.0, 0.0)
    assert G.sequence_prf(np.asarray([[0, 0, 0]])).tolist() == [[1.0, 1.0, 1.0]]
    assert [G.span_bin((0, span)) for span in (3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128)] == ["le3", "4_7", "4_7", "8_15", "8_15", "16_31", "16_31", "32_63", "32_63", "64_127", "64_127", "ge128"]


def test_competition_uses_fixed_crossing_or_endpoint_reference():
    truth = G.pairs([(0, 10), (20, 30)])
    reference = G.pairs([(1, 11), (20, 29)])
    selected = G.environment_pairs(truth, "competition", "reference_conflicts", reference)
    assert selected == set(truth)
    nested_reference = G.pairs([(1, 9)])
    selected = G.environment_pairs(truth, "competition", "reference_conflicts", nested_reference)
    assert selected == set()


def _records_for_macro():
    truth = G.pairs([(4 * k, 4 * k + 3) for k in range(17)])
    return [G.Record(f"s{index}", "RF1", truth) for index in range(30)]


def test_macro_bootstrap_matches_independent_loop_recomputation():
    records = _records_for_macro()
    boot = G.BootstrapSpace(records, resamples=8, seed=20261118)
    values = np.asarray([float(index % 2) for index in range(30)])
    observed, sampled = boot.macro(values)
    assert observed == 0.5
    expected_first = values[boot.indices[0]].mean()
    assert sampled[0] == pytest.approx(expected_first)
    identical, _ = boot.macro(np.ones(30))
    assert identical == 1.0


def test_small_stratum_cannot_select_route():
    records = [G.Record("only", "RF1", G.pairs([(0, 20)]))]
    counts = np.asarray([[1.0, 0.0, 0.0]])
    summary = G.concentration(records, counts, {
        "min_sequences": 100, "min_pairs": 1000, "min_known_families": 10,
        "max_family_hhi": 0.25, "max_top_family_fraction": 0.5,
        "max_unmapped_family_fraction": 0.5,
    })
    assert not summary["route_eligible"]
    assert {"min_sequences", "min_pairs", "min_known_families", "top_family_fraction", "family_hhi"} <= set(summary["route_failure_reasons"])
    evaluation = _passing_evaluation()
    evaluation["targets"] = {"pair_span:16_31": {"target_gate_pass": summary["route_eligible"] and True}}
    assert not G.verify_gates(evaluation)["pass"]


def test_single_family_gain_fails_leave_one_family_out():
    records = [
        G.Record("a", "RFA", G.pairs([(0, 20)])),
        G.Record("b", "RFB", G.pairs([(0, 20)])),
    ]
    ensemble = np.asarray([[10, 0, 0], [4, 0, 6]], dtype=float)
    control = np.asarray([[0, 0, 10], [10, 0, 0]], dtype=float)
    result = G.diversity(records, ensemble, control)
    assert result["recall_delta"] > 0
    assert result["min_leave_one_family_out_recall_delta"] < 0
    assert not result["non_single_family"]


def test_raw_auprc_recall_at_p08_hand_checked_and_auxiliary():
    truth = G.pairs([(0, 5), (1, 6)])
    scores = [(0, 5, 0.9), (2, 7, 0.8), (1, 6, 0.7)]
    records = [G.Record("s", "RF", truth, truth, scores)]
    result = G.raw_auxiliary({arm: records for arm in G.ARMS})
    for arm in G.ARMS:
        assert result["arms"][arm]["auprc"] == pytest.approx(5 / 6)
        assert result["arms"][arm]["recall_at_precision_0.8"] == 0.5
    assert result["role"] == "AUXILIARY_ONLY_NEVER_GATE"


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _identity(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _scientific_fixture(tmp_path):
    truth = [[4 * k, 4 * k + 3] for k in range(17)]
    sequence = "A" * 68
    sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
    raw_path = tmp_path / "raw_tr1c.jsonl"
    panel_records = tmp_path / "panel.jsonl"
    dev_ids = tmp_path / "dev.txt"
    train_ids = tmp_path / "train.txt"
    environment = tmp_path / "environment.json"
    contract = tmp_path / "contract.json"
    shared_init = tmp_path / "shared_init.pt"
    shared_init.write_bytes(b"shared-init")
    ids = [f"bprna_tr1c:row{index:06d}:s{index}" for index in range(31)]
    raw_rows = []
    panel_rows = []
    for index, sequence_id in enumerate(ids):
        raw_rows.append({
            "name": f"s{index}", "seq": sequence, "pairs": truth,
            "source": f"synthetic:s{index}",
        })
        panel_rows.append({
            "sequence_id": sequence_id, "source_row_index": index, "name": f"s{index}",
            "family_id": "RF1" if index < 30 else "RFT", "cluster_id": index,
            "cluster_assignment": "dev" if index < 30 else "train",
            "n_pairs": 17, "length": 68, "sequence_sha256": sequence_sha256,
            "pair_span_counts": {"le3": 17}, "isolated_pair_counts": {"isolated": 17},
        })
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    panel_records.write_text("".join(json.dumps(row) + "\n" for row in panel_rows))
    dev_ids.write_text("".join(f"{sequence_id}\n" for sequence_id in ids[:30]))
    train_ids.write_text(f"{ids[30]}\n")
    _write_json(environment, {"panels": {}})
    _write_json(contract, {"contract_id": "gec_four_arm_convergence_contract_v1"})
    registry = tmp_path / "registry.json"
    _write_json(registry, {
        "schema_version": "foldnar.model_dev_registry.v1", "frozen_utc": "synthetic",
        "assets": [{
            "dataset_id": "bprna_tr1c", "role": "legal_train_dev_source",
            "source": str(raw_path), "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        }],
        "artifacts": {"environment_summary": _identity(environment)},
        "environment_contracts": {
            "panel_records": _identity(panel_records),
            "pair_level": {"isolated_pair": "neither inward nor outward partner"},
        },
        "panels": [{
            "panel_id": "tr1c_cluster80_dev_v1", "split_unit": "existing 80% cluster",
            "dev_ids": _identity(dev_ids), "train_ids": _identity(train_ids),
        }],
    })
    config = tmp_path / "gec_config.json"
    _write_json(config, {
        "config_id": "gec_four_arm_implementation_v1", "arms": list(G.ARMS),
        "source_contract": str(contract),
        "assets": {
            "data": str(raw_path), "data_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "registry": str(registry), "train_ids": str(train_ids), "validation_ids": str(dev_ids),
            "shared_init": str(shared_init),
            "shared_init_sha256": hashlib.sha256(shared_init.read_bytes()).hexdigest(),
        },
        "training": {
            "seed": 0, "batch_size": 4, "head_lr": 1e-4, "backbone_lr": 1e-6,
            "warmup_head_steps": 1600, "unfreeze_every": 800, "unfreeze_per_step": 2,
            "optimizer_warmup_steps": 5, "grad_clip": 1.0,
            "n_blocks": 33, "max_unfrozen_blocks": 20, "terminal_step": 9600,
            "fixed_step_cap": None,
        },
        "convergence": {
            "validation_every": 500, "patience": 5, "min_delta": 1e-6,
            "stop_reason": "CONVERGED_VALIDATION_NLL_AFTER_TERMINAL_UNFREEZE",
        },
        "model": {"decoder": {
            "name": "MEA", "temperature": 1.0, "gamma": 1.0,
            "score_variant": "train_off/plana_off", "search_allowed": False,
        }},
    })
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    contract_sha256 = hashlib.sha256(contract.read_bytes()).hexdigest()
    prediction_paths = {}
    metadata_paths = {}
    for arm in G.ARMS:
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        best = arm_dir / "best.pt"
        resume = arm_dir / "resume.pt"
        convergence = {
            "terminal_step": 9600, "terminal_unfrozen_blocks": 20, "best_step": 9600,
            "best_validation_loss": 1.0, "bad_evaluations": 5, "eligible": True,
        }
        run_identity = {
            "schema_version": "foldnar.gec_run_identity.v2", "arm": arm,
            "engineering_smoke": False, "gec_config_sha256": config_sha256,
            "source_contract_sha256": contract_sha256,
            "shared_init_sha256": hashlib.sha256(shared_init.read_bytes()).hexdigest(),
            "gec_init_sha256": "common-gec-init",
            "initial_residual_tensor_sha256": "common-residual",
            "train_record_ids": [ids[30]], "validation_record_ids": ids[:30],
            "terminal_unfrozen_blocks": 20,
        }
        encoder = {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        head = {"weight": torch.tensor([3.0], dtype=torch.float64)}
        residual = {"weight": torch.tensor([4.0], dtype=torch.float32)}
        model_hashes = {
            "encoder": G.tensor_state_sha256(encoder),
            "head": G.tensor_state_sha256(head),
            "residual": G.tensor_state_sha256(residual),
        }
        common_checkpoint = {
            "format": "foldnar.gec_checkpoint.v2", "run_identity": run_identity,
            "step": 9600, "encoder": encoder, "head": head, "residual": residual,
            "n_unfrozen": 20, "optimizer_steps": 9600, "convergence": convergence,
            "model_tensor_sha256": model_hashes,
        }
        torch.save({
            **common_checkpoint, "checkpoint_phase": "best_eligible",
            "terminal_status": "running",
            "model": {"encoder": encoder, "head": head, "residual": residual},
        }, best)
        torch.save({
            **common_checkpoint, "checkpoint_phase": "restored_best_eligible",
            "terminal_status": "converged",
        }, resume)
        best_sha256 = hashlib.sha256(best.read_bytes()).hexdigest()
        resume_sha256 = hashlib.sha256(resume.read_bytes()).hexdigest()
        metadata = {
            "schema_version": "foldnar.gec_training_run.v2", "run_identity": run_identity,
            "arm": arm, "status": "converged", "converged": True, "engineering_smoke": False,
            "gec_config": str(config), "gec_config_sha256": config_sha256,
            "source_contract": str(contract), "source_contract_sha256": contract_sha256,
            "shared_init": str(shared_init),
            "shared_init_sha256": hashlib.sha256(shared_init.read_bytes()).hexdigest(),
            "gec_init_sha256": "common-gec-init", "initial_residual_tensor_sha256": "common-residual",
            "residual_parameter_count": 10, "trainable_head_and_residual_parameter_count": 20,
            "train_record_ids": [ids[30]], "validation_record_ids": ids[:30],
            "objective": {"name": "exact_structured_NLL_only"},
            "schedule": {
                "seed": 0, "batch_size": 4, "head_lr": 1e-4, "backbone_lr": 1e-6,
                "warmup_head_steps": 1600, "unfreeze_every": 800, "unfreeze_per_step": 2,
                "max_unfrozen_blocks": 20, "warmup_steps": 5, "grad_clip": 1.0,
                "validation_every": 500, "validation_patience": 5, "validation_min_delta": 1e-6,
                "engineering_smoke": False, "steps": 0, "train_record_limit": 0,
                "validation_record_limit": 0, "device": "cuda:0",
            },
            "convergence": convergence,
            "environment": {"requested_device": "cuda:0", "resolved_device": "cuda:0"},
            "decoder": json.loads(config.read_text())["model"]["decoder"],
            "backbone_report": {"n_layers": 33}, "steps_completed": 15000,
            "stop_reason": "CONVERGED_VALIDATION_NLL_AFTER_TERMINAL_UNFREEZE",
            "divergence_reason": "", "final_unfrozen_blocks": 20,
            "restored_best_step": 9600, "best_checkpoint_sha256": best_sha256,
            "resume_checkpoint_sha256": resume_sha256,
        }
        metadata_path = arm_dir / "run_meta.json"
        _write_json(metadata_path, metadata)
        metadata_paths[arm] = metadata_path
        prediction_path = arm_dir / "predictions.jsonl"
        prediction_path.write_text("".join(json.dumps({
            "sequence_id": ids[index], "true_pairs": truth, "predicted_pairs": truth,
            "sequence_length": 68, "sequence_sha256": sequence_sha256,
            "checkpoint_sha256": best_sha256,
            "raw_scores": [[4 * k, 4 * k + 3, 1.0] for k in range(17)],
        }) + "\n" for index in range(30)))
        prediction_paths[arm] = prediction_path
    return {
        "registry": registry, "panel_records": panel_records, "raw": raw_path,
        "dev_ids": dev_ids, "train_ids": train_ids, "environment": environment,
        "contract": contract, "shared_init": shared_init, "config": config,
        "prediction_paths": prediction_paths, "metadata_paths": metadata_paths,
        "ids": ids, "truth": truth, "sequence_sha256": sequence_sha256,
    }


def _main_args(fixture, output=None, include_panel=True):
    args = ["--registry", str(fixture["registry"]), "--gec-config", str(fixture["config"])]
    if include_panel:
        args += ["--panel-id", "tr1c_cluster80_dev_v1"]
    for arm in G.ARMS:
        args += ["--prediction", f"{arm}=" + str(fixture["prediction_paths"][arm])]
        args += ["--run-metadata", f"{arm}=" + str(fixture["metadata_paths"][arm])]
    if output is not None:
        args += ["--output", str(output)]
    return args


def _validated_fixture(fixture):
    registry, records, panel = G.load_panel(fixture["registry"], "tr1c_cluster80_dev_v1")
    validation = G.validate_run_metadata(
        fixture["registry"], fixture["config"], fixture["metadata_paths"], panel,
    )
    return registry, records, panel, validation


def test_end_to_end_failed_gate_is_reported_without_modifying_inputs(tmp_path):
    fixture = _scientific_fixture(tmp_path)
    paths = fixture["prediction_paths"]
    tracked = [
        fixture[name] for name in
        ("registry", "panel_records", "raw", "dev_ids", "train_ids", "environment",
         "contract", "shared_init", "config")
    ] + list(paths.values()) + list(fixture["metadata_paths"].values())
    tracked += [path.parent / name for path in fixture["metadata_paths"].values() for name in ("best.pt", "resume.pt")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    output = tmp_path / "report.json"
    rc = G.main(_main_args(fixture, output, include_panel=False))
    assert rc == 1
    report = json.loads(output.read_text())
    assert report["status"] == "FAIL_GEC_SEED0"
    assert not report["gates"]["pass"]
    assert "macro_pair_f1_strictly_positive" in report["gates"]["failed_gates"]
    assert report["bootstrap"]["resamples"] == 2000
    assert report["bootstrap"]["family_cluster_sensitivity"]["n_mapped_families"] == 1
    assert report["panel"]["panel_id"] == "tr1c_cluster80_dev_v1"
    assert report["panel"]["n_sequences"] == 30 and report["panel"]["n_pairs"] == 510
    assert report["panel"]["leakage_eligibility"] == {
        "exact_train_id_overlap": False, "train_cluster_overlap": False,
        "planned_train_ids_sha256": _identity(fixture["train_ids"])["sha256"],
    }
    metric = report["comparisons"]["matched_control"]["metrics"]["micro_pair_f1"]
    assert "ci95" in metric and "family_cluster_sensitivity_ci95" in metric
    assert report["auxiliary_raw_ranking"]["role"] == "AUXILIARY_ONLY_NEVER_GATE"
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    output_before = output.read_bytes()
    rc = G.main(_main_args(fixture, output))
    assert rc == 2
    assert output.read_bytes() == output_before


def test_family_cluster_bootstrap_resamples_whole_registered_families():
    records = _records_for_macro() + [G.Record("unmapped", None, _records_for_macro()[0].true_pairs)]
    boot = G.FamilyClusterBootstrapSpace(records, resamples=8, seed=20261118)
    assert boot.n_mapped_families == 1 and boot.n_unmapped_singletons == 1
    for row in range(8):
        expected = np.bincount(boot.indices[row], minlength=len(boot.rows))
        for group_index, rows in enumerate(boot.rows):
            assert np.all(boot.multiplicity[row, rows] == expected[group_index])
    counts = np.asarray([[1.0, 0.0, 0.0] for _ in records])
    sampled = boot.sample_counts(counts)
    assert sampled[0, 0] == boot.multiplicity[0].sum()
    values = np.ones(len(records))
    observed, sensitivity = boot.macro(values, {"RF1"})
    assert observed == 1.0 and np.all(sensitivity == 1.0)


@pytest.mark.parametrize("panel_id", ["tr1c_rfam_family_dev_v1", "legacy_vl0_control", "synthetic_dev"])
def test_non_cluster_panels_are_rejected(tmp_path, panel_id):
    fixture = _scientific_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires the registered cluster-dev panel"):
        G.load_panel(fixture["registry"], panel_id)


def test_exact_train_id_leakage_is_rejected(tmp_path):
    fixture = _scientific_fixture(tmp_path)
    fixture["train_ids"].write_text(fixture["train_ids"].read_text() + fixture["ids"][0] + "\n")
    registry = json.loads(fixture["registry"].read_text())
    registry["panels"][0]["train_ids"] = _identity(fixture["train_ids"])
    _write_json(fixture["registry"], registry)
    with pytest.raises(ValueError, match="panel rows occur in planned train IDs"):
        G.load_panel(fixture["registry"], "tr1c_cluster80_dev_v1")


def test_registered_cluster_leakage_is_rejected(tmp_path):
    fixture = _scientific_fixture(tmp_path)
    rows = [json.loads(line) for line in fixture["panel_records"].read_text().splitlines()]
    rows[-1]["cluster_id"] = rows[0]["cluster_id"]
    fixture["panel_records"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    registry = json.loads(fixture["registry"].read_text())
    registry["environment_contracts"]["panel_records"] = _identity(fixture["panel_records"])
    _write_json(fixture["registry"], registry)
    with pytest.raises(ValueError, match="share a registered train cluster"):
        G.load_panel(fixture["registry"], "tr1c_cluster80_dev_v1")


@pytest.mark.parametrize("key,value,match", [
    ("status", "engineering_smoke_completed", "not converged"),
    ("converged", False, "not converged"),
    ("engineering_smoke", True, "smoke or subsampled"),
    ("stop_reason", "MANUAL_STOP", "not converged"),
    ("final_unfrozen_blocks", 19, "not 20/33"),
    ("gec_config_sha256", "forged", "exact common GEC config"),
    ("train_record_ids", [], "training IDs differ"),
    ("validation_record_ids", [], "validation IDs differ"),
])
def test_invalid_run_metadata_is_rejected(tmp_path, key, value, match):
    fixture = _scientific_fixture(tmp_path)
    path = fixture["metadata_paths"]["ensemble_context"]
    metadata = json.loads(path.read_text())
    metadata[key] = value
    _write_json(path, metadata)
    _, _, panel, _ = _validated_fixture_without_runs(fixture)
    with pytest.raises(ValueError, match=match):
        G.validate_run_metadata(fixture["registry"], fixture["config"], fixture["metadata_paths"], panel)


def _validated_fixture_without_runs(fixture):
    registry, records, panel = G.load_panel(fixture["registry"], "tr1c_cluster80_dev_v1")
    return registry, records, panel, None


@pytest.mark.parametrize("section,key,value,match", [
    ("convergence", "eligible", False, "best checkpoint is not eligible"),
    ("convergence", "best_step", 8000, "best checkpoint is not eligible"),
    ("convergence", "bad_evaluations", 4, "best checkpoint is not eligible"),
    ("backbone_report", "n_layers", 32, "not 20/33"),
    ("environment", "resolved_device", "cpu", "explicit CUDA"),
    ("schedule", "device", "cpu", "explicit CUDA"),
    ("schedule", "steps", 9600, "common training config|smoke or subsampled"),
    ("schedule", "validation_record_limit", 2, "smoke or subsampled"),
])
def test_nested_run_metadata_exploits_are_rejected(tmp_path, section, key, value, match):
    fixture = _scientific_fixture(tmp_path)
    path = fixture["metadata_paths"]["independent_context"]
    metadata = json.loads(path.read_text())
    metadata[section][key] = value
    _write_json(path, metadata)
    _, _, panel, _ = _validated_fixture_without_runs(fixture)
    with pytest.raises(ValueError, match=match):
        G.validate_run_metadata(fixture["registry"], fixture["config"], fixture["metadata_paths"], panel)


def test_cross_arm_common_identity_and_best_checkpoint_hash_are_enforced(tmp_path):
    fixture = _scientific_fixture(tmp_path)
    path = fixture["metadata_paths"]["destroy_interval_shuffle"]
    metadata = json.loads(path.read_text())
    metadata["gec_init_sha256"] = "arm-specific-init"
    metadata["run_identity"]["gec_init_sha256"] = "arm-specific-init"
    _write_json(path, metadata)
    _, _, panel, _ = _validated_fixture_without_runs(fixture)
    with pytest.raises(ValueError, match="common four-arm run metadata mismatch"):
        G.validate_run_metadata(fixture["registry"], fixture["config"], fixture["metadata_paths"], panel)
    best = fixture["metadata_paths"]["matched_control"].parent / "best.pt"
    best.write_bytes(b"replaced-checkpoint")
    with pytest.raises(ValueError, match="best-checkpoint identity mismatch"):
        G.validate_run_metadata(fixture["registry"], fixture["config"], fixture["metadata_paths"], panel)



def test_valid_structured_v2_checkpoints_are_accepted(tmp_path):
    fixture = _scientific_fixture(tmp_path)
    _, _, _, validation = _validated_fixture(fixture)
    for arm in G.ARMS:
        metadata = json.loads(fixture["metadata_paths"][arm].read_text())
        assert metadata["schema_version"] == "foldnar.gec_training_run.v2"
        result = validation["arms"][arm]
        assert result["checkpoint_validation"] == "PASS_BEST_AND_RESUME"
        assert result["resume_checkpoint_sha256"] == metadata["resume_checkpoint_sha256"]
        assert set(result["selected_model_tensor_sha256"]) == {"encoder", "head", "residual"}


def test_v1_run_metadata_schema_is_rejected(tmp_path):
    fixture = _scientific_fixture(tmp_path)
    path = fixture["metadata_paths"]["matched_control"]
    metadata = json.loads(path.read_text())
    metadata["schema_version"] = "foldnar.gec_training_run.v1"
    _write_json(path, metadata)
    _, _, panel, _ = _validated_fixture_without_runs(fixture)
    with pytest.raises(ValueError, match="unexpected run metadata identity"):
        G.validate_run_metadata(fixture["registry"], fixture["config"], fixture["metadata_paths"], panel)


def _refresh_checkpoint_hash(fixture, arm, name):
    path = fixture["metadata_paths"][arm]
    metadata = json.loads(path.read_text())
    key = "best_checkpoint_sha256" if name == "best.pt" else "resume_checkpoint_sha256"
    metadata[key] = hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
    _write_json(path, metadata)


@pytest.mark.parametrize("name", ["best.pt", "resume.pt"])
def test_arbitrary_checkpoint_bytes_are_rejected_after_file_hash_validation(tmp_path, name):
    fixture = _scientific_fixture(tmp_path)
    arm = "ensemble_context"
    (fixture["metadata_paths"][arm].parent / name).write_bytes(b"not-a-torch-checkpoint")
    _refresh_checkpoint_hash(fixture, arm, name)
    _, _, panel, _ = _validated_fixture_without_runs(fixture)
    with pytest.raises(ValueError, match="invalid checkpoint bytes"):
        G.validate_run_metadata(fixture["registry"], fixture["config"], fixture["metadata_paths"], panel)


def _mutate_checkpoint(fixture, arm, name, mutate):
    path = fixture["metadata_paths"][arm].parent / name
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    mutate(checkpoint)
    torch.save(checkpoint, path)
    _refresh_checkpoint_hash(fixture, arm, name)


def _assert_checkpoint_mutation_rejected(fixture, arm, match):
    _, _, panel, _ = _validated_fixture_without_runs(fixture)
    with pytest.raises(ValueError, match=match):
        G.validate_run_metadata(fixture["registry"], fixture["config"], fixture["metadata_paths"], panel)


@pytest.mark.parametrize("name,mutation,match", [
    ("best.pt", ("format", "forged"), "unsupported checkpoint format"),
    ("best.pt", ("checkpoint_phase", "latest"), "checkpoint phase is not best_eligible"),
    ("resume.pt", ("checkpoint_phase", "latest"), "restored best eligible"),
    ("resume.pt", ("terminal_status", "running"), "restored best eligible"),
    ("best.pt", ("step", 10600), "selected best-step mismatch"),
    ("resume.pt", ("step", 10600), "selected best-step mismatch"),
    ("best.pt", ("n_unfrozen", 19), "terminal 20/33 unfreezing mismatch"),
    ("resume.pt", ("n_unfrozen", 19), "terminal 20/33 unfreezing mismatch"),
])
def test_checkpoint_phase_step_and_unfreeze_mismatches_are_rejected(tmp_path, name, mutation, match):
    fixture = _scientific_fixture(tmp_path)
    arm = "destroy_interval_shuffle"
    _mutate_checkpoint(fixture, arm, name, lambda cp: cp.__setitem__(*mutation))
    _assert_checkpoint_mutation_rejected(fixture, arm, match)


@pytest.mark.parametrize("key,value", [
    ("arm", "matched_control"),
    ("gec_config_sha256", "forged-config"),
    ("train_record_ids", []),
])
def test_checkpoint_arm_config_and_train_identity_mismatches_are_rejected(tmp_path, key, value):
    fixture = _scientific_fixture(tmp_path)
    arm = "independent_context"
    _mutate_checkpoint(
        fixture, arm, "best.pt", lambda cp: cp["run_identity"].__setitem__(key, value),
    )
    _assert_checkpoint_mutation_rejected(fixture, arm, "checkpoint run identity mismatch")


def test_best_tensor_content_cannot_be_changed_without_updating_hash(tmp_path):
    fixture = _scientific_fixture(tmp_path)
    arm = "matched_control"
    _mutate_checkpoint(
        fixture, arm, "best.pt", lambda cp: cp["encoder"]["weight"].add_(1.0),
    )
    _assert_checkpoint_mutation_rejected(fixture, arm, "serialized encoder tensor identity mismatch")


def test_best_and_resume_selected_model_tensors_must_match_exactly(tmp_path):
    fixture = _scientific_fixture(tmp_path)
    arm = "matched_control"
    def mutate(cp):
        cp["residual"]["weight"].add_(1.0)
        cp["model_tensor_sha256"]["residual"] = G.tensor_state_sha256(cp["residual"])
    _mutate_checkpoint(fixture, arm, "resume.pt", mutate)
    _assert_checkpoint_mutation_rejected(fixture, arm, "best-vs-resume selected model tensor identity mismatch")


def test_restored_best_step_and_checkpoint_convergence_must_match(tmp_path):
    fixture = _scientific_fixture(tmp_path)
    path = fixture["metadata_paths"]["matched_control"]
    metadata = json.loads(path.read_text())
    metadata["restored_best_step"] = 10600
    _write_json(path, metadata)
    _, _, panel, _ = _validated_fixture_without_runs(fixture)
    with pytest.raises(ValueError, match="restored_best_step"):
        G.validate_run_metadata(fixture["registry"], fixture["config"], fixture["metadata_paths"], panel)
    _mutate_checkpoint(
        fixture, "matched_control", "resume.pt",
        lambda cp: cp["convergence"].__setitem__("best_step", 10600),
    )
    _assert_checkpoint_mutation_rejected(fixture, "matched_control", "convergence state mismatch")

def _binding_case():
    sequence = "AAAAAAAAAAA"
    record = G.Record(
        "s", "RF", G.pairs([(0, 10)]), sequence_length=11,
        sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(), cluster_id=1,
    )
    run_validation = {"arms": {
        arm: {"best_checkpoint_sha256": f"checkpoint-{arm}", "checkpoint_validation": "PASS_BEST_AND_RESUME"} for arm in G.ARMS
    }}
    rows = {arm: [{
        "sequence_id": "s", "true_pairs": [[0, 10]], "predicted_pairs": [[0, 10]],
        "sequence_length": 11, "sequence_sha256": record.sequence_sha256,
        "checkpoint_sha256": f"checkpoint-{arm}",
    }] for arm in G.ARMS}
    return record, rows, run_validation


@pytest.mark.parametrize("arm", G.ARMS)
def test_same_count_same_span_forged_truth_is_rejected_in_every_arm(arm):
    record, rows, run_validation = _binding_case()
    rows[arm][0]["true_pairs"] = [[1, 9]]
    with pytest.raises(ValueError, match="authoritative true-pair mismatch"):
        G.bind_predictions([record], rows, run_validation)


@pytest.mark.parametrize("exploit", [
    "sequence_hash", "missing_sequence_hash", "sequence_length",
    "predicted_bounds", "raw_score_bounds", "checkpoint",
])
@pytest.mark.parametrize("arm", G.ARMS)
def test_prediction_identity_and_coordinate_exploits_are_rejected_in_every_arm(arm, exploit):
    record, rows, run_validation = _binding_case()
    row = rows[arm][0]
    if exploit == "sequence_hash":
        row["sequence_sha256"] = "wrong"
    elif exploit == "missing_sequence_hash":
        row.pop("sequence_sha256")
    elif exploit == "sequence_length":
        row["sequence_length"] = 10
    elif exploit == "predicted_bounds":
        row["predicted_pairs"] = [[0, 11]]
    elif exploit == "raw_score_bounds":
        row["raw_scores"] = [[0, 11, 0.5]]
    else:
        row["checkpoint_sha256"] = "wrong-checkpoint"
    match = {
        "sequence_hash": "sequence identity", "missing_sequence_hash": "sequence identity",
        "sequence_length": "sequence length", "predicted_bounds": "outside frozen sequence length",
        "raw_score_bounds": "outside frozen sequence length", "checkpoint": "prediction checkpoint mismatch",
    }[exploit]
    with pytest.raises(ValueError, match=match):
        G.bind_predictions([record], rows, run_validation)


def test_family_cluster_sensitivity_cannot_rescue_or_veto_paired_gates():
    evaluation = _passing_evaluation()
    matched = evaluation["comparisons"]["matched_control"]["metrics"]["micro_pair_f1"]
    matched["family_cluster_sensitivity_ci95"] = [-100.0, -100.0]
    assert G.verify_gates(evaluation)["pass"]
    evaluation["comparisons"]["matched_control"]["metrics"]["micro_pair_f1"]["delta"] = -1e-8
    matched["family_cluster_sensitivity_ci95"] = [100.0, 100.0]
    assert not G.verify_gates(evaluation)["pass"]
