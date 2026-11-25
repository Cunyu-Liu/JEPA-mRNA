from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/gec_seed0_independent_reeval_v1.py"
SPEC = importlib.util.spec_from_file_location("gec_seed0_independent_reeval_v1", MODULE_PATH)
G = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = G
SPEC.loader.exec_module(G)


def test_validation_reconstruction_and_reset_detection(tmp_path):
    losses = [0.4 - index * 0.001 for index in range(19)] + [1.0, 0.9, 0.95, 0.94, 0.96, 1.0, 1.1]
    rows = []
    best = float("inf")
    bad = 0
    best_step = None
    for index, loss in enumerate(losses):
        step = (index + 1) * 500
        n_unfrozen = min(20, max(0, ((step - 1600) // 800) * 2))
        eligible = step >= 9600 and n_unfrozen == 20
        improved = eligible and loss < best - 1e-6
        if improved:
            best, best_step, bad = loss, step, 0
        elif eligible:
            bad += 1
        rows.append({
            "step": step, "validation_loss": loss, "n_unfrozen": n_unfrozen,
            "eligible": eligible, "improved": improved, "save_best": improved,
            "best_step": best_step, "best_validation_loss": None if best_step is None else best,
            "bad_evaluations": bad, "stop": eligible and bad >= 5,
        })
    result = G.reconstruct_validation(rows)
    assert result["mismatches"] == []
    assert result["stop_step"] == 13000 and result["best_step"] == 10500
    assert result["bad_evaluations"] == 5 and result["stop"]
    assert result["final_convergence"]["bad_evaluations"] == 5

    state = {"weight": torch.tensor([1.0, 2.0])}
    model = {"encoder": state, "head": state, "residual": state}
    tensor_hashes = {name: G.tensor_state_sha256(state) for name in model}
    run_identity = {
        "arm": "matched_control", "gec_config_sha256": "config", "source_contract_sha256": "source",
        "engineering_smoke": False, "terminal_unfrozen_blocks": 20,
    }
    common = {
        "format": "foldnar.gec_checkpoint.v2", "step": 10500, "optimizer_steps": 10500,
        "n_unfrozen": 20, "run_identity": run_identity, "model_tensor_sha256": tensor_hashes,
        **model,
    }
    best_convergence = dict(result["final_convergence"], bad_evaluations=0)
    torch.save({
        **common, "checkpoint_phase": "best_eligible", "terminal_status": "running",
        "convergence": best_convergence, "validation_loss": 0.9, "model": model,
    }, tmp_path / "best.pt")
    torch.save({
        **common, "checkpoint_phase": "restored_best_eligible", "terminal_status": "converged",
        "convergence": dict(best_convergence),
    }, tmp_path / "resume.pt")
    audit = G.audit_checkpoint_pair(tmp_path, "matched_control", result, "config", "source")
    assert not audit["pass"]
    assert any("final_convergence_from_validation_log" in error for error in audit["errors"])
    resume = torch.load(tmp_path / "resume.pt", map_location="cpu", weights_only=False)
    resume["convergence"] = result["final_convergence"]
    torch.save(resume, tmp_path / "resume.pt")
    audit = G.audit_checkpoint_pair(tmp_path, "matched_control", result, "config", "source")
    assert audit["pass"], audit["errors"]


def test_hand_checked_pair_stem_and_environment_confusion():
    truth = ((0, 11), (1, 10), (4, 15))
    predicted = ((0, 11), (2, 9), (4, 15))
    assert G.confusion(truth, predicted) == (2, 1, 1)
    assert G.prf((2, 1, 1)) == pytest.approx((2 / 3, 2 / 3, 2 / 3))
    assert G.confusion(G.stems(truth), G.stems(predicted)) == (1, 2, 1)
    assert G.isolated(truth) == {(4, 15)}
    assert G.confusion(truth, predicted, G.isolated(truth), G.isolated(predicted)) == (1, 2, 0)
    assert G.span_bin((0, 16)) == "16_31" and G.span_bin((0, 128)) == "ge128"
    reference = ((1, 11),)
    assert G.selected_pairs(truth, "competition:reference_conflicts", reference) == set(truth)


def _records():
    truth = tuple((20 * index, 20 * index + 16) for index in range(17))
    records = []
    for family in range(10):
        for index in range(30):
            sequence_id = f"RF{family}:s{index}"
            records.append(G.Record(
                sequence_id, f"RF{family}", family * 30 + index, 400,
                hashlib.sha256(sequence_id.encode()).hexdigest(), truth,
            ))
    return records


def test_full_metric_and_gate_recomputation_from_records():
    frozen = _records()
    arms = {
        arm: [replace(record, predicted_pairs=(
            record.true_pairs if arm == "ensemble_context" else record.true_pairs[:-1] + ((360, 380),)
        )) for record in frozen]
        for arm in G.ARMS
    }
    result = G.evaluate_metrics(frozen, arms)
    for metric in G.GLOBAL_METRICS:
        assert result["arm_metrics"]["ensemble_context"][metric] == pytest.approx(1.0)
        assert result["arm_metrics"]["ensemble_context"][metric] >= result["arm_metrics"]["matched_control"][metric]
    assert result["arm_metrics"]["matched_control"]["micro_pair_f1"] < 1.0
    assert result["arm_metrics"]["matched_control"]["macro_pair_f1"] < 1.0
    assert result["bootstrap"]["n_macro_eligible_families"] == 10
    assert result["targets"]["isolated_pair:isolated"]["target_gate_pass"]
    assert result["targets"]["pair_span:16_31"]["target_gate_pass"]
    gates = G.verify_gates(result)
    assert gates["pass"], gates["failed_gates"]
    matched = result["comparisons"]["matched_control"]["metrics"]["micro_pair_f1"]
    assert matched["delta"] == pytest.approx(1 / 17)
    assert matched["ci95"] == pytest.approx([1 / 17, 1 / 17])


def _passing_evaluation():
    comparisons = {}
    for control in G.CONTROLS:
        comparisons[control] = {"metrics": {
            metric: {"delta": 0.02} for metric in G.GLOBAL_METRICS
        }}
    return {
        "ci_contract_pass": True, "comparisons": comparisons,
        "targets": {"pair_span:ge128": {"target_gate_pass": True}},
    }


@pytest.mark.parametrize("gate_name", [
    "paired_ci_contract", "micro_pair_f1_nonnegative", "macro_pair_f1_strictly_positive",
    "micro_precision_loss_at_most_0.005", "macro_precision_loss_at_most_0.005",
    "micro_stem_f1_loss_at_most_0.005", "macro_stem_f1_loss_at_most_0.005",
    "strictly_improved_diverse_target",
] + [
    f"{control}:{metric}_strict_advantage"
    for control in ("destroy_interval_shuffle", "independent_context")
    for metric in ("micro_pair_f1", "macro_pair_f1", "micro_stem_f1", "macro_stem_f1")
])
def test_every_atomic_scientific_gate_can_fail(gate_name):
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
        evaluation["comparisons"][control]["metrics"][metric.removesuffix("_strict_advantage")]["delta"] = 0.0
    verdict = G.verify_gates(evaluation)
    assert not verdict["pass"] and verdict["failed_gates"] == [gate_name]


def test_loss_threshold_boundary_and_ci_cannot_rescue_point_gate():
    evaluation = _passing_evaluation()
    matched = evaluation["comparisons"]["matched_control"]["metrics"]
    for metric in ("micro_pair_precision", "macro_pair_precision", "micro_stem_f1", "macro_stem_f1"):
        matched[metric]["delta"] = -0.005
    matched["micro_pair_f1"]["delta"] = 0.0
    assert G.verify_gates(evaluation)["pass"]
    matched["micro_pair_f1"].update(delta=-1e-8, ci95=[1.0, 1.0])
    assert not G.verify_gates(evaluation)["pass"]


def test_paired_sequence_and_family_bootstrap_multiplicity():
    records = _records()[:30]
    boot = G.PairedBootstrap(records, resamples=8, seed=20261118)
    assert len(boot.eligible_families) == 1
    assert np.all(boot.multiplicity.sum(axis=1) == len(records))
    counts = np.asarray([[1.0, 0.0, 0.0]] * len(records))
    sequence, family = boot.sample_counts(counts)
    assert np.all(sequence[:, 0] == len(records))
    assert np.all(family[:, 0] == boot.record_group_multiplicity.sum(axis=1) * len(records))
    values = np.arange(len(records), dtype=float)
    observed, sequence_macro, family_macro = boot.sample_macro(values)
    expected = (boot.multiplicity[0] * values).sum() / boot.multiplicity[0].sum()
    assert observed == pytest.approx(values.mean())
    assert sequence_macro[0] == pytest.approx(expected)
    assert np.all(family_macro == pytest.approx(values.mean()))


def test_family_sensitivity_handles_noncontiguous_eligible_group_indices():
    records = _records()[:30]
    records.append(replace(records[0], sequence_id="unmapped", family_id=None))
    boot = G.PairedBootstrap(records, resamples=8, seed=20261118)
    assert len(boot.eligible_families) == 1 and len(boot.group_ids) == 2
    values = np.arange(len(records), dtype=float)
    observed, _, family_macro = boot.sample_macro(values)
    assert observed == pytest.approx(values[:30].mean())
    group_means = np.asarray([values[:30].mean(), values[30]])
    expected = group_means[boot.group_indices[:, boot.eligible_groups]].mean(axis=1)
    assert np.all(family_macro == pytest.approx(expected))


def test_prediction_binding_checks_truth_coverage_and_checkpoint(tmp_path):
    record = G.Record("s", "RF", 1, 11, "hash", ((0, 10),))
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_identities = {
        arm: {"best_checkpoint": {"sha256": G.sha256(checkpoint)}} for arm in G.ARMS
    }
    paths = {}
    for arm in G.ARMS:
        path = tmp_path / f"{arm}.jsonl"
        path.write_text(json.dumps({
            "sequence_id": "s", "sequence_length": 11, "sequence_sha256": "hash",
            "true_pairs": [[0, 10]], "predicted_pairs": [[0, 10]],
            "checkpoint_sha256": G.sha256(checkpoint),
        }) + "\n")
        paths[arm] = path
    (tmp_path / "run_meta.json").write_text("not json and never read")
    bound, _ = G.bind_predictions([record], paths, checkpoint_identities)
    assert bound["ensemble_context"][0].predicted_pairs == ((0, 10),)
    row = json.loads(paths["matched_control"].read_text())
    row["true_pairs"] = [[1, 9]]
    paths["matched_control"].write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="frozen truth mismatch"):
        G.bind_predictions([record], paths, checkpoint_identities)
