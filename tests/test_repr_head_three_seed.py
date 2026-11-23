from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnajepa.repr_head_matrix import (active_repr_head_arms, load_repr_head_contract,
                                      validate_repr_head_training_args)


def _contract():
    return load_repr_head_contract(ROOT / "configs/representation_head_mechanism_4arm_v1.json")


def _valid_args(contract, arm="control_matched"):
    train = contract["training"]
    return SimpleNamespace(
        contract_mode="full", raw_score_arm=arm, train_turner="off",
        shared_init=train["shared_init_template"].format(seed=1), init_checkpoint=None,
        no_early_stop=True, head_init="fresh", data=train["data"], batch_size=4,
        head_lr=train["head_lr"], backbone_lr=train["backbone_lr"],
        warmup_head_steps=1600, unfreeze_every=800, unfreeze_per_step=2,
        warmup_steps=5, grad_clip=1.0, nll_normalization="length", lambda_nll=1.0,
        max_unfrozen_blocks=33, lambda_marginal_bce=0.0,
        d_z=contract["arms"][arm]["d_z"],
        resnet_blocks=contract["arms"][arm]["resnet_blocks"],
        resnet_hidden=contract["arms"][arm]["resnet_hidden"],
        seed=1, steps=9600, scorer="resnet2d", serial_encoder=False,
    )


def test_three_seed_active_arms_exclude_failed_repr_wide():
    contract = _contract()
    assert active_repr_head_arms(contract, 0) == (
        "control_matched", "repr_wide", "head_deep", "repr_head")
    for seed in (1, 2):
        assert active_repr_head_arms(contract, seed) == (
            "control_matched", "head_deep", "repr_head")


def test_full_mode_retrains_control_and_only_promoted_variants():
    contract = _contract()
    for arm in ("control_matched", "head_deep", "repr_head"):
        assert validate_repr_head_training_args(contract, _valid_args(contract, arm)) == "train_off"
    with pytest.raises(ValueError, match="did not pass"):
        validate_repr_head_training_args(contract, _valid_args(contract, "repr_wide"))


def test_three_seed_macro_pair_f1_must_be_strictly_positive():
    spec = importlib.util.spec_from_file_location(
        "raw_score_evaluator", ROOT / "tools/evaluate_raw_score_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keys = ("micro_auprc", "micro_recall_at_precision_0.8", "micro_pair_f1", "macro_pair_f1",
            "micro_pair_precision", "macro_pair_precision", "micro_stem_f1", "macro_stem_f1")
    control = {key: 0.5 for key in keys}
    candidate = dict(control, micro_auprc=0.6, micro_pair_f1=0.6)
    candidate["micro_recall_at_precision_0.8"] = 0.6
    rules = _contract()["pilot"]["per_arm_direction_gate"]
    _, checks, passed = module._direction_gate(candidate, control, rules)
    assert checks["macro_pair_f1_strictly_positive"] is False
    assert passed is False
    candidate["macro_pair_f1"] = 0.500001
    _, checks, passed = module._direction_gate(candidate, control, rules)
    assert checks["macro_pair_f1_strictly_positive"] is True
    assert passed is True


def _write_three_seed_fixture(root, contract, failing_seed=None):
    metric_keys = ("micro_auprc", "micro_recall_at_precision_0.8", "micro_pair_f1", "macro_pair_f1",
                   "micro_pair_precision", "macro_pair_precision", "micro_pair_recall",
                   "macro_pair_recall", "micro_stem_f1", "macro_stem_f1")
    contract_sha256 = __import__("hashlib").sha256(
        (ROOT / "configs/representation_head_mechanism_4arm_v1.json").read_bytes()).hexdigest()
    for seed in (0, 1, 2):
        stage = root / f"seed{seed}"
        (stage / "evaluation").mkdir(parents=True)
        arms = active_repr_head_arms(contract, seed)
        control = {key: 0.5 for key in metric_keys}
        verdict_arms = {"control_matched": control}
        comparisons = {}
        for arm in arms[1:]:
            candidate = dict(control, micro_auprc=0.6, micro_pair_f1=0.6,
                             macro_pair_f1=0.6, micro_pair_recall=0.6,
                             macro_pair_recall=0.6, micro_stem_f1=0.6, macro_stem_f1=0.6)
            candidate["micro_recall_at_precision_0.8"] = 0.6
            if seed == failing_seed and arm == "head_deep":
                candidate["macro_pair_f1"] = 0.5
            verdict_arms[arm] = candidate
            delta = {key: candidate[key] - control[key] for key in metric_keys[:4]}
            checks = {"micro_auprc_strictly_positive": True,
                      "micro_recall_at_precision_0.8_strictly_positive": True,
                      "micro_pair_f1_noninferior": True,
                      "macro_pair_f1_noninferior": True,
                      "macro_pair_f1_strictly_positive": delta["macro_pair_f1"] > 0,
                      "micro_pair_precision_preserved": True,
                      "macro_pair_precision_preserved": True,
                      "micro_stem_f1_preserved": True, "macro_stem_f1_preserved": True,
                      "at_least_one_pair_f1_strictly_positive": True}
            comparisons[arm] = {"delta_vs_control_matched": delta, "checks": checks,
                                "direction_passed": all(checks.values())}
        identity_sha = "seed0-frozen-contract" if seed == 0 else contract_sha256
        (stage / "evaluation/pilot_verdict.json").write_text(json.dumps({
            "contract_sha256": identity_sha, "arms": verdict_arms, "comparisons": comparisons}))
        (stage / "verification.json").write_text(json.dumps({
            "status": "PASS", "contract_sha256": identity_sha, "arms": list(arms)}))
        (stage / "final_independent_check.json").write_text(json.dumps({
            "overall": "PASS", "contract_sha256": identity_sha}))


@pytest.mark.parametrize("failing_seed,expected_decision,external_allowed", [
    (None, "THREE_SEED_CAPACITY_DIRECTION_CONFIRMED", True),
    (1, "STOP_REPRESENTATION_HEAD_CAPACITY_THREE_SEED", False),
])
def test_three_seed_aggregate_requires_all_seed_gates(tmp_path, failing_seed,
                                                       expected_decision, external_allowed):
    contract = _contract()
    _write_three_seed_fixture(tmp_path, contract, failing_seed)
    out = tmp_path / "three_seed_verdict.json"
    subprocess.run([
        sys.executable, str(ROOT / "tools/verify_repr_head_three_seed.py"),
        "--contract", str(ROOT / "configs/representation_head_mechanism_4arm_v1.json"),
        "--artifact-root", str(tmp_path), "--out", str(out),
    ], check=True, cwd=ROOT)
    result = json.loads(out.read_text())
    assert result["status"] == "PASS"
    assert result["decision"] == expected_decision
    assert result["external_evaluation_allowed"] is external_allowed
    assert result["candidate_selected"] is False
    assert result["sota_claim"] is False


def test_leave_one_sequence_out_detects_single_sequence_driven_gain():
    spec = importlib.util.spec_from_file_location(
        "three_seed_analysis", ROOT / "tools/analyze_repr_head_three_seed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def row(tp, fp, fn):
        return {"pair": {"tp": tp, "fp": fp, "fn": fn},
                "stem": {"tp": tp, "fp": fp, "fn": fn},
                "raw": {"auprc": tp / max(tp + fp + fn, 1)}}

    control = {"a": row(10, 10, 10), "b": row(10, 10, 10)}
    candidate = {"a": row(100, 0, 0), "b": row(10, 10, 10)}
    result = module.summarize(None, control, candidate)
    check = result["micro_pair_f1_delta_leave_one_sequence_out"]
    assert check["full"] > 0
    assert check["positive_after_every_single_sequence_removal"] is False
    assert module.summarize(None, {}, {})["n_sequences"] == 0
