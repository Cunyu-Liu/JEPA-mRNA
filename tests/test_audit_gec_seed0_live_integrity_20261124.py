import importlib.util
import math
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "audit_gec_seed0_live_integrity_20261124.py"
SPEC = importlib.util.spec_from_file_location("audit_gec_seed0_live_integrity_20261124", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def training_row(step, **changes):
    row = {
        "step": step, "loss": 0.5, "gradient_norm": 1.0,
        "residual_gradient_norm": 0.1, "context_log_z": [10.0],
        "peak_cuda_bytes": 1000 + step, "elapsed_seconds": float(step),
        "optimizer_step": True, "optimizer_steps": step, "divergence_reason": "",
    }
    row.update(changes)
    return row


def test_frozen_warmup_and_unfreeze_schedules_are_reconstructed_independently():
    assert [MODULE.expected_lr(0.0001, step, 5) for step in (1, 4, 5, 6)] == [0.00002, 0.00008, 0.0001, 0.0001]
    expected = {1: 0, 1600: 0, 1601: 0, 2399: 0, 2400: 2, 9600: 20, 9601: 20}
    for step, count in expected.items():
        assert MODULE.expected_unfrozen(step, 1600, 800, 2, 33, 20) == count


def test_training_audit_checks_finite_values_and_monotonic_ledgers():
    result, issues = MODULE.audit_training_rows([training_row(1), training_row(2)])
    assert result["completed_steps"] == 2
    assert result["all_losses_finite"] and result["all_gradient_norms_finite"]
    assert result["lr_observation"]["rows_missing_lr"] == 2
    assert not issues
    result, issues = MODULE.audit_training_rows([
        training_row(1), training_row(3, loss=math.nan, peak_cuda_bytes=1),
    ])
    codes = {row["code"] for row in issues}
    assert result["completed_steps"] is None
    assert {"NONMONOTONIC_TRAIN_STEP", "NONFINITE_TRAIN_VALUE", "NONMONOTONIC_CUDA_PEAK"} <= codes


def test_observed_lr_is_required_and_must_be_finite_nonnegative():
    result, issues = MODULE.audit_training_rows([
        training_row(1, head_lr=0.00002, backbone_lr=0.0000002),
        training_row(2, head_lr=math.inf, backbone_lr=0.0000004),
    ])
    assert result["lr_observation"]["rows_with_lr"] == 2
    assert result["lr_observation"]["all_steps_have_finite_lr"] is False
    assert [row["code"] for row in issues] == ["INVALID_OBSERVED_LR"]


def test_four_arm_batch_comparison_checks_common_prefix_and_mismatch():
    rows = [{"step": 1, "epoch": 0, "record_ids": ["a"], "sequence_sha256": ["x"]}]
    batch = {arm: list(rows) for arm in MODULE.ARMS}
    assert MODULE.compare_batch_orders(batch)["exact_common_prefix_identity"]
    batch["independent_context"] = [{"step": 1, "epoch": 0, "record_ids": ["b"], "sequence_sha256": ["y"]}]
    comparison = MODULE.compare_batch_orders(batch)
    assert not comparison["exact_common_prefix_identity"]
    assert comparison["mismatches"] == [{"row": 1, "reference_arm": "matched_control", "arm": "independent_context"}]


def test_checkpoint_absence_before_first_save_is_not_due(tmp_path):
    schedule = {"save_every": 500}
    result = MODULE.checkpoint_audit(tmp_path, {"converged": False}, [], schedule, [])
    assert result["status"] == "NOT_AVAILABLE_NOT_DUE"
    result = MODULE.checkpoint_audit(tmp_path, {"converged": False}, [{}] * 500, schedule, [])
    assert result["status"] == "MISSING_WHEN_DUE"
