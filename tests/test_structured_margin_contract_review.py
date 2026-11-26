import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rnajepa.harness import valid_pair_mask
from rnajepa.prior_alignment import sha256_file
from rnajepa.structured_margin import pair_hamming_structured_margin
from tools.verify_structured_margin_contract_review_v1 import (
    ARMS, METRIC, STOP_REASON, audit_converged_margin_arm, audit_full_contract,
    audit_resource_evidence, enumerate_legal_structures_by_subset, subset_oracle,
)


def test_subset_oracle_checks_value_gradient_and_mask_without_dp_recurrence():
    rng = np.random.default_rng(6026)
    cases = 0
    for min_loop in (0, 1, 3, 5):
        for sequence in ("AUAUGC", "AUGCAUGC", "CCCCCCCC"):
            mask = valid_pair_mask(sequence, min_loop)
            structures = list(enumerate_legal_structures_by_subset(mask, min_loop))
            scores = np.zeros((len(sequence), len(sequence)))
            scores[np.triu_indices(len(sequence), 1)] = rng.normal(
                size=len(sequence) * (len(sequence) - 1) // 2
            )
            for target in ((), structures[len(structures) // 2], structures[-1]):
                for normalization in ("length", "sum"):
                    expected, gradient, _, _ = subset_oracle(
                        scores, mask, target, min_loop, normalization,
                    )
                    tensor = torch.tensor(scores, requires_grad=True)
                    loss = pair_hamming_structured_margin(
                        tensor, torch.tensor(mask), target,
                        min_loop=min_loop, normalization=normalization,
                    )
                    loss.backward()
                    assert float(loss) == pytest.approx(expected, abs=1e-10)
                    np.testing.assert_allclose(tensor.grad.numpy(), gradient, atol=1e-12, rtol=0)
                    assert not tensor.grad.numpy()[~mask].any()
                    cases += 1
    assert cases == 72


def _write_converged_fixture(root):
    root.mkdir()
    best_step = 10500
    stopped_step = 13000
    stopped_state = {
        "terminal_step": 9600, "terminal_unfrozen_blocks": 20, "patience": 5,
        "min_delta": 1e-6, "best_step": best_step, "best_validation_loss": 0.25,
        "bad_evaluations": 5, "eligible": True,
    }
    best_state = {**stopped_state, "bad_evaluations": 0}
    identity = {"validation_metric": METRIC, "terminal_unfrozen_blocks": 20}
    meta = {
        "schema_version": "foldnar.structured_margin_training_run.v1",
        "arm": "ensemble_context", "status": "converged", "converged": True,
        "stop_reason": STOP_REASON, "validation_metric": METRIC,
        "objective": {"name": "exact_pair_hamming_cost_augmented_structured_max_margin", "lambda_nll": 0.0},
        "run_identity": identity, "final_unfrozen_blocks": 20,
        "training_stopped_step": stopped_step, "steps_completed": stopped_step,
        "restored_best_step": best_step, "convergence": stopped_state,
        "best_convergence": best_state,
    }
    validations = [{
        "step": 500, "validation_loss": 0.3, "validation_metric": METRIC,
        "n_unfrozen": 0, "eligible": False, "save_best": False, "stop": False,
    }, {
        "step": stopped_step, "validation_loss": 0.3, "validation_metric": METRIC,
        "n_unfrozen": 20, "eligible": True, "save_best": False, "stop": True,
        "best_step": best_step, "best_validation_loss": 0.25,
        "bad_evaluations": 5,
    }]
    (root / "validation_log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in validations))
    for name, phase, terminal in (
        ("best.pt", "best_eligible", "running"),
        ("resume.pt", "restored_best_eligible", "converged"),
    ):
        state = {
            "format": "foldnar.structured_margin_checkpoint.v1", "step": best_step,
            "n_unfrozen": 20, "run_identity": identity, "checkpoint_phase": phase,
            "terminal_status": terminal, "convergence": best_state,
            "encoder": {"weight": torch.tensor([1.0])},
            "head": {"weight": torch.tensor([2.0])},
            "residual": {"weight": torch.tensor([3.0])},
        }
        from rnajepa.prior_alignment import tensor_state_sha256
        state["model_tensor_sha256"] = {
            part: tensor_state_sha256(state[part]) for part in ("encoder", "head", "residual")
        }
        if name == "best.pt":
            state["validation_loss"] = 0.25
        torch.save(state, root / name)
    meta["best_checkpoint_sha256"] = sha256_file(root / "best.pt")
    meta["resume_checkpoint_sha256"] = sha256_file(root / "resume.pt")
    (root / "run_meta.json").write_text(json.dumps(meta))
    return meta


def test_convergence_verifier_requires_distinct_stopped_and_restored_best_states(tmp_path):
    root = tmp_path / "ensemble_context"
    meta = _write_converged_fixture(root)
    result = audit_converged_margin_arm(root)
    assert result["status"] == "PASS_CONVERGED_CONTRACT"
    assert result["best_step"] == 10500 and result["stopped_step"] == 13000
    broken = copy.deepcopy(meta)
    del broken["best_convergence"]
    (root / "run_meta.json").write_text(json.dumps(broken))
    with pytest.raises(AssertionError, match="distinct restored-best"):
        audit_converged_margin_arm(root)


def test_full_contract_fails_closed_when_all_margin_runs_are_absent(tmp_path):
    numerical = {"status": "PASS"}
    smoke = {"status": "PASS_TWO_STEP_ENGINEERING_SMOKE_ONLY", "full_ready": False}
    report = audit_full_contract(tmp_path / "absent", numerical, smoke)
    assert report["status"] == "FAIL_FULL_CONTRACT"
    assert set(report["margin_arms"]) == set(ARMS)
    assert all(row["status"] == "NOT_RUN" for row in report["margin_arms"].values())
    assert any("resource:" in blocker for blocker in report["blockers"])
    assert not report["scientific_training_authorized"]
    assert any("science:" in blocker for blocker in report["blockers"])


def test_resource_gate_requires_measured_worst_case_and_full_headroom():
    evidence = {
        "status": "PASS_B4_L500_TERMINAL20_RESOURCE_GATE", "batch_size": 4,
        "max_sequence_length": 500, "terminal_unfrozen_blocks": 20,
        "measured_peak_cuda_bytes": 10 * 1024**3,
        "free_cuda_bytes_at_launch": 22 * 1024**3,
        "total_cuda_bytes": 32 * 1024**3,
    }
    assert audit_resource_evidence(evidence) is None
    evidence["free_cuda_bytes_at_launch"] -= 1024**3
    with pytest.raises(AssertionError, match="headroom"):
        audit_resource_evidence(evidence)
    assert "not_measured" in audit_resource_evidence(None)
