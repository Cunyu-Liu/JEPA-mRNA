import json
from pathlib import Path

import pytest

from tools import audit_structured_margin_incident_recovery_20261127 as audit
from tools import run_structured_margin_incident_recovery_20261127 as recovery
from tools.run_frozen_margin_trainer_with_resume_audit_20261127 import ValidationAuditList


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def interrupted_fixture(tmp_path, step=500):
    for name in ("train_log.jsonl", "batch_order.jsonl", "compute_ledger.jsonl"):
        rows = []
        for index in range(1, step + 2):
            row = {"step": index, "record_ids": [f"record-{index}"]}
            if name == "train_log.jsonl":
                row.update({"optimizer_steps": index, "optimizer_step": True})
            rows.append(row)
        write_jsonl(tmp_path / name, rows)
    write_jsonl(tmp_path / "validation_log.jsonl", [
        {"step": 500, "eligible": False, "stop": False},
    ])
    return tmp_path


def test_next_batch_and_ledger_audit_preserve_resume_order(tmp_path):
    interrupted_fixture(tmp_path)
    report = audit.audit_ledgers(tmp_path, 500)
    assert report["train_log.jsonl"]["rows_through_checkpoint"] == 500
    assert report["train_log.jsonl"]["rows_before_resume_truncation"] == 501
    assert audit.next_batch_indices({"queue": [[8, 9, 10, 11]], "batch_size": 4}) == [8, 9, 10, 11]


def test_ledger_audit_rejects_missing_or_diverged_steps(tmp_path):
    interrupted_fixture(tmp_path)
    with pytest.raises(ValueError, match="ends before checkpoint"):
        audit.audit_ledgers(tmp_path, 502)
    rows = [json.loads(line) for line in (tmp_path / "compute_ledger.jsonl").read_text().splitlines()]
    rows[0]["record_ids"] = ["wrong"]
    write_jsonl(tmp_path / "compute_ledger.jsonl", rows)
    with pytest.raises(ValueError, match="batch/compute identity mismatch"):
        audit.audit_ledgers(tmp_path, 500)


def test_training_command_is_frozen_unbounded_20_of_33(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"assets": {"shared_init": "/shared.pt"}}))
    command = recovery.training_command(Path("/frozen"), config, "matched_control", tmp_path)
    joined = " ".join(command)
    assert command[1].endswith("tools/run_frozen_margin_trainer_with_resume_audit_20261127.py")
    assert "--steps 0" in joined
    assert "--max-unfrozen-blocks 20" in joined
    assert "--seed 0" in joined and "--batch-size 4" in joined
    assert "--validation-every 500" in joined and "--save-every 500" in joined


def test_interrupted_arm_requires_checkpoint_and_batch_preflight(tmp_path):
    arm_dir = tmp_path / "matched_control"
    arm_dir.mkdir()
    preflight = {
        "status": "PASS_INTERRUPTED_RESUME_AUDIT",
        "arms": {"matched_control": {
            "status": "PASS_INTERRUPTED_RESUME_AUDIT", "converged": False,
            "stream": {"matches_interrupted_batch_order": True},
        }},
    }
    with pytest.raises(ValueError, match="checkpoint audit"):
        recovery.launch_eligibility("matched_control", arm_dir, preflight)
    (arm_dir / "resume.pt").touch()
    assert recovery.launch_eligibility("matched_control", arm_dir, preflight) == "resume"
    preflight["arms"]["matched_control"]["stream"]["matches_interrupted_batch_order"] = False
    with pytest.raises(ValueError, match="batch continuity"):
        recovery.launch_eligibility("matched_control", arm_dir, preflight)


def test_running_or_incomplete_run_is_never_converged(tmp_path):
    path = tmp_path / "run_meta.json"
    path.write_text(json.dumps({"status": "running", "converged": False}))
    assert recovery.converged_meta(path) is None
    path.write_text(json.dumps({
        "status": "converged", "converged": True, "final_unfrozen_blocks": 20,
        "stop_reason": recovery.STOP_REASON,
    }))
    assert recovery.converged_meta(path) is not None


def test_validation_audit_list_preserves_truthiness():
    values = ValidationAuditList([{"id": "record"}])
    assert bool(values) is True
    assert values.truth_checks == 1
    values.clear()
    assert bool(values) is False


def test_evaluation_is_reattached_to_separate_output(tmp_path):
    command = recovery.evaluation_command(tmp_path / "artifacts", tmp_path / "evaluation")
    assert command[2] == "all"
    assert str(tmp_path / "evaluation") in command
    assert str(tmp_path / "artifacts") in command
