import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import run_gec_seed0_production_v1 as production


def test_production_commands_are_full_seed0_cuda_without_step_cap(tmp_path):
    for arm in production.ARMS:
        command = production.build_arm_command(arm, tmp_path)
        assert command[command.index("--device") + 1] == "cuda:0"
        assert command[command.index("--steps") + 1] == "0"
        assert command[command.index("--seed") + 1] == "0"
        assert command[command.index("--train-record-limit") + 1] == "0"
        assert command[command.index("--validation-record-limit") + 1] == "0"
        assert "--engineering-smoke" not in command
        assert command[command.index("--out") + 1] == str(tmp_path / arm)
    evaluator = production.build_evaluator_command(tmp_path)
    assert evaluator[evaluator.index("--panel-id") + 1] == "tr1c_cluster80_dev_v1"
    assert sum(item == "--prediction" for item in evaluator) == 4
    assert sum(item == "--run-metadata" for item in evaluator) == 4


def _long_run_audit(tmp_path):
    document = {
        "schema_version": "foldnar.independent_gec_seed0_live_integrity_audit_20261124.v1",
        "artifact_root": str(production.LONG_RUN_AUDIT.parent), "scientific_result": False,
        "arms": {
            arm: {
                "completed_steps": 57,
                "cuda_oom_observed": arm in ("destroy_interval_shuffle", "independent_context"),
                "training": {"cuda_peak_bytes": {"max": (
                    7723756544 if arm in ("matched_control", "ensemble_context") else 6659579392
                )}},
            } for arm in production.ARMS
        },
    }
    path = tmp_path / "long_run_audit.json"
    path.write_text(json.dumps(document))
    return path


def test_oom_long_run_peaks_yield_two_slots_and_two_explicit_waves(tmp_path):
    audit = _long_run_audit(tmp_path)
    gpu = {"index": 0, "name": "synthetic", "total_mib": 32607, "used_mib": 2}
    plan = production.plan_parallelism(audit, gpu)
    assert plan["parallelism"] == 2
    assert plan["waves"] == [list(production.ARMS[:2]), list(production.ARMS[2:])]
    assert plan["memory_safety_factor"] == 1.5
    assert plan["max_long_run_allocated_bytes"] == 7723756544
    assert plan["device"] == "cuda:0" and plan["cpu_fallback"] is False
    constrained = dict(gpu, total_mib=24000, used_mib=2)
    assert production.plan_parallelism(audit, constrained)["parallelism"] == 1
    with pytest.raises(RuntimeError, match="does not support even one"):
        production.plan_parallelism(audit, dict(gpu, total_mib=12000))


def test_invalid_long_run_audit_cannot_set_parallelism(tmp_path):
    audit = _long_run_audit(tmp_path)
    document = json.loads(audit.read_text())
    document["arms"]["matched_control"]["training"]["cuda_peak_bytes"]["max"] = 0
    audit.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="insufficient long-run memory evidence"):
        production.plan_parallelism(audit, {"total_mib": 32607, "used_mib": 2})


class _FakeProcess:
    def __init__(self, arm, completed, returncode=0):
        self.arm = arm
        self.completed = completed
        self.returncode = returncode
        self.pid = 10000 + production.ARMS.index(arm)
        self.poll_count = 0
        self.terminated = False

    def poll(self):
        self.poll_count += 1
        if self.poll_count == 1:
            return None
        if self.returncode == 0:
            self.completed.add(self.arm)
        return self.returncode

    def send_signal(self, signal):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def _patch_converged(completed):
    def check(root, arm):
        if arm not in completed:
            raise ValueError("not converged")
        return {"arm": arm}
    return patch.object(production, "converged_metadata", side_effect=check)


def test_wave_scheduling_is_two_then_two_without_duplicate_steps(tmp_path):
    completed = set()
    processes = []
    active = set()
    maximum_active = 0

    def popen(command, **kwargs):
        arm = command[command.index("--arm") + 1]
        process = _FakeProcess(arm, completed)
        processes.append(process)
        active.add(arm)
        nonlocal maximum_active
        maximum_active = max(maximum_active, len(active))
        original_poll = process.poll
        def poll():
            result = original_poll()
            if result is not None:
                active.discard(arm)
            return result
        process.poll = poll
        return process

    with _patch_converged(completed), \
         patch.object(production.subprocess, "Popen", side_effect=popen), \
         patch.object(production.time, "sleep", return_value=None):
        production.run_training(tmp_path, 2)
    assert [process.arm for process in processes] == list(production.ARMS)
    assert maximum_active == 2
    assert all(not process.terminated for process in processes)
    for arm in production.ARMS:
        status = json.loads((tmp_path / arm / "launcher_status.json").read_text())
        assert status["selected_parallelism"] == 2
        assert status["wave"] == (1 if arm in production.ARMS[:2] else 2)
    with _patch_converged(completed), patch.object(production.subprocess, "Popen") as no_restart:
        production.run_training(tmp_path, 2)
    no_restart.assert_not_called()


def test_peer_failure_stops_wave_and_never_launches_remaining_arms(tmp_path):
    processes = []
    def popen(command, **kwargs):
        arm = command[command.index("--arm") + 1]
        process = _FakeProcess(arm, set(), 1 if arm == "ensemble_context" else 0)
        if arm == "matched_control":
            process.returncode = None
        processes.append(process)
        return process

    def fake_terminate(active, root, reason):
        for arm, item in active.items():
            item["process"].send_signal(15)
            item["log"].close()
            production.release_lock(item["lock"])
            production.write_arm_status(root, arm, "failed", returncode=-15, failure_reason=reason)

    with _patch_converged(set()), \
         patch.object(production.subprocess, "Popen", side_effect=popen), \
         patch.object(production, "terminate_processes", side_effect=fake_terminate), \
         patch.object(production.time, "sleep", return_value=None):
        with pytest.raises(RuntimeError, match="failed.*no evaluation invoked"):
            production.run_training(tmp_path, 2)
    assert len(processes) == 2
    assert not (tmp_path / production.ARMS[2] / "production.pid").exists()
    assert not (tmp_path / production.ARMS[3] / "production.pid").exists()


def test_v3_root_is_fresh_and_never_reads_failed_run_artifacts(tmp_path):
    assert str(production.ARTIFACT_ROOT).endswith("/gec_seed0_v3")
    assert production.ARTIFACT_ROOT != production.LONG_RUN_AUDIT.parent
    for arm in production.ARMS:
        command = production.build_arm_command(arm)
        assert str(production.ARTIFACT_ROOT / arm) in command
        assert str(production.ARTIFACT_ROOT / "gec_shared_residual_init_seed0.pt") in command
        assert not any("/gec_seed0_v1/" in item or "/gec_seed0_v2/" in item for item in command)


def test_evaluator_uses_real_v2_run_metadata_without_adapter(tmp_path):
    command = production.build_evaluator_command(tmp_path)
    metadata_args = [command[index + 1] for index, value in enumerate(command) if value == "--run-metadata"]
    assert len(metadata_args) == 4
    for arm, item in zip(production.ARMS, metadata_args):
        assert item == f"{arm}={tmp_path / arm / 'run_meta.json'}"
        assert "evaluator_run_meta" not in item


def test_evaluator_gate_runs_only_after_all_four_converge(tmp_path):
    with patch.object(production.subprocess, "run") as run:
        with pytest.raises((OSError, ValueError, KeyError)):
            production.run_scientific_evaluator(tmp_path)
    run.assert_not_called()


def test_supervisor_and_arm_locks_are_nonduplicate(tmp_path):
    path = tmp_path / "arm.lock"
    held = production.acquire_lock(path)
    try:
        with pytest.raises(RuntimeError, match="another production process"):
            production.acquire_lock(path)
    finally:
        production.release_lock(held)
    reacquired = production.acquire_lock(path)
    production.release_lock(reacquired)


def test_dry_run_does_not_create_artifact_root_or_spawn_training():
    report = {
        "status": "DRY_RUN_NO_TRAINING", "artifact_root_created": False,
        "parallelism_plan": {"parallelism": 4},
    }
    with patch.object(production, "dry_run", return_value=report), \
         patch.object(production.subprocess, "Popen") as popen, \
         patch.object(production, "prepare_artifact_root") as prepare:
        assert production.main(["--dry-run"]) == 0
    popen.assert_not_called()
    prepare.assert_not_called()


def test_new_git_head_is_frozen_without_hardcoded_commit(tmp_path):
    plan = {"parallelism": 4}
    with patch.object(production, "require_clean_scientific_files"), \
         patch.object(production, "current_git_head", return_value="4a66d6e-new-reviewed-head"), \
         patch.object(production, "code_manifest", return_value={}), \
         patch.object(production, "sha256", return_value="frozen-hash"):
        identity = production.build_identity(tmp_path, plan)
        assert identity["git_commit"] == "4a66d6e-new-reviewed-head"
        production.validate_identity(identity, tmp_path)
    with patch.object(production, "require_clean_scientific_files"), \
         patch.object(production, "current_git_head", return_value="different-head"), \
         patch.object(production, "code_manifest", return_value={}), \
         patch.object(production, "sha256", return_value="frozen-hash"):
        with pytest.raises(ValueError, match="frozen production identity mismatch.*git_commit"):
            production.validate_identity(identity, tmp_path)


def test_dirty_scientific_or_launcher_files_are_rejected():
    with patch.object(production.subprocess, "check_output", return_value=" M tools/gec_seed0_scientific_evaluator.py\n"):
        with pytest.raises(ValueError, match="must be clean"):
            production.require_clean_scientific_files()
    with patch.object(production.subprocess, "check_output", return_value=""):
        production.require_clean_scientific_files()


def test_dirty_files_fail_before_artifact_root_creation():
    with patch.object(production, "require_clean_scientific_files", side_effect=ValueError("dirty")), \
         patch.object(production.Path, "mkdir") as mkdir:
        with pytest.raises(ValueError, match="dirty"):
            production.prepare_artifact_root()
    mkdir.assert_not_called()
