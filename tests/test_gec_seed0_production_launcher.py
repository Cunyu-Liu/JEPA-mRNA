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


def _smoke_document(tmp_path):
    manifest = {
        name: {"sha256": production.sha256(production.REPO / name)}
        for name in (
            "tools/train_plan_a_gec_convergence_v1.py",
            "src/rnajepa/gec_context.py",
        )
    }
    document = {
        "status": "PASS_ENGINEERING_SMOKE_ONLY", "cpu_fallback": False,
        "code_manifest": manifest,
        "arms": {
            arm: {"status": "engineering_smoke_completed", "peak_cuda_bytes": 2748794880}
            for arm in production.ARMS
        },
    }
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(document))
    return path


def test_measured_smoke_supports_four_and_reduces_parallelism_without_cpu(tmp_path):
    smoke = _smoke_document(tmp_path)
    gpu = {"index": 0, "name": "synthetic", "total_mib": 32607, "used_mib": 2}
    assert production.plan_parallelism(smoke, gpu)["parallelism"] == 4
    constrained = dict(gpu, total_mib=13000, used_mib=2)
    plan = production.plan_parallelism(smoke, constrained)
    assert 1 <= plan["parallelism"] < 4
    assert plan["device"] == "cuda:0" and plan["cpu_fallback"] is False
    with pytest.raises(RuntimeError, match="does not support even one"):
        production.plan_parallelism(smoke, dict(gpu, total_mib=5000))


def test_stale_smoke_code_cannot_determine_parallelism(tmp_path):
    smoke = _smoke_document(tmp_path)
    document = json.loads(smoke.read_text())
    name = next(iter(document["code_manifest"]))
    document["code_manifest"][name]["sha256"] = "0" * 64
    smoke.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="does not match current code"):
        production.plan_parallelism(smoke, {"total_mib": 32607, "used_mib": 2})


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
