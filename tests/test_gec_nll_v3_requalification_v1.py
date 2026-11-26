from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/gec_nll_v3_requalification_v1.py"
SPEC = importlib.util.spec_from_file_location("gec_nll_v3_requalification_v1", MODULE_PATH)
R = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)


def validation_reconstruction():
    rows = []
    best = float("inf")
    best_step = None
    bad = 0
    for index in range(26):
        step = (index + 1) * 500
        loss = 1.0 if step < 10000 else {10000: 1.0, 10500: 0.9}.get(step, 0.95)
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
            "stop": eligible and bad >= 5, "best_step": best_step,
            "best_validation_loss": None if best_step is None else best,
            "bad_evaluations": bad,
        })
    result = R.reconstruct_validation(rows)
    assert result["mismatches"] == []
    return result


def test_raw_ledger_is_authoritative_and_only_bad_counter_is_defective():
    reconstruction = validation_reconstruction()
    expected = reconstruction["final_convergence"]
    observed = dict(expected, bad_evaluations=0)
    defect = R.run_meta_defect({"convergence": observed}, observed, expected)
    assert defect["sole_qualification_defect"]
    assert defect["field_differences"] == {
        "bad_evaluations": {"observed": 0, "ledger_expected": 5}
    }
    other = dict(observed, best_step=10000)
    defect = R.run_meta_defect({"convergence": other}, other, expected)
    assert not defect["sole_qualification_defect"]
    assert set(defect["field_differences"]) == {"bad_evaluations", "best_step"}


def test_checkpoint_pair_allows_only_documented_metadata_defect(tmp_path):
    reconstruction = validation_reconstruction()
    state = {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([3.0])}
    model = {name: copy.deepcopy(state) for name in ("encoder", "head", "residual")}
    hashes = {name: R.tensor_state_sha256(value) for name, value in model.items()}
    run_identity = {
        "arm": "matched_control", "gec_config_sha256": "config",
        "source_contract_sha256": "source", "engineering_smoke": False,
        "terminal_unfrozen_blocks": 20,
    }
    common = {
        "format": "foldnar.gec_checkpoint.v2", "step": 10500, "optimizer_steps": 10500,
        "n_unfrozen": 20, "run_identity": run_identity, "model_tensor_sha256": hashes,
        **model,
    }
    best_convergence = dict(reconstruction["final_convergence"], bad_evaluations=0)
    torch.save({
        **common, "checkpoint_phase": "best_eligible", "convergence": best_convergence,
        "validation_loss": 0.9, "model": model,
    }, tmp_path / "best.pt")
    torch.save({
        **common, "checkpoint_phase": "restored_best_eligible", "terminal_status": "converged",
        "convergence": dict(best_convergence), "model": model,
    }, tmp_path / "resume.pt")
    run_meta = {
        "schema_version": "foldnar.gec_training_run.v2", "arm": "matched_control",
        "status": "converged", "converged": True, "steps_completed": 13000,
        "training_stopped_step": 13000, "training_stopped_optimizer_steps": 13000,
        "optimizer_steps": 10500, "restored_best_step": 10500, "final_unfrozen_blocks": 20,
        "stop_reason": "CONVERGED_VALIDATION_NLL_AFTER_TERMINAL_UNFREEZE",
        "best_checkpoint_sha256": R.sha256(tmp_path / "best.pt"),
        "resume_checkpoint_sha256": R.sha256(tmp_path / "resume.pt"),
        "final_residual_tensor_sha256": hashes["residual"], "run_identity": run_identity,
        "convergence": dict(best_convergence),
    }
    result = R.audit_checkpoint_pair(tmp_path, "matched_control", reconstruction, run_meta, "config", "source")
    assert result["pass"], [name for name, row in result["checks"].items() if not row["pass"]]
    assert result["best_tensor_sha256"] == result["resume_tensor_sha256"]
    assert result["run_meta_defect"]["sole_qualification_defect"]

    resume = torch.load(tmp_path / "resume.pt", map_location="cpu", weights_only=False)
    resume["encoder"]["weight"] = torch.tensor([9.0, 9.0])
    torch.save(resume, tmp_path / "resume.pt")
    run_meta["resume_checkpoint_sha256"] = R.sha256(tmp_path / "resume.pt")
    result = R.audit_checkpoint_pair(tmp_path, "matched_control", reconstruction, run_meta, "config", "source")
    assert not result["pass"]
    assert not result["checks"]["best_resume_model_tensor_identity"]["pass"]


def test_prediction_export_requires_best_hash_and_frozen_mea(tmp_path):
    (tmp_path / "dev_predictions.jsonl").write_bytes(b"predictions")
    prediction_hash = R.sha256(tmp_path / "dev_predictions.jsonl")
    export = {
        "schema_version": "foldnar.gec_seed0_prediction_export.v1", "arm": "matched_control",
        "panel_id": "tr1c_cluster80_dev_v1", "record_count": 1,
        "best_checkpoint_sha256": "best", "prediction_sha256": prediction_hash,
        "decoder": R.EXPECTED_DECODER, "external_test_sets": False,
    }
    (tmp_path / "prediction_export.json").write_text(json.dumps(export))
    result = R.audit_prediction_export(tmp_path, "matched_control", "best", prediction_hash, 1)
    assert result["pass"]
    export["best_checkpoint_sha256"] = "resume"
    (tmp_path / "prediction_export.json").write_text(json.dumps(export))
    result = R.audit_prediction_export(tmp_path, "matched_control", "best", prediction_hash, 1)
    assert not result["pass"] and not result["checks"]["best_checkpoint_sha256"]["pass"]


def test_source_identity_recheck_detects_mutation(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    inputs = {"input": R.identity(path)}
    assert R.unchanged_identities(inputs)
    path.write_text("{\"changed\":true}")
    assert not R.unchanged_identities(inputs)
