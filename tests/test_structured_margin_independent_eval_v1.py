from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/structured_margin_independent_eval_v1.py"
SPEC = importlib.util.spec_from_file_location("structured_margin_independent_eval_v1", MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
I = M.independent
REPO = MODULE_PATH.parents[1]
CONFIG_PATH = REPO / "configs/gec_structured_margin_convergence_v1.json"
CONFIG = json.loads(CONFIG_PATH.read_text())


def validation_rows():
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
            "bad_evaluations": bad, "validation_metric": M.MARGIN_METRIC,
        })
    return rows


def test_raw_ledger_stop_is_independent_and_incomplete_is_in_progress(tmp_path):
    arm_dir = tmp_path / "matched_control"
    arm_dir.mkdir()
    result = M.audit_validation_arm(arm_dir, "matched_control", CONFIG)
    assert result["status"] == "IN_PROGRESS"
    validation = arm_dir / "validation_log.jsonl"
    validation.write_text("")
    assert M.audit_validation_arm(arm_dir, "matched_control", CONFIG)["status"] == "IN_PROGRESS"
    rows = validation_rows()
    validation.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (arm_dir / "train_log.jsonl").write_text("".join(json.dumps({
        "step": step, "optimizer_steps": step, "optimizer_step": True, "divergence_reason": "",
    }) + "\n" for step in range(1, 13001)))
    (arm_dir / "run_meta.json").write_text("deliberately invalid and never read")
    result = M.audit_validation_arm(arm_dir, "matched_control", CONFIG)
    assert result["status"] == "STOPPED" and result["pass"]
    assert result["reconstruction"]["best_step"] == 10500
    assert result["reconstruction"]["stop_step"] == 13000
    assert result["reconstruction"]["bad_evaluations"] == 5
    changed = copy.deepcopy(rows)
    changed[-1]["bad_evaluations"] = 0
    validation.write_text("".join(json.dumps(row) + "\n" for row in changed))
    assert M.audit_validation_arm(arm_dir, "matched_control", CONFIG)["status"] == "INVALID"


def test_all_arm_progress_precedes_checkpoint_deserialization(tmp_path):
    with pytest.raises(M.TrainingInProgress) as caught:
        M.audit_validation(tmp_path, CONFIG)
    assert set(caught.value.arms) == set(M.ARMS)
    assert all(row["status"] == "IN_PROGRESS" for row in caught.value.arms.values())


def synthetic_checkpoint_audit(tmp_path):
    reconstruction = I.reconstruct_validation(validation_rows())
    state = {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([3.0])}
    model = {name: copy.deepcopy(state) for name in ("encoder", "head", "residual")}
    hashes = {name: I.tensor_state_sha256(value) for name, value in model.items()}
    run_identity = {
        "schema_version": M.MARGIN_IDENTITY, "arm": "matched_control",
        "engineering_smoke": False, "gec_config_sha256": I.sha256(CONFIG_PATH),
        "source_contract_sha256": I.sha256(CONFIG["source_contract"]),
        "candidate_contract_sha256": I.sha256(REPO / CONFIG["candidate_contract"]),
        "validation_metric": M.MARGIN_METRIC, "objective": CONFIG["objective"],
        "terminal_unfrozen_blocks": 20, "validation_record_ids": [],
    }
    common = {"format": M.MARGIN_CHECKPOINT, "step": 10500, "optimizer_steps": 10500,
              "n_unfrozen": 20, "run_identity": run_identity, "model_tensor_sha256": hashes,
              **model}
    best_convergence = dict(reconstruction["final_convergence"], bad_evaluations=0)
    torch.save({**common, "checkpoint_phase": "best_eligible", "convergence": best_convergence,
                "validation_loss": 0.9, "model": model}, tmp_path / "best.pt")
    torch.save({**common, "checkpoint_phase": "restored_best_eligible", "terminal_status": "converged",
                "convergence": dict(best_convergence)}, tmp_path / "resume.pt")
    return reconstruction, run_identity


def test_margin_checkpoint_content_and_identity_are_cpu_audited(tmp_path):
    reconstruction, _ = synthetic_checkpoint_audit(tmp_path)
    result = M.audit_checkpoint_arm(tmp_path, "matched_control", reconstruction, CONFIG, CONFIG_PATH, [])
    assert result["pass"], result["errors"]
    assert result["best_tensor_sha256"] == result["resume_tensor_sha256"]
    resume = torch.load(tmp_path / "resume.pt", map_location="cpu", weights_only=False)
    resume["encoder"]["weight"] = torch.tensor([8.0, 8.0])
    torch.save(resume, tmp_path / "resume.pt")
    result = M.audit_checkpoint_arm(tmp_path, "matched_control", reconstruction, CONFIG, CONFIG_PATH, [])
    assert not result["pass"]
    assert any("tensors differ" in error or "content hashes" in error for error in result["errors"])


def one_record():
    return I.Record("s", "RF", 1, 11, hashlib.sha256(b"s").hexdigest(), ((0, 10),))


def test_margin_prediction_export_binds_checkpoint_ledger_truth_and_mea(tmp_path):
    record = one_record()
    best = tmp_path / "best.pt"
    best.write_bytes(b"best")
    best_hash = I.sha256(best)
    validation_hash = hashlib.sha256(b"ledger").hexdigest()
    checkpoints = {arm: {"best_checkpoint": {"sha256": best_hash}, "best_tensor_sha256": {"encoder": arm},
                         "validation_log_sha256": validation_hash} for arm in M.ARMS}
    root = tmp_path / "predictions"
    for arm in M.ARMS:
        directory = root / arm
        directory.mkdir(parents=True)
        row = {"sequence_id": "s", "sequence_length": 11, "sequence_sha256": record.sequence_sha256,
               "true_pairs": [[0, 10]], "predicted_pairs": [[0, 10]], "checkpoint_sha256": best_hash}
        prediction = directory / "dev_predictions.jsonl"
        prediction.write_text(json.dumps(row) + "\n")
        export = {"schema_version": M.EXPORT_SCHEMA, "arm": arm,
                  "panel_id": "tr1c_cluster80_dev_v1", "record_count": 1,
                  "best_checkpoint_sha256": best_hash, "selected_model_tensor_sha256": {"encoder": arm},
                  "device": "cuda:0", "cpu_fallback": False, "prediction_sha256": I.sha256(prediction),
                  "decoder": CONFIG["model"]["decoder"], "validation_log_sha256": validation_hash,
                  "external_test_sets": False}
        (directory / "prediction_export.json").write_text(json.dumps(export))
    bound, _ = M.bind_margin_predictions([record], root, checkpoints)
    assert bound["ensemble_context"][0].predicted_pairs == ((0, 10),)
    export_path = root / "matched_control/prediction_export.json"
    export = json.loads(export_path.read_text())
    export["validation_log_sha256"] = "wrong"
    export_path.write_text(json.dumps(export))
    with pytest.raises(ValueError, match="validation_log_sha256 mismatch"):
        M.bind_margin_predictions([record], root, checkpoints)


def test_qualified_nll_raw_predictions_are_hash_checked_and_bound(tmp_path):
    record = one_record()
    inputs = {}
    for arm in M.ARMS:
        best = tmp_path / f"{arm}_best.pt"
        best.write_bytes(arm.encode())
        prediction = tmp_path / f"{arm}.jsonl"
        prediction.write_text(json.dumps({
            "sequence_id": "s", "sequence_length": 11, "sequence_sha256": record.sequence_sha256,
            "true_pairs": [[0, 10]], "predicted_pairs": [[0, 10]],
            "checkpoint_sha256": I.sha256(best),
        }) + "\n")
        inputs[f"{arm}:dev_predictions"] = I.identity(prediction)
        inputs[f"{arm}:best_checkpoint"] = I.identity(best)
    input_manifest = tmp_path / "input_manifest.json"
    input_manifest.write_text(json.dumps({"inputs": inputs}))
    summary = {"schema_version": M.NLL_SCHEMA, "qualification_pass": True,
               "status": "PASS_NLL_V3_REQUALIFIED",
               "reference_use": {"historical_nll_four_arm_structure_reference_eligible": True,
                                 "margin_superior_to_nll_claim_allowed": False},
               "per_arm": {arm: {"qualification": "QUALIFIED_STRUCTURE_REFERENCE"} for arm in M.ARMS}}
    summary_path = tmp_path / "qualification_summary.json"
    summary_path.write_text(json.dumps(summary))
    names = ("input_manifest.json", "qualification_summary.json")
    manifest = {"schema_version": M.NLL_SCHEMA,
                "artifacts": {name: I.identity(tmp_path / name) for name in names}}
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))
    bound, _, provenance = M.load_nll_predictions(REPO, [record], tmp_path)
    assert set(bound) == set(M.ARMS)
    assert provenance["same_frozen_mea"] and not provenance["validation_loss_comparison_allowed"]
    summary["qualification_pass"] = False
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="artifact changed"):
        M.load_nll_predictions(REPO, [record], tmp_path)


def science_records():
    truth = tuple((20 * index, 20 * index + 16) for index in range(17))
    return [I.Record(f"RF{family}:s{index}", f"RF{family}", family * 30 + index, 400,
                     hashlib.sha256(f"RF{family}:s{index}".encode()).hexdigest(), truth)
            for family in range(10) for index in range(30)]


def test_synthetic_full_science_and_same_arm_nll_comparison():
    frozen = science_records()
    margin = {arm: [replace(record, predicted_pairs=(record.true_pairs if arm == "ensemble_context"
                    else record.true_pairs[:-1] + ((360, 380),))) for record in frozen]
              for arm in M.ARMS}
    nll = {arm: [replace(record, predicted_pairs=record.true_pairs[:-2]) for record in frozen]
           for arm in M.ARMS}
    result = M.evaluate_bound(frozen, margin, nll)
    assert result["margin_gates"]["pass"], result["margin_gates"]["failed_gates"]
    assert len(result["margin_gates"]["gates"]) == 16
    matched = result["margin_vs_qualified_nll_same_arm"]["matched_control"]
    assert matched["metrics"]["micro_pair_f1"]["delta_margin_minus_nll"] > 0
    assert "paired_ci95" in matched["metrics"]["macro_stem_f1"]
    assert "family_cluster_sensitivity_ci95" in matched["subgroups"]["isolated_pair:isolated"]["recall"]
    assert result["bootstrap"]["family_cluster_sensitivity"]["role"].startswith("SENSITIVITY_ONLY")


def test_in_progress_final_eval_never_reads_exposure_registry(monkeypatch, tmp_path):
    def incomplete(repo, artifact_root):
        raise M.TrainingInProgress({arm: {"status": "IN_PROGRESS", "reason": "fixture"} for arm in M.ARMS})

    def forbidden_registry_read(*args, **kwargs):
        raise AssertionError("exposure registry was read before terminal qualification")

    monkeypatch.setattr(M, "qualify", incomplete)
    monkeypatch.setattr(M.exposure, "bind_exposure_registry", forbidden_registry_read)
    with pytest.raises(M.TrainingInProgress):
        M.evaluate_run(REPO, tmp_path / "artifacts", tmp_path / "predictions",
                       tmp_path / "nll", tmp_path / "output", tmp_path / "missing-exposure")


def legacy_all_args(tmp_path):
    return [
        "all", "--repo", str(REPO), "--artifact-root", str(tmp_path / "artifacts"),
        "--nll-artifact-root", str(tmp_path / "nll"), "--output-root", str(tmp_path / "output"),
    ]


def test_legacy_all_command_uses_frozen_exposure_default(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(M, "export_run", lambda *args: {})

    def fake_evaluate(repo, artifact_root, prediction_root, nll_root, output_root, exposure_root):
        captured["exposure_root"] = exposure_root
        return {"overall_pass": True}

    monkeypatch.setattr(M, "evaluate_run", fake_evaluate)
    assert M.main(legacy_all_args(tmp_path)) == 0
    assert captured["exposure_root"] == M.exposure.DEFAULT_EXPOSURE_ROOT
    assert M.exposure.DEFAULT_EXPOSURE_ROOT == Path(
        "/home/rna_jev_pretrain_exposure/manifests/pretrain_exposure_v1"
    )
    assert M.exposure.SOURCE_COMMIT == "4247d1e4a971ceb8c63e3f8338e026a23c18941b"


def test_legacy_all_command_in_progress_precedes_export_and_registry(monkeypatch, tmp_path):
    def incomplete(repo, artifact_root):
        raise M.TrainingInProgress({arm: {"status": "IN_PROGRESS", "reason": "fixture"} for arm in M.ARMS})

    monkeypatch.setattr(M, "qualify", incomplete)
    monkeypatch.setattr(M.exposure, "bind_exposure_registry", lambda *args: pytest.fail("registry read"))
    assert M.main(legacy_all_args(tmp_path)) == 3
    assert (tmp_path / "output/in_progress.json").is_file()
    assert not (tmp_path / "output/exposure_sensitivity.json").exists()
