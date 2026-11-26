from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/gec_nll_v3_requalification_independent_v1.py"
SPEC = importlib.util.spec_from_file_location("gec_nll_v3_requalification_independent_v1", MODULE_PATH)
I = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = I
SPEC.loader.exec_module(I)


def make_ledger(tmp_path):
    rows = []
    best, best_step, bad = math.inf, None, 0
    for index in range(26):
        step = (index + 1) * 500
        n_unfrozen = min(20, max(0, ((step - 1600) // 800) * 2))
        eligible = step >= 9600 and n_unfrozen == 20
        loss = 1.0 if step < 10000 else {10000: 1.0, 10500: 0.9}.get(step, 0.95)
        improved = eligible and loss < best - 1e-6
        if improved:
            best, best_step, bad = loss, step, 0
        elif eligible:
            bad += 1
        rows.append({"step": step, "validation_loss": loss, "n_unfrozen": n_unfrozen,
                     "eligible": eligible, "improved": improved, "save_best": improved,
                     "best_step": best_step, "best_validation_loss": None if best_step is None else best,
                     "bad_evaluations": bad, "stop": eligible and bad >= 5})
    (tmp_path / "validation_log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "train_log.jsonl").write_text("".join(
        json.dumps({"step": step, "optimizer_steps": step, "optimizer_step": True}) + "\n"
        for step in range(1, 13001)
    ))
    return rows


def test_second_ledger_recheck_and_tamper_detection(tmp_path):
    rows = make_ledger(tmp_path)
    result = I.ledger_recheck(tmp_path / "validation_log.jsonl")
    assert result["pass"], result["errors"]
    assert (result["stop_step"], result["best_step"], result["bad_evaluations"]) == (13000, 10500, 5)
    rows[-1]["bad_evaluations"] = 0
    (tmp_path / "validation_log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = I.ledger_recheck(tmp_path / "validation_log.jsonl")
    assert not result["pass"]


def test_direct_tensor_recheck_does_not_use_primary_hash_code(tmp_path):
    rows = make_ledger(tmp_path)
    ledger = I.ledger_recheck(tmp_path / "validation_log.jsonl")
    model = {name: {"weight": torch.tensor([float(index), 2.0])}
             for index, name in enumerate(("encoder", "head", "residual"))}
    identity = {"arm": "matched_control", "code_manifest": {}}
    common = {"format": "foldnar.gec_checkpoint.v2", "step": 10500, "optimizer_steps": 10500,
              "n_unfrozen": 20, "run_identity": identity, **model}
    best_convergence = dict(ledger["final_convergence"], bad_evaluations=0)
    torch.save({**common, "checkpoint_phase": "best_eligible", "convergence": best_convergence,
                "model": model}, tmp_path / "best.pt")
    torch.save({**common, "checkpoint_phase": "restored_best_eligible", "convergence": best_convergence},
               tmp_path / "resume.pt")
    run_meta = {"convergence": best_convergence, "run_identity": identity,
                "steps_completed": 13000, "training_stopped_step": 13000,
                "training_stopped_optimizer_steps": 13000, "optimizer_steps": 10500,
                "restored_best_step": 10500, "status": "converged", "converged": True,
                "best_checkpoint_sha256": I.sha256(tmp_path / "best.pt"),
                "resume_checkpoint_sha256": I.sha256(tmp_path / "resume.pt")}
    result = I.direct_tensor_recheck(tmp_path / "best.pt", tmp_path / "resume.pt", ledger, run_meta, "matched_control")
    assert result["pass"], result["errors"]
    assert result["all_model_tensors_equal"] and result["sole_defect_is_restored_bad_counter"]
    resume = torch.load(tmp_path / "resume.pt", map_location="cpu", weights_only=False)
    resume["head"]["weight"][0] = 99
    torch.save(resume, tmp_path / "resume.pt")
    run_meta["resume_checkpoint_sha256"] = I.sha256(tmp_path / "resume.pt")
    result = I.direct_tensor_recheck(tmp_path / "best.pt", tmp_path / "resume.pt", ledger, run_meta, "matched_control")
    assert not result["pass"] and not result["tensor_checks"]["head"]["equal"]


def test_second_prediction_binding_uses_raw_truth_and_best(tmp_path):
    best = tmp_path / "best.pt"
    best.write_bytes(b"best")
    best_hash = I.sha256(best)
    predictions = tmp_path / "dev_predictions.jsonl"
    row = {"sequence_id": "s", "sequence_length": 11,
           "sequence_sha256": "sequence", "true_pairs": [[0, 10]],
           "predicted_pairs": [[0, 10]], "checkpoint_sha256": best_hash}
    predictions.write_text(json.dumps(row) + "\n")
    export = {"arm": "matched_control", "panel_id": "tr1c_cluster80_dev_v1", "record_count": 1,
              "best_checkpoint_sha256": best_hash, "prediction_sha256": I.sha256(predictions),
              "decoder": I.EXPECTED_DECODER, "external_test_sets": False}
    (tmp_path / "prediction_export.json").write_text(json.dumps(export))
    raw = {"s": {"sequence_length": 11, "sequence_sha256": "sequence", "true_pairs": [(0, 10)]}}
    result = I.prediction_recheck(tmp_path, "matched_control", raw, {})
    assert result["pass"], result["errors"]
    row["checkpoint_sha256"] = "resume"
    predictions.write_text(json.dumps(row) + "\n")
    export["prediction_sha256"] = I.sha256(predictions)
    (tmp_path / "prediction_export.json").write_text(json.dumps(export))
    result = I.prediction_recheck(tmp_path, "matched_control", raw, {})
    assert not result["pass"] and any("not bound to best" in error for error in result["errors"])
