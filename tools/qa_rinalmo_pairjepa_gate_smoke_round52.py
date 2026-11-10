import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from rnajepa.decision_head import FlatDecisionHead
from structrfm_round48 import forbid_pairjepa_teacher_forward


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    meta = json.loads((root / "run_meta.json").read_text())
    assert meta["arm"] == "round52_rinalmo_pairjepa_gate_smoke" and meta["seed"] == 42
    assert meta["status"] == "completed" and meta["steps_completed"] == 1
    assert meta["stop_reason"] == "step_cap" and not meta["converged"]
    assert meta["schedule"]["steps"] == 1 and meta["schedule"]["batch_size"] == 1
    assert meta["objective"]["nll_normalization"] == "sum"
    assert meta["objective"]["lambda_jepa"] == 0.1
    assert meta["objective"]["jepa_target"] == "ema"
    assert meta["scorer"] == "pairjepa" and meta["pair_latent_dim"] == 490
    assert meta["environment"]["resolved_device"].startswith("cuda")
    assert meta["backbone_report"]["weights"] == "/root/rna_jepa_data/weights/rinalmo-giga"
    train = [json.loads(line) for line in (root / "train_log.jsonl").read_text().splitlines()]
    diagnostics = [json.loads(line) for line in (root / "jepa_diagnostics.jsonl").read_text().splitlines()]
    assert len(train) == len(diagnostics) == 1
    row = diagnostics[0]
    assert row["step"] == 1 and row["masked_pairs"] > 0
    assert math.isclose(row["teacher_student_mse"], row["loss"], rel_tol=0, abs_tol=1e-8)
    assert len(row["target_channel_variance"]) == len(row["prediction_channel_variance"]) == 490
    gradient = row["direct_nll_gradient"]
    assert math.isfinite(gradient["cosine"]) and -1 <= gradient["cosine"] <= 1
    assert gradient["nll_gradient_norm"] > 0 and gradient["jepa_gradient_norm"] > 0
    assert gradient["shared_gradient_elements"] > 0
    for key in ("target_effective_rank", "prediction_effective_rank",
                "target_cross_position_cosine", "prediction_cross_position_cosine",
                "target_cross_sequence_cosine", "prediction_cross_sequence_cosine"):
        assert math.isfinite(row[key])
    checkpoint = torch.load(root / "resume.pt", map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 1 and checkpoint["stream_state"]
    head_state = checkpoint["model"]["head"]
    changed = [key for key in head_state if key.startswith("pair_jepa.scorer.")
               and not torch.equal(head_state[key], head_state[key.replace("scorer", "teacher", 1)])]
    assert changed, "EMA must differ from the student after the optimizer step"
    head = FlatDecisionHead(d_model=1280, d_z=128, hidden=64, scorer="pairjepa",
                            chunk_size=0, pair_latent_dim=490, jepa_use_ema=False,
                            ripe_mode="r6")
    head.load_state_dict(head_state, strict=True)
    assert forbid_pairjepa_teacher_forward(head)
    head = head.cuda().eval()
    z = torch.randn(1, 4, 4, 128, device="cuda")
    ids = torch.ones(1, 4, dtype=torch.long, device="cuda")
    calls = {"student": 0}
    original = head.pair_jepa.scorer.forward
    def counted(*values, **kwargs):
        calls["student"] += 1
        return original(*values, **kwargs)
    head.pair_jepa.scorer.forward = counted
    with torch.no_grad():
        head.score_pair_matrix(z, ids, presymmetrized=True)
    assert calls["student"] == 1
    try:
        head.masked_pair_latent_loss(z, torch.ones(1, 4, 4, dtype=torch.bool, device="cuda"))
    except RuntimeError as error:
        assert "teacher forward is forbidden" in str(error)
    else:
        raise AssertionError("teacher guard did not fail closed")
    result = {
        "status": "PASS_RINALMO_PAIRJEPA_ONE_STEP_ENGINEERING_GATE",
        "scientific_conclusion": False, "allow_pairjepa_training": False,
        "run_meta_sha256": sha256_file(root / "run_meta.json"),
        "resume_sha256": sha256_file(root / "resume.pt"),
        "train_log_sha256": sha256_file(root / "train_log.jsonl"),
        "diagnostics_sha256": sha256_file(root / "jepa_diagnostics.jsonl"),
        "qa_sha256": sha256_file(__file__),
        "device": "cuda", "gpu_name": torch.cuda.get_device_name(0),
        "masked_pairs": row["masked_pairs"], "direct_nll_gradient": gradient,
        "ema_changed_tensors": len(changed), "teacher_forward_calls_at_inference": 0,
        "student_score_calls": calls["student"], "peak_gb": train[0]["peak_gb"],
        "limitations": [
            "One-step engineering smoke on VL0 assets only; no convergence, model selection, or structure-performance claim.",
            "This is not the J1 structured-block arm and not a three-seed experiment.",
            "Full J-gate also requires the frozen capacity decision and independently reviewed training contract.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
